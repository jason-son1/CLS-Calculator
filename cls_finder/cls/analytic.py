import numpy as np
from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly

# ── Complexity gates (tunable) ────────────────────────────────────────────────
# Symbolic CLS extraction uses exact Laurent/SymPy arithmetic whose cost grows
# super-exponentially with the orbital count Q and the number of hopping terms.
# These ceilings keep a large Hamiltonian from appearing to hang: above them the
# analytic routine degrades gracefully (returns fewer / no gauges) so the
# pipeline falls back to the numerical CLS instead of stalling. Override entries
# of GATES at runtime to trade speed for completeness.
GATES = {
    "max_symbolic_Q":  14,   # above this: skip the symbolic adjugate entirely
    "max_nullspace_Q":  7,   # above this: skip the SymPy nullspace (degenerate)
    "max_syzygy_Q":     5,   # above this: skip the Gröbner syzygy generators
    "max_syzygy_terms": 36,  # total H_k term budget for syzygy
    "verify_max_Q":     8,   # above this: trust adjugate, skip H_bar*x check
    "max_det_terms": 12000,  # Faddeev-LeVerrier intermediate term budget
    "max_det_seconds": 8.0,  # Faddeev-LeVerrier wall-clock backstop (per gauge)
    "max_det_ops": 2_000_000,  # projected term-ops of a single F-L multiply
}


def _total_terms(H_k):
    return sum(len(H_k.data[r][c].coefs)
               for r in range(H_k.rows) for c in range(H_k.cols))


def _eval_laurent_randX(poly, Xvals):
    """Evaluate a LaurentPoly at given complex variable values (no lattice)."""
    v = 0j
    for exp, coef in poly.coefs.items():
        t = complex(coef)
        for l, xv in enumerate(Xvals):
            t *= xv ** exp[l]
        v += t
    return v


def _numeric_screen(H_bar, d, n_samples=3, zero_tol=1e-5):
    """
    Cheap numerical pre-screen on the torus (|X_l| = 1): estimate the kernel
    dimension M of H_bar and rank the guide indices p by how non-singular the
    (p,p) submatrix is. Lets the symbolic routine compute only the gauge(s) it
    actually needs (one suffices when M == 1) instead of all Q determinants.
    """
    Q = H_bar.rows
    rng = np.random.default_rng(20240531)
    M_votes, det_acc = [], np.zeros(Q)
    for _ in range(n_samples):
        Xv = np.exp(1j * rng.uniform(0, 2 * np.pi, size=d))
        Hn = np.array([[_eval_laurent_randX(H_bar.data[r][c], Xv)
                        for c in range(Q)] for r in range(Q)], dtype=complex)
        try:
            ev = np.linalg.eigvalsh((Hn + Hn.conj().T) / 2.0)
        except Exception:
            ev = np.linalg.eigvals(Hn)
        M_votes.append(int(np.sum(np.abs(ev) < zero_tol)))
        for p in range(Q):
            sub = np.delete(np.delete(Hn, p, 0), p, 1)
            det_acc[p] += abs(np.linalg.det(sub)) if sub.size else 0.0
    M = max(1, int(np.median(M_votes))) if M_votes else 1
    return M, list(int(p) for p in np.argsort(-det_acc))


def _adjugate_gauge(H_bar, test_p, Q, d, verify=True):
    """Compute the single minor+adjugate gauge for guide index test_p.
    Returns the x_k vector (list of LaurentPoly) or None."""
    A = H_bar.submatrix(test_p, test_p)
    h_data = [[H_bar.data[r][test_p]] for r in range(Q) if r != test_p]
    h_p = MatrixPoly(h_data, d)

    det_A, adj_A = A.det_and_adjugate(max_terms=GATES["max_det_terms"],
                                      max_seconds=GATES["max_det_seconds"],
                                      max_ops=GATES["max_det_ops"])
    y = adj_A * h_p * -1.0

    test_x = [None] * Q
    test_x[test_p] = det_A
    for r in range(Q):
        if r < test_p:
            test_x[r] = y.data[r][0]
        elif r > test_p:
            test_x[r] = y.data[r - 1][0]

    if all(elem.is_zero(1e-12) for elem in test_x):
        return None
    if verify:
        # H_bar * x == 0 — guaranteed analytically for rank(H_bar)==Q-1, but
        # float Laurent arithmetic can drift. Skipped for large Q (cost ~ a full
        # matrix-poly product) where we trust the exact Faddeev-LeVerrier build.
        x_mat = MatrixPoly([[elem] for elem in test_x], d)
        if not (H_bar * x_mat).is_zero(1e-7):
            return None
    return test_x



def _degenerate_adjugate_gauges(H_bar, M, Q, d, verify=True):
    """
    Generalized minor+adjugate (Cramer's rule) for an M-fold degenerate band.

    The ordinary adjugate gauge collapses when M >= 2 (every (Q-1)x(Q-1) minor
    vanishes). Instead we pick r = Q-M independent rows R and columns C so that
    the r×r submatrix H_bar[R, C] is non-singular, then solve for the dependent
    components in terms of each free column via that submatrix's adjugate. This
    yields M independent polynomial CLS — division-free, FSBP-preserving, and
    far faster than a SymPy nullspace — uniformly for any Q.

    Returns a list of x_k vectors (each a list of LaurentPoly), possibly empty.
    """
    import numpy as np
    r = Q - M
    if r < 1:
        return []

    rng = np.random.default_rng(20240531)
    Xv = np.exp(1j * rng.uniform(0, 2 * np.pi, size=d))
    Hn = np.array([[_eval_laurent_randX(H_bar.data[i][j], Xv) for j in range(Q)]
                   for i in range(Q)], dtype=complex)

    # Independent columns C (rest are free) and independent rows R via
    # rank-revealing pivoted QR, so H_bar[R, C] is the non-singular r×r block.
    try:
        import scipy.linalg as sla
        _, _, pivC = sla.qr(Hn, pivoting=True, mode='economic')
        C = sorted(int(c) for c in pivC[:r])
        F = sorted(int(c) for c in pivC[r:])
        _, _, pivR = sla.qr(Hn.conj().T, pivoting=True, mode='economic')
        R = sorted(int(c) for c in pivR[:r])
    except Exception:
        return []

    A = MatrixPoly([[H_bar.data[R[i]][C[j]] for j in range(r)] for i in range(r)], d)
    det_A, adj_A = A.det_and_adjugate(max_terms=GATES["max_det_terms"],
                                      max_seconds=GATES["max_det_seconds"],
                                      max_ops=GATES["max_det_ops"])

    vecs = []
    for f in F:
        h = MatrixPoly([[H_bar.data[R[i]][f]] for i in range(r)], d)
        y = adj_A * h * -1.0           # dependent components x_C
        x = [LaurentPoly.zero(d) for _ in range(Q)]
        x[f] = det_A
        for j in range(r):
            x[C[j]] = y.data[j][0]
        if all(e.is_zero(1e-12) for e in x):
            continue
        if verify:
            xm = MatrixPoly([[e] for e in x], d)
            if not (H_bar * xm).is_zero(1e-6):
                continue
        vecs.append(x)
    return vecs


def _cramer_right_null(Mp, nr, nc, d):
    """
    Right null-space of a polynomial matrix Mp (nr x nc) via generalized Cramer.

    Estimates the numerical rank r on the torus, picks r independent rows R and
    columns C (rank-revealing pivoted QR) so Mp[R, C] is the non-singular r x r
    block, and solves for the dependent components in terms of each free column
    via that block's adjugate. Returns one (nc-length) LaurentPoly null vector
    per free column. Works for both rectangular (rank == nr, e.g. a rim-hub
    coupling B^dagger) and square-singular (rank < nr, e.g. a Schur complement)
    inputs. The determinant is r x r — far smaller and more stable than the
    full Q x Q Faddeev-LeVerrier adjugate.
    """
    import numpy as np
    rng = np.random.default_rng(20240531)
    Xv = np.exp(1j * rng.uniform(0, 2 * np.pi, size=d))
    Bn = np.array([[_eval_laurent_randX(Mp.data[i][j], Xv) for j in range(nc)]
                   for i in range(nr)], dtype=complex)

    sv = np.linalg.svd(Bn, compute_uv=False)
    if sv.size == 0:
        return []
    tol = max(nr, nc) * np.finfo(float).eps * sv[0] * 100.0
    r = int(np.sum(sv > max(tol, 1e-9)))
    if r == 0 or r >= nc:
        return []

    try:
        import scipy.linalg as sla
        _, _, pivC = sla.qr(Bn, pivoting=True, mode='economic')
        C = sorted(int(c) for c in pivC[:r])
        F = sorted(int(c) for c in pivC[r:])
        _, _, pivR = sla.qr(Bn.conj().T, pivoting=True, mode='economic')
        R = sorted(int(c) for c in pivR[:r])
    except Exception:
        return []

    A = MatrixPoly([[Mp.data[R[i]][C[j]] for j in range(r)] for i in range(r)], d)
    det_A, adj_A = A.det_and_adjugate(max_terms=GATES["max_det_terms"],
                                      max_seconds=GATES["max_det_seconds"],
                                      max_ops=GATES["max_det_ops"])
    vecs = []
    for f in F:
        h = MatrixPoly([[Mp.data[R[i]][f]] for i in range(r)], d)
        y = adj_A * h * -1.0
        u = [LaurentPoly.zero(d) for _ in range(nc)]
        u[f] = det_A
        for j in range(r):
            u[C[j]] = y.data[j][0]
        if all(e.is_zero(1e-12) for e in u):
            continue
        vecs.append(u)
    return vecs


def _structural_reduce(H_bar, Q, d, verify=True):
    r"""
    Structural Schur reduction for "rim-decorated" flat-band Hamiltonians.

    Many multi-orbital flat-band models split into a *hub* set S and a *rim*
    set T whose intra-rim block D = H_bar[T, T] is k-independent (constant) and
    invertible — e.g. a constant on-site energy with no rim-rim hopping. Writing
    H_bar = [[A, B], [B^dagger, D]] with A = H_bar[S, S], the kernel reduces to
    the Schur complement on the (much smaller) hub block:

        (A - B D^{-1} B^dagger) u = 0,   w = -D^{-1} B^dagger u,   x = (u, w).

    When the hub block A == 0 (no intra-hub hopping) this collapses further to
    x = (u, 0) with u in null(B^dagger): the CLS lives entirely on the hub.

    The payoff is numerical stability: the determinant shrinks from Q x Q to at
    most |S| x |S| (rectangular B^dagger needs only |T| x |T|), so a model whose
    full Faddeev-LeVerrier adjugate is float-unstable (8x8+ with sqrt(3)
    coefficients) is solved exactly via a small, well-conditioned block.

    Returns a list of verified Q-length CLS vectors, or None if the structure is
    absent / the reduction yields nothing.
    """
    import numpy as np
    zero_exp = (0,) * d

    def _is_const(p):
        return all(e == zero_exp for e in p.coefs)

    # Rim T: orbitals with a constant, non-zero diagonal (D's diagonal).
    T = [i for i in range(Q)
         if _is_const(H_bar.data[i][i])
         and abs(H_bar.data[i][i].coefs.get(zero_exp, 0j)) > 1e-9]
    if not T or len(T) >= Q:
        return None
    S = [i for i in range(Q) if i not in T]

    # The whole rim-rim block must be constant for D^{-1} to be constant.
    for i in T:
        for j in T:
            if not _is_const(H_bar.data[i][j]):
                return None
    D = np.array([[complex(H_bar.data[i][j].coefs.get(zero_exp, 0j))
                   for j in T] for i in T], dtype=complex)
    if abs(np.linalg.det(D)) < 1e-9:
        return None
    Dinv = np.linalg.inv(D)

    Bdag = MatrixPoly([[H_bar.data[i][j] for j in S] for i in T], d)  # |T| x |S|
    A_zero = all(H_bar.data[i][j].is_zero(1e-12) for i in S for j in S)

    out = []
    if A_zero:
        # CLS lives entirely on the hub: u in null(B^dagger), rim = 0.
        for u in _cramer_right_null(Bdag, len(T), len(S), d):
            x = [LaurentPoly.zero(d) for _ in range(Q)]
            for j, s in enumerate(S):
                x[s] = u[j]
            out.append(x)
    else:
        # General Schur complement S_A = A - B D^{-1} B^dagger on the hub.
        B = MatrixPoly([[H_bar.data[i][j] for j in T] for i in S], d)     # |S| x |T|
        A_mp = MatrixPoly([[H_bar.data[i][j] for j in S] for i in S], d)  # |S| x |S|
        Dinv_mp = MatrixPoly([[LaurentPoly.constant(Dinv[i][j], d)
                               for j in range(len(T))] for i in range(len(T))], d)
        DinvBdag = Dinv_mp * Bdag                       # |T| x |S|
        S_A = A_mp - (B * DinvBdag)                     # |S| x |S|
        for u in _cramer_right_null(S_A, len(S), len(S), d):
            um = MatrixPoly([[e] for e in u], d)
            w = (DinvBdag * um) * -1.0                  # rim components, |T| x 1
            x = [LaurentPoly.zero(d) for _ in range(Q)]
            for j, s in enumerate(S):
                x[s] = u[j]
            for j, t in enumerate(T):
                x[t] = w.data[j][0]
            out.append(x)

    final = []
    for x in out:
        if verify:
            xm = MatrixPoly([[e] for e in x], d)
            if not (H_bar * xm).is_zero(1e-6):
                continue
        final.append(x)
    return final or None


def _evaluate_candidate_worker(args):
    """
    Evaluate a single gauge candidate (minimize and check singularity).
    Must be top-level for multiprocessing compatibility.
    """
    key, x_k, symbols, test_k, lattice = args
    from cls_finder.cls.reduce import minimize_cls
    try:
        x_k_min, A_0_R_min = minimize_cls(x_k, symbols)
        support  = sum(len(v) for v in A_0_R_min.values())
        n_terms  = sum(len(p.coefs) for p in x_k_min)
        is_singular = _is_singular_gauge(x_k_min, symbols, test_k, lattice)
        return (int(is_singular), support, n_terms, key, x_k_min, A_0_R_min), None
    except Exception as e:
        return None, repr(e)


def extract_all_cls_analytic(H_k, eps0, max_gauges=None, verify=True):
    """
    Derive the unnormalized CLS eigenvectors (as a list of LaurentPolys)
    and their real-space amplitude representations A_{0, R} (as dicts).

    For a non-degenerate band (M == 1) every guide index yields the same minimal
    CLS after GCD reduction, so only the best-conditioned gauge is computed
    (capped by `max_gauges`) — this avoids computing all Q large determinants,
    the dominant cost for large Q. Degenerate bands (M >= 2) use the generalized
    minor+adjugate (Cramer) method, which is fast and division-free for any Q;
    the SymPy nullspace is only a last resort. All paths are size-gated so a
    large Hamiltonian degrades to the numerical CLS rather than hanging.

    Every gauge is verified (H_bar * x == 0) before being accepted: the
    division-free Faddeev-LeVerrier adjugate becomes float-unstable for large
    (~8x8+) submatrices with irrational coefficients, and an unverified result
    would be silently wrong. Verification is cheap (one matrix-poly product) and
    guarantees we never emit an invalid CLS — callers fall back to the numerical
    CLS instead.

    Returns: dict mapping p (int) or "sympy_nullspace_i" (str) -> (x_k, A_0_R)
    """
    Q = H_k.rows
    d = H_k.d

    if Q == 1:
        x_k = [LaurentPoly.constant(1.0, d)]
        A_0_R = {0: {(0,) * d: 1.0}}
        return {0: (x_k, A_0_R)}

    H_bar = H_k - MatrixPoly.identity(Q, d) * eps0
    results = {}

    M, p_order = _numeric_screen(H_bar, d)

    def _store_null(vectors):
        for idx, w in enumerate(vectors):
            A_0_R = {q: {exp: coef for exp, coef in w[q].coefs.items()}
                     for q in range(Q)}
            results[f"sympy_nullspace_{idx}"] = (w, A_0_R)

    def _store_named(vectors, prefix):
        for idx, w in enumerate(vectors):
            A_0_R = {q: {exp: coef for exp, coef in w[q].coefs.items()}
                     for q in range(Q)}
            results[f"{prefix}_{idx}"] = (w, A_0_R)

    # Structural Schur reduction first: "rim-decorated" models (a constant
    # invertible on-site block coupled to a hub) reduce to a much smaller and
    # numerically *stable* determinant. The full Q x Q Faddeev-LeVerrier
    # adjugate is float-unstable for large submatrices with irrational (sqrt 3)
    # coefficients, so try this exact reduction before the generic gauges.
    try:
        struct = _structural_reduce(H_bar, Q, d, verify=verify)
    except Exception:
        struct = None
    if struct:
        _store_named(struct, "structural")
        # A verified, stable CLS set. For a degenerate band this is already the
        # full M-dimensional basis, so skip the large (unstable) generic adjugate.
        if len(struct) >= max(1, M):
            return results

    if M == 1 and Q <= GATES["max_symbolic_Q"]:
        # One good adjugate gauge suffices (rank-1 free kernel).
        target = max_gauges if max_gauges else 1
        for test_p in p_order:
            try:
                test_x = _adjugate_gauge(H_bar, test_p, Q, d, verify=verify)
            except MemoryError:
                # Determinant blew past the budget. Every guide index has
                # comparable cost for M==1, so don't retry the rest — degrade.
                break
            except Exception:
                continue
            if test_x is None:
                continue
            A_0_R = {q: {exp: coef for exp, coef in test_x[q].coefs.items()}
                     for q in range(Q)}
            results[test_p] = (test_x, A_0_R)
            if len(results) >= target:
                break

    elif M >= 2 and Q <= GATES["max_symbolic_Q"]:
        # Degenerate band: generalized minor+adjugate (Cramer), no SymPy nullspace.
        try:
            _store_null(_degenerate_adjugate_gauges(H_bar, M, Q, d, verify=verify))
        except MemoryError:
            pass
        except Exception:
            pass

    # Last resort (small systems only): exact SymPy nullspace.
    if not results and Q <= GATES["max_nullspace_Q"]:
        try:
            from cls_finder.eigen.eigenstate import extract_eigenstate_analytical
            import sympy
            symbols = sympy.symbols(f'x1:{d+1}')
            _store_null(extract_eigenstate_analytical(H_k, eps0, symbols))
        except Exception:
            pass

    return results

def _is_singular_gauge(x_k_min, symbols, test_k, lattice):
    """
    Decide whether a minimized CLS gauge is singular (its components share a
    common zero on the Brillouin torus). Prefer the exact resultant test; fall
    back to grid sampling when it is inapplicable (d >= 3, inexact coeffs).
    """
    try:
        from cls_finder.classify.torus_zeros import has_common_zero_on_torus
        verdict, _ = has_common_zero_on_torus(x_k_min, symbols)
        if verdict is not None:
            return bool(verdict)
    except Exception:
        pass

    if test_k is not None and lattice is not None:
        import numpy as np
        try:
            vals = np.stack(
                [p.evaluate_batch(test_k, lattice.primitive_vectors)
                 for p in x_k_min],
                axis=1)                             # (K, Q)
            norms = np.linalg.norm(vals, axis=1)   # (K,)
            return bool(np.any(norms < 1e-8))
        except Exception:
            pass
    return False


def _mix_degenerate_gauges(raw_vectors, d):
    """
    Build candidate combinations of M degenerate kernel vectors (Rhim-Yang
    mixing, §8.2). For a removable (gauge) singularity a constant linear
    combination of the raw FSBP eigenvectors yields a singularity-free basis;
    the Kagome-3 model is the canonical example.

    raw_vectors : list of (list of LaurentPoly)  — the M raw kernel vectors
    Returns: list of (label, vector) candidates, including the originals and
             pairwise w_i ± w_j combinations.
    """
    M = len(raw_vectors)
    out = []
    for i, v in enumerate(raw_vectors):
        out.append((f"raw_{i}", v))
    if M < 2:
        return out

    Q = len(raw_vectors[0])
    for i in range(M):
        for j in range(i + 1, M):
            vi, vj = raw_vectors[i], raw_vectors[j]
            w_plus  = [vi[q] + vj[q] for q in range(Q)]
            w_minus = [vi[q] - vj[q] for q in range(Q)]
            out.append((f"mix_{i}+{j}", w_plus))
            out.append((f"mix_{i}-{j}", w_minus))
    return out


def extract_cls_analytic(H_k, eps0, p=None, verify=True):
    """
    Derive the unnormalized CLS eigenvector x_k (as a list of LaurentPolys)
    and its real-space amplitude representation A_{0, R} (as a dict)
    using the Minor + Adjugate method.
    """
    all_gauges = extract_all_cls_analytic(H_k, eps0, verify=verify)
    
    if p is not None:
        if p in all_gauges:
            return all_gauges[p]
        else:
            raise ValueError(f"Guide index p={p} yielded a zero eigenvector or failed.")
            
    # If p is not specified, try standard orbital indices first
    Q = H_k.rows
    for test_p in range(Q):
        if test_p in all_gauges:
            return all_gauges[test_p]
            
    # Fallback to SymPy nullspace
    for key in all_gauges:
        if str(key).startswith("sympy_nullspace"):
            return all_gauges[key]
            
    raise ValueError("Failed to find any valid analytical CLS gauge.")


def _build_gauge_candidates(H_k, eps0, symbols, lattice=None, verify=True):
    """
    Enumerate, minimize and rank every available CLS gauge.

    Returns a list of candidate tuples
        (is_singular: int, support: int, n_terms: int, gauge_id, x_k_min, A_0_R_min)
    sorted best-first: non-singular first, then smallest support, then fewest
    terms. Includes the Rhim-Yang mixed combinations for degenerate bands.
    """
    import numpy as np
    from itertools import product as iproduct
    from cls_finder.cls.reduce import minimize_cls

    all_gauges = extract_all_cls_analytic(H_k, eps0, verify=verify)
    if not all_gauges:
        raise ValueError("No analytical CLS gauge found.")

    d = H_k.d
    Q = H_k.rows

    # Degenerate flat bands (M >= 2): the minor+adjugate method collapses
    # (every (Q-1)x(Q-1) minor vanishes), so extract_all_cls_analytic falls
    # back to the raw SymPy nullspace vectors — each of which may carry a
    # *removable* singularity. Apply Rhim-Yang mixing (§8.2) to add
    # singularity-free combinations to the candidate pool, so the
    # nonsingular-first ranking below can select a complete CLS basis.
    # Rhim-Yang mixing applies to the raw SymPy-nullspace vectors, which may
    # individually carry a *removable* singularity (kagome_3). The structural
    # Schur reduction already returns a clean, verified, complete basis, so it is
    # *not* mixed: mixing those would only add cost (a GCD blow-up for irrational
    # coefficients) without improving support or singularity.
    nullspace_keys = [k for k in all_gauges if str(k).startswith("sympy_nullspace")]
    if len(nullspace_keys) >= 2:
        raw_vectors = [all_gauges[k][0] for k in nullspace_keys]
        for label, w in _mix_degenerate_gauges(raw_vectors, d):
            if label.startswith("raw_"):
                continue  # originals already in all_gauges
            A_w = {}
            for q in range(len(w)):
                A_w[q] = {exp: coef for exp, coef in w[q].coefs.items()}
            all_gauges[f"sympy_{label}"] = (w, A_w)

    # Rigorous Gröbner/syzygy generators: the *complete* generating set of the
    # CLS module (works uniformly for any Q and degeneracy, and often finds a
    # more compact representative than the adjugate/mixing heuristics). Added to
    # the candidate pool; the ranking below still chooses among them. Returns
    # None for irrational coefficients / Convention II. Gröbner cost explodes
    # super-exponentially, so it is gated by Q and term count — above the gate
    # we rely on the (already complete for these cases) adjugate/mixing gauges.
    if Q <= GATES["max_syzygy_Q"] and _total_terms(H_k) <= GATES["max_syzygy_terms"]:
        try:
            from cls_finder.cls.syzygy import compute_cls_generators
            syz = compute_cls_generators(H_k, eps0, symbols)
            if syz:
                for gi, g in enumerate(syz["generators"]):
                    A_g = {q: {exp: coef for exp, coef in g[q].coefs.items()}
                           for q in range(len(g))}
                    all_gauges[f"syzygy_{gi}"] = (g, A_g)
        except Exception:
            pass

    # Grid fallback test k-points: high-symmetry corners of BZ + a few interior points
    test_k = None
    if lattice is not None:
        A = np.array(lattice.primitive_vectors, dtype=float)
        B = 2.0 * np.pi * np.linalg.solve(A @ A.T, A)
        frac_list = list(iproduct([0.0, 0.25, 0.5, -0.25, -0.5], repeat=d))
        test_k = np.array([np.array(xi) @ B for xi in frac_list])

    # Evaluate (minimize + singularity-check) each gauge. After the algorithmic
    # fixes (single-gauge M=1, generalized adjugate for M>=2, bounded GCD) each
    # candidate is cheap and the pool is small, so a serial loop beats process-
    # pool spawn overhead — and works unchanged under Pyodide.
    candidates = []
    for key, (x_k, A_0_R_raw) in all_gauges.items():
        res, err = _evaluate_candidate_worker((key, x_k, symbols, test_k, lattice))
        if res is not None:
            candidates.append(res)

    candidates.sort(key=lambda c: (c[0], c[1], c[2]))
    return candidates


def select_best_cls_gauge(H_k, eps0, symbols, lattice=None, verify=True):
    """
    From all available gauge choices, return the single best one:
      1. Non-singular eigenvector (doesn't vanish in BZ) preferred
      2. Smallest real-space support (fewest non-zero sites)
      3. Fewest total polynomial terms (simplest form)

    Returns
    -------
    x_k_min   : list of LaurentPoly  — GCD-minimized eigenvector
    A_0_R_min : dict {q -> {cell -> coef}}
    meta      : dict {gauge_id, is_singular, support_size}
    """
    best = _build_gauge_candidates(H_k, eps0, symbols, lattice, verify=verify)[0]
    return best[4], best[5], {
        'gauge_id':    best[3],
        'is_singular': bool(best[0]),
        'support_size': best[1],
    }


def select_cls_basis(H_k, eps0, symbols, M, lattice=None, verify=True):
    """
    Select a *complete* basis of M independent CLS for an M-fold degenerate
    flat band. Greedily picks the best-ranked gauges (non-singular first, then
    most compact) that are mutually linearly independent — independence is
    judged by the rank of their amplitudes evaluated at a generic k-point, so
    the chosen translates span the full M-dimensional flat eigenspace.

    For M == 1 this returns the single best gauge.

    Returns
    -------
    list of (x_k_min, A_0_R_min, meta) of length min(M, #independent gauges),
    ordered best-first.
    """
    import numpy as np

    candidates = _build_gauge_candidates(H_k, eps0, symbols, lattice, verify=verify)
    if M <= 1:
        c = candidates[0]
        return [(c[4], c[5], {'gauge_id': c[3], 'is_singular': bool(c[0]),
                              'support_size': c[1]})]

    # Generic interior k-point (avoid high-symmetry nodes) for the rank test.
    if lattice is not None:
        A = np.array(lattice.primitive_vectors, dtype=float)
        B = 2.0 * np.pi * np.linalg.solve(A @ A.T, A)
        k_gen = np.array([0.123, 0.317, 0.211][:H_k.d]) @ B
        pv = lattice.primitive_vectors
    else:
        k_gen, pv = None, None

    basis, chosen_vecs = [], []
    for (is_sing, support, n_terms, key, x_k_min, A_0_R_min) in candidates:
        if len(basis) >= M:
            break
        if k_gen is not None:
            vec = np.array([p.evaluate(k_gen, pv) for p in x_k_min])
            if np.linalg.norm(vec) < 1e-9:
                continue  # vanishes at the generic point — not usable
            trial = chosen_vecs + [vec]
            if np.linalg.matrix_rank(np.array(trial), tol=1e-7) <= len(chosen_vecs):
                continue  # linearly dependent on already-chosen CLS
            chosen_vecs.append(vec)
        basis.append((x_k_min, A_0_R_min, {
            'gauge_id': key, 'is_singular': bool(is_sing), 'support_size': support}))

    if not basis:  # degenerate rank test failed everywhere — fall back to best
        c = candidates[0]
        basis = [(c[4], c[5], {'gauge_id': c[3], 'is_singular': bool(c[0]),
                              'support_size': c[1]})]
    return basis
