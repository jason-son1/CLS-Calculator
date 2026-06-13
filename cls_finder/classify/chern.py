r"""
Chern number of a flat band.

Two independent routes, cross-checked:

  * Numerical (robust ground truth) — the Fukui-Hatsugai-Suzuki lattice field
    strength of the flat-band eigenspace, obtained by diagonalizing H(k) on a
    BZ grid and selecting the M states nearest eps0 (non-Abelian for M >= 2).
    Gauge-invariant by construction; returns the integer C.

  * Analytic (Rhim-Yang style; see goal/finite_fourier_eigenvector_chern_conditions)
    — for the finite Fourier-sum eigenvector f(k) (= the CLS), a nonzero Chern
    requires a COMMON ZERO of all components; at each first-order common zero the
    rank-one Jacobian A = [d f/dk_x , d f/dk_y] must factor as A = v (m_x, m_y)
    (projector continuity), and the local winding is w = sgn Im(conj(m_x) m_y)
    = sgn Im<A_x, A_y>. The Chern number is C = sum_i w_i.

The analytic route both *explains* the value (which of the three conditions
hold) and cross-checks the numerical one.
"""
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# reciprocal lattice + BZ grid
# ──────────────────────────────────────────────────────────────────────────────
def reciprocal_vectors(primitive_vectors):
    """Rows b_l satisfy b_l . a_m = 2 pi delta_lm (for a square primitive matrix)."""
    A = np.asarray(primitive_vectors, dtype=float)
    return 2.0 * np.pi * np.linalg.solve(A @ A.T, A)


def _frac_grid(n1, n2):
    f1 = np.arange(n1) / n1
    f2 = np.arange(n2) / n2
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")
    return np.stack([F1.ravel(), F2.ravel()], axis=1)            # (n1*n2, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Numerical: Fukui-Hatsugai-Suzuki lattice Chern number
# ──────────────────────────────────────────────────────────────────────────────
def flat_band_eigvecs_on_grid(H_k, lattice, eps0, M, n):
    """(n, n, Q, M) array: the M eigenvectors nearest eps0 on an n x n BZ grid."""
    B = reciprocal_vectors(lattice.primitive_vectors)            # (2, spatial)
    frac = _frac_grid(n, n)                                       # (n*n, 2)
    k_cart = frac @ B                                             # (n*n, spatial)
    H_batch = H_k.evaluate_batch(k_cart, lattice.primitive_vectors)  # (n*n, Q, Q)
    w, v = np.linalg.eigh(H_batch)                               # ascending
    idx = np.argsort(np.abs(w - eps0), axis=1)[:, :M]            # (n*n, M)
    U = np.take_along_axis(v, idx[:, None, :], axis=2)          # (n*n, Q, M)
    Q = H_k.rows
    return U.reshape(n, n, Q, M)


def fhs_chern(U):
    """
    Fukui-Hatsugai-Suzuki Chern number of an (n1, n2, Q, M) eigenvector grid.

    Abelian (M=1) or non-Abelian (M>=2, via det of the M x M overlap). The grid
    is treated as periodic (k and k+G identified), so no boundary handling is
    needed. Returns a float that is an integer up to discretization error.
    """
    n1, n2 = U.shape[0], U.shape[1]

    def link(a, b):                                              # a,b: (Q, M)
        ov = a.conj().T @ b                                      # (M, M)
        det = np.linalg.det(ov)
        mag = abs(det)
        return det / mag if mag > 1e-300 else 1.0 + 0j

    total = 0.0
    for i in range(n1):
        ip = (i + 1) % n1
        for j in range(n2):
            jp = (j + 1) % n2
            u1 = link(U[i, j],  U[ip, j])
            u2 = link(U[ip, j], U[ip, jp])
            u3 = link(U[ip, jp], U[i, jp])
            u4 = link(U[i, jp], U[i, j])
            total += np.angle(u1 * u2 * u3 * u4)
    return total / (2.0 * np.pi)


def band_isolation(H_k, lattice, eps0, M, n=48):
    """
    Minimum energy gap separating the M-fold flat-band group (the M states
    nearest eps0) from the bands immediately below and above, over an n x n grid.

    A single-band Chern number is well defined only when the band (or M-group) is
    ISOLATED — gap_below and gap_above both > 0. If either is ~0 the band touches
    a neighbour and its Chern is not a clean integer (the touching point shares
    Berry curvature). Returns {gap_below, gap_above, isolated}.
    """
    B = reciprocal_vectors(lattice.primitive_vectors)
    frac = _frac_grid(n, n)
    w = np.linalg.eigh(H_k.evaluate_batch(frac @ B, lattice.primitive_vectors))[0]
    Q = w.shape[1]
    # at each k, the M indices nearest eps0 form the group; gaps to neighbours
    order = np.argsort(np.abs(w - eps0), axis=1)[:, :M]          # (P, M)
    gb = np.inf
    ga = np.inf
    for p in range(w.shape[0]):
        grp = np.sort(order[p])
        lo_i, hi_i = grp[0], grp[-1]
        if lo_i - 1 >= 0:
            gb = min(gb, w[p, lo_i] - w[p, lo_i - 1])
        if hi_i + 1 < Q:
            ga = min(ga, w[p, hi_i + 1] - w[p, hi_i])
    gb = None if not np.isfinite(gb) else float(gb)
    ga = None if not np.isfinite(ga) else float(ga)
    iso = ((gb is None or gb > 1e-4) and (ga is None or ga > 1e-4))
    return {"gap_below": gb, "gap_above": ga, "isolated": bool(iso)}


def numerical_chern(H_k, lattice, eps0, M, n=24, refine=True):
    """Robust FHS Chern of the flat band; returns dict with C (rounded int) and raw."""
    U = flat_band_eigvecs_on_grid(H_k, lattice, eps0, M, n)
    C_raw = fhs_chern(U)
    C = int(round(C_raw))
    result = {"C": C, "C_raw": float(C_raw), "grid": n, "M": M,
              "converged": bool(abs(C_raw - C) < 0.15)}
    if refine and not result["converged"]:
        # one denser pass to settle a borderline plaquette
        U2 = flat_band_eigvecs_on_grid(H_k, lattice, eps0, M, n * 2)
        C2 = fhs_chern(U2)
        result.update(C=int(round(C2)), C_raw=float(C2), grid=n * 2,
                      converged=bool(abs(C2 - round(C2)) < 0.15))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Analytic: common zeros + local winding (the paper)
# ──────────────────────────────────────────────────────────────────────────────
def _cls_field(x_k, k_cart, prim):
    """Evaluate the finite-Fourier vector f(k) at k-points → (N_pts, Q) complex."""
    return np.stack([p.evaluate_batch(k_cart, prim) for p in x_k], axis=1)


def _frac_to_cart(frac, B):
    return np.asarray(frac, dtype=float) @ B


def _cart_to_frac(k_cart, B):
    """Fractional coordinates of a Cartesian k (least squares for non-square B)."""
    return np.linalg.lstsq(B.T, np.asarray(k_cart, dtype=float), rcond=None)[0]


def _wrap_frac(fr):
    """Wrap fractional coordinates to the symmetric BZ cell [-0.5, 0.5)^d."""
    return (np.asarray(fr, dtype=float) + 0.5) % 1.0 - 0.5


def _jacobian_cart(x_k, k_cart, prim):
    """
    Full Cartesian first-derivative matrix A_{alpha,mu} = d f_alpha / d k_mu at
    k_cart, where f_alpha = sum_R c_{alpha,R} exp(i k . R), R = sum_l exp_l a_l.
    Returns (Q, spatial) complex.
    """
    a = np.asarray(prim, dtype=float)                            # (d, spatial)
    spatial = a.shape[1]
    N = len(x_k)
    A = np.zeros((N, spatial), dtype=complex)
    k_cart = np.asarray(k_cart, dtype=float)
    for alpha, poly in enumerate(x_k):
        for exp, coef in poly.coefs.items():
            R = np.zeros(spatial)
            for l, p in enumerate(exp):
                R += p * a[l]
            phase = np.exp(1j * np.dot(k_cart, R))
            A[alpha, :] += coef * (1j * R) * phase
    return A


def jacobian_at_zero(x_k, k_cart, prim):
    """
    First-derivative columns (A_x, A_y) of f at k_cart (Eq. 26/57). Each is a
    length-Q complex vector.
    """
    A = _jacobian_cart(x_k, k_cart, prim)
    return A[:, 0], A[:, 1]


def _polish_zero(x_k, frac0, prim, B, max_iter=24, tol=1e-13):
    """
    Damped Gauss-Newton polish of a candidate common zero in fractional space.

    Minimizes |f(k)|^2 with the analytic Jacobian (quadratic convergence at a
    first-order zero); a backtracking line search keeps it monotone and lets it
    crawl down higher-order (rank-deficient) zeros too. Returns (frac, |f|^2).
    """
    fr = np.array(frac0, dtype=float)
    f = np.array([p.evaluate(fr @ B, prim) for p in x_k])
    val = float(np.sum(np.abs(f) ** 2))
    for _ in range(max_iter):
        if val < tol:
            break
        kc = fr @ B
        A = _jacobian_cart(x_k, kc, prim)                        # (Q, spatial)
        Jf = A @ B.T                                             # (Q, d): df/dfrac
        Jr = np.vstack([Jf.real, Jf.imag])                       # (2Q, d)
        rr = np.concatenate([f.real, f.imag])                    # (2Q,)
        try:
            delta = np.linalg.lstsq(Jr, -rr, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)) or np.linalg.norm(delta) < 1e-15:
            break
        step = 1.0
        improved = False
        for _ls in range(12):
            fr_try = fr + step * delta
            f_try = np.array([p.evaluate(fr_try @ B, prim) for p in x_k])
            val_try = float(np.sum(np.abs(f_try) ** 2))
            if val_try < val:
                fr, f, val, improved = fr_try, f_try, val_try, True
                break
            step *= 0.5
        if not improved:
            break
    return fr, val


def find_common_zeros(x_k, lattice, n_scan=160, refine=True, zero_tol=1e-4,
                      max_refine=400, abs_seed_rel=1e-2):
    """
    Locate ALL common zeros k_i of the finite-Fourier vector f (every component
    vanishes; Eq. 10/71) inside the BZ — the singularities that carry the Chern
    number.

    Robust strategy:
      1. Scan |f(k)|^2 on an n_scan x n_scan grid.
      2. Seed from every grid cell whose relative |f|^2 is small OR is a discrete
         local minimum (so zeros falling *between* grid nodes are not missed),
         plus all high-symmetry points.
      3. Sort seeds by depth and polish each with damped Gauss-Newton (then a
         Nelder-Mead fallback), keeping those that reach |f|^2 < zero_tol^2.
      4. De-duplicate modulo a reciprocal-lattice vector, wrapped to [-0.5,0.5)^d.

    Returns the Cartesian zeros sorted by depth (deepest / most certain first).
    Numerical, so it works for any lattice/convention (hexagonal, sqrt3, ...).
    """
    prim = lattice.primitive_vectors
    B = reciprocal_vectors(prim)
    frac = _frac_grid(n_scan, n_scan)
    k_cart = frac @ B
    F = _cls_field(x_k, k_cart, prim)                            # (P, Q)
    norm2 = np.sum(np.abs(F) ** 2, axis=1)                       # (P,)
    scale = norm2.max() if norm2.size else 1.0
    if scale < 1e-300:
        return []
    rel = norm2 / scale
    grid = rel.reshape(n_scan, n_scan)

    # ── seeds: discrete local minima OR absolutely-small cells ───────────────
    gmin = float(grid.min())
    loc_thr = max(0.30, 30.0 * gmin)        # generous local-min gate
    abs_thr = max(abs_seed_rel, 5.0 * gmin)  # absolute small-value gate
    seeds = []                              # (depth, frac)
    for i in range(n_scan):
        for j in range(n_scan):
            v = grid[i, j]
            is_small = v <= abs_thr
            is_locmin = False
            if v <= loc_thr:
                nb = (grid[(i + 1) % n_scan, j], grid[(i - 1) % n_scan, j],
                      grid[i, (j + 1) % n_scan], grid[i, (j - 1) % n_scan])
                is_locmin = v <= min(nb) + 1e-12
            if is_small or is_locmin:
                seeds.append((v, np.array([i / n_scan, j / n_scan])))

    # high-symmetry points (Γ, X, M, K, ...) as extra seeds
    try:
        from cls_finder.classify.singularity import get_high_symmetry_k_points
        for kp in get_high_symmetry_k_points(lattice):
            seeds.append((0.0, _cart_to_frac(kp, B)))
    except Exception:
        pass

    seeds.sort(key=lambda t: t[0])          # deepest first

    zeros = []          # list of (frac_wrapped, |f|^2)

    def _objective(fr):
        f = np.array([p.evaluate(np.asarray(fr) @ B, prim) for p in x_k])
        return float(np.sum(np.abs(f) ** 2))

    # de-duplicate in fractional space (modulo the reciprocal lattice). The gate
    # is generous (1% of the BZ): genuine zeros of these trig polynomials are
    # O(0.1-0.5) apart, while the polish endpoints of one zero scatter at the
    # ~1e-3 level — a Cartesian 1e-3 gate was too tight and let copies through.
    dedup_frac = 1e-2

    def _dup_index(fr_w):
        for idx, (fz, _v) in enumerate(zeros):
            if np.linalg.norm(_wrap_frac(fr_w - fz)) < dedup_frac:
                return idx
        return -1

    for _depth, fr0 in seeds[:max_refine]:
        if refine:
            fr_opt, val = _polish_zero(x_k, fr0, prim, B)
            if val > zero_tol ** 2:          # Nelder-Mead fallback for hard zeros
                try:
                    from scipy.optimize import minimize
                    res = minimize(_objective, fr_opt, method="Nelder-Mead",
                                   options={"xatol": 1e-10, "fatol": 1e-18,
                                            "maxiter": 600})
                    if res.fun < val:
                        fr_opt, val = res.x, float(res.fun)
                except Exception:
                    pass
        else:
            fr_opt, val = fr0, _objective(fr0)
        if val > zero_tol ** 2:
            continue
        fr_w = _wrap_frac(fr_opt)
        idx = _dup_index(fr_w)
        if idx < 0:
            zeros.append((fr_w, val))
        elif val < zeros[idx][1]:            # keep the deeper representative
            zeros[idx] = (fr_w, val)

    zeros.sort(key=lambda t: t[1])           # deepest / most certain first
    return [fr @ B for (fr, _v) in zeros]


def local_winding(A_x, A_y, rank_tol=1e-6, zero_tol=1e-9):
    """
    First-order analysis of a common zero from its Jacobian columns (Eqs. 33/46).

    Returns dict:
      first_order : the Jacobian is non-zero (genuine first-order zero)
      rank        : numerical complex rank of [A_x | A_y] (1 => continuous projector)
      rank_one    : bool
      winding     : sgn Im<A_x, A_y>  (valid when rank_one); else best-effort sign
      im_inner    : Im<A_x, A_y>
    """
    Ax = np.asarray(A_x, dtype=complex)
    Ay = np.asarray(A_y, dtype=complex)
    Mmat = np.column_stack([Ax, Ay])                            # (N, 2)
    sv = np.linalg.svd(Mmat, compute_uv=False)
    s0 = sv[0] if sv.size else 0.0
    first_order = s0 > zero_tol
    rank = int(np.sum(sv > rank_tol * (s0 if s0 > 0 else 1.0)))
    inner = np.vdot(Ax, Ay)                                      # <A_x, A_y>
    im = float(np.imag(inner))
    w = int(np.sign(im)) if abs(im) > rank_tol * (s0 ** 2 + 1e-30) else 0
    return {"first_order": bool(first_order), "rank": rank,
            "rank_one": rank == 1, "winding": w, "im_inner": im,
            "sing_values": [float(x) for x in sv]}


def loop_winding(x_k, k_zero, prim, B, radius=None, n_loop=128, cont_tol=2e-2):
    """
    Robust local winding of a common zero by the contour integral
    w = (1/2pi) oint d arg h(q)  (Eqs. 38/65), valid for first-order AND
    higher-order zeros (|w| can exceed 1, e.g. h=(q_x+i q_y)^n -> w=n).

    Samples f on a small counter-clockwise circle around k_zero in the
    (k_x, k_y) plane, extracts the dominant section direction v (so the scalar
    s(k) = <v, f(k)> tracks the vortex factor h up to a constant), and counts
    the principal-branch phase increments of s. The relative weight of the
    second singular direction measures projector (dis)continuity directly on the
    loop — a clean numerical version of the rank-one test that also covers
    higher-order zeros.

    Returns: winding, radius, projector_continuous, rank_ratio, sing_values.
    """
    k0 = np.asarray(k_zero, dtype=float)
    spatial = B.shape[1]
    bmin = min(np.linalg.norm(B[i]) for i in range(B.shape[0]))
    if radius is None:
        radius = 0.02 * bmin
    radius = max(radius, 1e-9)
    phis = 2.0 * np.pi * np.arange(n_loop) / n_loop
    pts = np.tile(k0, (n_loop, 1)).astype(float)
    pts[:, 0] = k0[0] + radius * np.cos(phis)
    if spatial > 1:
        pts[:, 1] = k0[1] + radius * np.sin(phis)
    F = _cls_field(x_k, pts, prim)                               # (n_loop, Q)

    # projector continuity & dominant direction from the density matrix
    # rho = sum_j f_j f_j^dagger  (so its leading eigenvector v is the direction
    # the section f points along — NOT its conjugate). Using F^dagger F instead
    # would return conj(v), and the scalar <v,f> can then vanish identically
    # (e.g. v=conj(g) with sum_a g_a^2 = 0), killing the winding.
    rho = F.T @ F.conj()                                         # (Q, Q) = sum f f^+
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals.real, 0.0, None)
    s0 = evals[-1]
    s1 = evals[-2] if evals.size > 1 else 0.0
    rank_ratio = float(s1 / s0) if s0 > 1e-300 else 0.0
    continuous = rank_ratio < cont_tol
    v = evecs[:, -1]                                             # dominant v (∝ f)

    s = F @ v.conj()                                             # scalar <v,f>
    ph = np.angle(s)
    dph = np.diff(np.concatenate([ph, ph[:1]]))
    dph = (dph + np.pi) % (2.0 * np.pi) - np.pi                  # principal branch
    w = int(round(float(np.sum(dph)) / (2.0 * np.pi)))
    return {"winding": w, "radius": float(radius),
            "projector_continuous": bool(continuous),
            "rank_ratio": rank_ratio,
            "sing_values": [float(np.sqrt(x)) for x in evals[::-1]]}


def _min_zero_spacing(zeros, B):
    """Smallest pairwise BZ distance between zeros (modulo reciprocal lattice)."""
    if len(zeros) < 2:
        return np.inf
    dmin = np.inf
    for i in range(len(zeros)):
        fi = _cart_to_frac(zeros[i], B)
        for j in range(i + 1, len(zeros)):
            fj = _cart_to_frac(zeros[j], B)
            d = np.linalg.norm(_wrap_frac(fi - fj) @ B)
            dmin = min(dmin, d)
    return dmin


def analytic_chern(x_k, lattice, zeros=None, n_scan=160, loop_n=128):
    """
    Apply the finite-Fourier criterion to the CLS f = x_k (paper §11 checklist).

    For every common zero it computes BOTH the first-order Jacobian data
    (rank-one test + sgn Im<A_x,A_y>) and the robust contour winding
    w = (1/2pi) oint d arg h, which also resolves higher-order zeros and tests
    projector continuity on a loop. The Chern number is C = sum_i w_i with the
    contour winding as the primary value and the first-order formula as a
    cross-check. Returns the per-zero analysis and the three-condition verdict.
    """
    prim = lattice.primitive_vectors
    B = reciprocal_vectors(prim)
    if zeros is None:
        zeros = find_common_zeros(x_k, lattice, n_scan=n_scan)

    # loop radius: small, but inside the nearest-neighbour separation
    bmin = min(np.linalg.norm(B[i]) for i in range(B.shape[0]))
    radius = 0.02 * bmin
    if len(zeros) >= 2:
        radius = min(radius, 0.30 * _min_zero_spacing(zeros, B))

    per_zero = []
    C = 0                       # sum over CONTINUOUS-projector zeros (valid Chern)
    C_all = 0                   # sum over every zero (diagnostic, incl. touchings)
    all_continuous = True
    any_winding = False
    first_order_match = True
    for kz in zeros:
        Ax, Ay = jacobian_at_zero(x_k, kz, prim)
        fo = local_winding(Ax, Ay)
        lw = loop_winding(x_k, kz, prim, B, radius=radius, n_loop=loop_n)
        w = lw["winding"]
        cont = bool(lw["projector_continuous"])
        fr = _wrap_frac(_cart_to_frac(kz, B))
        info = {
            "k": [float(x) for x in np.asarray(kz)],
            "k_frac": [float(x) for x in fr],
            "label": _label_k_point(fr),
            "winding": int(w),
            "order": max(1, abs(int(w))),
            "projector_continuous": cont,
            "rank_ratio": lw["rank_ratio"],
            "first_order": fo,
            "loop": lw,
        }
        per_zero.append(info)
        C_all += w
        # Condition (ii): only a CONTINUOUS-projector zero contributes a
        # well-defined local Chern. A discontinuous zero (e.g. a band touching,
        # where the section spans a 2D subspace on the loop) does NOT define a
        # smooth rank-one bundle, so its winding is not a valid Chern term.
        if lw["projector_continuous"]:
            C += w
            if w != 0:
                any_winding = True
        else:
            all_continuous = False
        # cross-check: for a genuine first-order, continuous zero the two routes
        # must agree in sign
        if cont and fo["rank_one"] and fo["first_order"] and abs(w) == 1:
            if fo["winding"] != w:
                first_order_match = False
    return {
        "C": int(C),
        "C_all_zeros": int(C_all),
        "n_common_zeros": len(zeros),
        "has_common_zero": len(zeros) > 0,
        "projector_continuous": bool(all_continuous and len(zeros) > 0),
        "has_discontinuous_zero": bool(not all_continuous and len(zeros) > 0),
        "nonzero_winding": bool(any_winding),
        "first_order_consistent": bool(first_order_match),
        "loop_radius": float(radius),
        "per_zero": per_zero,
        # Proposition 1: no common zero => C = 0 with certainty.
        "trivial_no_zero": len(zeros) == 0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# High-symmetry labelling for the BZ explorer
# ──────────────────────────────────────────────────────────────────────────────
def _label_k_point(frac, tol=2e-2):
    """Name a fractional BZ point (Γ, X, M, K, K', ...) when it sits on a common
    high-symmetry fraction; otherwise return a coordinate string."""
    fr = _wrap_frac(frac)
    f = [round(float(x), 3) for x in fr]

    def near(a, b):
        return abs(((a - b + 0.5) % 1.0) - 0.5) < tol

    if len(f) == 2:
        x, y = f
        if near(x, 0) and near(y, 0):
            return "Γ"
        if (near(x, 0.5) and near(y, 0.5)):
            return "M"
        if (near(x, 0.5) and near(y, 0)) or (near(x, 0) and near(y, 0.5)):
            return "X"
        if (near(x, 1 / 3) and near(y, 1 / 3)) or (near(x, 2 / 3) and near(y, 2 / 3)):
            return "K"
        if (near(x, 1 / 3) and near(y, -1 / 3)) or (near(x, -1 / 3) and near(y, 1 / 3)) \
           or (near(x, 2 / 3) and near(y, 1 / 3)) or (near(x, 1 / 3) and near(y, 2 / 3)):
            return "K'"
        return f"({x:+.2f}, {y:+.2f})"
    if len(f) == 1:
        x = f[0]
        if near(x, 0):
            return "Γ"
        if near(x, 0.5):
            return "X"
        return f"({x:+.2f})"
    return "(" + ", ".join(f"{v:+.2f}" for v in f) + ")"


def bz_landscape(x_k, lattice, n=121):
    """
    |f(k)|^2 over the symmetric BZ cell [-0.5,0.5)^2 in fractional coordinates —
    the 'singularity landscape' whose minima are the common zeros. Returns a dict
    ready for a contour plot (fractional axes + Cartesian-projected meshes).
    """
    prim = lattice.primitive_vectors
    B = reciprocal_vectors(prim)
    if lattice.dimension != 2:
        return None
    fx = np.linspace(-0.5, 0.5, n)
    fy = np.linspace(-0.5, 0.5, n)
    FX, FY = np.meshgrid(fx, fy, indexing="ij")
    frac = np.stack([FX.ravel(), FY.ravel()], axis=1)
    kc = frac @ B
    F = _cls_field(x_k, kc, prim)
    norm2 = np.sum(np.abs(F) ** 2, axis=1).reshape(n, n)
    cart = frac @ B
    return {
        "frac_x": fx.tolist(),
        "frac_y": fy.tolist(),
        "norm2": norm2.T.tolist(),          # (y, x) for Plotly
        "log_norm2": np.log10(norm2 + 1e-12).T.tolist(),
        "kx": cart[:, 0].reshape(n, n).T.tolist(),
        "ky": cart[:, 1].reshape(n, n).T.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Combined report
# ──────────────────────────────────────────────────────────────────────────────
def analyze_flat_band_chern(H_k, lattice, eps0, M, x_k=None, symbols=None,
                            grid_n=24, scan_n=160, n_grid=60):
    """
    Full Chern analysis of a flat band.

    H_k, lattice, eps0, M : the flat band (M = degeneracy).
    x_k                   : the CLS finite-Fourier vector (list of LaurentPoly).
                            Required for the analytic structural route; if None,
                            only the numerical FHS Chern is returned.

    Returns dict: numerical C, analytic structural data, agreement, caveats.
    """
    num = numerical_chern(H_k, lattice, eps0, M, n=grid_n)
    iso = band_isolation(H_k, lattice, eps0, M)
    out = {
        "energy": float(eps0),
        "M": int(M),
        # For an ISOLATED band the FHS integer is the ground truth. For a band
        # that touches a neighbour, single-band FHS is grid-unstable (the band
        # selection swaps at the touching) — the value is set later from the
        # analytic section winding and flagged not well defined.
        "chern_number": num["C"],
        "well_defined": bool(iso["isolated"]),
        "numerical": num,
        "isolation": iso,
        "method": "Fukui-Hatsugai-Suzuki (lattice field strength)",
        "caveats": [],
    }
    if M > 1:
        out["caveats"].append(
            "M>1: 비가환(non-Abelian) FHS로 M차원 평탄 부분공간의 총 Chern을 계산함.")
    if not iso["isolated"]:
        out["caveats"].append(
            f"평탄 밴드가 이웃 밴드와 접촉(gap_below={iso['gap_below']}, "
            f"gap_above={iso['gap_above']}) → 고립 밴드가 아니므로 단일 밴드 Chern이 "
            "위상적으로 잘 정의되지 않음(접촉점이 Berry 곡률을 공유). 접촉 그룹 전체의 "
            "비가환 Chern을 봐야 함.")

    if x_k is not None:
        try:
            ana = analytic_chern(x_k, lattice, n_scan=scan_n)
            out["analytic"] = ana
            out["agreement"] = (ana["C"] == num["C"]) or \
                               (abs(ana["C"]) == abs(num["C"]))

            # compact per-singularity summary for the UI / explorer
            out["singularities"] = [
                {"label": z["label"], "k_frac": z["k_frac"], "k": z["k"],
                 "winding": z["winding"], "order": z["order"],
                 "projector_continuous": z["projector_continuous"]}
                for z in ana["per_zero"]
            ]
            if lattice.dimension == 2:
                try:
                    out["bz_landscape"] = bz_landscape(x_k, lattice)
                except Exception:
                    pass

            if lattice.dimension == 2:
                # Generate 2D orthogonal grid data centered around (0,0) for contour plots
                B = reciprocal_vectors(lattice.primitive_vectors)
                b1 = B[0]
                b2 = B[1]
                max_len = max(np.linalg.norm(b1), np.linalg.norm(b2))
                pad = max_len * 1.25
                
                n_grid = n_grid or 60
                x_ticks = np.linspace(-pad, pad, n_grid)
                y_ticks = np.linspace(-pad, pad, n_grid)
                X, Y = np.meshgrid(x_ticks, y_ticks, indexing="ij")
                cart_grid = np.stack([X.ravel(), Y.ravel()], axis=1)  # (P, 2)
                
                F_grid = _cls_field(x_k, cart_grid, lattice.primitive_vectors)  # (P, Q)
                F_grid = F_grid.reshape(n_grid, n_grid, -1)  # (n_grid, n_grid, Q)
                
                # Amplitude norm of the wave function
                norm_grid = np.linalg.norm(F_grid, axis=2)  # (n_grid, n_grid)
                
                # Phase of dominant orbital
                amp_sums = np.sum(np.abs(F_grid), axis=(0, 1))  # (Q,)
                dom_idx = int(np.argmax(amp_sums))
                phase_grid = np.angle(F_grid[:, :, dom_idx])  # (n_grid, n_grid)
                
                # Spacings for gradient calculation
                dx = x_ticks[1] - x_ticks[0]
                dy = y_ticks[1] - y_ticks[0]
                
                # Normalize wavefunction on the grid: u = f / |f|
                n_x, n_y, Q = F_grid.shape
                norm2 = np.sum(np.abs(F_grid)**2, axis=2, keepdims=True)
                norm2_reg = np.where(norm2 < 1e-20, 1e-20, norm2)
                u = F_grid / np.sqrt(norm2_reg)  # (n_grid, n_grid, Q)
                
                # Projector: P[i, j, a, b] = u_a * conj(u_b)
                P_grid = u[:, :, :, None] * u[:, :, None, :].conj()  # (n_grid, n_grid, Q, Q)
                
                # Calculate derivatives of P on the grid using central differences
                dP_di, dP_dj = np.gradient(P_grid, axis=(0, 1))
                dP_dx = dP_di / dx
                dP_dy = dP_dj / dy
                
                # Curvature F_xy = i * Tr( P * [dP_dx, dP_dy] )
                term1 = np.einsum('xyab,xybc->xyac', dP_dx, dP_dy)
                term2 = np.einsum('xyab,xybc->xyac', dP_dy, dP_dx)
                comm = term1 - term2
                
                P_comm = np.einsum('xyab,xybc->xyac', P_grid, comm)
                trace = np.einsum('xyaa->xy', P_comm)
                F_xy = (1j * trace).real
                
                # Serialize P_real and P_imag as nested lists of shape (Q, Q, n_grid, n_grid)
                # and transpose to align with Plotly format (y, x)
                P_real_list = [[[] for _ in range(Q)] for _ in range(Q)]
                P_imag_list = [[[] for _ in range(Q)] for _ in range(Q)]
                for a in range(Q):
                    for b in range(Q):
                        P_real_list[a][b] = P_grid[:, :, a, b].real.T.tolist()
                        P_imag_list[a][b] = P_grid[:, :, a, b].imag.T.tolist()
                
                grid_data = {
                    "x": x_ticks.tolist(),
                    "y": y_ticks.tolist(),
                    "z_amp": norm_grid.T.tolist(),      # Transpose to align with Plotly's (y, x) contour grid format
                    "z_phase": phase_grid.T.tolist(),  # Transpose to align with Plotly's (y, x) contour grid format
                    "dom_orbital": dom_idx,
                    "dom_orbital_label": symbols[dom_idx] if (symbols and dom_idx < len(symbols)) else f"Orbital {dom_idx}",
                    "P_real": P_real_list,
                    "P_imag": P_imag_list,
                    "berry_curvature": F_xy.T.tolist()
                }
                out["grid_data"] = grid_data

            # When the band is not isolated, the FHS single-band value is
            # unreliable; report the (continuous-zero) section winding instead.
            if not iso["isolated"]:
                out["chern_number"] = int(ana["C"])
            # interpret, with isolation taking priority (a non-isolated band has
            # no clean single-band Chern regardless of the section's structure).
            if not iso["isolated"]:
                base = ("평탄 밴드가 이웃 밴드와 접촉 → 고립 밴드가 아니므로 단일 밴드 "
                        "Chern이 정수로 잘 정의되지 않음.")
                if ana.get("has_discontinuous_zero") and ana.get("C_all_zeros"):
                    wlist = ", ".join(
                        f"{z['label']}:w={z['winding']:+d}"
                        f"{'(불연속)' if not z['projector_continuous'] else ''}"
                        for z in ana["per_zero"])
                    out["explanation"] = (
                        base + f" 접촉점에서 단면이 winding을 갖지만(영점별 [{wlist}]) "
                        "사영자가 불연속이라 유효 Chern 항이 아님 — 접촉을 갭으로 열면 "
                        f"|C|={abs(ana['C_all_zeros'])}까지 실현 가능. 연속 영점 winding 합 = "
                        f"{ana['C']}.")
                else:
                    out["explanation"] = (
                        base + " (접촉을 갭으로 열어야 정수 C를 가짐.) 연속 영점 winding 합 = "
                        f"{ana['C']}.")
            elif not ana["has_common_zero"]:
                out["explanation"] = ("공통 영점 없음 → 전역적으로 매끄러운 단면 → C=0 "
                                      "(Proposition 1).")
            elif not ana["projector_continuous"]:
                out["explanation"] = (
                    f"공통 영점 {ana['n_common_zeros']}개가 있으나 일부에서 사영자가 불연속"
                    "(loop 위 단면 rank>1, rank(A)≠1) → 이 CLS 게이지에서 사영자가 매끄럽게 "
                    "확장되지 않음. 수치 FHS 값을 신뢰.")
            elif not ana["nonzero_winding"]:
                out["explanation"] = (
                    f"공통 영점 {ana['n_common_zeros']}개 + 연속 사영자이나 모든 국소 "
                    "winding=0 → C=0.")
            elif ana["C"] == 0:
                wlist = ", ".join(f"{z['label']}:w={z['winding']:+d}"
                                  for z in ana["per_zero"])
                out["explanation"] = (
                    f"공통 영점들의 국소 winding이 0이 아니나 서로 상쇄되어 총합 "
                    f"C=Σwᵢ=0 (caveat iv). 영점별 [{wlist}].")
            else:
                wlist = ", ".join(f"{z['label']}:w={z['winding']:+d}"
                                  for z in ana["per_zero"])
                ho = any(z["order"] > 1 for z in ana["per_zero"])
                tag = " (고차 영점 포함)" if ho else ""
                out["explanation"] = (
                    "공통 영점 + 연속 사영자 + 0이 아닌 winding → "
                    f"C=Σwᵢ={ana['C']}{tag} (위상학적으로 비자명). 영점별 [{wlist}].")
        except Exception as e:
            out["analytic_error"] = repr(e)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Brillouin-zone explorer: scan the whole BZ, locate every singularity, sum C
# ──────────────────────────────────────────────────────────────────────────────
def explore_brillouin_zone(H_k, lattice, eps0, M, x_k=None, symbols=None,
                           grid_n=32, scan_n=200, loop_n=128):
    """
    Full Brillouin-zone analysis of a flat band: scan the entire BZ, locate ALL
    common-zero singularities, classify each (order, local winding, projector
    continuity, high-symmetry label) and sum the windings into a Chern number,
    cross-checked against the gauge-invariant FHS lattice Chern.

    Returns a self-contained report:
      chern_number, well_defined, isolation,
      numerical (FHS), analytic (per-zero), agreement,
      singularities (clean per-zero list), bz_landscape (|f|^2 map),
      summary (human-readable).
    """
    rep = analyze_flat_band_chern(H_k, lattice, eps0, M, x_k=x_k,
                                  symbols=symbols, grid_n=grid_n, scan_n=scan_n)

    sings = rep.get("singularities", [])
    iso = rep.get("isolation", {})
    num = rep.get("numerical", {})
    ana = rep.get("analytic", {})

    lines = []
    lines.append(f"평탄 밴드 E={rep['energy']:+.4f} (M={rep['M']}) BZ 전면 탐색")
    lines.append(f"  · 밴드 고립: {'예' if iso.get('isolated') else '아니오 (이웃 밴드와 접촉)'}"
                 f"  (gap_below={iso.get('gap_below')}, gap_above={iso.get('gap_above')})")
    lines.append(f"  · 수치 FHS Chern: C = {num.get('C')} "
                 f"(raw {num.get('C_raw'):+.4f}, grid {num.get('grid')})"
                 if num else "  · 수치 FHS: 미계산")
    if x_k is not None and "analytic_error" not in rep:
        lines.append(f"  · 공통 영점(특이점) {len(sings)}개 발견:")
        if sings:
            for z in sings:
                cont = "연속" if z["projector_continuous"] else "불연속"
                fr = z["k_frac"]
                lines.append(
                    f"      - {z['label']:>10s}  frac=({fr[0]:+.3f},{fr[1]:+.3f})"
                    if len(fr) == 2 else f"      - {z['label']}  frac={fr}")
                lines[-1] += (f"  w={z['winding']:+d}  order={z['order']}  사영자:{cont}")
            lines.append(f"  · 국소 winding 합 C = Σwᵢ = {ana.get('C')}")
        else:
            lines.append("      (없음 → Proposition 1에 의해 C=0)")
        lines.append(f"  · 두 방법 일치: {'예' if rep.get('agreement') else '아니오'}")
    lines.append(f"  ⇒ 결론: Chern = {rep['chern_number']}"
                 f"{'' if rep['well_defined'] else '  (단일밴드로 잘 정의되지 않음 — 접촉 갭 필요)'}")
    rep["summary"] = "\n".join(lines)
    return rep
