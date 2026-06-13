"""
Module 5 — Hamiltonian Engineering & Inverse Fourier Transform (Phase 5).

Implements Note B Phase 5 / Eq. 127:

    H(k) = E0 * P(k) + [I - P(k)] M(k) [I - P(k)]

where P(k) = |psi(k)><psi(k)| (Phase 4) is a rank-1 projector with
P(k_i) = E0-flat-band eigenvalue exactly E0 at every k (E0*P contributes E0
on the P-subspace, and (I-P)M(I-P) annihilates it there), so the band
projected by P(k) is PERFECTLY FLAT at E0 for ANY Hermitian M(k) -- M(k) only
shapes the remaining N-1 dispersive bands.

`NumericHk` duck-types the (.rows, .cols, .d, .evaluate, .evaluate_batch)
interface of cls_finder.core.matrixpoly.MatrixPoly so it can be passed
UNCHANGED into cls_finder.band.bands / cls_finder.classify.chern /
cls_finder.viz.plot for verification and visualization.

`inverse_fourier_transform` recovers the real-space tight-binding hoppings

    t_{alpha,beta}(R) = (1/N_k) * sum_k H_{alpha,beta}(k) e^{-i k.R}

on an n_grid x n_grid BZ grid (cls_finder.classify.chern._frac_grid), with a
Parseval-based truncation diagnostic (||H||^2 retained for |R| <= R_cut vs
the full sum) so the pipeline's Rhim-Yang feedback loop can decide whether
R_cut needs to grow.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.classify import chern as _chern
from cls_finder.engineer.spec import LatticeSpec, DesignTarget
from cls_finder.engineer.assembly import vortex_vector


def default_M_k(lattice_spec: LatticeSpec, t: float = 0.3, delta: float = 0.5) -> MatrixPoly:
    """A generic local Hermitian N x N dispersive matrix (Note B Phase 5
    step 2):

        M(k) = t * (S e^{ik.a1} + S^T e^{-ik.a1} + S e^{ik.a2} + S^T e^{-ik.a2})
               + delta * diag(0, 1, ..., N-1)

    where S is the cyclic shift S_{alpha,(alpha+1) mod N} = 1 (S^T its
    transpose). Each off-diagonal pair (alpha, beta=(alpha+1)%N) acquires
    M_{alpha,beta}(k) = t(e^{ik.a1}+e^{ik.a2}) and M_{beta,alpha}(k) = its
    Laurent conjugate -- so the hopping part of M(k) is Hermitian for every k
    and N. The on-site (R=0) term delta*diag(0,...,N-1) is the shortest
    possible local term (locality is preserved) and is added so that, for
    N=2, M(k) is not a pure scalar-times-fixed-matrix t*g(k)*sigma_x: without
    it (I-P(k))M(k)(I-P(k)) would vanish identically along the nodal line
    g(k)=cos(k.a1)+cos(k.a2)=0 and the dispersive band would touch the flat
    band at E0 along that whole line.

    With generic delta!=0 the line-touching is broken to (generically)
    isolated points. The exact P(k) (a rational, non-local function of e^{ik}
    via Eq. 100's normalization f/||f||) can still touch the dispersive
    sector AT such an isolated point if it happens to coincide with another
    feature of f(k) or of g(k)=0 (e.g. for the default Gamma/w=+1 example,
    H(k) touches at (kx,ky)=(pi/2,-pi/2) where g(k)=0 AND f_0(k)=0
    simultaneously) -- band_isolation(H_k, ...) may report isolated=False.
    This is harmless: Phase4's analytic_chern(x_k) (the ground truth for the
    flat band's topology) does not depend on M(k) at all, and the
    IFT-truncated H_trunc -- the actual locality-respecting tight-binding
    deliverable, whose long-range P(k) tail is cut off -- is generically
    gapped (this is what pipeline.design_flat_band's Phase5 R_cut loop
    actually checks).

    N=1 reduces to the real diagonal scalar 2t(cos k.a1 + cos k.a2), which
    has no effect on H(k) since (I-P)=0 identically when N=1.
    """
    N = lattice_spec.N
    d = 2
    data = [[LaurentPoly.zero(d) for _ in range(N)] for _ in range(N)]
    for alpha in range(N):
        beta = (alpha + 1) % N
        for exp in ((1, 0), (0, 1)):
            neg = (-exp[0], -exp[1])
            data[alpha][beta] = data[alpha][beta] + LaurentPoly.monomial(exp, t)
            data[beta][alpha] = data[beta][alpha] + LaurentPoly.monomial(neg, t)
    for alpha in range(N):
        if alpha != 0:
            data[alpha][alpha] = data[alpha][alpha] + LaurentPoly.constant(delta * alpha, d)
    return MatrixPoly(data, d)


class NumericHk:
    """H(k) = E0*P(k) + (I-P(k)) M(k) (I-P(k)) (Eq. 127), duck-typing
    MatrixPoly's (.rows, .cols, .d, .evaluate, .evaluate_batch).

    P(k) = f(k) f(k)^dagger / ||f(k)||^2 is computed directly from x_k at
    every k. At the designed common zeros k_i (||f(k_i)||=0), P is replaced
    by its analytic limit P(k_i) = v_i v_i^dagger / <v_i|v_i> (Note A Eq. 16,
    `assembly.vortex_vector`); `vortex_vectors` is the list of (k_i_cart,
    v_i_normalized) pairs used for this patch, matched to query points by
    fractional-coordinate distance modulo the reciprocal lattice (so periodic
    images of k_i are patched too).
    """

    def __init__(self, x_k: List[LaurentPoly], M: MatrixPoly, E0: float,
                 lattice_spec: LatticeSpec,
                 vortex_vectors: List[Tuple[np.ndarray, np.ndarray]],
                 zero_tol: float = 1e-8):
        self.x_k = list(x_k)
        self.M = M
        self.E0 = float(E0)
        self.lattice_spec = lattice_spec
        self.N = len(x_k)
        self.rows = self.N
        self.cols = self.N
        self.d = 2
        self.vortex_vectors = list(vortex_vectors)
        self.zero_tol = float(zero_tol)

    def _projector_batch(self, k_vals: np.ndarray,
                         primitive_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P(k), f(k), ||f(k)||^2 for a batch of k-points, with the analytic
        patch applied at points where ||f(k)||^2 < zero_tol."""
        F = np.stack([p.evaluate_batch(k_vals, primitive_vectors) for p in self.x_k],
                     axis=1)                                          # (Np, N)
        norm2 = np.sum(np.abs(F) ** 2, axis=1)                        # (Np,)
        small = norm2 < self.zero_tol
        safe = np.where(small, 1.0, norm2)
        P = np.einsum('ka,kb->kab', F, F.conj()) / safe[:, None, None]

        idx = np.where(small)[0]
        if idx.size and self.vortex_vectors:
            B = self.lattice_spec.reciprocal_vectors()
            vortex_fracs = [(_chern._cart_to_frac(k_i, B), v)
                           for k_i, v in self.vortex_vectors]
            for i in idx:
                f_k = _chern._cart_to_frac(k_vals[i], B)
                best_v, best_d = None, None
                for f_i, v in vortex_fracs:
                    diff = _chern._wrap_frac(f_k - f_i)
                    dd = float(np.dot(diff, diff))
                    if best_d is None or dd < best_d:
                        best_d, best_v = dd, v
                P[i] = np.outer(best_v, best_v.conj())
        return P, F, norm2

    def projector(self, k_vals: np.ndarray, primitive_vectors: np.ndarray) -> np.ndarray:
        """P(k) for a batch of k-points, shape (Np, N, N). Exposed for
        diagnostics (e.g. checking Tr P = 1 and P^2 = P)."""
        k_vals = np.atleast_2d(np.asarray(k_vals, dtype=float))
        P, _, _ = self._projector_batch(k_vals, primitive_vectors)
        return P

    def evaluate_batch(self, k_vals: np.ndarray, primitive_vectors: np.ndarray) -> np.ndarray:
        k_vals = np.asarray(k_vals, dtype=float)
        P, _, _ = self._projector_batch(k_vals, primitive_vectors)
        I = np.eye(self.N, dtype=complex)
        IminusP = I[None, :, :] - P
        Mb = self.M.evaluate_batch(k_vals, primitive_vectors)
        H = self.E0 * P + np.einsum('kab,kbc,kcd->kad', IminusP, Mb, IminusP)
        return H

    def evaluate(self, k_val: np.ndarray, primitive_vectors: np.ndarray) -> np.ndarray:
        k_val = np.asarray(k_val, dtype=float).reshape(1, -1)
        return self.evaluate_batch(k_val, primitive_vectors)[0]


def build_local_flat_hamiltonian(x_k: List[LaurentPoly], E0: float = 0.0,
                                 t: float = 1.0,
                                 M0: Optional[np.ndarray] = None) -> MatrixPoly:
    """A FINITE-RANGE tight-binding Hamiltonian whose flat band at E0 is
    EXACT (Rhim-Yang singular flat band), built directly from the CLS Bloch
    vector f(k) = x_k -- WITHOUT the inverse-Fourier truncation that breaks
    flatness.

        Q(k) = ||f(k)||^2 I - f(k) f(k)^dagger        (Hermitian, PSD, finite-range)
        H(k) = E0 I + t Q(k)            (M0 None, fewest terms)
             = E0 I + Q(k) M0 Q(k)      (M0 a constant NxN Hermitian, to split
                                         the N-1 dispersive bands for N >= 3)

    Key properties (all EXACT, not numerical/approximate):
      * Q(k) f(k) = ||f||^2 f - f (f^dagger f) = ||f||^2 f - ||f||^2 f = 0,
        so H(k) f(k) = E0 f(k): the band spanned by f is PERFECTLY FLAT at E0
        for every k -- and stays flat when this model is reloaded into the
        band/CLS analyser, because every H_{ab}(k) is a genuine finite Laurent
        polynomial (real-space hopping of bounded range = 2x the CLS extent),
        not a truncation of the non-local projector f f^dagger/||f||^2.
      * H is Hermitian (Q^dagger = Q; with M0 Hermitian, (Q M0 Q)^dagger = Q M0 Q).
      * The flat band TOUCHES the dispersive sector exactly where ||f(k)||^2 = 0,
        i.e. at the designed common zero / singularity k_i -- which is precisely
        where the flat band's Chern winding is generated (Note A Sec.8). A
        finite-range flat band carrying nonzero C MUST be such a touching
        (singular) band; this construction realises that by symmetry rather
        than forcing an (impossible) isolated finite-range Chern flat band.

    The dispersive bands are the nonzero eigenvalues of t*Q (= t||f||^2 with
    multiplicity N-1 when M0 is None). For N=2 this is the complete minimal
    topological flat-band model; for N>=3, pass a constant Hermitian M0 with
    distinct spectrum to lift the dispersive degeneracy.
    """
    N = len(x_k)
    d = 2
    f = list(x_k)
    fbar = [p.conjugate() for p in f]                       # f_a^*(k), finite-range

    norm2 = LaurentPoly.zero(d)
    for a in range(N):
        norm2 = norm2 + f[a] * fbar[a]                      # ||f(k)||^2 (real)

    Qdata = [[LaurentPoly.zero(d) for _ in range(N)] for _ in range(N)]
    for a in range(N):
        for b in range(N):
            term = (norm2 if a == b else LaurentPoly.zero(d)) - f[a] * fbar[b]
            Qdata[a][b] = term
    Q = MatrixPoly(Qdata, d)

    I = MatrixPoly.identity(N, d)
    if M0 is None:
        H_disp = Q * float(t)
    else:
        M0 = np.asarray(M0, dtype=complex)
        if M0.shape != (N, N) or np.linalg.norm(M0 - M0.conj().T) > 1e-9:
            raise ValueError("M0 must be a constant N x N Hermitian matrix")
        M0mp = MatrixPoly([[LaurentPoly.constant(M0[a, b], d) for b in range(N)]
                           for a in range(N)], d)
        H_disp = Q * M0mp * Q
    return I * float(E0) + H_disp


def build_hamiltonian(x_k: List[LaurentPoly], lattice_spec: LatticeSpec,
                      target: DesignTarget, E0: float = 0.0,
                      M: Optional[MatrixPoly] = None, t: float = 0.3,
                      delta: float = 0.5, zero_tol: float = 1e-8) -> NumericHk:
    """Phase 5 assembly: pick M(k) (default_M_k if not given), extract the
    vortex vector v_i (Eq. 16) at every target singularity from x_k, and
    return the NumericHk H(k) = E0*P(k) + (I-P(k)) M(k) (I-P(k))."""
    if M is None:
        M = default_M_k(lattice_spec, t=t, delta=delta)
    vortex_vectors = []
    for sing in target.singularities:
        k_i_cart = sing.k_cartesian(lattice_spec)
        v = vortex_vector(x_k, k_i_cart, lattice_spec.primitive_vectors)
        vortex_vectors.append((k_i_cart, v))
    return NumericHk(x_k, M, E0, lattice_spec, vortex_vectors, zero_tol=zero_tol)


def inverse_fourier_transform(H_k, lattice_spec: LatticeSpec, n_grid: int = 24,
                              R_cut: int = 3, tol: float = 1e-10) -> Dict:
    """t_{alpha,beta}(R) = (1/N_k) sum_k H_{alpha,beta}(k) e^{-i k.R}
    (R = n*a1 + m*a2) on an n_grid x n_grid BZ grid (chern._frac_grid),
    using k.R = 2*pi*(f1*n + f2*m).

    Returns a dict with:
      hoppings        : {(alpha, beta, (n, m)): complex} for |n|,|m| <= R_cut
                         and |t| > tol  (the truncated tight-binding model)
      hoppings_full   : {(n, m): (N,N) complex} for the FULL n_grid x n_grid
                         range of (n, m) (untruncated)
      total_weight    : sum_R ||t(R)||_F^2 over the full range (Parseval:
                         equals the mean of ||H(k)||_F^2 over the BZ grid)
      truncated_weight: sum_{|R|<=R_cut} ||t(R)||_F^2
      truncation_ratio: 1 - truncated_weight/total_weight -- spectral weight
                         DROPPED by truncating to |R| <= R_cut (the pipeline's
                         Rhim-Yang feedback loop increases R_cut while this is
                         non-negligible AND the truncated H is gapped but
                         C_trunc != C_target).
    """
    prim = lattice_spec.primitive_vectors
    B = lattice_spec.reciprocal_vectors()
    frac = _chern._frac_grid(n_grid, n_grid)            # (n_grid^2, 2) in [0,1)
    k_cart = frac @ B
    Hb = H_k.evaluate_batch(k_cart, prim)               # (Np, N, N)
    Nk = frac.shape[0]
    N = H_k.rows

    half = n_grid // 2
    offsets = range(-half, n_grid - half)
    hoppings_full: Dict[Tuple[int, int], np.ndarray] = {}
    total_weight = 0.0
    for n in offsets:
        for m in offsets:
            phase = np.exp(-1j * 2.0 * np.pi * (frac[:, 0] * n + frac[:, 1] * m))  # (Np,)
            tR = np.einsum('k,kab->ab', phase, Hb) / Nk
            hoppings_full[(n, m)] = tR
            total_weight += float(np.sum(np.abs(tR) ** 2))

    hoppings: Dict[Tuple[int, int, Tuple[int, int]], complex] = {}
    cut_weight = 0.0
    for (n, m), tR in hoppings_full.items():
        if max(abs(n), abs(m)) <= R_cut:
            cut_weight += float(np.sum(np.abs(tR) ** 2))
            for alpha in range(N):
                for beta in range(N):
                    val = complex(tR[alpha, beta])
                    if abs(val) > tol:
                        hoppings[(alpha, beta, (n, m))] = val

    truncation_ratio = 1.0 - (cut_weight / total_weight if total_weight > 1e-300 else 1.0)
    return {
        "hoppings": hoppings,
        "hoppings_full": hoppings_full,
        "R_cut": R_cut,
        "n_grid": n_grid,
        "total_weight": total_weight,
        "truncated_weight": cut_weight,
        "truncation_ratio": truncation_ratio,
    }


def hoppings_to_matrixpoly(hoppings: Dict[Tuple[int, int, Tuple[int, int]], complex],
                           N: int, d: int = 2) -> MatrixPoly:
    """Re-assemble the truncated tight-binding hoppings as a MatrixPoly
    H_trunc(k) = sum_{alpha,beta,R} t_{alpha,beta}(R) e^{ik.R} |alpha><beta|,
    ready for cls_finder.classify.chern.numerical_chern / band_isolation /
    cls_finder.viz.plot.plot_bands."""
    data = [[LaurentPoly.zero(d) for _ in range(N)] for _ in range(N)]
    for (alpha, beta, (n, m)), val in hoppings.items():
        data[alpha][beta] = data[alpha][beta] + LaurentPoly.monomial((n, m), val)
    return MatrixPoly(data, d)
