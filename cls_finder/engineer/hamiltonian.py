"""
Module 5 — Hamiltonian Engineering (Phase 5).

Implements the "Hamiltonian Engineering (Finite Hopping TBM 구축)" note's
generalized Phase 5:

    H_core(k) = ||f(k)||^2 I - f(k) f(k)^dagger        (polynomial, Hermitian, PSD)
    H(k)      = E0 I + alpha * H_core(k) + H_core(k) M_tilde(k) H_core(k)

for the CLS Bloch eigenvector f(k) = x_k (Phase 1-4) and any Hermitian,
finite-Fourier N x N "dispersive shape" M_tilde(k)
(cls_finder.engineer.dispersive). Every term is built from polynomial
multiplication/addition of finite Laurent polynomials (cls_finder.core.laurent
/ cls_finder.core.matrixpoly), so H(k) is itself a finite Laurent polynomial
matrix -- i.e. a STRICTLY FINITE-RANGE tight-binding model
(matrixpoly_to_hoppings).

Key exact properties (hold for ANY alpha, ANY Hermitian M_tilde(k)):
  * H_core(k) f(k) = (||f||^2 - ||f||^2) f(k) = 0, so H(k) f(k) = E0 f(k):
    the band spanned by f is PERFECTLY FLAT at E0 for every k -- and stays
    flat when this model is reloaded into the band/CLS analyser.
  * At a designed common zero f(k_i) = 0 => H_core(k_i) = 0 => H(k_i) = E0 I:
    the flat band TOUCHES every dispersive band at E0 (band touching), which
    is where the flat band's Chern winding lives (Note A Sec.8).
  * H(k) is Hermitian: H_core^dagger = H_core (it is real and symmetric in
    f f^dagger), and with M_tilde Hermitian, (H_core M_tilde H_core)^dagger
    = H_core M_tilde^dagger H_core = H_core M_tilde H_core.

For N=1, H_core(k) = 0 identically, so H(k) = E0 I -- a flat band has no
dispersive sector to engineer.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.classify import chern as _chern
from cls_finder.engineer.spec import LatticeSpec


def build_H_core(x_k: List[LaurentPoly]) -> MatrixPoly:
    """H_core(k) = ||f(k)||^2 I - f(k) f(k)^dagger (Hermitian, PSD, finite-range).

    H_core(k) f(k) = 0 for every k, and H_core(k_i) = 0 at every common zero
    f(k_i) = 0 (Sec.1/3 of the note)."""
    N = len(x_k)
    d = 2
    f = list(x_k)
    fbar = [p.conjugate() for p in f]                       # f_a^*(k), finite-range

    norm2 = LaurentPoly.zero(d)
    for a in range(N):
        norm2 = norm2 + f[a] * fbar[a]                      # ||f(k)||^2 (real)

    data = [[LaurentPoly.zero(d) for _ in range(N)] for _ in range(N)]
    for a in range(N):
        for b in range(N):
            term = (norm2 if a == b else LaurentPoly.zero(d)) - f[a] * fbar[b]
            data[a][b] = term
    return MatrixPoly(data, d)


def build_engineered_hamiltonian(x_k: List[LaurentPoly], E0: float = 0.0,
                                  alpha: float = 1.0,
                                  M_tilde: Optional[MatrixPoly] = None) -> MatrixPoly:
    """H(k) = E0 I + alpha * H_core(k) + H_core(k) M_tilde(k) H_core(k)
    (Sec.2 of the note).

    M_tilde, if given, must be an N x N Hermitian MatrixPoly with the same
    Laurent dimension d=2 as x_k (cls_finder.engineer.dispersive.build_dispersive_M).
    """
    N = len(x_k)
    d = 2
    Q = build_H_core(x_k)
    I = MatrixPoly.identity(N, d)
    H_disp = Q * float(alpha)
    if M_tilde is not None:
        if M_tilde.rows != N or M_tilde.cols != N or M_tilde.d != d:
            raise ValueError(
                f"M_tilde must be an {N}x{N} MatrixPoly with d={d}, got "
                f"{M_tilde.rows}x{M_tilde.cols} d={M_tilde.d}")
        if not (M_tilde - M_tilde.dagger()).is_zero(1e-9):
            raise ValueError("M_tilde must be Hermitian")
        H_disp = H_disp + Q * M_tilde * Q
    return I * float(E0) + H_disp


def matrixpoly_to_hoppings(H: MatrixPoly, tol: float = 1e-9) -> Dict:
    """{(alpha, beta, (n, m)): complex} for the nonzero hoppings of a
    finite-range MatrixPoly (the real-space tight-binding model)."""
    hops: Dict = {}
    for a in range(H.rows):
        for b in range(H.cols):
            for (n, m), val in H.data[a][b].coefs.items():
                if abs(val) > tol:
                    hops[(a, b, (int(n), int(m)))] = complex(val)
    return hops


def max_hopping_range(hoppings: Dict) -> int:
    return max((max(abs(n), abs(m)) for (_, _, (n, m)) in hoppings), default=0)


def flatness_deviation(H, x_k: List[LaurentPoly], lattice_spec: LatticeSpec,
                       E0: float, n: int = 16) -> float:
    """max_k || H(k) psi(k) - E0 psi(k) || over an n x n BZ grid, with
    psi(k)=f(k)/||f(k)|| from x_k (skipping the common zeros). Works for any
    H exposing .evaluate_batch (e.g. MatrixPoly)."""
    try:
        prim = lattice_spec.primitive_vectors
        B = lattice_spec.reciprocal_vectors()
        frac = _chern._frac_grid(n, n)
        k_cart = frac @ B
        F = np.stack([p.evaluate_batch(k_cart, prim) for p in x_k], axis=1)
        norm2 = np.sum(np.abs(F) ** 2, axis=1)
        Hb = H.evaluate_batch(k_cart, prim)
        worst = 0.0
        for i in range(k_cart.shape[0]):
            if norm2[i] < 1e-6:
                continue
            psi = F[i] / np.sqrt(norm2[i])
            worst = max(worst, float(np.linalg.norm(Hb[i] @ psi - E0 * psi)))
        return worst
    except Exception:
        return -1.0


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
