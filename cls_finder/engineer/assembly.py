"""
Module 4 — Analytical Assembly & Projector Continuity Verification (Phase 4).

Combines the per-singularity CLS designs of Phase 2/3 into the final
finite-Fourier Bloch eigenvector

    f_alpha(k) = e^{i zeta^(alpha)} * Sum_R c_{alpha,R} e^{i k.R}            (Eq. 71)
    c_{alpha,R} = A_R * e^{i theta_R}

as a list of cls_finder.core.laurent.LaurentPoly (one per sublattice), and
re-uses cls_finder.classify.chern's Jacobian / rank-1 / winding machinery
(unchanged) to verify, at each target singularity k_i:

  - condition (ii) projector continuity: [A_x | A_y] has C-rank 1
    (chern.jacobian_at_zero + chern.local_winding), and
  - condition (iii) nonzero local winding: sgn Im<A_x, A_y> == w_i, confirmed
    robustly by the contour integral (chern.loop_winding).

`vortex_vector` extracts the normalized limit vector |l> = v_i (Note A
Eq. 16) used by Module 5 to analytically patch P(k_i) = |l><l| / <l|l>.

ZETA NOTE (ARCHITECTURE.md Sec. 0.1): Phase 2/3 (pairing.py / chiral.py) work
in the zeta=0 gauge. zeta^(alpha) is injected here, exactly once, as an
overall per-sublattice phase factor e^{i zeta^(alpha)} multiplying every
coefficient of f_alpha(k). Because the common-zero condition, the rank-1
condition and Im<A_x,A_y> are all invariant under independent per-row phase
factors (the zeta's cancel in <.,.> = sum_alpha conj(.)_alpha (.)_alpha when
multiplied by conj(e^{i zeta_alpha}) e^{i zeta_alpha} = 1), every verification
in this module gives IDENTICAL results regardless of the chosen zeta's.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from cls_finder.core.laurent import LaurentPoly
from cls_finder.classify import chern as _chern
from cls_finder.engineer.spec import LatticeSpec
from cls_finder.engineer.pairing import CLSDesign


def build_x_k(lattice_spec: LatticeSpec, cls_designs: List[CLSDesign]) -> List[LaurentPoly]:
    """Assemble f(k) = [f_1(k), ..., f_N(k)] (Eq. 71) from one CLSDesign per
    target singularity.

    Each CLSDesign already satisfies Condition 1 (Module 2) and the chiral
    condition (Module 3) FOR ITS OWN singularity, in the zeta=0 gauge. Their
    real-space coefficients are summed (a CLS centered on a different
    singularity contributes additively to the same c_{alpha,R}; this is the
    superposition used to realize |C| = sum|w_i| > 1 from several
    w_i = +-1 zeros), and zeta^(alpha) is applied as a final overall phase
    per sublattice.
    """
    N = lattice_spec.N
    x_k: List[LaurentPoly] = []
    for alpha in range(N):
        zeta = lattice_spec.sublattices[alpha].zeta
        phase_factor = np.exp(1j * zeta)
        coefs: Dict[Tuple[int, int], complex] = {}
        for design in cls_designs:
            sub = design.sublattices[alpha]
            for (n, m), (A, theta) in sub.sites().items():
                c = A * np.exp(1j * theta) * phase_factor
                coefs[(n, m)] = coefs.get((n, m), 0j) + c
        x_k.append(LaurentPoly(coefs, d=2))
    return x_k


def evaluate_psi(x_k: List[LaurentPoly], k_cart: np.ndarray,
                 primitive_vectors: np.ndarray) -> Tuple[np.ndarray, "np.ndarray | None", float]:
    """f(k), the normalized eigenvector psi(k) = f(k)/||f(k)|| (Eq. 100/Phase4
    step 2), and ||f(k)|| at one Cartesian k-point.

    Returns (f, psi, norm). At a designed common zero ||f(k_i)|| = 0 (Eq. 71
    common-zero condition), so psi is returned as None there -- use
    `vortex_vector` (the analytic limit, Eq. 16) instead, e.g. when assembling
    P(k_i) directly.
    """
    f = np.array([p.evaluate(k_cart, primitive_vectors) for p in x_k], dtype=complex)
    norm = float(np.linalg.norm(f))
    psi = f / norm if norm > 1e-12 else None
    return f, psi, norm


def vortex_vector(x_k: List[LaurentPoly], k_i_cart: np.ndarray,
                  primitive_vectors: np.ndarray) -> np.ndarray:
    """The normalized analytic-limit vector |l> = v_i (Note A Eq. 16) such
    that P(k_i) = |l><l| / <l|l>, obtained from the Jacobian columns
    (A_x, A_y) = jacobian_at_zero(x_k, k_i, prim) at the designed common zero
    k_i.

    By the rank-1 condition (Phase 3), A_x and A_y are (to numerical
    precision) proportional to the same vector v_i = |l>; whichever column
    has the larger norm is used as v_i for numerical stability.
    """
    Ax, Ay = _chern.jacobian_at_zero(x_k, k_i_cart, primitive_vectors)
    v = Ay if np.linalg.norm(Ay) >= np.linalg.norm(Ax) else Ax
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        raise ValueError(
            "Jacobian vanishes at k_i: f(k) has no first-order zero there "
            "(the chiral construction degenerated) -- this should not "
            "happen for a valid CLSDesign; check the bond-vector shell "
            "(chiral.DEFAULT_SHELLS / shell_offset)"
        )
    return v / nrm


def verify_projector_continuity(x_k: List[LaurentPoly], lattice_spec: LatticeSpec,
                                k_i_cart: np.ndarray) -> Dict:
    """Verify conditions (ii) projector continuity and (iii) nonzero local
    winding (Note B Phase 3) at one designed singularity k_i, by reusing
    cls_finder.classify.chern unchanged.

    Returns:
      Ax, Ay        : Jacobian columns at k_i (chern.jacobian_at_zero)
      first_order   : chern.local_winding(Ax, Ay) -- rank-1 + sgn Im<Ax,Ay>
      loop          : chern.loop_winding(x_k, k_i, prim, B) -- robust contour
                       winding + projector_continuous flag (handles |w|>1 too)
    """
    prim = lattice_spec.primitive_vectors
    B = lattice_spec.reciprocal_vectors()
    Ax, Ay = _chern.jacobian_at_zero(x_k, k_i_cart, prim)
    first_order = _chern.local_winding(Ax, Ay)
    loop = _chern.loop_winding(x_k, k_i_cart, prim, B)
    return {"Ax": Ax, "Ay": Ay, "first_order": first_order, "loop": loop}
