"""
Module 5b — Dispersive Band Shape Library.

Provides M_tilde(k): Hermitian, finite-Fourier N x N MatrixPoly "shapes" that
dress the exactly-flat H_core(k) sector
(cls_finder.engineer.hamiltonian.build_H_core) via

    H(k) = E0 I + alpha * H_core(k) + H_core(k) M_tilde(k) H_core(k)

("Hamiltonian Engineering (Finite Hopping TBM 구축)" note, Sec.2/4). Since
H_core(k) f(k) = 0 and H_core(k_i) = 0 at every designed singularity k_i,
M_tilde(k) has NO effect on the flat band's energy/flatness or on the
band-touching at k_i (Sec.3) -- it only shapes the N-1 dispersive bands and
their gap to the flat band away from k_i.

For N=1, H_core(k) = 0 identically, so every shape below is a no-op
regardless of `strength`.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.engineer.spec import LatticeSpec


def shape_none(lattice_spec: LatticeSpec, strength: float) -> Optional[MatrixPoly]:
    """No dispersive dressing: H(k) = E0 I + alpha * H_core(k)."""
    return None


def shape_nn_real(lattice_spec: LatticeSpec, strength: float) -> MatrixPoly:
    """Real nearest-neighbour cyclic hopping, time-reversal-symmetric:

        M_{a,(a+1)%N}(k) += strength * (e^{ik.a1} + e^{ik.a2})

    with the conjugate Laurent terms added to M_{(a+1)%N,a}(k), so M(k) is
    Hermitian for any N >= 1."""
    N = lattice_spec.N
    d = 2
    data = [[LaurentPoly.zero(d) for _ in range(N)] for _ in range(N)]
    for a in range(N):
        b = (a + 1) % N
        for exp in ((1, 0), (0, 1)):
            neg = (-exp[0], -exp[1])
            data[a][b] = data[a][b] + LaurentPoly.monomial(exp, strength)
            data[b][a] = data[b][a] + LaurentPoly.monomial(neg, strength)
    return MatrixPoly(data, d)


def shape_onsite_mass(lattice_spec: LatticeSpec, strength: float) -> MatrixPoly:
    """Diagonal on-site mass staggering M_{a,a}(k) = strength * a
    (k-independent), lifting the degeneracy of the N-1 dispersive bands for
    N >= 3."""
    N = lattice_spec.N
    d = 2
    data = [[LaurentPoly.zero(d) for _ in range(N)] for _ in range(N)]
    for a in range(N):
        if a != 0:
            data[a][a] = LaurentPoly.constant(strength * a, d)
    return MatrixPoly(data, d)


def shape_haldane(lattice_spec: LatticeSpec, strength: float) -> MatrixPoly:
    """Time-reversal-symmetry-breaking imaginary next-nearest-neighbour
    (Haldane-like) on-site terms: for each sublattice a, sign = +1 if a is
    even else -1, and for the diagonal NNN bond vectors R in {a1+a2, a1-a2},

        M_{a,a}(k) += i*sign*strength * (e^{ik.R} - e^{-ik.R})
                    = -2*sign*strength * sin(k.R)

    -- a real diagonal entry (Hermitian), but the underlying real-space
    hopping t_{aa}(+-R) = +-i*sign*strength is genuinely complex/imaginary,
    the TRS-breaking ingredient needed for a nonzero-Chern dispersive sector
    (analogous to the Haldane model's NNN imaginary hopping)."""
    N = lattice_spec.N
    d = 2
    data = [[LaurentPoly.zero(d) for _ in range(N)] for _ in range(N)]
    for a in range(N):
        sign = 1.0 if a % 2 == 0 else -1.0
        coeff = 1j * sign * strength
        for R in ((1, 1), (1, -1)):
            neg = (-R[0], -R[1])
            data[a][a] = (data[a][a]
                          + LaurentPoly.monomial(R, coeff)
                          + LaurentPoly.monomial(neg, -coeff))
    return MatrixPoly(data, d)


def shape_combo(lattice_spec: LatticeSpec, strength: float) -> MatrixPoly:
    """shape_nn_real(strength) + shape_onsite_mass(strength)
    + shape_haldane(0.5*strength) -- a combined TRS-breaking, mass-staggered
    NN+NNN dispersive sector."""
    return (shape_nn_real(lattice_spec, strength)
            + shape_onsite_mass(lattice_spec, strength)
            + shape_haldane(lattice_spec, 0.5 * strength))


DISPERSIVE_SHAPES: Dict[str, Callable[[LatticeSpec, float], Optional[MatrixPoly]]] = {
    "none": shape_none,
    "nn_real": shape_nn_real,
    "onsite_mass": shape_onsite_mass,
    "haldane": shape_haldane,
    "combo": shape_combo,
}


def build_dispersive_M(lattice_spec: LatticeSpec, shape: str = "nn_real",
                       strength: float = 0.3) -> Optional[MatrixPoly]:
    """Build M_tilde(k) for the named dispersive shape (DISPERSIVE_SHAPES)."""
    if shape not in DISPERSIVE_SHAPES:
        raise ValueError(
            f"unknown dispersive_shape {shape!r}; choices: {sorted(DISPERSIVE_SHAPES)}")
    return DISPERSIVE_SHAPES[shape](lattice_spec, strength)
