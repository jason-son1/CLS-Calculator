"""
Module 3 — Chiral Condition Optimizer (Phase 3).

For one sublattice alpha and one singularity k_i with target winding
w_i = +-1, this module solves the Chiral Symmetry Equation

    a^(alpha) = -i * w_i * b^(alpha)                       (Note A Eq. 33-46)

where (a^(alpha), b^(alpha)) are the bond-vector moments of the two
destructive-interference pairs built in pairing.py:

    a^(alpha) = Sum_p A_p e^{i Phi_p} d_{p,x}
    b^(alpha) = Sum_p A_p e^{i Phi_p} d_{p,y}
    Phi_p = Theta_{j1,p} + pi/2

Writing D_p = d_{p,x} + i*w_i*d_{p,y}, the chiral condition is the single
complex equation

    Sum_p A_p e^{i Phi_p} D_p = 0.

For P=2 pairs (the minimal construction) this has the closed form (gauge
Phi_2 = 0):

    A_1 = |D_2|,  A_2 = |D_1|,  Phi_1 = pi + arg(D_2) - arg(D_1)

which is well-defined whenever d_1, d_2 != 0 and linearly independent
(cross(d_1, d_2) != 0) -- ALWAYS realizable with a primitive-vector bond pair
such as (a1, a2).

Verifying a^(alpha) + i*w_i*b^(alpha) = Sum_p A_p e^{i Phi_p} D_p == 0
identically gives a^(alpha) = -i*w_i*b^(alpha), and since
\\bar a b = i*w_i|b|^2, sgn Im<A_x,A_y> = w_i Sum_alpha|b^(alpha)|^2 has sign
w_i -- i.e. chern.local_winding() will report exactly w_i (ARCHITECTURE.md
Sec. 0.3).

To avoid the "identical CLS per sublattice" degeneracy (f(k) = g(k)*v_0,
constant projector, C=0 globally despite a correct LOCAL winding at k_i --
ARCHITECTURE.md Sec. 0.4), each sublattice alpha is assigned a DIFFERENT
bond-vector shell pair (d_1, d_2) from DEFAULT_SHELLS, cycling modulo an
`offset` that the feedback loop (pipeline.py) can vary on retry.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from cls_finder.engineer.spec import LatticeSpec, SingularityTarget
from cls_finder.engineer.pairing import CLSDesign, SublatticeCLS, make_pair


# ──────────────────────────────────────────────────────────────────────────
# Shell library: integer bond-vector pairs (d1, d2) = ((n1,m1), (n2,m2)).
#
# Only requirement is cross(d1, d2) = n1*m2 - n2*m1 != 0 (linear independence
# in Cartesian space, for ANY a1, a2). Diverse shapes/orientations are
# included so that cycling them per sublattice breaks the global-degeneracy
# described in ARCHITECTURE.md Sec. 0.4.
# ──────────────────────────────────────────────────────────────────────────
DEFAULT_SHELLS: List[Tuple[Tuple[int, int], Tuple[int, int]]] = [
    ((1, 0), (0, 1)),
    ((1, 0), (1, 1)),
    ((0, 1), (1, 1)),
    ((1, 1), (1, -1)),
    ((1, 0), (-1, 1)),
    ((0, 1), (2, 1)),
    ((2, 1), (1, 1)),
    ((1, 2), (2, 1)),
]


def shell_for_sublattice(alpha: int, offset: int = 0) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """The (d1, d2) bond-vector shell assigned to sublattice `alpha`. The
    feedback loop varies `offset` to try alternative geometries."""
    return DEFAULT_SHELLS[(alpha + offset) % len(DEFAULT_SHELLS)]


def shells_of_size(size: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Bond-vector shell pairs (d1, d2) that make a CLS of a chosen SIZE.

    The CLS of a sublattice has sites at {0, d1, d2}, so its spatial extent
    (Chebyshev radius) is max(|d1|_inf, |d2|_inf). This returns every linearly
    independent pair whose extent is EXACTLY `size` (at least one bond reaches
    the requested radius), ordered for diversity so cycling them per sublattice
    still breaks the global-degeneracy of ARCHITECTURE.md Sec. 0.4.

    Larger `size` => CLS reaches farther => the exactly-flat local Hamiltonian
    H = E0 I + (||f||^2 I - f f^dagger) has a correspondingly longer (but still
    finite) hopping range. size=1 recovers nearest-neighbour CLS.
    """
    size = max(1, int(size))
    ring = [(n, m) for n in range(-size, size + 1) for m in range(-size, size + 1)
            if max(abs(n), abs(m)) == size]                       # outer ring
    smalls = [(1, 0), (0, 1), (1, 1), (1, -1), (2, 1), (1, 2)]
    smalls = [c for c in smalls if 1 <= max(abs(c[0]), abs(c[1])) <= size]
    shells: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    seen = set()
    for d1 in ring:
        for d2 in ring + smalls:
            if d1 == d2:
                continue
            if d1[0] * d2[1] - d1[1] * d2[0] == 0:               # linearly dependent
                continue
            key = frozenset((d1, d2))
            if key in seen:
                continue
            seen.add(key)
            shells.append((d1, d2))
    return shells if shells else list(DEFAULT_SHELLS)


def bond_cartesian(nm: Tuple[int, int], primitive_vectors: np.ndarray) -> np.ndarray:
    """Cartesian bond vector for integer cell coordinates (n, m)."""
    n, m = nm
    a1, a2 = np.asarray(primitive_vectors, dtype=float)
    return n * a1 + m * a2


def solve_two_pair_chiral(d1_cart: np.ndarray, d2_cart: np.ndarray, w: int) -> Dict:
    """Closed-form solution of Sum_p A_p e^{i Phi_p} D_p = 0 for two pairs,
    D_p = d_{p,x} + i*w*d_{p,y} (gauge Phi_2 = 0).

    Raises ValueError if d1, d2 are linearly dependent (degenerate: the
    construction would give a = b = 0, no first-order vortex).
    """
    d1 = np.asarray(d1_cart, dtype=float)
    d2 = np.asarray(d2_cart, dtype=float)
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        raise ValueError(
            f"bond vectors d1={d1}, d2={d2} are linearly dependent "
            f"(cross={cross:.3e}): the chiral condition degenerates to "
            f"a=b=0 (no first-order vortex)"
        )
    if w not in (-1, 1):
        raise ValueError(f"w must be +-1, got {w}")

    D1 = d1[0] + 1j * w * d1[1]
    D2 = d2[0] + 1j * w * d2[1]
    A1 = abs(D2)
    A2 = abs(D1)
    Phi2 = 0.0
    Phi1 = float(np.pi + np.angle(D2) - np.angle(D1))
    return {"A1": A1, "A2": A2, "Phi1": Phi1, "Phi2": Phi2,
            "D1": D1, "D2": D2, "cross": cross}


def build_sublattice_cls(alpha: int, singularity: SingularityTarget,
                         lattice_spec: LatticeSpec,
                         shell: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
                         ) -> SublatticeCLS:
    """Module 2 + 3 combined for one sublattice: pick a bond-vector shell
    (d1, d2), solve the chiral condition for (A_p, Phi_p), and build the two
    destructive-interference pairs (anchored at the unit-cell origin) whose
    theta_{j1,p} = Phi_p - pi/2 - k_i.d_p reproduces that solution exactly."""
    if shell is None:
        shell = shell_for_sublattice(alpha)
    d1_nm, d2_nm = shell
    prim = lattice_spec.primitive_vectors
    d1_cart = bond_cartesian(d1_nm, prim)
    d2_cart = bond_cartesian(d2_nm, prim)
    sol = solve_two_pair_chiral(d1_cart, d2_cart, singularity.w)

    pairs = []
    for (n, m), A, Phi in ((d1_nm, sol["A1"], sol["Phi1"]),
                           (d2_nm, sol["A2"], sol["Phi2"])):
        theta1 = Phi - np.pi / 2.0 - singularity.k_dot_R(n, m)
        pairs.append(make_pair(n, m, theta1, A, 0, 0, singularity))
    return SublatticeCLS(alpha=alpha, pairs=pairs)


def build_cls_design(singularity: SingularityTarget, lattice_spec: LatticeSpec,
                     shell_offset: int = 0,
                     cls_size: Optional[int] = None) -> CLSDesign:
    """Phase 2+3 for ALL sublattices targeting one singularity, cycling the
    shell library by `shell_offset` (used by the pipeline's feedback loop).

    cls_size : if None (default), shells are drawn from DEFAULT_SHELLS (the
    original, minimal behaviour). If a positive integer, shells are drawn from
    shells_of_size(cls_size) instead, so the resulting CLS has the requested
    spatial extent (and the final exactly-flat model the corresponding hopping
    range). Each sublattice still gets a DISTINCT shell (cycled by
    alpha+shell_offset) to avoid the global degeneracy of Sec. 0.4."""
    if cls_size is None:
        shell_lib = None
    else:
        shell_lib = shells_of_size(cls_size)
    sublattices = []
    for alpha in range(lattice_spec.N):
        if shell_lib is None:
            shell = shell_for_sublattice(alpha, offset=shell_offset)
        else:
            shell = shell_lib[(alpha + shell_offset) % len(shell_lib)]
        sublattices.append(
            build_sublattice_cls(alpha, singularity, lattice_spec, shell=shell))
    return CLSDesign(singularity=singularity, sublattices=sublattices)


def moments(sub: SublatticeCLS, singularity: SingularityTarget,
           lattice_spec: LatticeSpec) -> Tuple[complex, complex]:
    """Direct evaluation of (a^(alpha), b^(alpha)) = Sum_p A_p e^{i Phi_p} d_p
    from a built SublatticeCLS, for diagnostics/tests. Phi_p = Theta_{j1,p} +
    pi/2 = theta_{j1,p} + k_i.d_p + pi/2 (site1 of each pair is j1)."""
    prim = lattice_spec.primitive_vectors
    a = 0j
    b = 0j
    for pair in sub.pairs:
        s1 = pair.site1
        d_cart = bond_cartesian(pair.bond, prim)
        Phi = s1.phase + singularity.k_dot_R(s1.n, s1.m) + np.pi / 2.0
        a += s1.amplitude * np.exp(1j * Phi) * d_cart[0]
        b += s1.amplitude * np.exp(1j * Phi) * d_cart[1]
    return a, b
