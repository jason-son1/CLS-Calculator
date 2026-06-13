"""
Module 2 — Common Zero Solver via Destructive Interference (Phase 2).

Implements the Pairing Rule of Note A / Note B:

    A_{j1}^(alpha) = A_{j2}^(alpha) == A^(alpha)
    theta_{j1}^(alpha) + k_i . R_{j1}^(alpha)
        = theta_{j2}^(alpha) + k_i . R_{j2}^(alpha) + (2m+1) pi

Anchoring R_{j2,p} = (0,0) (the unit-cell origin) and R_{j1,p} = d_p (an
integer bond vector (n, m), Cartesian R = n*a1 + m*a2), the rule reduces to a
single algebraic relation that fixes theta_{j2,p} given theta_{j1,p} (or vice
versa) -- see `make_pair`. For ANY choice of (A_p, theta_{j1,p}) this makes

    Sum_j A_j^(alpha) e^{i Theta_j^(alpha)} = 0   (Theta_j = theta_j + k_i.R_j)

hold EXACTLY for that pair, hence for the whole sublattice (Condition 1 /
Note A Eq. 11 / Proposition 1). theta_{j1,p} (equivalently the pair phase
Phi_p, see chiral.py) remains a free design variable for Phase 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from cls_finder.engineer.spec import SingularityTarget


@dataclass
class CLSSite:
    """A single CLS contribution A * e^{i theta} at lattice position
    R = n*a1 + m*a2 (integer cell-vector coordinates), within one
    sublattice."""
    n: int
    m: int
    amplitude: float
    phase: float

    @property
    def R(self) -> Tuple[int, int]:
        return (self.n, self.m)

    @property
    def c(self) -> complex:
        """Complex CLS coefficient A*e^{i theta} (excludes zeta^(alpha))."""
        return self.amplitude * np.exp(1j * self.phase)


@dataclass
class CLSPair:
    """A destructive-interference pair (j1, j2) within one sublattice,
    satisfying the Module-2 pairing rule for one singularity k_i."""
    site1: CLSSite
    site2: CLSSite

    @property
    def bond(self) -> Tuple[int, int]:
        """Bond vector d_p = R_{j1} - R_{j2} (integer cell coordinates)."""
        return (self.site1.n - self.site2.n, self.site1.m - self.site2.m)

    def residual(self, singularity: SingularityTarget) -> complex:
        """A_{j1} e^{i Theta_{j1}} + A_{j2} e^{i Theta_{j2}}, Theta_j =
        theta_j + k_i.R_j -- must vanish exactly for Condition 1."""
        s1, s2 = self.site1, self.site2
        Theta1 = s1.phase + singularity.k_dot_R(s1.n, s1.m)
        Theta2 = s2.phase + singularity.k_dot_R(s2.n, s2.m)
        return s1.amplitude * np.exp(1j * Theta1) + s2.amplitude * np.exp(1j * Theta2)


def make_pair(n1: int, m1: int, theta1: float, A: float,
              n2: int, m2: int,
              singularity: SingularityTarget, m_int: int = 0) -> CLSPair:
    """Build a CLS pair (site1 at (n1,m1), site2 at (n2,m2)) of one
    sublattice that satisfies the Module-2 destructive-interference pairing
    rule EXACTLY for k = singularity.k_frac:

        A_{j1} = A_{j2} = A
        theta_{j1} + k_i.R_{j1} = theta_{j2} + k_i.R_{j2} + (2*m_int+1)*pi

    `theta1` (= theta_{j1}) and `A` are free design parameters (set by
    Phase 3 / chiral.py); theta2 (= theta_{j2}) is DERIVED so that Condition
    1 holds identically (to floating-point precision).
    """
    phi1 = singularity.k_dot_R(n1, m1)
    phi2 = singularity.k_dot_R(n2, m2)
    theta2 = theta1 + phi1 - phi2 - (2 * m_int + 1) * np.pi
    site1 = CLSSite(n1, m1, A, theta1)
    site2 = CLSSite(n2, m2, A, theta2)
    return CLSPair(site1, site2)


@dataclass
class SublatticeCLS:
    """The CLS configuration of one sublattice alpha, as a set of
    destructive-interference pairs targeting one singularity."""
    alpha: int
    pairs: List[CLSPair]

    def sites(self) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """Merge pair-sites that land on the same lattice position (n, m)
        (their complex CLS coefficients add). Returns {(n,m): (A, theta)}
        with A >= 0."""
        merged: Dict[Tuple[int, int], complex] = {}
        for pair in self.pairs:
            for site in (pair.site1, pair.site2):
                merged[site.R] = merged.get(site.R, 0j) + site.c
        out: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for R, c in merged.items():
            A = abs(c)
            theta = float(np.angle(c)) if A > 1e-15 else 0.0
            out[R] = (A, theta)
        return out


@dataclass
class CLSDesign:
    """Per-singularity CLS configuration for all N sublattices."""
    singularity: SingularityTarget
    sublattices: List[SublatticeCLS]

    def verify_condition1(self, tol: float = 1e-9) -> List[complex]:
        """Return the per-sublattice residual f_alpha(k_i) (excluding
        zeta^(alpha), which is an overall phase and does not affect whether
        this is zero). Each entry should have |residual| < tol."""
        residuals = []
        for sub in self.sublattices:
            total = 0j
            for pair in sub.pairs:
                total += pair.residual(self.singularity)
            residuals.append(total)
        return residuals
