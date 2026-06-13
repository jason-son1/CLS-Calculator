"""
Module 1 — Initialization & Target Setting (Phase 1).

Defines the user-facing input data structures for the Real-Space CLS-based
Topological Flat Band Engineering pipeline:

  - LatticeSpec / SublatticeSpec : lattice vectors a1,a2 and the N
    sublattices (each with an internal site-shift phase zeta^(alpha)).
  - SingularityTarget            : a BZ singularity k_i (fractional
    reciprocal coordinates) with a target local winding/Chern w_i = +-1.
  - DesignTarget                 : the full topological target, C = sum w_i,
    with the pre-validation required by Note B Phase 1
    ("local windings must not trivially cancel").

Only Phase 1 logic lives here: geometry/target bookkeeping and validation.
Phases 2-5 (cls_finder.engineer.pairing/chiral/assembly/hamiltonian) consume
these objects but do not modify them.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Tuple

import numpy as np

from cls_finder.core.lattice import Lattice
from cls_finder.classify import chern as _chern


# ──────────────────────────────────────────────────────────────────────────
# Named high-symmetry points, in FRACTIONAL reciprocal coordinates
# (k = f1*b1 + f2*b2). These are convenience aliases; any (f1, f2) tuple may
# be used directly as a singularity location.
# ──────────────────────────────────────────────────────────────────────────
NAMED_K_POINTS = {
    "Gamma": (0.0, 0.0), "G": (0.0, 0.0), "Γ": (0.0, 0.0),
    "M": (0.5, 0.5),
    "X": (0.5, 0.0),
    "Y": (0.0, 0.5),
    "K": (1.0 / 3.0, 1.0 / 3.0),
    "K'": (2.0 / 3.0, 2.0 / 3.0), "Kp": (2.0 / 3.0, 2.0 / 3.0),
}


def resolve_k_frac(name_or_frac) -> Tuple[float, float]:
    """Resolve a named high-symmetry point or an explicit (f1, f2) tuple."""
    if isinstance(name_or_frac, str):
        if name_or_frac not in NAMED_K_POINTS:
            raise KeyError(
                f"unknown high-symmetry point '{name_or_frac}'; known names: "
                f"{sorted(set(NAMED_K_POINTS) - {'G', 'Kp', 'Γ'})} "
                f"or pass an explicit (f1, f2) fractional tuple"
            )
        return NAMED_K_POINTS[name_or_frac]
    f1, f2 = name_or_frac
    return (float(f1), float(f2))


@dataclass
class SublatticeSpec:
    """One sublattice (orbital) alpha = 1..N.

    zeta: the internal site-shift phase zeta^(alpha) (radians). This is a
    pure gauge choice -- it does not affect the common-zero condition, the
    rank-1/projector-continuity condition, or the local winding number (see
    ARCHITECTURE.md Sec. 0.1). It only re-phases the assembled f_alpha(k) and
    therefore the resulting hopping parameters t_ab(R).
    """
    label: str
    zeta: float = 0.0
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass
class LatticeSpec:
    """2D lattice geometry: primitive vectors a1, a2 and N sublattices."""
    primitive_vectors: np.ndarray  # shape (2, 2), rows = a1, a2
    sublattices: List[SublatticeSpec]

    def __post_init__(self):
        a = np.asarray(self.primitive_vectors, dtype=float)
        if a.shape != (2, 2):
            raise ValueError(
                f"primitive_vectors must have shape (2, 2), got {a.shape}"
            )
        if abs(np.linalg.det(a)) < 1e-12:
            raise ValueError("primitive vectors a1, a2 are linearly dependent")
        self.primitive_vectors = a
        if len(self.sublattices) == 0:
            raise ValueError("at least one sublattice is required")

    @property
    def N(self) -> int:
        return len(self.sublattices)

    @property
    def dimension(self) -> int:
        return 2

    def reciprocal_vectors(self) -> np.ndarray:
        """Rows b1, b2 with b_l . a_m = 2*pi*delta_lm."""
        return _chern.reciprocal_vectors(self.primitive_vectors)

    def to_lattice(self) -> Lattice:
        """A cls_finder.core.lattice.Lattice exposing .primitive_vectors /
        .dimension (and orbital bookkeeping), so the engineered system can be
        fed directly into cls_finder.classify.chern / band.bands / viz.plot
        without modification."""
        orbitals = [
            {"label": s.label,
             "position": [float(s.position[0]), float(s.position[1])],
             "sublattice": idx}
            for idx, s in enumerate(self.sublattices)
        ]
        return Lattice(dimension=2, primitive_vectors=self.primitive_vectors,
                       orbitals=orbitals)


@dataclass
class SingularityTarget:
    """A BZ singularity k_i (Eq. 11/63 of Note A) with target local winding
    number w_i = +-1 (the sign convention of chern.local_winding: w_i = +1
    means h(q) ~ q_x + i q_y, w_i = -1 means h(q) ~ q_x - i q_y)."""
    name: str
    k_frac: Tuple[float, float]
    w: int

    def __post_init__(self):
        self.k_frac = resolve_k_frac(self.k_frac)
        if self.w not in (-1, 1):
            raise ValueError(
                f"singularity '{self.name}': the chiral condition (Note A "
                f"Sec. 6-7) is defined for first-order vortices w_i = +-1 "
                f"only; got w={self.w}. (A target C=0 with no singularities "
                f"is expressed via DesignTarget(C=0, singularities=[]).)"
            )

    def k_cartesian(self, lattice_spec: LatticeSpec) -> np.ndarray:
        B = lattice_spec.reciprocal_vectors()
        return np.asarray(self.k_frac, dtype=float) @ B

    def k_dot_R(self, n: int, m: int) -> float:
        """k_i . R for R = n*a1 + m*a2, using k_i.R = 2*pi*(f1*n + f2*m)
        (since b_l . a_m = 2*pi*delta_lm)."""
        f1, f2 = self.k_frac
        return 2.0 * np.pi * (f1 * n + f2 * m)


def derive_zeta(lattice_spec: LatticeSpec, singularity: SingularityTarget,
                 alpha: int) -> float:
    """The site-shift phase zeta^(alpha) that ties sublattice alpha's gauge
    to its real-space position and the target singularity k_i:

        zeta^(alpha) = k_i . r_alpha   (mod 2*pi)

    where r_alpha = sublattices[alpha].position (fractional) expressed in
    Cartesian coordinates via primitive_vectors, and
    k_i = singularity.k_cartesian(lattice_spec).

    This is the gauge transform that converts the pipeline's zeta=0
    convention (Sec 0.1: f_alpha(k) = g_alpha(k), "position-independent"
    Bloch phase) into the standard "position-dependent" Bloch convention,
    evaluated AT k_i. Per Sec 0.1 it does not change f(k)'s common zeros,
    local windings, or analytic_chern (all zeta-invariant) -- but it DOES
    change how the flat band's projector P(k) sits relative to M(k) in
    H(k) = E0 P(k) + (I-P(k)) M(k) (I-P(k)), and therefore can change the
    gap to the dispersive bands (band_isolation) and how robustly that gap
    -- and the resulting Chern number -- survive IFT truncation.

    Opt-in only: SublatticeSpec.zeta still defaults to 0.0; callers apply
    this via with_derived_zetas() when they want the derived gauge.
    """
    sub = lattice_spec.sublattices[alpha]
    r_alpha = np.asarray(sub.position, dtype=float) @ lattice_spec.primitive_vectors
    k_i = singularity.k_cartesian(lattice_spec)
    return float(np.dot(k_i, r_alpha) % (2.0 * np.pi))


def with_derived_zetas(lattice_spec: LatticeSpec,
                        singularity: SingularityTarget) -> LatticeSpec:
    """Return a copy of lattice_spec with every sublattice's zeta replaced by
    derive_zeta(lattice_spec, singularity, alpha) (zeta^(alpha) = k_i.r_alpha).

    For a DesignTarget with multiple singularities this gauge-fixes to ONE
    of them (the caller's choice, typically target.singularities[0]); per
    Sec 0.1 the local windings and analytic Chern number at every
    singularity are zeta-invariant regardless, so this choice only affects
    isolation/truncation behavior, not which topology is realized.
    """
    new_subs = [
        replace(s, zeta=derive_zeta(lattice_spec, singularity, alpha))
        for alpha, s in enumerate(lattice_spec.sublattices)
    ]
    return replace(lattice_spec, sublattices=new_subs)


@dataclass
class DesignTarget:
    """The full target topology: total Chern number C = sum_i w_i, realized
    via a set of BZ singularities each carrying a local winding w_i."""
    C: int
    singularities: List[SingularityTarget]

    def validate(self) -> None:
        """Phase 1 pre-validation (Note B): sum_i w_i must equal C, i.e. the
        local windings must not (accidentally) cancel against a nonzero
        target, and a nonzero target requires at least one singularity."""
        if not self.singularities:
            if self.C != 0:
                raise ValueError(
                    f"target C={self.C} != 0 but no singularities were given"
                )
            return

        total = sum(s.w for s in self.singularities)
        if total != self.C:
            raise ValueError(
                f"sum(w_i)={total} != target C={self.C}: the requested local "
                f"windings {[s.w for s in self.singularities]} do not sum to "
                f"the target Chern number (they may trivially cancel)."
            )

        # distinct singularity locations (mod reciprocal lattice)
        seen = []
        for s in self.singularities:
            fr = np.asarray(s.k_frac, dtype=float)
            for fr2 in seen:
                if np.allclose((fr - fr2 + 0.5) % 1.0 - 0.5, 0.0, atol=1e-9):
                    raise ValueError(
                        f"singularities '{s.name}' and a previous one sit at "
                        f"the same BZ point {s.k_frac} (mod reciprocal "
                        f"lattice) -- merge them into one entry"
                    )
            seen.append(fr)


def validate_design(lattice_spec: LatticeSpec, target: DesignTarget) -> None:
    """Combined Phase-1 validation: target self-consistency (DesignTarget.
    validate) plus the structural N>=2 requirement for a nonzero Chern
    number (a single-component f(k) has P(k)=[1] identically, so C=0 always
    -- ARCHITECTURE.md Sec. 0.5)."""
    target.validate()
    if target.C != 0 and lattice_spec.N < 2:
        raise ValueError(
            f"target C={target.C} != 0 requires at least 2 sublattices "
            f"(N={lattice_spec.N}): a single-component CLS has a constant "
            f"projector P(k)=[1] and is always topologically trivial."
        )
