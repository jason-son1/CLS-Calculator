"""
Real-Space CLS-based Topological Flat Band Engineering.

Reverse-engineers a finite-Fourier-sum (CLS) Bloch eigenvector f(k) -- and
the flat-band Hamiltonian H(k) = E0 P(k) + (I-P(k)) M(k) (I-P(k)) built on
its projector -- from a TARGET total Chern number C and a set of Brillouin
zone singularities k_i carrying local winding w_i = sum-to-C, by directly
constructing the real-space CLS amplitudes/phases (A_j, theta_j, R_j) that
make the common-zero (Eq.11), rank-1/projector-continuity (Eq.33-34) and
winding (Eq.45-46) conditions hold by algebraic construction. See
``goal/CLSEngineering/ARCHITECTURE.md`` for the full design and the two
source notes it implements.

    design_flat_band(LatticeSpec(...), DesignTarget(C=1, [SingularityTarget(...)]))
        -> DesignResult(x_k, H_k, hoppings, verification, ...)
"""
from cls_finder.engineer.spec import (
    NAMED_K_POINTS,
    resolve_k_frac,
    SublatticeSpec,
    LatticeSpec,
    SingularityTarget,
    DesignTarget,
    validate_design,
    derive_zeta,
    with_derived_zetas,
)
from cls_finder.engineer.pairing import (
    CLSSite,
    CLSPair,
    make_pair,
    SublatticeCLS,
    CLSDesign,
)
from cls_finder.engineer.chiral import (
    DEFAULT_SHELLS,
    shell_for_sublattice,
    shells_of_size,
    bond_cartesian,
    solve_two_pair_chiral,
    build_sublattice_cls,
    build_cls_design,
    moments,
)
from cls_finder.engineer.assembly import (
    build_x_k,
    evaluate_psi,
    vortex_vector,
    verify_projector_continuity,
)
from cls_finder.engineer.hamiltonian import (
    default_M_k,
    NumericHk,
    build_hamiltonian,
    inverse_fourier_transform,
    hoppings_to_matrixpoly,
)
from cls_finder.engineer.pipeline import DesignResult, design_flat_band
from cls_finder.engineer.manual import (
    build_x_k_from_sites,
    validate_manual_cls,
    analyze_manual_cls,
)
from cls_finder.engineer.explore import (
    DesignCandidate,
    explore_designs,
    iter_design_attempts,
    dedupe_and_rank,
)

__all__ = [
    "NAMED_K_POINTS", "resolve_k_frac",
    "SublatticeSpec", "LatticeSpec", "SingularityTarget", "DesignTarget",
    "validate_design", "derive_zeta", "with_derived_zetas",
    "CLSSite", "CLSPair", "make_pair", "SublatticeCLS", "CLSDesign",
    "DEFAULT_SHELLS", "shell_for_sublattice", "shells_of_size", "bond_cartesian",
    "solve_two_pair_chiral", "build_sublattice_cls", "build_cls_design", "moments",
    "build_x_k", "evaluate_psi", "vortex_vector", "verify_projector_continuity",
    "default_M_k", "NumericHk", "build_hamiltonian",
    "inverse_fourier_transform", "hoppings_to_matrixpoly",
    "DesignResult", "design_flat_band",
    "build_x_k_from_sites", "validate_manual_cls", "analyze_manual_cls",
    "DesignCandidate", "explore_designs", "iter_design_attempts", "dedupe_and_rank",
]
