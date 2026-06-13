"""
Robust Boundary Mode (RBM) module.

Computes and visualizes the boundary mode that emerges when the destructive
interference of a singular flat band's CLS — exact on the periodic torus
(sum_R c_R |chi_R> = 0) — is truncated by an open boundary. See
``goal/Robust Boundary Mode (RBM) ...`` for the full specification and
Rhim & Yang, "Classification of flat bands according to the band-crossing
singularity of Bloch wave functions".
"""
from cls_finder.rbm.boundary_mode import (
    cls_amplitude_to_cells,
    calculate_boundary_mode,
    verify_bulk_cancellation,
    skin_depth_profile,
    boundary_mode_sites,
    boundary_mode_with_defect,
    compute_rbm,
    cls_support_radius,
)

__all__ = [
    "cls_amplitude_to_cells",
    "calculate_boundary_mode",
    "verify_bulk_cancellation",
    "skin_depth_profile",
    "boundary_mode_sites",
    "boundary_mode_with_defect",
    "compute_rbm",
    "cls_support_radius",
]
