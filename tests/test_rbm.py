"""
Robust Boundary Mode (RBM) module tests.

Covers the spec's two mandatory control groups:
  - Kagome (singular)  -> exact bulk cancellation, boundary band, local-robust to a defect.
  - Bilayer (nonsingular) -> no protected boundary mode (bulk does not cancel).
Plus a hand-built dimer to lock in the translate+truncate+sum algorithm, and the
Lieb case to exercise the k0 = (pi, pi) phase factor.
"""
import numpy as np
import sympy
import pytest

from cls_finder.io.parser import parse_input
from cls_finder.band.bands import detect_flat_bands
from cls_finder.eigen.eigenstate import extract_eigenstate_analytical
from cls_finder.classify.singularity import classify_singularity
from cls_finder.cls.analytic import select_cls_basis
from cls_finder.core.lattice import Lattice
from cls_finder.models import library
from cls_finder.rbm import (cls_amplitude_to_cells, calculate_boundary_mode,
                            verify_bulk_cancellation, boundary_mode_with_defect,
                            compute_rbm, cls_support_radius)

SYMS = sympy.symbols('x1 x2 x3')


def _extract(builder):
    """(lattice, A_0_R, k0, singular) for the first flat band of a library model."""
    lat, H = parse_input(builder())
    grid = builder()["options"].get("k_grid", [24, 24])
    tol = builder()["options"].get("flat_tol", 1e-4)
    fb = detect_flat_bands(H, lat, grid, tol)[0]
    eps0, deg = fb["energy"], fb["degenerate_indices"]
    s = SYMS[:lat.dimension]
    try:
        w = extract_eigenstate_analytical(H, eps0, s)
        sc = classify_singularity(w, lat, H, eps0, grid)
        singular, k0l = sc["singular"], sc["k0_list"]
    except Exception:
        singular, k0l = None, []
    A_0_R = select_cls_basis(H, eps0, s, len(deg), lattice=lat)[0][1]
    return lat, A_0_R, (k0l[0] if k0l else None), singular


# ── algorithm-level: a 1D telescoping dimer ──────────────────────────────────
def test_dimer_translation_and_truncation():
    """chi = e_R - e_{R+1}; summing over an open chain telescopes to a single
    boundary site. Verifies Steps 1-3 (translate, truncate, accumulate)."""
    lat = Lattice(1, [[1.0]], [{"label": "A", "position": [0.0]}])
    A_0_R = {0: {(0,): 1.0, (1,): -1.0}}   # offsets 0 and +1
    res = calculate_boundary_mode(lat, A_0_R, (8,), k_singularity=None)
    psi = res["psi"]
    # interior sites cancel exactly
    interior = [abs(psi.get(((n,), 0), 0j)) for n in range(1, 8)]
    assert max(interior) < 1e-12
    # exactly one boundary site (n=0) survives with |amp| = 1
    assert abs(abs(psi.get(((0,), 0), 0j)) - 1.0) < 1e-12


def test_amplitude_to_cells_integer():
    lat = Lattice(2, [[1.0, 0.0], [0.0, 1.0]],
                  [{"label": "A", "position": [0.0, 0.0]}])
    A_0_R = {0: {(0, 0): 1.0, (1, 0): -1.0, (0, 1): 2.0}}
    cells = cls_amplitude_to_cells(lat, A_0_R)
    assert cells[0][(0, 0)] == 1.0 and cells[0][(1, 0)] == -1.0
    assert cls_support_radius(cells) == 1


# ── control group 1: Kagome (singular) ───────────────────────────────────────
def test_kagome_singular_bulk_cancels():
    lat, A_0_R, k0, singular = _extract(library.kagome_nn)
    assert singular is True
    out = compute_rbm(lat, A_0_R, (12, 12), k_singularity=k0, singular=singular)
    v = out["validation"]
    assert v["has_bulk_region"]
    assert v["passed"] and v["max_bulk_amp"] < 1e-10, \
        f"singular bulk must cancel, got {v['max_bulk_amp']:.2e}"
    assert out["n_nonzero_sites"] > 0           # a boundary mode exists
    assert v["n_edge_sites"] > 0


def test_kagome_defect_is_local():
    """Removing one interior CLS copy spoils cancellation only within ~one CLS
    radius of that cell (robustness); the rest of the bulk stays 0."""
    lat, A_0_R, k0, singular = _extract(library.kagome_nn)
    N = (14, 14)
    defect = (7, 7)
    dres = boundary_mode_with_defect(lat, A_0_R, N, k_singularity=k0,
                                     omit_centers=[defect])
    rad = dres["support_radius"]
    # every newly-nonzero deep-bulk site must sit within rad+1 of the defect
    far_bulk_amp = 0.0
    for (cell, q), amp in dres["psi"].items():
        deep = all(rad <= cell[l] <= N[l] - 1 - rad for l in range(2))
        if deep:
            cheb = max(abs(cell[l] - defect[l]) for l in range(2))
            if cheb > rad + 1:
                far_bulk_amp = max(far_bulk_amp, abs(amp))
    assert far_bulk_amp < 1e-10, \
        f"defect must stay local; far-bulk amp leaked: {far_bulk_amp:.2e}"


# ── control group 2: Bilayer (nonsingular) ───────────────────────────────────
def test_bilayer_nonsingular_no_robust_mode():
    lat, A_0_R, k0, singular = _extract(library.bilayer_square)
    assert singular is False
    out = compute_rbm(lat, A_0_R, (12, 12), k_singularity=k0, singular=singular)
    # nonsingular: the naive CLS sum has no topological protection -> bulk does
    # NOT cancel (a robust boundary mode is not guaranteed/formed).
    assert not out["validation"]["passed"]


# ── Lieb: k0 = (pi, pi) phase must be applied for cancellation ────────────────
def test_lieb_kpi_phase_cancels():
    lat, A_0_R, k0, singular = _extract(library.lieb)
    assert singular is True and k0 is not None
    out = compute_rbm(lat, A_0_R, (12, 12), k_singularity=k0, singular=singular)
    assert out["validation"]["passed"] and out["validation"]["max_bulk_amp"] < 1e-9
    # without the phase, the same singular CLS sum does NOT cancel
    out0 = compute_rbm(lat, A_0_R, (12, 12), k_singularity=None, singular=singular)
    assert not out0["validation"]["passed"]
