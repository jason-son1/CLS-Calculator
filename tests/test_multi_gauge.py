import pytest
import numpy as np
import sympy
from cls_finder.io.parser import parse_input
from cls_finder.cls.analytic import extract_all_cls_analytic, extract_cls_analytic
from cls_finder.cls.reduce import minimize_cls
from cls_finder.models import library
from web.bridge import _cls_plot_data, validate_gauge

def test_extract_all_cls_analytic():
    # Load Lieb model (singular, E=0, Q=3)
    model_spec = library.lieb()
    lattice, H_k = parse_input(model_spec)
    
    eps0 = 0.0
    
    # Extract all analytical gauges
    all_gauges = extract_all_cls_analytic(H_k, eps0)
    assert len(all_gauges) > 0, "Should find at least one valid analytic gauge"
    
    # Verify each gauge
    for p, (x_k, A_0_R) in all_gauges.items():
        assert len(x_k) == H_k.rows
        # Verify it's a non-zero vector
        is_zero = True
        for poly in x_k:
            if not poly.is_zero(1e-12):
                is_zero = False
                break
        assert not is_zero, f"Gauge for p={p} should not be a zero vector"

def test_validate_gauge():
    # Load zigzag chain
    model_spec = library.zigzag_chain()
    lattice, H_k = parse_input(model_spec)
    eps0 = -2.0
    
    x_k, A_0_R = extract_cls_analytic(H_k, eps0)
    x1 = sympy.symbols('x1')
    x_k_min, A_0_R_min = minimize_cls(x_k, [x1])
    
    # Validate eigenvector
    k_points = [np.array([0.0]), np.array([0.15]), np.array([0.5])]
    success, msg = validate_gauge(H_k, eps0, x_k_min, lattice, k_points)
    assert success, f"Zigzag CLS verification failed: {msg}"

def test_orbital_offsets():
    # Build a custom 2D model spec where orbitals are at the exact same location (same site)
    spec = {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A1", "position": [0.0, 0.0]},
                {"label": "A2", "position": [0.0, 0.0]}  # Same location as A1
            ]
        },
        "H_symbolic": [
            ["0", "1"],
            ["1", "0"]
        ],
        "options": {
            "k_grid": [10, 10],
            "flat_tol": 1e-4
        }
    }
    
    lattice, H_k = parse_input(spec)
    A_0_R = {
        0: {(0, 0): 1.0},
        1: {(0, 0): -1.0}
    }
    
    # Compute plot data
    plot_data = _cls_plot_data(lattice, H_k, A_0_R, plot_range=1)
    
    # Check that sublattice center and link coordinates are empty since we use concentric rendering
    assert "sublattices" in plot_data
    assert "sublattice_links" in plot_data
    assert len(plot_data["sublattices"]) == 0
    assert len(plot_data["sublattice_links"]) == 0
    
    # Check that positions are identical (concentric)
    s0, s1 = plot_data["sites"][0], plot_data["sites"][1]
    assert abs(s0["x"] - s1["x"]) < 1e-9 and abs(s0["y"] - s1["y"]) < 1e-9, "Orbitals A1 and A2 should be at the exact same concentric position"

