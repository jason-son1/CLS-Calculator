import pytest
import numpy as np
import sympy
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.io.parser import parse_input
from cls_finder.band.bands import detect_flat_bands
from cls_finder.eigen.eigenstate import extract_eigenstate_analytical
from cls_finder.classify.singularity import classify_singularity
from cls_finder.cls.analytic import extract_cls_analytic
from cls_finder.cls.reduce import minimize_cls
from cls_finder.cls.numeric import extract_cls_numeric, cross_validate_cls
from cls_finder.cls.noncontractible import build_noncontractible
from cls_finder.models import library

# Global list of SymPy symbols
kx, ky, kz = sympy.symbols('kx ky kz')
x1, x2, x3 = sympy.symbols('x1 x2 x3')
symbols = [x1, x2, x3]

def run_model_test(model_spec, expected_singular, expected_energy):
    lattice, H_k = parse_input(model_spec)
    grid_size = model_spec.get("options", {}).get("k_grid", [30] * lattice.dimension)
    flat_tol = model_spec.get("options", {}).get("flat_tol", 1e-4)
    
    # 1. Detect flat bands
    flat_bands = detect_flat_bands(H_k, lattice, grid_size, flat_tol)
    assert len(flat_bands) > 0, "Should detect at least one flat band"
    
    # Check energy of flat band
    eps0 = flat_bands[0]["energy"]
    assert abs(eps0 - expected_energy) < 1e-2, f"Expected energy {expected_energy}, got {eps0}"
    
    # 2. Extract analytical eigenstates
    w_k_list = extract_eigenstate_analytical(H_k, eps0, symbols[:lattice.dimension])
    assert len(w_k_list) == len(flat_bands[0]["degenerate_indices"])
    
    # 3. Classify singularity
    sing_res = classify_singularity(w_k_list, lattice, H_k, eps0, grid_size)
    assert sing_res["singular"] == expected_singular, f"Expected singular={expected_singular}, got {sing_res['singular']}"
    
    # 4. Extract and minimize CLS
    x_k, A_0_R = extract_cls_analytic(H_k, eps0)
    x_k_min, A_0_R_min = minimize_cls(x_k, symbols[:lattice.dimension])
    
    # Verify CLS is a null vector of H_bar: H_bar * x_k == 0
    H_bar = H_k - MatrixPoly.identity(H_k.rows, H_k.d) * eps0
    null_check = H_bar * MatrixPoly([[val] for val in x_k_min], H_k.d)
    assert null_check.is_zero(1e-9), "Analytical CLS should be a null vector of H - eps0*I"
    
    # 5. Extract numerical CLS and cross-validate
    A_numeric = extract_cls_numeric(H_k, lattice, eps0, grid_size)
    success, msg = cross_validate_cls(A_0_R_min, A_numeric)
    assert success, f"Numerical cross-check failed: {msg}"
    
    # 6. Build NLS/NPS if singular
    if expected_singular:
        k0 = sing_res["k0_list"][0]
        nls_nps = build_noncontractible(H_k, lattice, eps0, k0)
        assert len(nls_nps) == lattice.dimension, "Should construct NLS/NPS along all axes"

def test_zigzag_chain():
    # 1D is always non-singular
    run_model_test(library.zigzag_chain(), expected_singular=False, expected_energy=-2.0)

def test_kagome_nn():
    # Singular flat band at E = -2 (for t=1, flat band is at -2t = -2)
    # Singularity at (0,0)
    run_model_test(library.kagome_nn(), expected_singular=True, expected_energy=-2.0)

def test_bilayer_square():
    # Non-singular flat band at E = 2
    run_model_test(library.bilayer_square(), expected_singular=False, expected_energy=2.0)

def test_lieb():
    # Singular flat band at E = 0, touching at (pi, pi)
    run_model_test(library.lieb(), expected_singular=True, expected_energy=0.0)

def test_modified_lieb():
    # Singular flat band at E = 0, touching at (0,0)
    run_model_test(library.modified_lieb(), expected_singular=True, expected_energy=0.0)

def test_checkerboard_1():
    # Singular flat band at E = 0, touching at (0,0)
    run_model_test(library.checkerboard_1(), expected_singular=True, expected_energy=0.0)

def test_checkerboard_2():
    # Singular flat band at E = 0, touching at (pi, pi)
    run_model_test(library.checkerboard_2(), expected_singular=True, expected_energy=0.0)

def test_checkerboard_3():
    # Non-singular isolated flat band at E = 0
    run_model_test(library.checkerboard_3(), expected_singular=False, expected_energy=0.0)

def test_honeycomb_flat():
    # Non-singular flat band at E = 0
    run_model_test(library.honeycomb_flat(), expected_singular=False, expected_energy=0.0)

def test_cubic_3d():
    # Singular 3D flat band at E = 0
    run_model_test(library.cubic_3D(), expected_singular=True, expected_energy=0.0)

def test_kagome_3():
    # Doubly degenerate non-singular flat bands at E = -2
    run_model_test(library.kagome_3(), expected_singular=False, expected_energy=-2.0)

def test_fractional_complex_cross_validate():
    # Construct a custom model spec containing fractional hopping terms and complex amplitudes
    spec = {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.0, 0.0]},
                {"label": "Cx", "position": [0.5, 0.0]},
                {"label": "Cy", "position": [0.0, 0.5]}
            ]
        },
        "H_symbolic": [
            ["0", "I * (1 - exp(-I*kx))", "0"],
            ["-I * (1 - exp(I*kx))", "0", "0"],
            ["0", "0", "0"]
        ],
        "options": {
            "k_grid": [15, 15],
            "flat_tol": 1e-4
        }
    }
    
    lattice, H_k = parse_input(spec)
    
    flat_bands = detect_flat_bands(H_k, lattice, [15, 15], 1e-4)
    assert len(flat_bands) > 0
    eps0 = flat_bands[0]["energy"]
    
    x_k, A_0_R = extract_cls_analytic(H_k, eps0)
    x1, x2 = sympy.symbols('x1 x2')
    x_k_min, A_0_R_min = minimize_cls(x_k, [x1, x2])
    
    # Compute numerical CLS
    A_numeric = extract_cls_numeric(H_k, lattice, eps0, [15, 15])
    
    # Run cross validation
    success, msg = cross_validate_cls(A_0_R_min, A_numeric)
    assert success, f"Cross validation failed for fractional complex model: {msg}"


def test_cls_plot_data_conventions():
    from web.bridge import _cls_plot_data
    
    # Test 1: Convention I (Zigzag Chain)
    spec_zigzag = library.zigzag_chain()
    lattice, H_k = parse_input(spec_zigzag)
    # Mock A_0_R
    A_0_R_mock = {0: {(0,): 1.0}}
    plot_data = _cls_plot_data(lattice, H_k, A_0_R_mock)
    
    # Grid cell coordinates must all be integers
    for s in plot_data["sites"]:
        for c in s["cell"]:
            assert abs(c - round(c)) < 1e-9, "Grid cells must be integers"
            
    # Test 2: Convention II (5-Orbital model with fractional exponents)
    spec_fractional = {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0.5, 0.5]},
                {"label": "Cx", "position": [0, 0.5]},
                {"label": "Cy", "position": [0.5, 0]},
                {"label": "B0", "position": [0, 0]},
                {"label": "Bp", "position": [0, 0]}
            ]
        },
        "H_symbolic": [
            ["0", "0", "0", "2*I*sin(kx/2+ky/2) + 2*sin(kx/2-ky/2)", "2*cos(kx/2+ky/2) + 2*cos(kx/2-ky/2)"],
            ["0", "0", "0", "2*cos(ky/2)", "2*I*sin(ky/2)"],
            ["0", "0", "0", "-2*I*cos(kx/2)", "2*I*sin(kx/2)"],
            ["2*(-I)*sin(kx/2+ky/2) + 2*sin(kx/2-ky/2)", "2*cos(ky/2)", "-2*(-I)*cos(kx/2)", "-1", "0"],
            ["2*cos(kx/2+ky/2) + 2*cos(kx/2-ky/2)", "2*(-I)*sin(ky/2)", "2*(-I)*sin(kx/2)", "0", "-1"]
        ]
    }
    
    lattice2, H_k2 = parse_input(spec_fractional)
    # A_0_R mock with fractional keys
    A_0_R_mock2 = {
        0: {(0.5, 0.5): 1.0},
        1: {(0.0, 0.5): 1.0}
    }
    plot_data2 = _cls_plot_data(lattice2, H_k2, A_0_R_mock2)
    
    # 1. Grid cell coordinates must all be integers
    for s in plot_data2["sites"]:
        for c in s["cell"]:
            assert abs(c - round(c)) < 1e-9, "Grid cells must be integers in Convention II"
            
    # 2. Check site positions: e.g. for cell [0, 0]
    site_A = next(s for s in plot_data2["sites"] if s["cell"] == [0, 0] and s["label"] == "A")
    assert abs(site_A["x"] - 0.5) < 1e-9
    assert abs(site_A["y"] - 0.5) < 1e-9
    
    site_Cx = next(s for s in plot_data2["sites"] if s["cell"] == [0, 0] and s["label"] == "Cx")
    assert abs(site_Cx["x"] - 0.0) < 1e-9
    assert abs(site_Cx["y"] - 0.5) < 1e-9

    # 3. Check amplitude mapping (A mock had amplitude at (0.5, 0.5), which is cell [0,0] + orbital A position)
    assert site_A["is_cls"] is True
    assert abs(site_A["amplitude"] - 1.0) < 1e-9
    
    # 4. Check bonds: bond between B0 and A at origin cell must be length close to sqrt(0.5) = 0.7071
    bonds = plot_data2["bonds"]
    assert len(bonds) > 0
    origin_bonds = [b for b in bonds if abs(b["x0"]) < 0.6 and abs(b["y0"]) < 0.6 and abs(b["x1"]) < 0.6 and abs(b["y1"]) < 0.6]
    assert len(origin_bonds) > 0
    # There should be a bond between B0 (0, 0) and A (0.5, 0.5)
    # B0 position is at (0, 0) exactly now since multi-orbital offset is removed
    b_B0_A = next(b for b in origin_bonds if (abs(b["x0"]) < 1e-9 and abs(b["y0"]) < 1e-9 and abs(b["x1"] - 0.5) < 1e-9 and abs(b["y1"] - 0.5) < 1e-9) or
                                             (abs(b["x1"]) < 1e-9 and abs(b["y1"]) < 1e-9 and abs(b["x0"] - 0.5) < 1e-9 and abs(b["y0"] - 0.5) < 1e-9))
    dist = np.sqrt((b_B0_A["x1"] - b_B0_A["x0"])**2 + (b_B0_A["y1"] - b_B0_A["y0"])**2)
    assert abs(dist - np.sqrt(0.5)) < 1e-9


def test_chern3():
    # Test C=3 Chern flatband model
    # Because it is a Chern flat band with non-zero Chern number,
    # it must have a flat band, but it must not have a localized CLS (topological obstruction)
    spec = library.chern_3_flatband()
    lattice, H_k = parse_input(spec)
    
    flat_bands = detect_flat_bands(H_k, lattice, [24, 24], 1e-4)
    assert len(flat_bands) > 0, "Should detect flat band"
    eps0 = flat_bands[0]["energy"]
    assert abs(eps0) < 1e-3, "Flat band should be at zero energy"
    
    # Verify analytical CLS extraction fails as expected due to topological obstruction
    with pytest.raises(ValueError, match="Failed to find any valid analytical CLS gauge"):
        extract_cls_analytic(H_k, eps0)


def test_detect_high_symmetry_and_k_path():
    from cls_finder.band.bands import detect_high_symmetry_and_k_path
    
    # Test hexagonal Kagome model without H_k (pure geometric)
    spec_kagome = library.kagome_nn()
    lattice, H_k = parse_input(spec_kagome)
    hs_pts, path_labels = detect_high_symmetry_and_k_path(lattice, H_k=None)
    assert "Γ" in hs_pts
    assert "M" in hs_pts
    assert "K" in hs_pts
    assert path_labels == ["Γ", "M", "K", "K'", "Γ"]
    
    # Test hexagonal Kagome model WITH H_k (dynamic touching point detection)
    hs_pts_h, path_labels_h = detect_high_symmetry_and_k_path(lattice, H_k)
    assert "Γ" in hs_pts_h
    assert "K0_1" in hs_pts_h
    assert "K0_1" in path_labels_h
    
    # Test square Lieb model
    spec_lieb = library.lieb()
    lattice, H_k_lieb = parse_input(spec_lieb)
    hs_pts_l, path_labels_l = detect_high_symmetry_and_k_path(lattice, H_k_lieb)
    assert "Γ" in hs_pts_l
    assert "X" in hs_pts_l
    assert "M" in hs_pts_l
    assert "Γ" in path_labels_l
    assert "M" in path_labels_l



