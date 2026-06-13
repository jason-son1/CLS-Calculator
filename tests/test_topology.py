import numpy as np
import pytest
from cls_finder.io.parser import parse_input
from cls_finder.topology.fhs_chern import compute_fhs_chern
from cls_finder.topology.wilson_loop import compute_wilson_loop
from cls_finder.topology.entangle_spec import compute_entanglement_spectrum
from cls_finder.topology.fu_kane import compute_fu_kane
from cls_finder.models import library

def _qwz(m):
    return {
        "lattice": {
            "dimension": 2,
            "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
            "orbitals": [
                {"label": "A", "position": [0, 0]},
                {"label": "B", "position": [0, 0]}
            ]
        },
        "H_symbolic": [
            [f"{m} + cos(kx) + cos(ky)", "sin(kx) - I*sin(ky)"],
            ["sin(kx) + I*sin(ky)", f"-({m} + cos(kx) + cos(ky))"]
        ],
        "options": {"k_grid": [40, 40], "flat_tol": 1e-5}
    }

def test_fhs_chern_qwz():
    # Topological phase (m = 1.0) -> |C| = 1
    spec = _qwz(1.0)
    lat, H = parse_input(spec)
    res = compute_fhs_chern(H, lat, [0], n_x=24, n_y=24)
    assert abs(res["C"]) == 1
    
    # Trivial phase (m = 3.0) -> C = 0
    spec_triv = _qwz(3.0)
    lat_triv, H_triv = parse_input(spec_triv)
    res_triv = compute_fhs_chern(H_triv, lat_triv, [0], n_x=24, n_y=24)
    assert res_triv["C"] == 0

def test_wilson_loop_qwz():
    spec = _qwz(1.0)
    lat, H = parse_input(spec)
    
    # Run Wilson Loop
    res = compute_wilson_loop(H, lat, [0], n_x=30, n_y=30)
    assert "tracks" in res
    assert len(res["tracks"]) == 1
    assert len(res["tracks"][0]) == 30
    assert abs(res["chern"]) == 1

def test_entanglement_spectrum_lieb():
    spec = library.lieb()
    lat, H = parse_input(spec)
    
    # Run Cylinder Entanglement Spectrum for the lowest band (band 0)
    N_x = 10
    res = compute_entanglement_spectrum(H, lat, [0], N_x=N_x, n_y=10)
    assert "spectrum" in res
    # Region A size is N_x // 2 = 5
    # Orbitals per unit cell Q = 3
    # Dimension of Region A = 5 * 3 = 15
    assert len(res["spectrum"]) == 10  # n_y points
    assert len(res["spectrum"][0]) == 15

def test_fu_kane_lieb():
    spec = library.lieb()
    lat, H = parse_input(spec)
    
    # Parity matrix P = diag(1, 1, 1) or diag(1, -1, -1) depending on inversion mapping
    # Let's test with identity diag(1, 1, 1)
    res = compute_fu_kane(H, lat, [0], P_matrix=np.eye(3))
    assert res["symmetric"] is True  # Lieb should be symmetric under identity at TRIM points
    assert "z2" in res
    assert len(res["parity_details"]) == 4  # 4 TRIM points
