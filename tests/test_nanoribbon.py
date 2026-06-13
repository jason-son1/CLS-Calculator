import json
import pytest
import numpy as np
from cls_finder.io.parser import parse_input
from cls_finder.models.library import kagome_nn, lieb, bilayer_square
from web.bridge import get_nanoribbon_data, get_nanoribbon_state, _recip

def test_nanoribbon_kagome():
    # Load kagome model spec
    spec = kagome_nn()
    spec_json = json.dumps(spec)
    
    # Calculate nanoribbon bands
    Nx = 20
    Nk = 50
    result_str = get_nanoribbon_data(spec_json, Nx, Nk)
    result = json.loads(result_str)
    
    assert result["error"] is None
    
    ky_space = result["ky_space"]
    energies = result["energies"]
    iprs = result["iprs"]
    edge_weights = result["edge_weights"]
    
    # Assert dimensions
    # Q = 3 orbitals for Kagome
    # Total bands in ribbon = Nx * Q = 20 * 3 = 60
    assert len(ky_space) == Nk
    assert len(energies) == 60
    assert len(energies[0]) == Nk
    assert len(iprs) == 60
    assert len(edge_weights) == 60
    
    # Verify edge weights are between 0 and 1
    for w in edge_weights:
        for val in w:
            assert 0.0 <= val <= 1.0 + 1e-9

def test_nanoribbon_state_lieb():
    # Load lieb model spec
    spec = lieb()
    spec_json = json.dumps(spec)
    
    Nx = 15
    # Q = 3 orbitals for Lieb
    # Total bands = 15 * 3 = 45
    
    # Get state profile at ky = 0.0, band 22 (near zero energy flat band)
    result_str = get_nanoribbon_state(spec_json, Nx, 0.0, 22)
    result = json.loads(result_str)
    
    assert result["error"] is None
    assert "energy" in result
    assert "ipr" in result
    assert "edge_weight" in result
    assert len(result["layer_prob"]) == Nx
    assert len(result["sites"]) == Nx * 3
    
    # Verify coordinates are calculated
    site = result["sites"][0]
    assert "x" in site
    assert "y" in site
    assert "prob" in site
    assert "re" in site
    assert "im" in site

def test_nanoribbon_with_band_selection():
    # Load kagome model spec
    spec = kagome_nn()
    spec_json = json.dumps(spec)
    
    Nx = 20
    Nk = 50
    # Select only the middle bulk band (Band 1)
    result_str = get_nanoribbon_data(spec_json, Nx, Nk, selected_bands=[1])
    result = json.loads(result_str)
    
    assert result["error"] is None
    assert result["lo"] == 15
    
    ky_space = result["ky_space"]
    energies = result["energies"]
    iprs = result["iprs"]
    edge_weights = result["edge_weights"]
    
    # Expected number of bands = hi - lo + 1 = 44 - 15 + 1 = 30
    assert len(ky_space) == Nk
    assert len(energies) == 30
    assert len(energies[0]) == Nk
    assert len(iprs) == 30
    assert len(edge_weights) == 30

def test_nanoribbon_with_k_range():
    spec = kagome_nn()
    spec_json = json.dumps(spec)
    
    Nx = 10
    Nk = 20
    ky_min = 0.0
    ky_max = 1.0
    result_str = get_nanoribbon_data(spec_json, Nx, Nk, selected_bands=[0], ky_min=ky_min, ky_max=ky_max)
    result = json.loads(result_str)
    
    assert result["error"] is None
    ky_space = result["ky_space"]
    assert len(ky_space) == Nk
    assert abs(ky_space[0] - ky_min) < 1e-9
    assert abs(ky_space[-1] - ky_max) < 1e-9

def test_nanoribbon_x_periodic():
    spec = kagome_nn()
    spec_json = json.dumps(spec)
    
    Nx = 10
    Nk = 20
    # Test get_nanoribbon_data along x
    result_str = get_nanoribbon_data(spec_json, Nx, Nk, selected_bands=[0], periodic_dir="x")
    result = json.loads(result_str)
    
    assert result["error"] is None
    ky_space = result["ky_space"]
    assert len(ky_space) == Nk
    
    # Test get_nanoribbon_state along x
    result_state_str = get_nanoribbon_state(spec_json, Nx, 0.0, 5, periodic_dir="x")
    result_state = json.loads(result_state_str)
    assert result_state["error"] is None
    assert len(result_state["sites"]) == Nx * 3
    
    # Verify coordinate shifting matches y-layered structure
    site_0 = result_state["sites"][0]
    site_Q = result_state["sites"][3] # next layer, same orbital
    assert "x" in site_0
    assert "y" in site_0


def test_nanoribbon_with_bulk_overlay():
    spec = kagome_nn()
    spec_json = json.dumps(spec)

    Nx = 10
    Nk = 20
    # Evaluate nanoribbon data along y with fixed k_x = 0.5
    result_str = get_nanoribbon_data(spec_json, Nx, Nk, selected_bands=[0, 1], k_fixed=0.5)
    result = json.loads(result_str)

    assert result["error"] is None
    assert "bulk_energies" in result

    bulk_energies = result["bulk_energies"]
    # Q = 3 for Kagome model
    assert len(bulk_energies) == 3
    assert len(bulk_energies[0]) == Nk


# ── Band-index selection correctness (physically important) ───────────────────
def _full_bz_band_ranges(spec, n=41):
    """Per-band energy [min, max] over the full 2D BZ (containment reference)."""
    lattice, H_k = parse_input(spec)
    g = np.linspace(-np.pi, np.pi, n)
    KX, KY = np.meshgrid(g, g)
    kp = np.column_stack([KX.ravel(), KY.ravel()])
    B = _recip(lattice)
    Hb = H_k.evaluate_batch(kp @ (B / (2 * np.pi)), lattice.primitive_vectors)
    E = np.array([np.linalg.eigvalsh(Hb[i]) for i in range(len(kp))])
    return E.min(axis=0), E.max(axis=0)


@pytest.mark.parametrize("builder,flat_idx,eps0", [
    (kagome_nn, 0, -2.0),     # flat band is the lowest band (singular, touch at Γ)
    (lieb, 1, 0.0),           # flat band is the middle band (singular, touch at M)
    (bilayer_square, 1, 2.0), # flat band is the upper band (nonsingular)
])
def test_band_index_brackets_flat_band(builder, flat_idx, eps0):
    """Selecting the flat band's energy-sorted index must return a ribbon window
    that brackets eps0 and contains the full set of Nx flat-band states at it."""
    spec = builder()
    Nx, Nk = 24, 21
    res = json.loads(get_nanoribbon_data(json.dumps(spec), Nx, Nk,
                                         selected_bands=[flat_idx]))
    assert res["error"] is None
    E = np.array(res["energies"])               # (num_selected, Nk)
    assert E.min() - 1e-6 <= eps0 <= E.max() + 1e-6, "window must bracket eps0"
    # the flat band contributes Nx degenerate states pinned at eps0
    n_at_eps0 = int(np.sum(np.abs(E[:, Nk // 2] - eps0) < 1e-3))
    assert n_at_eps0 >= Nx, f"expected >= {Nx} states at eps0, got {n_at_eps0}"


def test_band_counting_selects_the_right_band_lieb():
    """Selecting band b must center the returned window ON band b (no off-by-one
    and no cross-band shift). Robust criterion: the MEDIAN returned energy lies in
    band b's full-BZ range, and band b is bracketed by the window. (The window's
    intentional `margin` also pulls in a few neighbouring states to expose gap
    edge states, so we test the centre, not every state.)"""
    spec = lieb()
    bmin, bmax = _full_bz_band_ranges(spec)
    Q = len(bmin)
    Nx, Nk = 24, 11
    span = bmax.max() - bmin.min()
    gap_tol = 0.05 * span
    medians = []
    for b in range(Q):
        res = json.loads(get_nanoribbon_data(json.dumps(spec), Nx, Nk,
                                             selected_bands=[b]))
        E = np.array(res["energies"])
        med = float(np.median(E))
        medians.append(med)
        assert bmin[b] - gap_tol <= med <= bmax[b] + gap_tol, \
            f"band {b}: window median {med:.3f} not in band range " \
            f"[{bmin[b]:.3f},{bmax[b]:.3f}] (off-by-one?)"
        # the window must bracket band b
        assert E.min() - gap_tol <= bmin[b] and bmax[b] <= E.max() + gap_tol
    # the per-band medians must be strictly increasing (band 0 < 1 < 2)
    assert medians[0] < medians[1] < medians[2], \
        f"band windows not monotonic in energy: {medians}"



