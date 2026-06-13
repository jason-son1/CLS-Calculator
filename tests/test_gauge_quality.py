import pytest
import numpy as np
from cls_finder.cls.gauge_analysis import (
    _symmetry_score,
    _phase_pattern,
    _calculate_representation_score,
    analyze_cls_representations,
    extract_canonical_minimal_cls,
    canonicalize_amplitudes
)
from cls_finder.models import library
from cls_finder.io.parser import parse_input

def test_within_orbital_symmetry():
    # Orbital 0 has symmetric amplitudes [1.0, -1.0]
    # Orbital 1 has symmetric amplitudes [0.5, -0.5] (different scale)
    # This should have a perfect symmetry score of 1.0
    A_0_R = {
        0: {(0, 0): 1.0, (1, 0): -1.0},
        1: {(0, 0): 0.5, (1, 0): -0.5}
    }
    
    score = _symmetry_score(A_0_R)
    assert abs(score - 1.0) < 1e-9, "Symmetric amplitudes with different scales should have score 1.0"

    # Non-symmetric case
    A_0_R_bad = {
        0: {(0, 0): 1.0, (1, 0): -0.5},  # asymmetric
        1: {(0, 0): 0.5, (1, 0): -0.5}   # symmetric
    }
    score_bad = _symmetry_score(A_0_R_bad)
    assert score_bad < 0.9, "Asymmetric amplitudes should have a lower score"

def test_relative_phase_pattern():
    # Orbital 0 has phases 0 and pi (relative phase pi)
    # Orbital 1 has phases pi/8 and pi/8 + pi (relative phase pi)
    # This should detect 'π' relative phase pattern even though absolute phases are not globally aligned.
    A_0_R = {
        0: {(0, 0): 1.0, (1, 0): -1.0},
        1: {(0, 0): 0.5 * np.exp(1j * np.pi/8), (1, 0): -0.5 * np.exp(1j * np.pi/8)}
    }
    
    pat = _phase_pattern(A_0_R)
    assert pat == 'π', f"Should detect 'π' phase pattern, got {pat}"

def test_canonicalize_amplitudes():
    # Input has arbitrary global phase exp(1j * 0.45)
    f = np.exp(1j * 0.45)
    A_0_R = {
        0: {(0, 0): 1.0 * f, (1, 0): -1.0 * f},
        1: {(0, 0): 0.5 * f, (1, 0): -0.5 * f}
    }
    
    A_canon = canonicalize_amplitudes(A_0_R, Q=2, d=2)
    
    # After canonicalization, the amplitudes should be real and normalized (max abs is 1.0)
    for q, qd in A_canon.items():
        for cell, val in qd.items():
            assert abs(val.imag) < 1e-9, "Imaginary part should be zero after canonicalization"
            
    # Max magnitude should be 1.0
    max_val = max(abs(val) for qd in A_canon.values() for val in qd.values())
    assert abs(max_val - 1.0) < 1e-9, "Max magnitude should be 1.0"

def test_extract_canonical_minimal_cls():
    A_0_R = {
        0: {(0, 0): 1.0, (1, 0): -1.0},
        1: {(0, 0): 0.5, (1, 0): -0.5}
    }
    
    res = extract_canonical_minimal_cls(A_0_R, Q=2, d=2)
    assert res is not None
    assert res["support_size"] == 4
    assert res["display_mode"] == "amplitude"
    assert res["realness"] > 0.99
    assert res["phase_pattern"] == "π"

def test_quality_aware_gauge_selection():
    # Load Lieb model and verify that the canonicalization is correct
    model_spec = library.lieb()
    lattice, H_k = parse_input(model_spec)
    
    from cls_finder.cls.analytic import extract_all_cls_analytic
    from cls_finder.cls.reduce import minimize_cls
    import sympy
    
    eps0 = 0.0
    symbols = sympy.symbols('x1 x2')
    
    all_gauges = extract_all_cls_analytic(H_k, eps0)
    assert len(all_gauges) > 0
    
    for g_key, (x_k, A_0_R) in all_gauges.items():
        x_k_min, A_0_R_min = minimize_cls(x_k, symbols)
        A_canon = canonicalize_amplitudes(A_0_R_min, H_k.rows, lattice.dimension)
        
        # Verify that max amplitude is 1.0 and is real/imaginary (quantized to pi/2)
        max_val = max(abs(val) for qd in A_canon.values() for val in qd.values())
        assert abs(max_val - 1.0) < 1e-9
        
        for q, qd in A_canon.items():
            for cell, val in qd.items():
                # Check if quantized to pi/2, i.e. real or imaginary
                assert abs(val.real) < 1e-9 or abs(val.imag) < 1e-9 or abs(abs(val) - 1.0) < 1e-9
