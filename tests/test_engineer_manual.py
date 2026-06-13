"""
Manual Real-Space CLS Placement (cls_finder.engineer.manual).

Covers:
  - SublatticeSpec.position default + round-trip through LatticeSpec.to_lattice().
  - build_x_k_from_sites: per-site coefficient convention
    c_{alpha,(n,m)} = A * e^{i(theta + zeta_alpha)}, accumulation of multiple
    sites at the same (alpha, (n, m)), and reconstruction of a known
    (build_cls_design-based) x_k from its own coefficients.
  - validate_manual_cls: f(k)==0 (no sites) is a hard error; an
    all-(0,0) placement is valid but flagged trivial with a warning;
    a placement with a neighbor-cell site is valid and non-trivial.
  - analyze_manual_cls end-to-end: H(k) (Eq. 127) is exactly flat at E0
    across a k-grid for a hand-reconstructed nontrivial CLS, and the
    auto-discovered chern_report matches the known target Chern number.
"""
import numpy as np
import pytest

from cls_finder.engineer import (
    SublatticeSpec, LatticeSpec, SingularityTarget,
    build_cls_design, build_x_k, evaluate_psi,
    build_x_k_from_sites, validate_manual_cls, analyze_manual_cls,
)
from cls_finder.engineer.pipeline import _chern_matches

_PRIM = np.array([[1.0, 0.0], [0.0, 1.0]])


def _square_lattice(zetas=(0.0, 0.0), positions=((0.0, 0.0), (0.0, 0.0))):
    return LatticeSpec(primitive_vectors=_PRIM,
                        sublattices=[SublatticeSpec('A', zetas[0], positions[0]),
                                      SublatticeSpec('B', zetas[1], positions[1])])


def _x_k_to_cls_sites(lat, x_k):
    """Invert build_x_k_from_sites's coefficient convention: one site per
    nonzero (alpha, (n, m)) coefficient, with the per-sublattice gauge
    zeta_alpha subtracted back out of the phase."""
    sites = []
    for alpha, p in enumerate(x_k):
        zeta = lat.sublattices[alpha].zeta
        for (n, m), c in p.coefs.items():
            A = abs(c)
            if A < 1e-12:
                continue
            theta = float(np.angle(c) - zeta)
            sites.append({"alpha": alpha, "n": int(n), "m": int(m), "A": float(A), "theta": theta})
    return sites


# ── SublatticeSpec.position ─────────────────────────────────────────────────
def test_sublattice_position_default_and_to_lattice_roundtrip():
    lat = _square_lattice()
    assert lat.sublattices[0].position == (0.0, 0.0)
    lattice = lat.to_lattice()
    assert np.allclose(lattice.orbitals[0]['position'], [0.0, 0.0])

    lat2 = _square_lattice(positions=((0.0, 0.0), (0.5, 0.5)))
    lattice2 = lat2.to_lattice()
    assert np.allclose(lattice2.orbitals[1]['position'], [0.5, 0.5])
    assert lattice2.orbitals[1]['_sublattice_hint'] == 1
    assert np.allclose(lattice2.sublattices[1]['position'], [0.5, 0.5])


# ── build_x_k_from_sites ─────────────────────────────────────────────────────
def test_build_x_k_from_sites_basic_coefficient_convention():
    lat = _square_lattice(zetas=(0.0, 0.5))
    sites = [
        {"alpha": 0, "n": 0, "m": 0, "A": 2.0, "theta": 0.0},
        {"alpha": 1, "n": 1, "m": -1, "A": 1.0, "theta": 0.3},
    ]
    x_k = build_x_k_from_sites(lat, sites)
    assert x_k[0].coefs[(0, 0)] == pytest.approx(2.0 + 0j)
    assert x_k[1].coefs[(1, -1)] == pytest.approx(1.0 * np.exp(1j * (0.3 + 0.5)))


def test_build_x_k_from_sites_accumulates_same_site():
    lat = _square_lattice()
    sites = [
        {"alpha": 0, "n": 0, "m": 0, "A": 1.0, "theta": 0.0},
        {"alpha": 0, "n": 0, "m": 0, "A": 1.0, "theta": np.pi},  # cancels
    ]
    x_k = build_x_k_from_sites(lat, sites)
    assert x_k[0].is_zero(1e-9)


def test_build_x_k_from_sites_rejects_out_of_range_sublattice():
    lat = _square_lattice()
    with pytest.raises(ValueError):
        build_x_k_from_sites(lat, [{"alpha": 5, "n": 0, "m": 0, "A": 1.0, "theta": 0.0}])


def test_build_x_k_from_sites_reconstructs_known_design():
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', 1)
    x_k_ref = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    sites = _x_k_to_cls_sites(lat, x_k_ref)
    x_k_test = build_x_k_from_sites(lat, sites)
    for p_ref, p_test in zip(x_k_ref, x_k_test):
        assert (p_ref - p_test).is_zero(1e-9)


# ── validate_manual_cls ───────────────────────────────────────────────────────
def test_validate_manual_cls_empty_is_invalid():
    lat = _square_lattice()
    x_k = build_x_k_from_sites(lat, [])
    val = validate_manual_cls(x_k)
    assert not val['valid']


def test_validate_manual_cls_all_central_cell_is_trivial():
    lat = _square_lattice()
    sites = [{"alpha": 0, "n": 0, "m": 0, "A": 1.0, "theta": 0.0},
             {"alpha": 1, "n": 0, "m": 0, "A": 1.0, "theta": 0.5}]
    x_k = build_x_k_from_sites(lat, sites)
    val = validate_manual_cls(x_k)
    assert val['valid']
    assert val['trivial']
    assert val['warnings']


def test_validate_manual_cls_neighbor_cell_is_nontrivial():
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', 1)
    x_k_ref = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    val = validate_manual_cls(x_k_ref)
    assert val['valid']
    assert not val['trivial']
    assert val['warnings'] == []


# ── analyze_manual_cls end-to-end ────────────────────────────────────────────
def test_analyze_manual_cls_invalid_when_no_sites():
    lat = _square_lattice()
    result = analyze_manual_cls(lat, [])
    assert not result['valid']
    assert 'reason' in result


def test_analyze_manual_cls_trivial_all_central_cell():
    lat = _square_lattice()
    sites = [{"alpha": 0, "n": 0, "m": 0, "A": 1.0, "theta": 0.0},
             {"alpha": 1, "n": 0, "m": 0, "A": 1.0, "theta": 0.7}]
    result = analyze_manual_cls(lat, sites)
    assert result['valid']
    assert result['trivial']
    assert result['warnings']
    assert result['chern_report']['chern_number'] == 0


@pytest.mark.parametrize('w', [1, -1])
def test_analyze_manual_cls_end_to_end_nontrivial(w):
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', w)
    x_k_ref = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    sites = _x_k_to_cls_sites(lat, x_k_ref)

    result = analyze_manual_cls(lat, sites, E0=0.0, t=0.3, delta=0.5)
    assert result['valid']
    assert not result['trivial']
    assert len(result['zeros']) >= 1

    chern_report = result['chern_report']
    assert _chern_matches(chern_report['chern_number'], w)

    # Eq. 127: H(k) psi(k) = E0 psi(k) for ANY f(k) -- exact flatness check.
    H_k = result['H_k']
    for k in ([0.3, 0.6], [1.1, -0.4], [2.0, 2.5]):
        _, psi, norm = evaluate_psi(x_k_ref, np.array(k), _PRIM)
        assert norm > 1e-9
        Hk = H_k.evaluate(np.array(k), _PRIM)
        assert np.linalg.norm(Hk @ psi - 0.0 * psi) < 1e-8

    # Truncated real-space model is gapped (isolated) at some R_cut.
    assert result['iso_trunc']['isolated']
