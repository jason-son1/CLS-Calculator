"""
Real-Space CLS-based Topological Flat Band Engineering (cls_finder.engineer).

Covers Phase 1-5 of design_flat_band():
  - Module1 (spec): LatticeSpec/DesignTarget validation, named k-points.
  - Module2 (pairing): the destructive-interference Pairing Rule (Condition 1)
    holds exactly for an arbitrary singularity.
  - Module3 (chiral): the closed-form two-pair solver satisfies the chiral
    equation Sum_p A_p e^{i Phi_p} D_p = 0 (a^(alpha) = -i w b^(alpha)).
  - Module4 (assembly): f(k) vanishes at the target singularity, the
    projector is rank-1/continuous there with chern.local_winding /
    chern.loop_winding == w_i, chern.analytic_chern(x_k).C == target C, and
    the result is independent of the per-sublattice gauge zeta^(alpha).
  - Module5 (hamiltonian): H(k) psi(k) = E0 psi(k) exactly (the flat band),
    P(k) is Hermitian/idempotent/unit-trace everywhere including at the
    designed singularity (vortex_vector patch), M(k) is Hermitian, and the
    inverse-Fourier-transformed H_trunc converges to H(k) as R_cut grows.
  - pipeline: design_flat_band() end-to-end for the |C|=1 minimal
    construction (w=+-1) and an S=2, C=0 (Gamma/+1, M/-1) case, including the
    documented analytic/FHS sign-convention relation (_chern_matches).
"""
import numpy as np
import pytest

from cls_finder.classify import chern as _chern
from cls_finder.engineer import (
    NAMED_K_POINTS, resolve_k_frac,
    SublatticeSpec, LatticeSpec, SingularityTarget, DesignTarget, validate_design,
    make_pair, build_cls_design, solve_two_pair_chiral, moments, bond_cartesian,
    build_x_k, evaluate_psi, vortex_vector, verify_projector_continuity,
    default_M_k, build_hamiltonian, inverse_fourier_transform, hoppings_to_matrixpoly,
    design_flat_band,
)
from cls_finder.engineer.pipeline import _chern_matches, _phase5_success

_PRIM = np.array([[1.0, 0.0], [0.0, 1.0]])


def _square_lattice(zetas=(0.0, 0.0)):
    return LatticeSpec(primitive_vectors=_PRIM,
                        sublattices=[SublatticeSpec('A', zetas[0]),
                                      SublatticeSpec('B', zetas[1])])


# ── Module 1: spec ────────────────────────────────────────────────────────
def test_resolve_k_frac_named_and_explicit():
    assert resolve_k_frac('Gamma') == (0.0, 0.0)
    assert resolve_k_frac('M') == (0.5, 0.5)
    assert resolve_k_frac((0.1, 0.2)) == (0.1, 0.2)
    with pytest.raises(KeyError):
        resolve_k_frac('NotAPoint')


def test_lattice_spec_rejects_degenerate_primitive_vectors():
    with pytest.raises(ValueError):
        LatticeSpec(primitive_vectors=np.array([[1.0, 0.0], [2.0, 0.0]]),
                    sublattices=[SublatticeSpec('A')])


def test_reciprocal_vectors_orientation_preserving_for_square_lattice():
    """B = 2*pi*I for prim=I: (f1,f2) -> (kx,ky) is orientation-preserving,
    which is what the pipeline's analytic/FHS sign-convention discussion
    (_chern_matches) for this lattice relies on."""
    B = _square_lattice().reciprocal_vectors()
    assert np.allclose(B, 2.0 * np.pi * np.eye(2), atol=1e-12)


def test_singularity_target_rejects_w_other_than_pm1():
    with pytest.raises(ValueError):
        SingularityTarget('Gamma', 'Gamma', 2)


def test_design_target_validate_sum_mismatch():
    target = DesignTarget(C=1, singularities=[SingularityTarget('Gamma', 'Gamma', -1)])
    with pytest.raises(ValueError):
        target.validate()


def test_design_target_nonzero_C_without_singularities_raises():
    with pytest.raises(ValueError):
        DesignTarget(C=1, singularities=[]).validate()


def test_design_target_validate_duplicate_singularity_location():
    target = DesignTarget(C=2, singularities=[
        SingularityTarget('G1', 'Gamma', 1), SingularityTarget('G2', 'Gamma', 1),
    ])
    with pytest.raises(ValueError):
        target.validate()


def test_validate_design_requires_two_sublattices_for_nonzero_C():
    lat1 = LatticeSpec(primitive_vectors=_PRIM, sublattices=[SublatticeSpec('A')])
    target = DesignTarget(C=1, singularities=[SingularityTarget('Gamma', 'Gamma', 1)])
    with pytest.raises(ValueError):
        validate_design(lat1, target)


# ── Module 2: pairing ─────────────────────────────────────────────────────
def test_make_pair_condition1_holds_exactly():
    sing = SingularityTarget('K', 'K', 1)
    pair = make_pair(1, 0, 0.37, 1.0, 0, 0, sing)
    assert abs(pair.residual(sing)) < 1e-12


@pytest.mark.parametrize('w', [1, -1])
def test_cls_design_condition1_residuals_vanish(w):
    sing = SingularityTarget('Gamma', 'Gamma', w)
    design = build_cls_design(sing, _square_lattice(), shell_offset=0)
    for r in design.verify_condition1():
        assert abs(r) < 1e-9


# ── Module 3: chiral ──────────────────────────────────────────────────────
@pytest.mark.parametrize('w', [1, -1])
def test_solve_two_pair_chiral_satisfies_chiral_equation(w):
    d1 = bond_cartesian((1, 0), _PRIM)
    d2 = bond_cartesian((0, 1), _PRIM)
    sol = solve_two_pair_chiral(d1, d2, w)
    lhs = (sol['A1'] * np.exp(1j * sol['Phi1']) * sol['D1']
           + sol['A2'] * np.exp(1j * sol['Phi2']) * sol['D2'])
    assert abs(lhs) < 1e-12


def test_solve_two_pair_chiral_rejects_parallel_bonds():
    d1 = bond_cartesian((1, 0), _PRIM)
    d2 = bond_cartesian((2, 0), _PRIM)
    with pytest.raises(ValueError):
        solve_two_pair_chiral(d1, d2, 1)


@pytest.mark.parametrize('w', [1, -1])
def test_moments_satisfy_chiral_condition(w):
    sing = SingularityTarget('Gamma', 'Gamma', w)
    design = build_cls_design(sing, _square_lattice(), shell_offset=0)
    for sub in design.sublattices:
        a, b = moments(sub, sing, _square_lattice())
        assert abs(a - (-1j * w * b)) < 1e-9


# ── Module 4: assembly ────────────────────────────────────────────────────
@pytest.mark.parametrize('w', [1, -1])
def test_build_x_k_vanishes_at_target_singularity(w):
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', w)
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    f, psi, norm = evaluate_psi(x_k, sing.k_cartesian(lat), _PRIM)
    assert norm < 1e-9
    assert psi is None


@pytest.mark.parametrize('w', [1, -1])
def test_projector_continuity_and_winding_match_target(w):
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', w)
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    info = verify_projector_continuity(x_k, lat, sing.k_cartesian(lat))
    assert info['first_order']['rank_one']
    assert info['first_order']['winding'] == w
    assert info['loop']['projector_continuous']
    assert info['loop']['winding'] == w


@pytest.mark.parametrize('w', [1, -1])
def test_analytic_chern_matches_target(w):
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', w)
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    ana = _chern.analytic_chern(x_k, lat.to_lattice())
    assert ana['C'] == w
    assert ana['n_common_zeros'] == 1
    assert ana['projector_continuous']


def test_zeta_gauge_does_not_affect_topology():
    """zeta^(alpha) is an overall per-sublattice phase (ARCHITECTURE.md Sec.
    0.1): analytic_chern(x_k).C must be identical whether zeta = 0 or not."""
    sing = SingularityTarget('Gamma', 'Gamma', 1)
    cls_designs = [build_cls_design(sing, _square_lattice(), shell_offset=0)]
    lattice = _square_lattice().to_lattice()
    x0 = build_x_k(_square_lattice((0.0, 0.0)), cls_designs)
    x1 = build_x_k(_square_lattice((0.7, -1.3)), cls_designs)
    assert _chern.analytic_chern(x0, lattice)['C'] == _chern.analytic_chern(x1, lattice)['C']


@pytest.mark.parametrize('w', [1, -1])
def test_vortex_vector_is_normalized(w):
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', w)
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    v = vortex_vector(x_k, sing.k_cartesian(lat), _PRIM)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-12


# ── Module 5: hamiltonian ──────────────────────────────────────────────────
@pytest.mark.parametrize('E0', [0.0, 0.7])
def test_flat_band_eigenvalue_exact(E0):
    """H(k) psi(k) = E0 psi(k) (Eq. 127) at generic k-points, for E0 = 0
    and a nonzero E0 (exercising the E0*P term)."""
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', 1)
    target = DesignTarget(C=1, singularities=[sing])
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    H_k = build_hamiltonian(x_k, lat, target, E0=E0)
    for k in ([0.3, 0.6], [1.1, -0.4], [2.0, 2.5]):
        _, psi, _ = evaluate_psi(x_k, np.array(k), _PRIM)
        Hk = H_k.evaluate(np.array(k), _PRIM)
        assert np.linalg.norm(Hk @ psi - E0 * psi) < 1e-10


def test_flat_band_eigenvalue_exact_at_singularity_via_vortex_vector():
    """At k_i itself, ||f(k_i)||=0 so psi is undefined; H(k_i) v_i = E0 v_i
    must hold for the analytic-limit vortex_vector v_i instead."""
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', 1)
    target = DesignTarget(C=1, singularities=[sing])
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    H_k = build_hamiltonian(x_k, lat, target, E0=0.3)
    v = vortex_vector(x_k, sing.k_cartesian(lat), _PRIM)
    Hk = H_k.evaluate(sing.k_cartesian(lat), _PRIM)
    assert np.linalg.norm(Hk @ v - 0.3 * v) < 1e-10


def test_projector_hermitian_idempotent_unit_trace():
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', 1)
    target = DesignTarget(C=1, singularities=[sing])
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    H_k = build_hamiltonian(x_k, lat, target, E0=0.0)
    ks = np.array([[0.0, 0.0], [0.3, 0.6], [1.1, -0.4], [2.0, 2.5]])  # incl. Gamma
    P = H_k.projector(ks, _PRIM)
    for Pi in P:
        assert np.linalg.norm(Pi - Pi.conj().T) < 1e-10
        assert np.linalg.norm(Pi @ Pi - Pi) < 1e-8
        assert abs(np.trace(Pi).real - 1.0) < 1e-10


def test_default_M_k_hermitian():
    M = default_M_k(_square_lattice(), t=0.3, delta=0.5)
    for k in ([0.3, 0.6], [1.1, -0.4], [np.pi, np.pi]):
        Mk = M.evaluate(np.array(k), _PRIM)
        assert np.linalg.norm(Mk - Mk.conj().T) < 1e-10


def test_inverse_fourier_transform_truncation_improves_with_rcut():
    lat = _square_lattice()
    sing = SingularityTarget('Gamma', 'Gamma', 1)
    target = DesignTarget(C=1, singularities=[sing])
    x_k = build_x_k(lat, [build_cls_design(sing, lat, shell_offset=0)])
    H_k = build_hamiltonian(x_k, lat, target, E0=0.0)
    sample_k = np.array([0.3, 0.6])
    ratios, errs = [], []
    for rc in (3, 6):
        ift = inverse_fourier_transform(H_k, lat, n_grid=24, R_cut=rc)
        ratios.append(ift['truncation_ratio'])
        H_trunc = hoppings_to_matrixpoly(ift['hoppings'], lat.N)
        Mk_trunc = H_trunc.evaluate(sample_k, _PRIM)
        Mk_full = H_k.evaluate(sample_k, _PRIM)
        assert np.linalg.norm(Mk_trunc - Mk_trunc.conj().T) < 1e-8  # Hermitian for any R_cut
        errs.append(np.linalg.norm(Mk_trunc - Mk_full))
    assert ratios[1] < ratios[0]
    assert errs[1] < errs[0]


# ── pipeline ────────────────────────────────────────────────────────────────
def test_chern_matches_helper():
    assert _chern_matches(1, 1)
    assert _chern_matches(-1, 1)   # documented chern.py analytic/FHS sign flip
    assert _chern_matches(0, 0)
    assert not _chern_matches(2, 1)


def test_phase5_success_accepts_touching_band():
    """A topological CLS flat band TOUCHES the dispersive sector at its
    singularity (Note A Sec.8); it is NOT required to be isolated. The success
    criterion is therefore the analytic design (analytic_match), with isolation
    only adding the FHS confirmation when the truncated band happens to be
    gapped."""
    # analytically-correct design, truncation touches (gapless) -> SUCCESS
    # regardless of the (unreliable-across-a-touching) FHS Chern match.
    assert _phase5_success(analytic_match=True, trunc_isolated=False, trunc_match=False)
    assert _phase5_success(analytic_match=True, trunc_isolated=False, trunc_match=True)
    # gapped truncation: additionally require the FHS Chern to confirm.
    assert _phase5_success(analytic_match=True, trunc_isolated=True, trunc_match=True)
    assert not _phase5_success(analytic_match=True, trunc_isolated=True, trunc_match=False)
    # wrong designed topology fails no matter what the truncation looks like.
    assert not _phase5_success(analytic_match=False, trunc_isolated=False, trunc_match=False)
    assert not _phase5_success(analytic_match=False, trunc_isolated=True, trunc_match=True)


def test_design_flat_band_derives_zeta_from_position():
    """zeta^(alpha) is auto-derived from the orbital position and the primary
    singularity (zeta = k_0 . r_alpha), not taken from the input. For B at
    cell-center (0.5, 0.5) with the M singularity, derived zeta_B = pi; the
    flat band is still exactly flat (Eq.127) and the analytic Chern matches."""
    lat = LatticeSpec(primitive_vectors=_PRIM, sublattices=[
        SublatticeSpec('A', zeta=0.0, position=(0.0, 0.0)),
        SublatticeSpec('B', zeta=0.0, position=(0.5, 0.5)),
    ])
    target = DesignTarget(C=1, singularities=[SingularityTarget('M', 'M', 1)])
    res = design_flat_band(lat, target, verbose=False)
    v = res.verification
    assert v['analytic_match']
    # auto-derived zeta_B = k_M . r_B = (pi,pi).(0.5,0.5) = pi
    assert abs(res.lattice_spec.sublattices[1].zeta - np.pi) < 1e-9
    assert res.lattice_spec.sublattices[0].zeta == 0.0
    # Eq.127 flatness sanity check is reported and ~0.
    assert v['flat_band_max_dev'] < 1e-9


@pytest.mark.parametrize('w', [1, -1])
def test_design_flat_band_minimal_construction(w):
    lat = _square_lattice()
    target = DesignTarget(C=w, singularities=[SingularityTarget('Gamma', 'Gamma', w)])
    res = design_flat_band(lat, target, verbose=False)
    v = res.verification
    assert v['feedback_success']
    assert v['analytic_C'] == w
    assert v['phase5_success']
    # The deliverable is the FINITE-RANGE, EXACTLY-FLAT local model (Rhim-Yang),
    # which TOUCHES the dispersive sector at the singularity (NOT isolated) --
    # that touching is where the Chern winding lives (Note A Sec.8).
    assert v['exact_flat'] and v['flat_band_max_dev'] < 1e-9
    assert v['trunc_singular'] and not v['trunc_isolated']
    assert v['truncation_ratio'] == 0.0       # exact, nothing truncated
    assert v['max_hopping_range'] <= 2        # genuinely short-range
    assert _chern_matches(v['trunc_numerical_C'], w)

    # The deliverable H_trunc (local exact MatrixPoly) is itself exactly flat:
    f, psi, norm = res.evaluate_psi(np.array([0.37, 1.21]))
    Hk = res.H_trunc.evaluate(np.array([0.37, 1.21]), lat.primitive_vectors)
    assert np.linalg.norm(Hk @ psi - 0.0 * psi) < 1e-9


def test_design_flat_band_S2_C0_two_singularities():
    """A C=0 target realized by two opposite-winding singularities (Gamma
    w=+1, M w=-1). The per-singularity feedback criterion (feedback_success)
    is conservative and is not guaranteed to pass for S>=2 (pipeline.py
    SCOPE NOTE), but the GLOBAL analytic Chern and the IFT-truncated
    Hamiltonian's numerical Chern both come out at the target C=0, gapped."""
    lat = _square_lattice()
    target = DesignTarget(C=0, singularities=[
        SingularityTarget('Gamma', 'Gamma', 1),
        SingularityTarget('M', 'M', -1),
    ])
    res = design_flat_band(lat, target, verbose=False)
    v = res.verification
    assert v['analytic_C'] == 0
    assert v['phase5_success']
    assert v['exact_flat'] and v['flat_band_max_dev'] < 1e-9
    assert v['truncation_ratio'] == 0.0
    assert _chern_matches(v['trunc_numerical_C'], 0)


def test_design_flat_band_rejects_inconsistent_target():
    lat = _square_lattice()
    target = DesignTarget(C=1, singularities=[SingularityTarget('Gamma', 'Gamma', -1)])
    with pytest.raises(ValueError):
        design_flat_band(lat, target, verbose=False)
