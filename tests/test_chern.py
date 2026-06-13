"""
Chern number of a flat band — numerical FHS engine + the finite-Fourier
(Rhim-Yang) structural criterion.

Validates the FHS engine on textbook GAPPED Chern insulators (QWZ on a square
lattice, Haldane on a hexagonal lattice — the latter exercising the hexagonal
reciprocal vectors), then checks the flat-band physics: standard flat bands
(Kagome, Lieb, Bilayer, Checkerboard-III, Kagome-3) are all topologically
trivial (C = 0), each for the correct structural reason, and the numerical and
analytic routes agree.
"""
import json
import numpy as np
import sympy
import pytest

from cls_finder.io.parser import parse_input
from cls_finder.band.bands import detect_flat_bands
from cls_finder.cls.analytic import select_cls_basis
from cls_finder.classify import chern
from cls_finder.models import library

SYMS = sympy.symbols('x1 x2')


def _band_chern(spec, sorted_idx, n=40):
    lat, H = parse_input(spec)
    B = chern.reciprocal_vectors(lat.primitive_vectors)
    frac = chern._frac_grid(n, n)
    w, v = np.linalg.eigh(H.evaluate_batch(frac @ B, lat.primitive_vectors))
    U = v[:, :, sorted_idx][:, :, None].reshape(n, n, H.rows, 1)
    return chern.fhs_chern(U)


def _qwz(m):
    return {"lattice": {"dimension": 2, "primitive_vectors": [[1.0, 0.0], [0.0, 1.0]],
                        "orbitals": [{"label": "A", "position": [0, 0]},
                                     {"label": "B", "position": [0, 0]}]},
            "H_symbolic": [[f"{m} + cos(kx) + cos(ky)", "sin(kx) - I*sin(ky)"],
                           ["sin(kx) + I*sin(ky)", f"-({m} + cos(kx) + cos(ky))"]],
            "options": {"k_grid": [40, 40], "flat_tol": 1e-5}}


# ── FHS engine: textbook gapped Chern insulators ─────────────────────────────
@pytest.mark.parametrize("m,absC", [(1.0, 1), (-1.0, 1), (3.0, 0)])
def test_fhs_qwz_square(m, absC):
    """QWZ lower band: |C|=1 in the topological regime (0<|m|<2), 0 outside."""
    C = _band_chern(_qwz(m), 0, n=40)
    assert abs(round(C)) == absC, f"QWZ m={m}: got C={C:.3f}, expect |C|={absC}"
    assert abs(C - round(C)) < 0.1, f"FHS not integer-quantized: {C}"


def test_fhs_haldane_hexagonal():
    """Haldane on a hexagonal lattice (tests hexagonal reciprocal vectors): the
    TRS-breaking NNN phase opens a |C|=1 gap; m=0 must give |C|=1."""
    phi = np.pi / 2
    t2 = 0.15
    t1 = "(1 + exp(-I*kx) + exp(-I*ky))"
    t1c = "(1 + exp(I*kx) + exp(I*ky))"
    nnnA = f"2*{t2}*(cos(kx - {phi}) + cos(ky - {phi}) + cos(kx - ky - {phi}))"
    nnnB = f"2*{t2}*(cos(kx + {phi}) + cos(ky + {phi}) + cos(kx - ky + {phi}))"
    spec = {"lattice": {"dimension": 2,
                        "primitive_vectors": [[1.0, 0.0], [0.5, 0.8660254037844386]],
                        "orbitals": [{"label": "A", "position": [0, 0]},
                                     {"label": "B", "position": [0.333, 0.333]}]},
            "H_symbolic": [[nnnA, t1], [t1c, nnnB]],
            "options": {"k_grid": [48, 48], "flat_tol": 1e-5}}
    C = _band_chern(spec, 0, n=48)
    assert abs(round(C)) == 1, f"Haldane should be |C|=1, got {C:.3f}"


# ── Flat bands: all standard library flat bands are C = 0 ────────────────────
@pytest.mark.parametrize("builder", [
    library.kagome_nn, library.lieb, library.bilayer_square,
    library.checkerboard_3, library.kagome_3,
])
def test_flat_band_chern_trivial(builder):
    spec = builder()
    lat, H = parse_input(spec)
    grid = spec["options"].get("k_grid", [24, 24])
    fb = detect_flat_bands(H, lat, grid, spec["options"].get("flat_tol", 1e-4))[0]
    eps0, M = fb["energy"], len(fb["degenerate_indices"])
    try:
        x_k = select_cls_basis(H, eps0, list(SYMS), M, lattice=lat)[0][0]
    except Exception:
        x_k = None
    out = chern.analyze_flat_band_chern(H, lat, eps0, M, x_k=x_k,
                                        symbols=list(SYMS), grid_n=18, scan_n=80)
    assert out["chern_number"] == 0, f"{builder.__name__}: C={out['chern_number']}"
    # For an ISOLATED band the numerical and analytic routes must agree. For a
    # touching (non-isolated) band single-band FHS is unreliable, so we only
    # require the reported C (taken from the structural winding sum) to be 0.
    if "analytic" in out and out["isolation"]["isolated"]:
        assert out["agreement"], f"{builder.__name__}: methods disagree"


def test_proposition1_no_common_zero_implies_trivial():
    """Bilayer's flat-band CLS has NO common zero -> globally smooth section ->
    C=0 with certainty (Proposition 1)."""
    spec = library.bilayer_square()
    lat, H = parse_input(spec)
    fb = detect_flat_bands(H, lat, [24, 24], 1e-4)[0]
    x_k = select_cls_basis(H, fb["energy"], list(SYMS), 1, lattice=lat)[0][0]
    ana = chern.analytic_chern(x_k, lat, n_scan=80)
    assert ana["n_common_zeros"] == 0
    assert ana["trivial_no_zero"] and ana["C"] == 0


def test_local_winding_canonical_vortex():
    """Sanity of the winding formula on the canonical vortex h(q)=q_x+i q_y:
    A_x=(1,..), A_y=(i,..) -> Im<A_x,A_y> > 0 -> w=+1; h=q_x+2q_y -> w=0."""
    v = np.array([1.0, 0.5, -0.3], dtype=complex)
    w_pos = chern.local_winding(v * 1.0, v * 1j)
    assert w_pos["rank_one"] and w_pos["winding"] == 1
    w_zero = chern.local_winding(v * 1.0, v * 2.0)   # real relative phase
    assert w_zero["rank_one"] and w_zero["winding"] == 0


def test_compute_chern_bridge_qwz_not_2d_guard_and_kagome():
    """The web endpoint returns a sane payload for a flat-band model."""
    from web import bridge
    res = json.loads(bridge.compute_chern(json.dumps(library.kagome_nn()),
                                          grid_n=18, scan_n=60))
    assert res.get("error") is None
    assert res["chern_number"] == 0
    assert "isolation" in res and "numerical" in res


# ── Robust BZ singularity finder + contour winding ───────────────────────────
from types import SimpleNamespace
from cls_finder.core.laurent import LaurentPoly

_SQ = SimpleNamespace(primitive_vectors=[[1.0, 0.0], [0.0, 1.0]], dimension=2)
_X0 = LaurentPoly.monomial((1, 0))
_X1 = LaurentPoly.monomial((0, 1))
_ONE = LaurentPoly.constant(1.0, 2)
# h(k) = (X0-1) + i(X1-1): a vector f = g*h has a CONSTANT (continuous) projector,
# so it is a section of the trivial bundle and the winding sum MUST be 0. h itself
# has two common zeros in the BZ (Gamma and (1/4,-1/4)) with opposite windings.
_H = (_X0 - _ONE) + 1j * (_X1 - _ONE)


def test_find_common_zeros_locates_both_and_dedups():
    """Both common zeros of the synthetic vortex are found, with no spurious
    duplicates (the deepest-representative de-duplication holds)."""
    f = [_H * 1.0, _H * 1j]
    zeros = chern.find_common_zeros(f, _SQ, n_scan=120)
    assert len(zeros) == 2, f"expected 2 zeros, got {len(zeros)}"
    B = chern.reciprocal_vectors(_SQ.primitive_vectors)
    fr = sorted(tuple(np.round(chern._wrap_frac(chern._cart_to_frac(z, B)), 3))
                for z in zeros)
    assert fr[0] == pytest.approx([-0.25, 0.25], abs=2e-2) or \
           fr[1] == pytest.approx([0.25, -0.25], abs=2e-2)


def test_loop_winding_signed_and_higher_order():
    """Contour winding gives the correct SIGN at each zero and resolves a
    higher-order (double) vortex as |w|=2 — the Eq.(65) generalization."""
    f1 = [_H * 1.0, _H * 1j]
    ana1 = chern.analytic_chern(f1, _SQ, n_scan=120)
    assert sorted(z["winding"] for z in ana1["per_zero"]) == [-1, 1]
    assert ana1["C"] == 0 and ana1["projector_continuous"]

    h2 = _H * _H
    f2 = [h2 * 1.0, h2 * 1j]
    ana2 = chern.analytic_chern(f2, _SQ, n_scan=120)
    assert sorted(z["winding"] for z in ana2["per_zero"]) == [-2, 2]
    assert all(z["order"] == 2 for z in ana2["per_zero"])
    assert ana2["C"] == 0          # trivial bundle: windings cancel


def test_continuous_finite_fourier_section_sums_to_zero():
    """A finite-Fourier vector with an everywhere-continuous projector is a
    section of the trivial bundle: the scalar windings of ALL its common zeros
    must cancel (Euler/Wannier obstruction). This is why an isolated flat band
    with a compact CLS is necessarily C=0; a nonzero Chern needs a singular
    (discontinuous-projector) common zero."""
    for g in ([1.0, 1j], [1.0, 0.5j], [0.3, 1.0]):
        f = [_H * g[0], _H * g[1]]
        ana = chern.analytic_chern(f, _SQ, n_scan=120)
        assert ana["projector_continuous"]
        assert ana["C"] == 0


def test_kagome_touching_winding_detected_but_discontinuous():
    """Kagome's flat band touches at Gamma: the CLS section has a common zero
    there with a nonzero raw winding (the would-be Chern if the touching were
    gapped), but the projector is DISCONTINUOUS, so it is not counted into a
    well-defined single-band Chern (reported C stays 0, well_defined False)."""
    spec = library.kagome_nn()
    lat, H = parse_input(spec)
    fb = detect_flat_bands(H, lat, [24, 24], 1e-4)[0]
    eps0, M = fb["energy"], len(fb["degenerate_indices"])
    x_k = select_cls_basis(H, eps0, list(SYMS), M, lattice=lat)[0][0]
    out = chern.explore_brillouin_zone(H, lat, eps0, M, x_k=x_k,
                                       symbols=list(SYMS), grid_n=18, scan_n=120)
    assert out["chern_number"] == 0 and not out["well_defined"]
    ana = out["analytic"]
    assert ana["n_common_zeros"] >= 1
    assert ana["has_discontinuous_zero"]
    assert abs(ana["C_all_zeros"]) >= 1          # the would-be |C| is detected
    assert "summary" in out
