"""
Structural Schur reduction for "rim-decorated" flat-band models.

These are the three user-provided test models (now library presets). A constant
invertible rim block coupled to a hub reduces the CLS extraction to a small,
numerically *stable* determinant, where the full Q x Q Faddeev-LeVerrier adjugate
would be float-unstable (8x8+ with sqrt(3) coefficients). The flat band sits at
E = 0 in every case.
"""
import time
import sympy
import pytest

from cls_finder.io.parser import parse_input
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.cls.analytic import select_cls_basis, _structural_reduce
from cls_finder.models import library

X1, X2 = sympy.symbols('x1 x2')


@pytest.mark.parametrize("builder,Q,M", [
    (library.flatband_5_trig,       5, 1),
    (library.flatband_5_sqrt3,      5, 1),
    (library.flatband_10_sqrt3_deg, 10, 2),
])
def test_structural_models_give_valid_cls(builder, Q, M):
    """Each model must yield exactly M CLS, all verified null vectors of H_bar at
    E=0, selected via the structural Schur reduction, and computed quickly."""
    lat, H = parse_input(builder())
    H_bar = H - MatrixPoly.identity(Q, 2) * 0.0

    t = time.perf_counter()
    basis = select_cls_basis(H, 0.0, [X1, X2], M=M, lattice=lat)
    elapsed = time.perf_counter() - t

    assert elapsed < 15.0, f"structural extraction too slow: {elapsed:.1f}s"
    assert len(basis) == M, f"expected {M} CLS, got {len(basis)}"
    for x_k, _A, meta in basis:
        xm = MatrixPoly([[p] for p in x_k], 2)
        assert (H_bar * xm).is_zero(1e-6), \
            f"{builder.__name__}: gauge {meta['gauge_id']} is not a null vector"
        assert str(meta["gauge_id"]).startswith("structural"), \
            f"{builder.__name__}: expected structural gauge, got {meta['gauge_id']}"


def test_structural_reduce_detects_rim_block():
    """The reducer must find the constant-invertible rim and return the hub-only
    (rim == 0) CLS for the A == 0 family."""
    lat, H = parse_input(library.flatband_5_trig())
    H_bar = H - MatrixPoly.identity(5, 2) * 0.0
    vecs = _structural_reduce(H_bar, 5, 2, verify=True)
    assert vecs, "structural reduction found nothing"
    for x in vecs:
        # rim orbitals (3,4) carry zero amplitude; CLS lives on the hub
        assert x[3].is_zero(1e-9) and x[4].is_zero(1e-9)


def test_structural_reduce_absent_returns_none():
    """kagome NN at E=-2: every H_bar diagonal becomes the constant +2 (so the
    whole matrix is 'rim', len(T) >= Q) and the rim-rim block is non-constant —
    either way the reducer must decline (return None), leaving the standard
    adjugate gauge to handle it."""
    lat, H = parse_input(library.kagome_nn())
    H_bar = H - MatrixPoly.identity(3, 2) * (-2.0)   # flat band at E=-2
    assert _structural_reduce(H_bar, 3, 2, verify=True) is None
