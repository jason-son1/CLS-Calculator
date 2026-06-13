"""
Tests for the analytical-CLS refinements:
  - Rhim-Yang degenerate mixing + complete basis selection (Kagome-3)
  - resultant-based singularity detection on the Brillouin torus
  - module-minimality verification
  - eigenstate-residual cross-check for degenerate bands
"""
import numpy as np
import sympy
import pytest

from cls_finder.io.parser import parse_input
from cls_finder.cls.analytic import select_best_cls_gauge, select_cls_basis
from cls_finder.cls.reduce import verify_cls_minimal
from cls_finder.cls.numeric import verify_cls_eigenstate
from cls_finder.cls.syzygy import compute_cls_generators
from cls_finder.classify.torus_zeros import has_common_zero_on_torus
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.models import library

X1, X2 = sympy.symbols('x1 x2')

# (model, eps0, expected_singular) — known physics
KNOWN = [
    ("zigzag_chain",   -2.0, False),
    ("checkerboard_1",  0.0, True),
    ("checkerboard_2",  0.0, True),
    ("checkerboard_3",  0.0, False),
    ("lieb",            0.0, True),
    ("modified_lieb",   0.0, True),
    ("kagome_nn",      -2.0, True),
    ("kagome_3",       -2.0, False),   # degenerate but NON-singular (§8.2)
    ("bilayer_square",  2.0, False),
]


@pytest.mark.parametrize("name,eps0,expected", KNOWN)
def test_resultant_singularity_matches_physics(name, eps0, expected):
    spec = getattr(library, name)()
    lat, H = parse_input(spec)
    syms = [X1, X2][:lat.dimension]
    xk, _, _ = select_best_cls_gauge(H, eps0, syms, lattice=lat)
    verdict, _ = has_common_zero_on_torus(xk, syms)
    assert verdict is not None, f"{name}: resultant test should apply (d<=2, exact)"
    assert verdict == expected, f"{name}: expected singular={expected}, got {verdict}"


def test_kagome3_degenerate_basis_is_complete_and_nonsingular():
    """Kagome-3 hosts a doubly degenerate flat band; the basis must contain
    TWO independent, non-singular CLS (the Rhim-Yang mixed pair)."""
    lat, H = parse_input(library.kagome_3())
    basis = select_cls_basis(H, -2.0, [X1, X2], M=2, lattice=lat)
    assert len(basis) == 2, "degenerate band must yield 2 independent CLS"
    for _, _, meta in basis:
        assert meta["is_singular"] is False
    # The two CLS must differ (not the same gauge repeated).
    assert basis[0][2]["gauge_id"] != basis[1][2]["gauge_id"]


def test_kagome3_basis_members_are_independent_eigenstates():
    lat, H = parse_input(library.kagome_3())
    basis = select_cls_basis(H, -2.0, [X1, X2], M=2, lattice=lat)
    # Each must be a genuine flat-band eigenstate.
    for x_k_min, _, _ in basis:
        ok, msg = verify_cls_eigenstate(H, lat, -2.0, x_k_min)
        assert ok, msg
    # Independent at a generic k-point.
    A = np.array(lat.primitive_vectors, dtype=float)
    B = 2.0 * np.pi * np.linalg.solve(A @ A.T, A)
    kg = np.array([0.137, 0.291]) @ B
    vecs = np.array([[p.evaluate(kg, lat.primitive_vectors) for p in basis[i][0]]
                     for i in range(2)])
    assert np.linalg.matrix_rank(vecs, tol=1e-7) == 2


@pytest.mark.parametrize("name,eps0", [("lieb", 0.0), ("modified_lieb", 0.0),
                                       ("bilayer_square", 2.0), ("kagome_nn", -2.0)])
def test_selected_gauge_is_module_minimal(name, eps0):
    spec = getattr(library, name)()
    lat, H = parse_input(spec)
    syms = [X1, X2][:lat.dimension]
    xk, _, _ = select_best_cls_gauge(H, eps0, syms, lattice=lat)
    ok, msg = verify_cls_minimal(xk, syms)
    assert ok, f"{name}: selected CLS should be module-minimal — {msg}"


def test_lieb_minimal_support_is_four():
    """Regression for the GCD-cutoff fix: Lieb CLS must reduce to 4 sites."""
    lat, H = parse_input(library.lieb())
    _, A, _ = select_best_cls_gauge(H, 0.0, [X1, X2], lattice=lat)
    support = sum(len(v) for v in A.values())
    assert support == 4, f"Lieb CLS support should be 4, got {support}"


# ── Gröbner / syzygy complete generating set ──────────────────────────────────

@pytest.mark.parametrize("name,eps0,rank,is_free", [
    ("lieb",           0.0, 1, True),
    ("checkerboard_3", 0.0, 1, True),
    ("bilayer_square", 2.0, 1, True),
    ("kagome_3",      -2.0, 2, False),   # non-free polynomial module (n_gen=3 > rank)
])
def test_syzygy_module_structure(name, eps0, rank, is_free):
    spec = getattr(library, name)()
    lat, H = parse_input(spec)
    syms = [X1, X2][:lat.dimension]
    info = compute_cls_generators(H, eps0, syms)
    assert info is not None, f"{name}: syzygy should apply (rational coeffs)"
    assert info["rank"] == rank
    assert info["is_free"] == is_free
    assert info["n_generators"] >= rank


def test_syzygy_generators_are_eigenstates():
    """Every syzygy generator must annihilate H_bar (be a genuine CLS)."""
    lat, H = parse_input(library.kagome_3())
    info = compute_cls_generators(H, -2.0, [X1, X2])
    Hbar = H - MatrixPoly.identity(H.rows, H.d) * (-2.0)
    for g in info["generators"]:
        x_mat = MatrixPoly([[p] for p in g], H.d)
        assert (Hbar * x_mat).is_zero(1e-7)


def test_syzygy_irrational_coeffs_fall_back():
    """Models with irrational coefficients (sqrt(2)/sqrt(3)) can't use exact
    Gröbner and must return None so the caller falls back to the heuristic."""
    for name, eps0 in [("zigzag_chain", -2.0), ("honeycomb_flat", -1.0)]:
        spec = getattr(library, name)()
        lat, H = parse_input(spec)
        syms = [X1, X2][:lat.dimension]
        assert compute_cls_generators(H, eps0, syms) is None, name
