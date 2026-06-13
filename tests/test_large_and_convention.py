"""
Tests for large/degenerate Hamiltonians and irrational (hexagonal) exponents:
  - generalized minor+adjugate (Cramer) for M>=2 degenerate bands at large Q
  - real-GCD minimization of CLS with irrational-but-commensurate exponents
"""
import numpy as np
import sympy
import time

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.cls.analytic import select_cls_basis, extract_all_cls_analytic

X1, X2 = sympy.symbols('x1 x2')


def _bipartite(nA, nB, d=2, seed=1):
    """Q=nA+nB, M=nB-nA zero-energy flat bands; a standard degenerate testbed."""
    rng = np.random.default_rng(seed)
    Q = nA + nB
    data = [[LaurentPoly.zero(d) for _ in range(Q)] for _ in range(Q)]
    for b in range(nB):
        for a in range(nA):
            p = LaurentPoly.zero(d)
            for _ in range(int(rng.integers(2, 4))):
                e = tuple(int(rng.integers(-1, 2)) for _ in range(d))
                p = p + LaurentPoly.monomial(e, float(rng.integers(1, 3)))
            data[nA + b][a] = p
            data[a][nA + b] = p.conjugate()
    return MatrixPoly(data, d)


def test_degenerate_large_Q_resolves():
    """Q=10, M=2 degenerate band must yield a 2-CLS basis quickly (regression
    for the previously-failing 'No analytical CLS gauge found')."""
    H = _bipartite(4, 6)            # Q=10, M=2
    t = time.perf_counter()
    basis = select_cls_basis(H, 0.0, [X1, X2], M=2, lattice=None)
    elapsed = time.perf_counter() - t
    assert len(basis) == 2, "degenerate Q=10 band must give 2 independent CLS"
    assert elapsed < 10.0, f"degenerate extraction too slow: {elapsed:.1f}s"


def test_degenerate_adjugate_vectors_are_eigenstates():
    H = _bipartite(3, 5)            # Q=8, M=2
    Hbar = H - MatrixPoly.identity(8, 2) * 0.0
    gauges = extract_all_cls_analytic(H, 0.0)
    assert len(gauges) >= 2
    for key, (x_k, _) in gauges.items():
        xm = MatrixPoly([[p] for p in x_k], 2)
        assert (Hbar * xm).is_zero(1e-6), f"gauge {key} not a null vector"


def test_real_gcd_extracts_irrational_common_factor():
    """A common factor with irrational (√3) exponents must be divided out —
    this is what keeps hexagonal-geometry CLS minimal instead of expanded."""
    s3 = np.sqrt(3)
    F = LaurentPoly({(0, 0): 1.0, (0, s3 / 6): 1.0, (0.5, 0): 1.0}, 2)
    a = LaurentPoly({(0.5, 0): 1.0, (0, -s3 / 6): 1.0}, 2)
    b = LaurentPoly({(0, s3 / 3): 1.0, (-0.5, 0): 1.0}, 2)
    g, divs = LaurentPoly.gcd_multiple([F * a, F * b], [X1, X2])
    assert len(g.coefs) == 3, "GCD should recover the 3-term irrational factor F"
    # Each quotient drops from 6 terms (product) to 2 (the bare factor).
    assert all(len(dp.coefs) == 2 for dp in divs)


def test_real_axis_unit_rational_and_irrational():
    assert abs(LaurentPoly._real_axis_unit([0.0, 0.5, 1.0]) - 0.5) < 1e-9
    s3 = np.sqrt(3)
    u = LaurentPoly._real_axis_unit([s3 / 6, s3 / 3, s3 / 2])
    assert abs(u - s3 / 6) < 1e-6
    assert LaurentPoly._real_axis_unit([1.0, np.pi]) is None  # incommensurate
