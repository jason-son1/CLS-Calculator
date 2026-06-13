"""
Tests for the complexity guards that keep large Hamiltonians from hanging:
  - det_and_adjugate cooperative budgets (terms / ops / time)
  - size-gated analytic extraction degrading to an empty result (→ numerical)
"""
import sympy
import pytest

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.cls import analytic
from cls_finder.cls.analytic import extract_all_cls_analytic, GATES


def _dense_matrix(Q, d=2, terms=3):
    """Q×Q MatrixPoly whose entries each have `terms` monomials — its
    Faddeev-LeVerrier intermediates grow fast, useful for budget tests."""
    data = []
    for i in range(Q):
        row = []
        for j in range(Q):
            p = LaurentPoly.zero(d)
            for t in range(terms):
                exp = tuple(((i + j + t) % 3 - 1,) * 1 + (0,) * (d - 1))
                p = p + LaurentPoly.monomial(exp, float(1 + t))
            row.append(p)
        data.append(row)
    return MatrixPoly(data, d)


def test_det_small_no_budget_ok():
    M = _dense_matrix(3)
    det, adj = M.det_and_adjugate()          # no budget → completes
    assert det is not None and adj.rows == 3


def test_det_term_budget_raises():
    M = _dense_matrix(5, terms=3)
    with pytest.raises(MemoryError):
        M.det_and_adjugate(max_terms=8)      # tiny budget → abort


def test_det_ops_budget_raises():
    M = _dense_matrix(6, terms=3)
    with pytest.raises(MemoryError):
        M.det_and_adjugate(max_ops=50)       # tiny op budget → abort


def test_det_generous_budget_completes():
    M = _dense_matrix(3, terms=2)
    det, adj = M.det_and_adjugate(max_terms=10**6, max_ops=10**9, max_seconds=30)
    assert det is not None


def test_symbolic_gate_degrades_gracefully(monkeypatch):
    """Above max_symbolic_Q / max_nullspace_Q the analytic extractor returns an
    empty dict (caller then falls back to the numerical CLS) instead of
    attempting an explosive symbolic computation."""
    monkeypatch.setitem(GATES, "max_symbolic_Q", 4)
    monkeypatch.setitem(GATES, "max_nullspace_Q", 4)
    M = _dense_matrix(6, terms=3)            # Q=6 > gate
    res = extract_all_cls_analytic(M, 0.0)
    assert res == {}


def test_mul_ops_matches_actual_growth():
    """_mul_ops must be a cheap upper-bound proxy for real multiply cost."""
    A = _dense_matrix(3, terms=2)
    ops = A._mul_ops(A)
    assert ops == sum(len(A.data[i][k].coefs) * len(A.data[k][j].coefs)
                      for i in range(3) for j in range(3) for k in range(3))
