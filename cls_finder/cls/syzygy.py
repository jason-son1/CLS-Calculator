"""
Rigorous CLS generators via the syzygy (Gröbner) module of H_k - eps0 I.

The compact-localized states at energy eps0 are exactly the solutions of
    (H_k - eps0 I) x_k = 0
over the Laurent ring R = C[X_1^±, ..., X_d^±].  Writing H_bar = H_k - eps0 I,
the solution module is the *syzygy module of the columns of H_bar* — the set of
coefficient vectors x with  sum_c x_c * (column c) = 0.

The minor+adjugate method only produces *one* solution (and collapses entirely
for degenerate bands), and the Rhim-Yang mixing heuristic guesses combinations.
A Gröbner-basis syzygy computation instead returns the *complete generating set*
of the whole CLS module — uniformly for any Q, any degeneracy M, singular or
not. Working over the polynomial ring after a unit monomial shift is valid
because every Laurent solution is a monomial multiple of a polynomial one.

For Convention II (non-integer exponents from orbital offsets) or coefficients
that cannot be made exact, the routine returns None so the caller falls back to
the heuristic gauge selection.
"""
from fractions import Fraction

import numpy as np
import sympy

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly

# Exact Gröbner requires the flatness relation det(H_bar)=0 to hold exactly,
# which an approximate rational of an irrational coefficient (e.g. sqrt(2))
# destroys. So we only accept coefficients that are *clean* small rationals.
_MAX_DEN = 256
_RAT_TOL = 1e-9


def _to_rational(v):
    """Return an exact sympy.Rational for v if v is a clean small rational
    (denominator <= _MAX_DEN), else None."""
    frac = Fraction(v).limit_denominator(_MAX_DEN)
    if abs(float(frac) - v) < _RAT_TOL:
        return sympy.Rational(frac.numerator, frac.denominator)
    return None


def _entry_to_poly_expr(poly, symbols, shift):
    """LaurentPoly entry -> SymPy expr over Gaussian rationals, with exponents
    shifted by `shift` (per-variable) so the result is an ordinary polynomial.
    Returns (expr, exact)."""
    if not poly.coefs:
        return sympy.Integer(0), True
    expr = sympy.Integer(0)
    exact = True
    for exp, coef in poly.coefs.items():
        c = complex(coef)
        re = _to_rational(c.real)
        im = _to_rational(c.imag)
        if re is None or im is None:
            exact = False
            break
        term = re + im * sympy.I
        for l, s in enumerate(symbols):
            e = exp[l] - shift[l]
            if abs(e - round(e)) > 1e-9:
                exact = False
                break
            term *= s ** int(round(e))
        expr += term
    return sympy.expand(expr), exact


def compute_cls_generators(H_k, eps0, symbols, verify=True):
    """
    Compute the complete generating set of the CLS module via a syzygy/Gröbner
    basis.

    Returns a dict, or None when the algebraic route is inapplicable:
        {
          "generators" : list of (list of LaurentPoly)  # full minimal poly gen set
          "basis"      : list of (list of LaurentPoly)  # M rank-independent CLS
          "rank"       : int   # generic kernel dimension M
          "n_generators": int  # size of the polynomial generating set
          "is_free"    : bool  # n_generators == rank  (free module ⇒ complete)
        }
    """
    Q = H_k.rows
    d = H_k.d

    H_bar = H_k - MatrixPoly.identity(Q, d) * eps0

    # Global per-variable minimum exponent → unit monomial shift to clear
    # negative exponents (multiplying H_bar by a torus unit preserves the
    # solution set).
    shift = [0] * d
    seen = False
    for r in range(Q):
        for c in range(Q):
            for exp in H_bar.data[r][c].coefs.keys():
                seen = True
                for l in range(d):
                    e = exp[l]
                    if abs(e - round(e)) > 1e-9:
                        return None  # Convention II — non-integer exponents
                    shift[l] = min(shift[l], int(round(e)))
    if not seen:
        return None

    # Build the SymPy polynomial matrix over Gaussian rationals.
    rows_expr = []
    for r in range(Q):
        row = []
        for c in range(Q):
            expr, exact = _entry_to_poly_expr(H_bar.data[r][c], symbols, shift)
            if not exact:
                return None
            row.append(expr)
        rows_expr.append(row)

    try:
        R = sympy.QQ_I.old_poly_ring(*symbols)
        F = R.free_module(Q)
        # Columns of H_bar generate the module; syzygies are the CLS solutions.
        cols = [[rows_expr[r][c] for r in range(Q)] for c in range(Q)]
        syz = F.submodule(*cols).syzygy_module()
        raw_gens = list(syz.gens)
    except Exception:
        return None

    # Decode each generator back to a GCD-minimized LaurentPoly vector.
    from cls_finder.cls.reduce import minimize_cls
    generators = []
    for g in raw_gens:
        comps = []
        ok = True
        for e in g:
            try:
                expr = R.to_sympy(e)
            except Exception:
                ok = False
                break
            comps.append(LaurentPoly.from_sympy(sympy.expand(expr), symbols))
        if not ok or all(p.is_zero(1e-12) for p in comps):
            continue
        x_min, _ = minimize_cls(comps, symbols)
        if verify:
            x_mat = MatrixPoly([[p] for p in x_min], d)
            if not (H_bar * x_mat).is_zero(1e-7):
                continue
        generators.append(x_min)

    if not generators:
        return None

    # Rank-reduce at a generic k to extract the M independent CLS (the basis).
    k_gen = np.array([0.137, 0.291, 0.211][:d])  # fractional, generic
    # Convert fractional to Cartesian using identity primitive vectors is fine
    # here because we only need linear independence, not absolute coords.
    def _eval(vec):
        out = []
        for p in vec:
            val = 0j
            for exp, coef in p.coefs.items():
                phase = np.exp(2j * np.pi * sum(k_gen[l] * exp[l] for l in range(d)))
                val += coef * phase
            out.append(val)
        return np.array(out)

    # Order generators the same way gauge selection does: non-singular first,
    # then most compact. A compact-but-singular generator must not pre-empt a
    # non-singular one in the completeness basis.
    try:
        from cls_finder.classify.torus_zeros import has_common_zero_on_torus
        def _sing(g):
            v, _ = has_common_zero_on_torus(g, symbols)
            return bool(v) if v is not None else False
    except Exception:
        def _sing(g):
            return False

    basis, chosen = [], []
    order = sorted(range(len(generators)),
                   key=lambda i: (int(_sing(generators[i])),
                                  sum(len(p.coefs) for p in generators[i])))
    for i in order:
        vec = _eval(generators[i])
        if np.linalg.norm(vec) < 1e-9:
            continue
        trial = chosen + [vec]
        if np.linalg.matrix_rank(np.array(trial), tol=1e-7) <= len(chosen):
            continue
        chosen.append(vec)
        basis.append(generators[i])

    rank = len(basis)
    return {
        "generators": generators,
        "basis": basis,
        "rank": rank,
        "n_generators": len(generators),
        "is_free": len(generators) == rank,
    }
