"""
Analytical detection of common zeros of a CLS eigenvector on the Brillouin
torus  |X_l| = 1.

A flat band is *singular* iff the (GCD-minimized, hence component-coprime)
eigenvector x_k has a common zero k0 in the BZ — equivalently, the Laurent
components share a point (X_1,...,X_d) with every |X_l| = 1 at which they all
vanish (the "forced node" α_k(k0) = 0 of §3.2 / §5).

Grid sampling can miss a node that falls between grid points. Here we instead
eliminate variables with resultants and test the candidate roots for unit
modulus, which is exact for the integer/half-integer Hamiltonians that occur
in practice. Falls back to None (=unknown) for d >= 3 or when the coefficients
cannot be made exact, so the caller can revert to grid sampling.
"""
import numpy as np
import sympy


def _laurent_to_ordinary_sympy(poly, symbols, tol=1e-6):
    """
    Convert a LaurentPoly to an ordinary SymPy polynomial by multiplying out
    negative exponents (a unit on the torus, so zeros on |X_l|=1 are preserved)
    and snapping coefficients to exact rationals/Gaussian-rationals.

    Returns (sympy_expr, exact: bool). `exact` is False if any coefficient or
    exponent could not be represented exactly within `tol`.
    """
    d = len(symbols)
    if not poly.coefs:
        return sympy.Integer(0), True

    exact = True
    # Shift so every exponent is a non-negative integer.
    min_exp = [0] * d
    for exp in poly.coefs.keys():
        for l in range(d):
            e = exp[l]
            if abs(e - round(e)) > 1e-9:
                exact = False  # non-integer exponent (Convention II)
            min_exp[l] = min(min_exp[l], e)

    expr = sympy.Integer(0)
    for exp, coef in poly.coefs.items():
        c = complex(coef)
        re = sympy.nsimplify(c.real, rational=True)
        im = sympy.nsimplify(c.imag, rational=True)
        if abs(float(re) - c.real) > tol or abs(float(im) - c.imag) > tol:
            exact = False
        c_sym = re + im * sympy.I
        term = c_sym
        for l, s in enumerate(symbols):
            p = int(round(exp[l] - min_exp[l]))
            term *= s ** p
        expr += term
    return sympy.expand(expr), exact


def _roots_on_unit_circle(expr, var, tol=1e-6):
    """Numerical roots of a univariate SymPy expr that lie on |z| = 1."""
    poly = sympy.Poly(expr, var)
    if poly.degree() <= 0:
        return []
    try:
        roots = np.roots([complex(c) for c in poly.all_coeffs()])
    except Exception:
        return []
    return [r for r in roots if abs(abs(r) - 1.0) < 1e-4]


# SymPy resultant cost explodes with polynomial size; above this total term
# count the exact test is abandoned (→ grid fallback) to avoid a near-hang.
_MAX_RESULTANT_TERMS = 300


def has_common_zero_on_torus(polys, symbols, tol=1e-6):
    """
    Returns (is_singular: bool, k0_candidates: list) or (None, []) when the test
    is not applicable (d >= 3, inexact coefficients, or polynomials too large for
    an exact resultant) so the caller can fall back to grid sampling.

    k0_candidates are (X_1, ..., X_d) tuples of complex unit-modulus values.
    """
    d = len(symbols)
    nz = [p for p in polys if p.coefs]
    # A component that is a nonzero constant can never vanish → nonsingular.
    for p in nz:
        if len(p.coefs) == 1 and all(e == 0 for e in next(iter(p.coefs))):
            return False, []
    if len(nz) < 2 or d >= 3:
        return None, []
    # Guard against resultant blow-up on large (non-minimal) eigenvectors.
    if sum(len(p.coefs) for p in nz) > _MAX_RESULTANT_TERMS:
        return None, []

    sp, exact = [], True
    for p in nz:
        e, ok = _laurent_to_ordinary_sympy(p, symbols, tol)
        exact = exact and ok
        sp.append(e)
    if not exact:
        return None, []

    if d == 1:
        x = symbols[0]
        g = sp[0]
        for e in sp[1:]:
            g = sympy.gcd(g, e)
        if g == 0 or not g.free_symbols:
            return False, []
        roots = _roots_on_unit_circle(g, x)
        return (len(roots) > 0, [(complex(r),) for r in roots])

    # d == 2 : eliminate x1 via resultant, test candidate x2 roots, back-substitute.
    x1, x2 = symbols[0], symbols[1]
    P0, P1 = sp[0], sp[1]
    try:
        res = sympy.resultant(sympy.Poly(P0, x1), sympy.Poly(P1, x1))
    except Exception:
        return None, []

    res = sympy.expand(res)
    if res == 0:
        # P0, P1 share a common factor (a curve of zeros). Check whether that
        # curve meets the torus by intersecting the gcd with |x2|=1 sample roots.
        g = sympy.gcd(P0, P1)
        cand_x2 = _roots_on_unit_circle(g, x2) if g.has(x2) else \
                  _roots_on_unit_circle(g, x1)
        return (len(cand_x2) > 0, [])

    cand_x2 = _roots_on_unit_circle(res, x2)
    found = []
    for z2 in cand_x2:
        # Back-substitute and require ALL components to vanish at a shared x1.
        subs_polys = []
        ok = True
        for e in sp:
            es = e.subs(x2, complex(z2))
            es = sympy.expand(es)
            if es == 0:
                continue
            if not es.has(x1):
                ok = False  # nonzero constant ⇒ no common zero at this z2
                break
            subs_polys.append(es)
        if not ok:
            continue
        if not subs_polys:
            # every component vanished identically ⇒ common zero for any x1 on torus
            found.append((1.0 + 0j, complex(z2)))
            continue
        g1 = subs_polys[0]
        for e in subs_polys[1:]:
            g1 = sympy.gcd(g1, e)
        for z1 in _roots_on_unit_circle(g1, x1):
            found.append((complex(z1), complex(z2)))

    return (len(found) > 0, found)
