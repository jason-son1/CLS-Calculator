import pytest
import sympy
import numpy as np
from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.core.lattice import Lattice

def test_laurent_poly_basic():
    # P1 = 2 + 3*X1 - X2^-1
    d = 2
    p1 = LaurentPoly({
        (0, 0): 2.0,
        (1, 0): 3.0,
        (0, -1): -1.0
    }, d)
    
    # P2 = X1^-1 + X2
    p2 = LaurentPoly({
        (-1, 0): 1.0,
        (0, 1): 1.0
    }, d)
    
    # Test Add
    p_add = p1 + p2
    assert p_add.coefs[(0, 0)] == 2.0
    assert p_add.coefs[(1, 0)] == 3.0
    assert p_add.coefs[(0, -1)] == -1.0
    assert p_add.coefs[(-1, 0)] == 1.0
    assert p_add.coefs[(0, 1)] == 1.0
    
    # Test Sub
    p_sub = p1 - p2
    assert p_sub.coefs[(0, 1)] == -1.0
    
    # Test Mul
    # (2 + 3*X1 - X2^-1) * (X1^-1 + X2)
    # = 2*X1^-1 + 2*X2 + 3 + 3*X1*X2 - X1^-1*X2^-1 - 1
    # = 2 + 2*X1^-1 + 2*X2 + 3*X1*X2 - X1^-1*X2^-1
    p_mul = p1 * p2
    assert p_mul.coefs[(0, 0)] == 2.0
    assert p_mul.coefs[(-1, 0)] == 2.0
    assert p_mul.coefs[(0, 1)] == 2.0
    assert p_mul.coefs[(1, 1)] == 3.0
    assert p_mul.coefs[(-1, -1)] == -1.0

def test_sympy_conversion():
    d = 2
    x, y = sympy.symbols('x y')
    p = LaurentPoly({
        (0, 0): 2.0,
        (1, -1): -3.0
    }, d)
    
    expr = p.to_sympy([x, y])
    # Should be 2.0 - 3.0*x/y
    assert sympy.expand(expr - (2.0 - 3.0 * x / y)) == 0
    
    p2 = LaurentPoly.from_sympy(expr, [x, y])
    assert p2.coefs[(0, 0)] == 2.0
    assert p2.coefs[(1, -1)] == -3.0

def test_gcd():
    d = 2
    x, y = sympy.symbols('x y')
    # p1 = x^-1 - 1
    p1 = LaurentPoly({(-1, 0): 1.0, (0, 0): -1.0}, d)
    # p2 = 1 - x
    p2 = LaurentPoly({(0, 0): 1.0, (1, 0): -1.0}, d)
    
    g, divs = LaurentPoly.gcd_multiple([p1, p2], [x, y])
    
    # GCD of x^-1 - 1 and 1 - x should be proportional to 1 - x
    # g = x^-1 - 1
    # div1 = 1, div2 = -x (or similar depending on standard sign)
    assert len(g.coefs) == 2
    assert len(divs[0].coefs) == 1
    assert len(divs[1].coefs) == 1
    
    # Check if g * divs[0] == p1
    assert (g * divs[0] - p1).is_zero()
    assert (g * divs[1] - p2).is_zero()

def test_matrix_poly_fl():
    d = 2
    # A = [[X1, 1], [1, X2]]
    # det(A) = X1*X2 - 1
    # adj(A) = [[X2, -1], [-1, X1]]
    a11 = LaurentPoly({(1, 0): 1.0}, d)
    a12 = LaurentPoly({(0, 0): 1.0}, d)
    a21 = LaurentPoly({(0, 0): 1.0}, d)
    a22 = LaurentPoly({(0, 1): 1.0}, d)
    
    A = MatrixPoly([[a11, a12], [a21, a22]], d)
    det_val, adj_matrix = A.det_and_adjugate()
    
    assert det_val.coefs[(1, 1)] == 1.0
    assert det_val.coefs[(0, 0)] == -1.0
    assert len(det_val.coefs) == 2
    
    assert adj_matrix.data[0][0].coefs[(0, 1)] == 1.0
    assert adj_matrix.data[0][1].coefs[(0, 0)] == -1.0
    assert adj_matrix.data[1][0].coefs[(0, 0)] == -1.0
    assert adj_matrix.data[1][1].coefs[(1, 0)] == 1.0

def test_fractional_exponents_and_gcd():
    d = 2
    x, y = sympy.symbols('x y')
    
    # Test LaurentPoly with float exponents
    p1 = LaurentPoly({(0.5, 0.0): 2.0, (0.0, -0.5): -3.0}, d)
    assert p1.coefs[(0.5, 0.0)] == 2.0
    assert p1.coefs[(0.0, -0.5)] == -3.0
    
    # Test conversion to sympy with fractional powers
    expr = p1.to_sympy([x, y])
    assert expr.has(x**0.5) or expr.has(sympy.sqrt(x))
    
    # Test conversion back from sympy
    p2 = LaurentPoly.from_sympy(expr, [x, y])
    assert p2.coefs[(0.5, 0.0)] == 2.0
    assert p2.coefs[(0.0, -0.5)] == -3.0
    
    # Test GCD of fractional exponent polynomials
    # p_a = x**0.5 - 1
    p_a = LaurentPoly({(0.5, 0.0): 1.0, (0.0, 0.0): -1.0}, d)
    # p_b = 1 - x
    p_b = LaurentPoly({(0.0, 0.0): 1.0, (1.0, 0.0): -1.0}, d)
    
    # Note: 1 - x = (1 - x**0.5)*(1 + x**0.5)
    # So GCD should be proportional to x**0.5 - 1
    g, divs = LaurentPoly.gcd_multiple([p_a, p_b], [x, y])
    
    assert (g * divs[0] - p_a).is_zero()
    assert (g * divs[1] - p_b).is_zero()
