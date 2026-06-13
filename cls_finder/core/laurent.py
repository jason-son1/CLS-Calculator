import numpy as np
import sympy

class LaurentPoly:
    """
    Represents a multi-variable Laurent polynomial:
    P(X_1, ..., X_d) = sum_{m} c_m X_1^{m_1} ... X_d^{m_d}
    Internal representation: dict mapping tuple[int, ...] (exponents) to complex (coefficients)
    """
    def __init__(self, coefs=None, d=None):
        if coefs is None:
            self.coefs = {}
        else:
            self.coefs = {}
            for k, v in coefs.items():
                if abs(v) > 1e-12:
                    new_key = tuple(int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in k)
                    self.coefs[new_key] = complex(v)
        
        if d is not None:
            self.d = d
        elif self.coefs:
            self.d = len(next(iter(self.coefs.keys())))
        else:
            self.d = 1  # Default to 1D if empty

    @classmethod
    def zero(cls, d):
        return cls({}, d)

    @classmethod
    def constant(cls, val, d):
        if abs(val) < 1e-12:
            return cls.zero(d)
        exponent = tuple([0] * d)
        return cls({exponent: val}, d)

    @classmethod
    def monomial(cls, exponent, val=1.0):
        if abs(val) < 1e-12:
            return cls.zero(len(exponent))
        return cls({tuple(exponent): val}, len(exponent))

    def clean(self, tol=1e-12):
        self.coefs = {k: v for k, v in self.coefs.items() if abs(v) > tol}
        return self

    def is_zero(self, tol=1e-9):
        self.clean(tol)
        return len(self.coefs) == 0

    def __add__(self, other):
        if isinstance(other, (int, float, complex)):
            other = LaurentPoly.constant(other, self.d)
        if self.d != other.d:
            raise ValueError(f"Dimension mismatch: {self.d} vs {other.d}")
        
        new_coefs = self.coefs.copy()
        for exp, coef in other.coefs.items():
            new_coefs[exp] = new_coefs.get(exp, 0.0) + coef
        return LaurentPoly(new_coefs, self.d).clean()

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)):
            other = LaurentPoly.constant(other, self.d)
        if self.d != other.d:
            raise ValueError(f"Dimension mismatch: {self.d} vs {other.d}")
        
        new_coefs = self.coefs.copy()
        for exp, coef in other.coefs.items():
            new_coefs[exp] = new_coefs.get(exp, 0.0) - coef
        return LaurentPoly(new_coefs, self.d).clean()

    def __rsub__(self, other):
        if isinstance(other, (int, float, complex)):
            other = LaurentPoly.constant(other, self.d)
        return other.__sub__(self)

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return LaurentPoly({k: v * other for k, v in self.coefs.items()}, self.d).clean()
        
        if self.d != other.d:
            raise ValueError(f"Dimension mismatch: {self.d} vs {other.d}")
        
        new_coefs = {}
        for exp1, coef1 in self.coefs.items():
            for exp2, coef2 in other.coefs.items():
                new_exp = tuple(e1 + e2 for e1, e2 in zip(exp1, exp2))
                new_coefs[new_exp] = new_coefs.get(new_exp, 0.0) + coef1 * coef2
        return LaurentPoly(new_coefs, self.d).clean()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)):
            if abs(other) < 1e-12:
                raise ZeroDivisionError("Division by zero in LaurentPoly")
            return LaurentPoly({k: v / other for k, v in self.coefs.items()}, self.d).clean()
        raise TypeError("Division is only supported by scalars")

    def conjugate(self):
        """Returns the Hermitian conjugate P^+(X) = sum_m c_m^* X^{-m}"""
        new_coefs = {tuple(-x for x in exp): np.conj(coef) for exp, coef in self.coefs.items()}
        return LaurentPoly(new_coefs, self.d)

    def evaluate(self, k_val, primitive_vectors):
        """
        Evaluate P(X) at momentum k_val.
        k_val: array-like of shape (d,)
        primitive_vectors: array-like of shape (d, spatial_dim)
        Returns: complex
        """
        val = 0.0
        k_val = np.array(k_val)
        a = np.array(primitive_vectors)

        for exp, coef in self.coefs.items():
            # Real space position R = sum_l m_l a_l
            R = np.zeros(a.shape[1])
            for l, power in enumerate(exp):
                R += power * a[l]
            # Term phase is exp(i * k_val . R)
            val += coef * np.exp(1j * np.dot(k_val, R))
        return val

    def evaluate_batch(self, k_vals, primitive_vectors):
        """Vectorized evaluation at N k-points. k_vals: (N, spatial_dim) → (N,) complex."""
        k_vals = np.asarray(k_vals, dtype=float)
        if not self.coefs:
            return np.zeros(len(k_vals), dtype=complex)
        a = np.asarray(primitive_vectors, dtype=float)          # (d, spatial_dim)
        exps = np.array(list(self.coefs.keys()), dtype=float)   # (T, d)
        coefs_arr = np.array(list(self.coefs.values()))          # (T,) complex
        R = exps @ a                                              # (T, spatial_dim)
        
        # GPU evaluation for large grids (N > 500)
        from cls_finder.core.gpu import USE_GPU
        if USE_GPU and len(k_vals) > 500:
            import cupy as cp
            k_vals_gpu = cp.asarray(k_vals)
            R_gpu = cp.asarray(R)
            coefs_gpu = cp.asarray(coefs_arr)
            phases_gpu = cp.exp(1j * (k_vals_gpu @ R_gpu.T))
            return cp.asnumpy(phases_gpu @ coefs_gpu)
            
        phases = np.exp(1j * (k_vals @ R.T))                    # (N, T)
        return phases @ coefs_arr                                # (N,)

    def to_sympy(self, symbols):
        """Convert to a SymPy expression using the provided symbols list."""
        if len(symbols) != self.d:
            raise ValueError(f"Number of symbols ({len(symbols)}) must match polynomial dimension ({self.d})")
        expr = 0
        for exp, coef in self.coefs.items():
            r = coef.real
            i = coef.imag
            if abs(r) < 1e-12:
                r = 0.0
            if abs(i) < 1e-12:
                i = 0.0
                
            if i == 0.0:
                clean_coef = r
            elif r == 0.0:
                clean_coef = i * 1j
            else:
                clean_coef = r + i * 1j
                
            c_sympy = sympy.nsimplify(clean_coef)
            term = c_sympy
            for s, power in zip(symbols, exp):
                term *= s**power
            expr += term
        return expr

    def slice_poly(self, keep_axis, fixed_fractional_coords):
        """
        Slices the d-dimensional LaurentPoly to a 1D LaurentPoly along keep_axis,
        by fixing other axes at fractional coordinates.
        fixed_fractional_coords: dict mapping axis_index -> xi_val (float)
        """
        new_coefs = {}
        for exp, coef in self.coefs.items():
            phase = 1.0
            for axis_idx, xi in fixed_fractional_coords.items():
                phase *= np.exp(2j * np.pi * exp[axis_idx] * xi)
            
            keep_power = exp[keep_axis]
            new_exp = (keep_power,)
            new_coefs[new_exp] = new_coefs.get(new_exp, 0.0) + coef * phase
            
        return LaurentPoly(new_coefs, d=1).clean()

    @classmethod
    def from_sympy(cls, expr, symbols):
        """Convert a SymPy expression back to a LaurentPoly."""
        d = len(symbols)
        expr = sympy.expand(expr)
        
        # Helper to check if expression is a monomial in the variables
        def is_monomial_in_symbols(e, syms):
            if not e.free_symbols.intersection(syms):
                return True
            if e in syms:
                return True
            if e.is_Pow:
                return is_monomial_in_symbols(e.base, syms)
            if e.is_Mul:
                return all(is_monomial_in_symbols(arg, syms) for arg in e.args)
            return False

        # Helper to get exponent of symbol in term
        def get_exponent_of_symbol(term_expr, s):
            if term_expr == s:
                return 1
            elif term_expr.is_Pow and term_expr.base == s:
                try:
                    val = float(term_expr.exp)
                    return val if np.isfinite(val) else 0
                except (ValueError, TypeError):
                    return 0
            elif term_expr.is_Mul:
                return sum(get_exponent_of_symbol(arg, s) for arg in term_expr.args)
            else:
                return 0

        coefs = {}
        if expr == sympy.nan or expr == sympy.zoo:
            return cls.zero(d)
            
        terms = expr.args if expr.is_Add else [expr]
        for term in terms:
            # Check for variables in denominator
            numer, denom = term.as_numer_denom()
            if not is_monomial_in_symbols(denom, symbols):
                # Skip terms with complex denominators (not Laurent polynomial terms)
                continue
                
            exponent = []
            has_bad_exp = False
            for s in symbols:
                exp_s = get_exponent_of_symbol(term, s)
                try:
                    exp_val = float(exp_s)
                    if not np.isfinite(exp_val):
                        has_bad_exp = True
                        break
                    exponent.append(exp_val)
                except (ValueError, TypeError):
                    has_bad_exp = True
                    break
            if has_bad_exp:
                continue
                
            exponent = tuple(exponent)
            
            # Evaluate coefficient by substituting all symbols with 1
            c_expr = term.subs({s: 1 for s in symbols})
            if c_expr == sympy.nan or c_expr == sympy.zoo:
                continue
            try:
                c_val = complex(c_expr.evalf())
                if not np.isfinite(c_val.real) or not np.isfinite(c_val.imag):
                    continue
            except Exception:
                continue
            
            coefs[exponent] = coefs.get(exponent, 0.0) + c_val
            
        return cls(coefs, d).clean()

    @staticmethod
    def _real_axis_unit(values, tol=1e-6):
        """
        Largest u > 0 such that every value is (within tol) an integer multiple
        of u, found by a real-number Euclidean algorithm. Handles both rational
        (e.g. 1/2) and irrational-but-commensurate (e.g. multiples of √3/6 from
        hexagonal geometry) exponents uniformly. Returns None if the values are
        not commensurate (no common unit), so the caller can fall back.
        """
        import math
        vs = sorted({abs(float(v)) for v in values if abs(v) > tol})
        if not vs:
            return 1.0
        g = vs[0]
        for v in vs[1:]:
            a, b = g, v
            for _ in range(200):
                if b < tol:
                    break
                a, b = b, a - math.floor(a / b + 1e-9) * b
            g = a if a > tol else g
        if g < tol:
            return None
        for v in vs:
            if abs(round(v / g) - v / g) > 1e-4:
                return None  # not an integer multiple of g → incommensurate
        return g

    @staticmethod
    def gcd_multiple(polys, symbols):
        """
        Compute the GCD of a list of LaurentPolys.
        Returns: g (LaurentPoly), and divided_polys (list of LaurentPolys)
        """
        if not polys:
            return LaurentPoly.constant(1.0, len(symbols)), []
        
        d = len(symbols)

        # Step 1: Per-axis exponent unit u_l (largest real number of which every
        # exponent on axis l is an integer multiple). Real-Euclid handles both
        # rational (1/2) and irrational-but-commensurate (√3/6) exponents, so
        # hexagonal-geometry models reduce correctly instead of staying expanded.
        unit = [1.0] * d
        for l in range(d):
            vals = [exp[l] for poly in polys for exp in poly.coefs.keys()
                    if np.isfinite(exp[l])]
            u = LaurentPoly._real_axis_unit(vals)
            if u is None:
                # Incommensurate exponents on this axis → cannot integer-scale;
                # skip the polynomial GCD (return unreduced) rather than corrupt.
                return LaurentPoly.constant(1.0, d), [poly.clean() for poly in polys]
            unit[l] = u

        # Step 2: Scale all polynomials to integer exponents using the units.
        scaled_polys = []
        for poly in polys:
            scaled_coefs = {}
            for exp, coef in poly.coefs.items():
                scaled_exp = tuple(int(round(exp[l] / unit[l])) for l in range(d))
                scaled_coefs[scaled_exp] = coef
            scaled_polys.append(LaurentPoly(scaled_coefs, d))
            
        # Step 3: Find the minimum power of each symbol across all scaled polynomials
        p_min = [0] * d
        for poly in scaled_polys:
            for exp in poly.coefs.keys():
                for l in range(d):
                    if exp[l] < p_min[l]:
                        p_min[l] = exp[l]
        
        # Step 4: Shift each scaled polynomial to standard polynomial (all powers >= 0)
        shifted_polys = []
        for poly in scaled_polys:
            shifted_coefs = {}
            for exp, coef in poly.coefs.items():
                shifted_exp = tuple(exp[l] - p_min[l] for l in range(d))
                shifted_coefs[shifted_exp] = coef
            shifted_polys.append(LaurentPoly(shifted_coefs, d))

        # Early exit: if any component is a constant after shifting (all exponents zero),
        # GCD must be a scalar → dividing by it changes nothing structurally.
        for sp in shifted_polys:
            if sp.coefs and all(all(e == 0 for e in exp) for exp in sp.coefs.keys()):
                return LaurentPoly.constant(1.0, d), [poly.clean() for poly in polys]

        # Early exit: if all components are monomials (1 term each), GCD = monomial
        # with min exponents. After shifting, min exponent is already 0, so GCD = 1.
        if all(len(sp.coefs) <= 1 for sp in shifted_polys):
            return LaurentPoly.constant(1.0, d), [poly.clean() for poly in polys]

        # Term-count guard. The cheap monomial GCD has already been factored out
        # above (p_min shift); what remains is the *polynomial* GCD via SymPy,
        # whose cost explodes for large inputs (a ~1300-term bivariate GCD takes
        # ~10s). Above this ceiling we skip it and return the vector unreduced —
        # a valid but possibly non-minimal CLS — rather than (apparently) hang.
        # Real CLS are compact (<100 terms); the ceiling leaves headroom while
        # keeping the GCD fast.
        total_terms = sum(len(sp.coefs) for sp in shifted_polys)
        if total_terms > 250:
            return LaurentPoly.constant(1.0, d), [poly.clean() for poly in polys]

        sympy_polys = [sp.to_sympy(symbols) for sp in shifted_polys]

        # Step 5: Compute polynomial GCD in SymPy
        g_sympy = sympy_polys[0]
        for p in sympy_polys[1:]:
            g_sympy = sympy.gcd(g_sympy, p)
            # Early exit: if GCD is already a constant, no further reduction possible
            if not g_sympy.free_symbols:
                return LaurentPoly.constant(1.0, d), [poly.clean() for poly in polys]
            
        if g_sympy == 0 or g_sympy.is_zero:
            return LaurentPoly.zero(d), [LaurentPoly.zero(d) for _ in polys]
            
        # Step 6: Divide shifted polynomials by the GCD
        divided_sympy_polys = []
        for p in sympy_polys:
            try:
                # Use polynomial division to prevent denominators with variables
                q, r = sympy.div(p, g_sympy, *symbols)
                divided_p = q
            except Exception:
                divided_p = sympy.cancel(p / g_sympy)
            divided_sympy_polys.append(divided_p)
            
        # Convert back GCD and rescale exponents by the per-axis units.
        g_shifted = LaurentPoly.from_sympy(g_sympy, symbols)
        # Shift GCD back: multiply by X^p_min
        g_coefs = {}
        for exp, coef in g_shifted.coefs.items():
            original_exp = tuple((exp[l] + p_min[l]) * unit[l] for l in range(d))
            g_coefs[original_exp] = coef
        g = LaurentPoly(g_coefs, d)

        # Convert back divided polynomials and rescale exponents by the units.
        divided_polys = []
        for dp in divided_sympy_polys:
            dp_poly = LaurentPoly.from_sympy(dp, symbols)
            dp_coefs = {}
            for exp, coef in dp_poly.coefs.items():
                original_exp = tuple(exp[l] * unit[l] for l in range(d))
                dp_coefs[original_exp] = coef
            divided_polys.append(LaurentPoly(dp_coefs, d))
        
        return g, divided_polys

    def __repr__(self):
        if self.is_zero():
            return "0"
        terms = []
        for exp, coef in sorted(self.coefs.items()):
            r = coef.real
            i = coef.imag
            if abs(r) < 1e-9:
                r = 0.0
            if abs(i) < 1e-9:
                i = 0.0
                
            if i == 0.0:
                c_str = f"{r:.4g}"
            elif r == 0.0:
                c_str = f"{i:.4g}i"
            else:
                sign = "+" if i > 0 else "-"
                c_str = f"({r:.4g}{sign}{abs(i):.4g}i)"
            # Format variables
            var_parts = []
            for l, p in enumerate(exp):
                if p == 1:
                    var_parts.append(f"X{l+1}")
                elif p != 0:
                    var_parts.append(f"X{l+1}^{p}")
            var_str = "*".join(var_parts)
            if var_str:
                terms.append(f"{c_str}*{var_str}")
            else:
                terms.append(c_str)
        return " + ".join(terms)
