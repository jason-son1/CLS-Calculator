import json
import re
import numpy as np
import sympy
from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.core.lattice import Lattice

def parse_input(spec):
    """
    Parses a specification dictionary or JSON string.
    spec: dict or str
    Returns: lattice (Lattice), H_k (MatrixPoly)
    """
    if isinstance(spec, str):
        spec = json.loads(spec)
        
    # 1. Parse Lattice
    lat_spec = spec["lattice"] if "lattice" in spec else spec
    dim = int(lat_spec["dimension"])
    primitive_vectors = lat_spec["primitive_vectors"]
    orbitals = lat_spec["orbitals"]
    
    lattice = Lattice(dim, primitive_vectors, orbitals)
    Q = lattice.num_orbitals
    
    # 2. Parse Options (optional)
    options = spec.get("options", {})
    
    # 3. Parse Parameters (optional)
    params = spec.get("parameters", {})
    param_symbols = {}
    for pname, pval in params.items():
        if isinstance(pval, (int, float, complex)):
            param_symbols[sympy.Symbol(pname)] = complex(pval)
        else:
            try:
                expr_val = sympy.sympify(str(pval), locals={'I': sympy.I, 'pi': sympy.pi, 'sqrt': sympy.sqrt, 'exp': sympy.exp, 'cos': sympy.cos, 'sin': sympy.sin})
                param_symbols[sympy.Symbol(pname)] = complex(expr_val.evalf())
            except Exception:
                param_symbols[sympy.Symbol(pname)] = complex(pval)
    
    # 4. Parse Hamiltonian
    if "hoppings" in spec:
        # Mode A: Hopping list
        # H_ij = sum_R t_ij(R) exp(i * k . R)
        # Note: in real space, if we have hopping t from j to i with cell displacement R,
        # it is t * c_i,R^\dagger c_j,0. The Fourier transform is t * exp(i * k . R).
        # We construct MatrixPoly with d = dim
        data = [[LaurentPoly.zero(dim) for _ in range(Q)] for _ in range(Q)]
        
        for hop in spec["hoppings"]:
            i = int(hop["i"])
            j = int(hop["j"])
            R_vec = tuple(int(x) for x in hop["R"])
            t_val = hop["t"]
            # Support string expressions with parameters
            if isinstance(t_val, str):
                kx, ky, kz = sympy.symbols('kx ky kz')
                t_expr = sympy.sympify(t_val, locals={'I': sympy.I, 'pi': sympy.pi, 'sqrt': sympy.sqrt, 'exp': sympy.exp, 'cos': sympy.cos, 'sin': sympy.sin})
                if param_symbols:
                    t_expr = t_expr.subs(param_symbols)
                t_val = complex(t_expr.evalf())
            elif isinstance(t_val, dict):
                t_val = complex(t_val.get("re", 0.0), t_val.get("im", 0.0))
            else:
                t_val = complex(t_val)
                
            # Add to H_ij
            # exp(i * k . R) corresponds to X^R
            data[i][j] = data[i][j] + LaurentPoly.monomial(R_vec, t_val)
            
        H_k = MatrixPoly(data, dim)
        
        # Verify Hermiticity: H_ji == H_ij^\dagger
        # We check if H - H^\dagger is zero
        H_diff = H_k - H_k.dagger()
        if not H_diff.is_zero(1e-9):
            raise ValueError("The provided hopping list does not construct a Hermitian Hamiltonian!")
            
    elif "H_symbolic" in spec:
        # Mode B: Symbolic matrix
        # Parse expressions using SymPy and convert to LaurentPoly
        kx, ky, kz = sympy.symbols('kx ky kz')
        x1, x2, x3 = sympy.symbols('x1 x2 x3')
        symbols = [x1, x2, x3][:dim]
        
        raw_H = spec["H_symbolic"]
        if len(raw_H) != Q or len(raw_H[0]) != Q:
            raise ValueError(f"Symbolic Hamiltonian size ({len(raw_H)}x{len(raw_H[0])}) must match number of orbitals ({Q}x{Q})")
            
        data = [[LaurentPoly.zero(dim) for _ in range(Q)] for _ in range(Q)]
        for r in range(Q):
            for c in range(Q):
                expr_str = raw_H[r][c]
                # Pre-process: add implicit multiplication between digit and letter
                preprocessed = expr_str.strip()
                preprocessed = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', preprocessed)
                preprocessed = re.sub(r'(\))(\w)', r'\1*\2', preprocessed)

                try:
                    expr = sympy.sympify(preprocessed, locals={
                        'I': sympy.I, 'pi': sympy.pi,
                        'sqrt': sympy.sqrt, 'exp': sympy.exp,
                        'cos': sympy.cos, 'sin': sympy.sin,
                        'tan': sympy.tan, 'abs': sympy.Abs
                    })
                except (sympy.SympifyError, SyntaxError, TypeError):
                    expr = sympy.sympify(expr_str)
                # Substitute user-defined parameters
                if param_symbols:
                    expr = expr.subs(param_symbols)
                # rewrite trig to exp
                expr = expr.rewrite(sympy.exp)
                # expand power of exp
                expr = sympy.expand(expr)
                
                # substitute using a helper that handles fractional exponentials of k
                def map_exp(arg):
                    if not arg.has(kx) and not arg.has(ky) and not arg.has(kz):
                        return None
                    expr_no_i = sympy.expand(arg / sympy.I)
                    cx = expr_no_i.coeff(kx)
                    cy = expr_no_i.coeff(ky)
                    cz = expr_no_i.coeff(kz)
                    rem = sympy.simplify(expr_no_i - (cx*kx + cy*ky + cz*kz))
                    if rem != 0:
                        return None
                    term = 1
                    if dim >= 1 and cx != 0:
                        term *= x1**cx
                    if dim >= 2 and cy != 0:
                        term *= x2**cy
                    if dim >= 3 and cz != 0:
                        term *= x3**cz
                    return term

                expr = expr.replace(sympy.exp, lambda arg: map_exp(arg) if map_exp(arg) is not None else sympy.exp(arg))
                # Parse to LaurentPoly
                data[r][c] = LaurentPoly.from_sympy(expr, symbols)
                
        H_k = MatrixPoly(data, dim)
        
        # Verify Hermiticity
        H_diff = H_k - H_k.dagger()
        if not H_diff.is_zero(1e-9):
            raise ValueError("The provided symbolic Hamiltonian is not Hermitian!")
            
    else:
        raise ValueError("Specification must contain either 'hoppings' or 'H_symbolic'")
        
    return lattice, H_k
