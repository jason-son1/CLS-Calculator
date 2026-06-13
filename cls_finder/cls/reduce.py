import sympy
from cls_finder.core.laurent import LaurentPoly

def minimize_cls(x_k, symbols):
    """
    Minimize the CLS by dividing out the greatest common divisor (GCD) of its components.
    x_k: list of LaurentPoly (the CLS eigenvector components)
    symbols: list of SymPy symbols representing the variables
    Returns:
      x_k_min: list of LaurentPoly (minimized eigenvector)
      A_0_R_min: dict (real-space amplitudes of minimized CLS)
    """
    g, divs = LaurentPoly.gcd_multiple(x_k, symbols)
    
    # divs is the minimized set of LaurentPolys
    x_k_min = divs
    
    # Center the CLS around (0, ..., 0)
    d = len(symbols)
    all_exps = []
    for poly in x_k_min:
        for exp in poly.coefs.keys():
            all_exps.append(exp)
            
    if all_exps:
        shift = []
        for l in range(d):
            coords_l = [exp[l] for exp in all_exps]
            min_val = min(coords_l)
            max_val = max(coords_l)
            midpoint = (min_val + max_val) / 2.0
            # Use floor(x + 0.5) to round 0.5 up consistently
            import math
            shift.append(int(math.floor(midpoint + 0.5)))
            
        if any(s != 0 for s in shift):
            shifted_x_k = []
            for poly in x_k_min:
                shifted_coefs = {}
                for exp, coef in poly.coefs.items():
                    new_exp = tuple(e - s for e, s in zip(exp, shift))
                    shifted_coefs[new_exp] = coef
                shifted_x_k.append(LaurentPoly(shifted_coefs, d))
            x_k_min = shifted_x_k
            
    # Construct real-space representation
    Q = len(x_k_min)
    A_0_R_min = {}
    for q in range(Q):
        A_0_R_min[q] = {}
        for exp, coef in x_k_min[q].coefs.items():
            A_0_R_min[q][exp] = coef

    return x_k_min, A_0_R_min


def verify_cls_minimal(x_k_min, symbols):
    """
    Verify that a CLS is module-minimal, i.e. cannot be shrunk further.

    For a non-degenerate (M=1) flat band the kernel of H_k - eps0 I over the
    Laurent ring is a rank-1 free module, so its generator is unique up to a
    unit (monomial × scalar): the GCD-reduced vector is then *provably* the
    minimal-support CLS. This routine confirms the witness of that fact —
    that the components are coprime (their polynomial GCD is a unit / monomial).

    Returns (is_minimal: bool, message: str).
    """
    g, _ = LaurentPoly.gcd_multiple(x_k_min, symbols)
    # A trivial GCD is a single monomial (a ring unit on the torus).
    is_unit = (len(g.coefs) == 1)
    if is_unit:
        return True, "최소(primitive) — 성분 GCD가 unit이므로 rank-1 free 가군의 최소 생성원"
    n_terms = len(g.coefs)
    return False, f"비최소 — 성분이 {n_terms}항 공통인자를 공유 (추가 약분 가능)"

