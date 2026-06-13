import numpy as np
import sympy
from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.cls.analytic import extract_cls_analytic

def build_noncontractible(H_k, lattice, eps0, k0):
    """
    Construct the non-contractible states (NLS in 2D, NPS in 3D) associated with singularity k0.
    Returns:
      A list of dicts, one for each keep_axis.
      Each dict represents the real-space amplitudes of the state evaluated on a finite range:
      {
         "keep_axis": int,
         "amplitudes": {q: {(m_1, ..., m_d): complex}}
      }
    """
    d = lattice.dimension
    Q = H_k.rows
    A_prim = lattice.primitive_vectors
    
    # 1. Convert k0 to fractional coordinates xi0
    # xi0 = (k0 @ A.T) / (2 * pi)
    xi0 = (k0 @ A_prim.T) / (2.0 * np.pi)
    
    states = []
    
    # Dummy symbol for 1D GCD
    dummy_symbol = sympy.Symbol('y')
    
    is_convention_ii = False
    for r in range(H_k.rows):
        for c in range(H_k.cols):
            for exp in H_k.data[r][c].coefs.keys():
                if any(abs(x - round(x)) > 1e-9 for x in exp):
                    is_convention_ii = True
                    break
            if is_convention_ii: break
        if is_convention_ii: break

    # 2. For each keep_axis, slice the matrix and solve the 1D problem
    for keep_axis in range(d):
        # Fix all axes other than keep_axis
        fixed_coords = {l: xi0[l] for l in range(d) if l != keep_axis}
        
        # Slice H_k to H_1D
        H_1D = H_k.slice_matrix(keep_axis, fixed_coords)
        
        # Find 1D CLS
        x_1D, _ = extract_cls_analytic(H_1D, eps0)
        
        # Minimize 1D CLS
        g_1D, divs_1D = LaurentPoly.gcd_multiple(x_1D, [dummy_symbol])
        
        # Extract 1D real-space amplitudes: dict mapping cell_coord (tuple of length 1) -> coef
        # coefs are of shape { (m_keep,): coef }
        
        # 3. Extend to d-dimensional real space on a finite range
        # We scan other axes in [-5, 5]
        import itertools
        other_axes = [l for l in range(d) if l != keep_axis]
        other_ranges = [list(range(-5, 6)) for _ in other_axes]
        other_tuples = list(itertools.product(*other_ranges))
        
        state_amps = {q: {} for q in range(Q)}
        
        for q in range(Q):
            tau_q = lattice.orbitals[q]["position"]
            for exp_1D, coef_1D in divs_1D[q].coefs.items():
                m_keep = exp_1D[0]
                
                # Combine with other axes
                for other_val in other_tuples:
                    # Construct full m tuple
                    m = [0] * d
                    m[keep_axis] = m_keep
                    for l_idx, val in zip(other_axes, other_val):
                        m[l_idx] = val + tau_q[l_idx] if is_convention_ii else val
                    m = tuple(m)
                    
                    # Compute phase: exp(2 * pi * i * sum_{l!=keep} m_l * xi_0_l)
                    phase = 1.0
                    for l_idx, val in zip(other_axes, other_val):
                        phase *= np.exp(2j * np.pi * val * xi0[l_idx])
                        
                    val_full = coef_1D * phase
                    if abs(val_full) > 1e-5:
                        state_amps[q][m] = val_full
                        
        states.append({
            "keep_axis": keep_axis,
            "amplitudes": state_amps
        })
        
    return states
