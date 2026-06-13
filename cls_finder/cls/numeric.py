import numpy as np
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.band.bands import compute_bz_grid
from cls_finder.eigen.eigenstate import extract_eigenstate_numerical
from cls_finder.core.gpu import idft_matmul

def extract_cls_numeric(H_k, lattice, eps0, grid_size, p=None, cutoff=1e-5, M=None):
    """
    Numerically derives the CLS real-space amplitudes using IDFT.
    """
    Q = H_k.rows
    d = lattice.dimension
    
    # BZ fractional grid for IDFT
    k_points, xi_points = compute_bz_grid(lattice, grid_size)
    
    # Auto-detect degeneracy M of the flat band at eps0 if M is None or M == 1
    if M is None or M == 1:
        test_k = k_points[0]
        H_num = H_k.evaluate(test_k, lattice.primitive_vectors)
        evals = np.linalg.eigvalsh(H_num)
        M_detected = np.sum(np.abs(evals - eps0) < 1e-3)
        M = max(1, int(M_detected))
        
    # 1. Get numerical eigenvectors on BZ grid
    _, evecs_grid = extract_eigenstate_numerical(H_k, lattice, eps0, grid_size, M=M)
    
    # Reshape evecs_grid to (N_total, Q, M)
    N_total = len(k_points)
    evecs_flat = evecs_grid.reshape((N_total, Q, M))
    
    # Try to find a reference analytical minimized CLS to align the phase and norm
    x_k_min = None
    try:
        from cls_finder.cls.analytic import extract_cls_analytic
        from cls_finder.cls.reduce import minimize_cls
        import sympy
        symbols = sympy.symbols(f'x1:{d+1}')
        x_k, _ = extract_cls_analytic(H_k, eps0)
        x_k_min, _ = minimize_cls(x_k, symbols)
    except Exception:
        try:
            from cls_finder.eigen.eigenstate import extract_eigenstate_analytical
            import sympy
            symbols = sympy.symbols(f'x1:{d+1}')
            w_k_list = extract_eigenstate_analytical(H_k, eps0, symbols)
            x_k_min = w_k_list[0]
        except Exception:
            pass
            
    # 3. Construct x(k)
    x_k_num = np.zeros((N_total, Q), dtype=complex)
    
    if x_k_min is not None:
        # Vectorized analytical reference alignment
        # x_ref_all[idx, q] = x_k_min[q].evaluate(k_points[idx])
        x_ref_all = np.stack(
            [poly.evaluate_batch(k_points, lattice.primitive_vectors) for poly in x_k_min],
            axis=1)  # (N_total, Q)

        if M == 1:
            v_all = evecs_flat[:, :, 0]                          # (N_total, Q)
            q_max_all = np.argmax(np.abs(v_all), axis=1)         # (N_total,)
            v_at_qmax = v_all[np.arange(N_total), q_max_all]     # (N_total,)
            x_at_qmax = x_ref_all[np.arange(N_total), q_max_all] # (N_total,)
            valid = np.abs(v_at_qmax) > 1e-12
            phase_factor = np.where(valid, np.exp(1j * np.angle(x_at_qmax / np.where(valid, v_at_qmax, 1.0))), 1.0)
            norm_x = np.linalg.norm(x_ref_all, axis=1)           # (N_total,)
            x_k_num = (norm_x * phase_factor)[:, np.newaxis] * v_all
        else:
            for idx in range(N_total):
                evecs_k = evecs_flat[idx, :, :]
                c = evecs_k.conj().T @ x_ref_all[idx]
                x_k_num[idx, :] = evecs_k @ c
    else:
        # Fallback to the original minor-based guide logic if no analytical reference is available
        alpha_k = np.ones(N_total, dtype=complex)
        if Q > 1:
            if p is None:
                test_k = k_points[0]
                H_bar = H_k - MatrixPoly.identity(Q, d) * eps0
                best_p = 0
                best_det = 0.0
                for test_p in range(Q):
                    A = H_bar.submatrix(test_p, test_p)
                    H_num = A.evaluate(test_k, lattice.primitive_vectors)
                    det_val = np.linalg.det(H_num)
                    if abs(det_val) > best_det:
                        best_det = abs(det_val)
                        best_p = test_p
                p = best_p
                
            H_bar = H_k - MatrixPoly.identity(Q, d) * eps0
            A = H_bar.submatrix(p, p)
            A_batch = A.evaluate_batch(k_points, lattice.primitive_vectors)  # (N, Q-1, Q-1)
            alpha_k = np.linalg.det(A_batch)                                 # (N,)
                
        for idx in range(N_total):
            x_k_num[idx, :] = alpha_k[idx] * evecs_flat[idx, :, 0]
            
    # 4. Perform direct summation IDFT to find real-space amplitudes
    import itertools
    from fractions import Fraction
    import math

    lcm = 1
    if x_k_min is not None:
        denominators = [1]
        for poly in x_k_min:
            for exp in poly.coefs.keys():
                for val in exp:
                    frac = Fraction(str(val)).limit_denominator(100)
                    denominators.append(frac.denominator)
        lcm = denominators[0]
        for den in denominators[1:]:
            lcm = (lcm * den) // math.gcd(lcm, den)
        lcm = min(lcm, 4)  # Cap LCM to prevent overflow of grid points

    m_range = [float(i) / lcm for i in range(-5 * lcm, 5 * lcm + 1)]
    m_tuples = list(itertools.product(m_range, repeat=d))
    
    # Vectorized IDFT: compute all (q, m) amplitudes (GPU-accelerated if CuPy available)
    m_arr_all = np.array(m_tuples, dtype=float)                      # (M_total, d)
    all_vals = idft_matmul(x_k_num, xi_points, m_arr_all)           # (Q, M_total)

    A_0_R_num = {q: {} for q in range(Q)}
    for m_idx, m in enumerate(m_tuples):
        clean_m = tuple(int(round(x)) if abs(x - round(x)) < 1e-9 else float(x) for x in m)
        for q in range(Q):
            val = all_vals[q, m_idx]
            if abs(val) > cutoff:
                A_0_R_num[q][clean_m] = val

    return A_0_R_num

def verify_cls_eigenstate(H_k, lattice, eps0, x_k_min, n_k=16, tol=1e-5):
    """
    Gauge-independent verification that a CLS is a genuine flat-band state:
    checks the residual  ||H(k) x(k) - eps0 x(k)|| / ||x(k)||  on a set of
    generic k-points. Unlike amplitude matching this is well-defined for an
    M-fold degenerate band, where any vector inside the degenerate eigenspace
    is a valid CLS and there is no canonical numerical partner to compare to.

    Returns (success: bool, message: str).
    """
    d = lattice.dimension
    # Generic, non-high-symmetry k-points (irrational offsets) to avoid nodes.
    base = np.array([0.1123, 0.3317, 0.2113])[:d]
    ks = np.array([((base + 0.137 * i) % 1.0) @ (2.0 * np.pi * np.linalg.solve(
        np.array(lattice.primitive_vectors) @ np.array(lattice.primitive_vectors).T,
        np.array(lattice.primitive_vectors)))
        for i in range(n_k)])

    H_batch = H_k.evaluate_batch(ks, lattice.primitive_vectors)            # (K, Q, Q)
    x_batch = np.stack(
        [poly.evaluate_batch(ks, lattice.primitive_vectors) for poly in x_k_min],
        axis=1)                                                            # (K, Q)
    lhs = np.einsum('kij,kj->ki', H_batch, x_batch)
    diff = np.linalg.norm(lhs - eps0 * x_batch, axis=1)
    norm_x = np.linalg.norm(x_batch, axis=1)
    valid = norm_x > 1e-12
    rel = np.where(valid, diff / np.where(valid, norm_x, 1.0), 0.0)
    max_err = float(np.max(rel)) if len(rel) else 0.0
    if max_err < tol:
        return True, f"고유상태 잔차 검증 통과 (최대 상대오차 {max_err:.2e})"
    return False, f"고유상태 잔차 검증 실패 (최대 상대오차 {max_err:.2e})"


def cross_validate_cls(A_analytic, A_numeric, tol=1e-3):
    """
    Compares the analytical and numerical CLS amplitudes,
    handling global gauge/phase scaling.
    """
    Q = len(A_analytic)
    
    # 1. Find a reference site and orbital with non-zero amplitude in both
    ref_q = None
    ref_m = None
    ref_val_a = None
    
    def find_matching_numeric(q, m_target):
        for m_num, val_num in A_numeric[q].items():
            if len(m_num) == len(m_target) and all(abs(k - c) < 1e-9 for k, c in zip(m_num, m_target)):
                return val_num
        return 0.0
    
    for q in range(Q):
        for m, val_a in A_analytic[q].items():
            val_n = find_matching_numeric(q, m)
            if abs(val_a) > 1e-4 and abs(val_n) > 1e-4:
                ref_q = q
                ref_m = m
                ref_val_a = val_a
                break
        if ref_q is not None:
            break
            
    if ref_q is None:
        return False, "No common non-zero amplitude found to establish scale factor."
        
    ref_val_n = find_matching_numeric(ref_q, ref_m)
    # Scaling factor: C * num_val = ana_val -> C = ana_val / num_val
    C = ref_val_a / ref_val_n
    
    # 2. Scale and compare all elements
    max_diff = 0.0
    for q in range(Q):
        # Gather all unique coordinates from both analytic and numeric for orbital q
        all_keys = list(A_analytic[q].keys())
        for m_num in A_numeric[q].keys():
            if not any(len(m_num) == len(k) and all(abs(x - y) < 1e-9 for x, y in zip(m_num, k)) for k in all_keys):
                all_keys.append(m_num)
                
        for m in all_keys:
            val_a = A_analytic[q].get(m, 0.0)
            if m not in A_analytic[q]:
                # Find with tolerance in analytic
                for k, v in A_analytic[q].items():
                    if len(k) == len(m) and all(abs(x - y) < 1e-9 for x, y in zip(k, m)):
                        val_a = v
                        break
            val_n = find_matching_numeric(q, m) * C
            diff = abs(val_a - val_n)
            if diff > max_diff:
                max_diff = diff
                
    if max_diff < tol:
        return True, f"Match successful! Max difference: {max_diff:.4e}"
    else:
        return False, f"Match failed. Max difference: {max_diff:.4e}"
