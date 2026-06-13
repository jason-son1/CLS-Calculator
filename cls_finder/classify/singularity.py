import numpy as np
from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.band.bands import compute_bands, get_reciprocal_vectors
from cls_finder.core.gpu import eigvalsh_batch

def get_high_symmetry_k_points(lattice):
    """
    Generate high-symmetry k-points in the BZ using fractional coordinates.
    Includes standard fractions for rectangular, hexagonal, FCC, and BCC lattices
    across a wider range [-1.0, 1.0] to handle complex BZs robustly.
    """
    d = lattice.dimension
    B = get_reciprocal_vectors(lattice)
    
    # We include standard fractional coordinates for all major lattice types:
    # - 0.0, 0.5, -0.5: standard square/rectangular BZ points
    # - 1/3, -1/3, 2/3, -2/3: hexagonal/triangular BZ points (K, K' points)
    # - 0.25, -0.25, 0.75, -0.75: FCC/BCC BZ points (W, P, etc.)
    # - 1.0, -1.0: boundary limits
    frac_coords = [
        0.0, 0.5, -0.5,
        1.0/3.0, -1.0/3.0, 2.0/3.0, -2.0/3.0,
        0.25, -0.25, 0.75, -0.75,
        1.0, -1.0
    ]
    import itertools
    combinations = list(itertools.product(frac_coords, repeat=d))
    
    k_points = []
    for xi in combinations:
        k_points.append(np.array(xi) @ B)
        
    return k_points

def find_singularity_points(w_k_list, lattice, H_k, eps0, n_scan=80, tol=1e-8, zero_tol=1e-4, max_refine=100):
    """
    Find all singularity points (common zeros or rank-drop points) of the Bloch wave functions.
    w_k_list: list of list of LaurentPoly (M eigenvectors of size Q)
    Returns: list of Cartesian k points (np.arrays)
    """
    if not w_k_list or not w_k_list[0]:
        return []

    d = lattice.dimension
    M = len(w_k_list)
    Q = len(w_k_list[0])
    prim = lattice.primitive_vectors
    B = get_reciprocal_vectors(lattice)
    
    # 1. Generate BZ scan grid
    xi_axes = [np.linspace(0.0, 1.0, n_scan, endpoint=False) for _ in range(d)]
    xi_grids = np.meshgrid(*xi_axes, indexing="ij")
    frac_grid = np.stack([grid.flatten() for grid in xi_grids], axis=-1)  # (P, d)
    k_cart = frac_grid @ B
    
    # Evaluate each of the M vectors on the grid
    V = []
    for j in range(M):
        V.append(np.stack([poly.evaluate_batch(k_cart, prim) for poly in w_k_list[j]], axis=1))  # (P, Q)
        
    # Compute the singularity metric at each grid point
    if M == 1:
        vals = np.sum(np.abs(V[0]) ** 2, axis=1)  # (P,)
    else:
        # W_batch shape: (P, Q, M)
        W_batch = np.stack(V, axis=2)
        # Compute eigenvalues of W^\dagger W, which is M x M
        W_dagger_W = np.einsum('pqi,pqj->pij', W_batch.conjugate(), W_batch)  # (P, M, M)
        eigvals = np.linalg.eigvalsh(W_dagger_W)  # (P, M)
        vals = eigvals[:, 0]  # smallest eigenvalue is square of smallest singular value
        
    scale = vals.max() if vals.size else 1.0
    if scale < 1e-300:
        scale = 1.0
    rel = vals / scale
    
    # 2. Find local minima on the grid
    grid = rel.reshape(*([n_scan] * d))
    cand = []
    # Relaxed candidate threshold to catch minima between grid points
    thr = max(0.3, 20.0 * grid.min())
    
    import itertools
    ranges = [range(n_scan) for _ in range(d)]
    for coords in itertools.product(*ranges):
        v = grid[coords]
        if v > thr:
            continue
            
        # Get 2*d neighbors modulo n_scan
        is_min = True
        for axis in range(d):
            n1_coords = list(coords)
            n1_coords[axis] = (coords[axis] + 1) % n_scan
            n2_coords = list(coords)
            n2_coords[axis] = (coords[axis] - 1) % n_scan
            
            if v > grid[tuple(n1_coords)] + 1e-12 or v > grid[tuple(n2_coords)] + 1e-12:
                is_min = False
                break
        if is_min:
            cand.append((v, coords))
            
    cand.sort(key=lambda t: t[0])
    
    # Convert to fractional coordinates
    cand_frac = [np.array([c / n_scan for c in coords]) for (_, coords) in cand[:max_refine]]
    
    # Add high-symmetry points as extra candidates
    hs_k = get_high_symmetry_k_points(lattice)
    for k in hs_k:
        frac_k = np.linalg.solve(B @ B.T, B @ k)
        cand_frac.append(frac_k)
        
    # Unique fractional candidates
    unique_cand_frac = []
    for fr in cand_frac:
        fr_wrapped = (fr + 0.5) % 1.0 - 0.5
        if not any(np.linalg.norm(fr_wrapped - uc) < 1e-3 for uc in unique_cand_frac):
            unique_cand_frac.append(fr_wrapped)
            
    zeros = []
    
    def _objective(fr):
        kc = np.array(fr) @ B
        W_k = np.zeros((Q, M), dtype=complex)
        for jj in range(M):
            for qq in range(Q):
                W_k[qq, jj] = w_k_list[jj][qq].evaluate(kc, prim)
        if M == 1:
            return float(np.sum(np.abs(W_k) ** 2)) / scale
        else:
            W_dagger_W = W_k.conjugate().T @ W_k
            evs = np.linalg.eigvalsh(W_dagger_W)
            return float(evs[0]) / scale
            
    for fr0 in unique_cand_frac:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, fr0, method="Nelder-Mead",
                           options={"xatol": 1e-9, "fatol": 1e-16, "maxiter": 400})
            fr_opt, val = res.x, res.fun
        except Exception:
            fr_opt, val = fr0, _objective(fr0)
            
        unnorm_val = val * scale
        if unnorm_val > tol:
            continue
            
        # Wrap to symmetric BZ range [-0.5, 0.5)^d
        fr_wrapped = (fr_opt + 0.5) % 1.0 - 0.5
        k_opt = fr_wrapped @ B
        dup = False
        for kz in zeros:
            fr_kz = np.linalg.solve(B @ B.T, B @ kz)
            dphase = ((fr_wrapped - fr_kz) + 0.5) % 1.0 - 0.5
            if np.linalg.norm(dphase @ B) < 1e-3:
                dup = True
                break
        if not dup:
            zeros.append(k_opt)
            
    return zeros

def classify_singularity(w_k_list, lattice, H_k, eps0, grid_size=None, tol=1e-8, touching_tol=1e-2, degenerate_indices=None):
    """
    Classify if the flat band is singular or non-singular.
    w_k_list: list of list of LaurentPoly (M eigenvectors of size Q)
    Returns: dict with "singular" (bool), "k0_list" (list of arrays)
    """
    d = lattice.dimension
    if d == 1:
        return {
            "singular": False,
            "k0_list": []
        }
        
    n_scan = 80
    if grid_size is not None:
        if isinstance(grid_size, int):
            n_scan = grid_size
        elif isinstance(grid_size, (list, tuple)) and len(grid_size) > 0:
            n_scan = grid_size[0]
            
    zeros = find_singularity_points(w_k_list, lattice, H_k, eps0, n_scan=n_scan, tol=tol)
    
    # For single flat bands (M=1), any common zero is a genuine wave function singularity (topological or touching)
    M = len(w_k_list)
    if M == 1:
        return {
            "singular": len(zeros) > 0,
            "k0_list": zeros
        }
        
    # Filter to keep only physical band-touching points (only for degenerate bands M >= 2)
    physical_zeros = []
    for k0 in zeros:
        H_eval = H_k.evaluate(k0, lattice.primitive_vectors)
        evals = np.linalg.eigvalsh(H_eval)
        
        Q = H_k.rows
        if degenerate_indices is not None:
            flat_indices = list(degenerate_indices)
        else:
            flat_indices = [n for n in range(Q) if abs(evals[n] - eps0) < 1e-3]
            
        dispersive_evals = [evals[n] for n in range(Q) if n not in flat_indices]
        if dispersive_evals:
            min_gap = min(abs(ev - eps0) for ev in dispersive_evals)
            if min_gap < touching_tol:
                physical_zeros.append(k0)
                
    return {
        "singular": len(physical_zeros) > 0,
        "k0_list": physical_zeros
    }

