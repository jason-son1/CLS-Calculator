import numpy as np
import sympy
from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.band.bands import compute_bz_grid
from cls_finder.core.gpu import eigh_batch

def extract_eigenstate_analytical(H_k, eps0, symbols):
    """
    Finds the nullspace of H_k - eps0 * I symbolically.
    Returns: list of list of LaurentPoly (list of eigenvectors)
    """
    Q = H_k.rows
    d = H_k.d
    
    # 1. H_bar = H_k - eps0 * I
    H_bar = H_k - MatrixPoly.identity(Q, d) * eps0
    
    # Fast path: Adjugate Method (works for M=1 flat bands, division-free and avoids SymPy)
    try:
        # Cooperative term/time budget (mirrors analytic.GATES) so a large/dense
        # Hamiltonian aborts to the SymPy path instead of hanging.
        det_val, adj_matrix = H_bar.det_and_adjugate(max_terms=12000, max_seconds=8.0,
                                                      max_ops=2_000_000)
        for col_idx in range(Q):
            col_vec = [adj_matrix.data[r][col_idx] for r in range(Q)]
            # Check if column vector is non-zero
            is_zero_col = True
            for poly in col_vec:
                if not poly.is_zero(1e-12):
                    is_zero_col = False
                    break
            if not is_zero_col:
                total_col_terms = sum(len(p.coefs) for p in col_vec)
                if total_col_terms > 250:
                    # Large poly: H*adj = det*I = 0 by construction — skip the
                    # expensive SymPy GCD reduction/verification (it explodes for
                    # big inputs). Below this ceiling we still GCD-minimize so the
                    # CLS stays compact.
                    return [[p.clean() for p in col_vec]]
                # Minimize vector using GCD
                g, divs = LaurentPoly.gcd_multiple(col_vec, symbols)
                # Verify that it is indeed a null vector of H_bar: H_bar * divs == 0
                V_poly = MatrixPoly([[val] for val in divs], d)
                check = H_bar * V_poly
                if check.is_zero(1e-9):
                    return [divs]
    except Exception:
        # Fall back if Faddeev-LeVerrier or GCD fails
        pass

    # Fallback path: exact SymPy nullspace. This is exponential and HANGS for
    # large multi-orbital models with irrational (e.g. sqrt(3)) coefficients, so
    # it is size-gated — above the gate the caller (singularity classifier /
    # numerical CLS reference) proceeds without an analytical eigenvector rather
    # than stalling.
    try:
        from cls_finder.cls.analytic import GATES as _GATES
        _max_q = _GATES.get("max_nullspace_Q", 7)
    except Exception:
        _max_q = 7
    if Q > _max_q:
        raise ValueError(
            f"SymPy nullspace skipped: Q={Q} exceeds gate ({_max_q}); "
            f"use numerical extraction for this model.")

    # Convert to SymPy Matrix
    sympy_matrix = sympy.Matrix([[elem.to_sympy(symbols) for elem in row] for row in H_bar.data])
    
    # Solve Nullspace using simplify=sympy.cancel to avoid SymPy's slow general simplify()
    nullspace_vectors = sympy_matrix.nullspace(simplify=sympy.cancel)
    
    if not nullspace_vectors:
        raise ValueError(f"No analytical nullspace found at energy {eps0}")
        
    laurent_vectors = []
    
    for v in nullspace_vectors:
        # v is a Q x 1 Matrix
        # Clear denominators using cancel() (much faster than simplify())
        denominators = []
        for i in range(Q):
            val = sympy.cancel(v[i])
            num, denom = val.as_numer_denom()
            denominators.append(denom)
            
        # Product of unique denominators
        prod_denom = 1
        seen = set()
        for denom in denominators:
            if denom not in seen and denom != 1:
                prod_denom *= denom
                seen.add(denom)
                
        # Multiply vector by the product of denominators to clear fractional terms
        v_cleared = v * prod_denom
        
        # Convert each component to LaurentPoly using cancel/expand
        components = []
        for i in range(Q):
            comp_expr = sympy.expand(sympy.cancel(v_cleared[i]))
            components.append(LaurentPoly.from_sympy(comp_expr, symbols))
            
        # Remove GCD of components to minimize
        g, divs = LaurentPoly.gcd_multiple(components, symbols)
        laurent_vectors.append(divs)
        
    return laurent_vectors

def extract_eigenstate_numerical(H_k, lattice, eps0, grid_size, M=1):
    """
    Extracts the numerical eigenvectors corresponding to eigenvalue close to eps0,
    and applies continuous gauge fixing.
    grid_size: list of int
    M: int (degeneracy of the flat band)
    Returns:
      k_points: array of shape (N_total, spatial_dim)
      eigenvectors: array of shape (grid_size_1, ..., grid_size_d, Q, M)
    """
    d = lattice.dimension
    Q = H_k.rows
    
    k_points, _ = compute_bz_grid(lattice, grid_size)
    N_total = len(k_points)

    # Batch-evaluate and diagonalize all k-points at once
    H_batch = H_k.evaluate_batch(k_points, lattice.primitive_vectors)  # (N, Q, Q)
    evals_batch, evecs_batch = eigh_batch(H_batch)                     # (N, Q), (N, Q, Q)

    # Select M eigenvectors closest to eps0 for each k-point
    sort_idx = np.argsort(np.abs(evals_batch - eps0), axis=1)[:, :M]   # (N, M)
    sort_idx_bc = np.broadcast_to(sort_idx[:, np.newaxis, :], (N_total, Q, M))
    flat_vectors = np.take_along_axis(evecs_batch, sort_idx_bc, axis=2) # (N, Q, M)

    grid_shape = tuple(grid_size) + (Q, M)
    eigenvectors = flat_vectors.reshape(grid_shape)
    
    # Apply continuous gauge alignment
    if M == 1:
        # Align phases for single band
        if d == 1:
            N1 = grid_size[0]
            for i in range(1, N1):
                c = np.dot(np.conj(eigenvectors[i-1, :, 0]), eigenvectors[i, :, 0])
                eigenvectors[i, :, 0] *= np.exp(-1j * np.angle(c))
        elif d == 2:
            N1, N2 = grid_size
            for i in range(1, N1):
                c = np.dot(np.conj(eigenvectors[i-1, 0, :, 0]), eigenvectors[i, 0, :, 0])
                eigenvectors[i, 0, :, 0] *= np.exp(-1j * np.angle(c))
            for i in range(N1):
                for j in range(1, N2):
                    c = np.dot(np.conj(eigenvectors[i, j-1, :, 0]), eigenvectors[i, j, :, 0])
                    eigenvectors[i, j, :, 0] *= np.exp(-1j * np.angle(c))
        elif d == 3:
            N1, N2, N3 = grid_size
            # i-axis at (0,0)
            for i in range(1, N1):
                c = np.dot(np.conj(eigenvectors[i-1, 0, 0, :, 0]), eigenvectors[i, 0, 0, :, 0])
                eigenvectors[i, 0, 0, :, 0] *= np.exp(-1j * np.angle(c))
            # j-axis at (i,0)
            for i in range(N1):
                for j in range(1, N2):
                    c = np.dot(np.conj(eigenvectors[i, j-1, 0, :, 0]), eigenvectors[i, j, 0, :, 0])
                    eigenvectors[i, j, 0, :, 0] *= np.exp(-1j * np.angle(c))
            # k-axis at (i,j)
            for i in range(N1):
                for j in range(N2):
                    for k_idx in range(1, N3):
                        c = np.dot(np.conj(eigenvectors[i, j, k_idx-1, :, 0]), eigenvectors[i, j, k_idx, :, 0])
                        eigenvectors[i, j, k_idx, :, 0] *= np.exp(-1j * np.angle(c))
    else:
        # Degenerate case: gauge fixing for multiple bands is more complex (parallel transport)
        # We can align each column individually as a simple approximation
        if d == 1:
            N1 = grid_size[0]
            for m in range(M):
                for i in range(1, N1):
                    c = np.dot(np.conj(eigenvectors[i-1, :, m]), eigenvectors[i, :, m])
                    eigenvectors[i, :, m] *= np.exp(-1j * np.angle(c))
        elif d == 2:
            N1, N2 = grid_size
            for m in range(M):
                for i in range(1, N1):
                    c = np.dot(np.conj(eigenvectors[i-1, 0, :, m]), eigenvectors[i, 0, :, m])
                    eigenvectors[i, 0, :, m] *= np.exp(-1j * np.angle(c))
                for i in range(N1):
                    for j in range(1, N2):
                        c = np.dot(np.conj(eigenvectors[i, j-1, :, m]), eigenvectors[i, j, :, m])
                        eigenvectors[i, j, :, m] *= np.exp(-1j * np.angle(c))
        elif d == 3:
            N1, N2, N3 = grid_size
            for m in range(M):
                for i in range(1, N1):
                    c = np.dot(np.conj(eigenvectors[i-1, 0, 0, :, m]), eigenvectors[i, 0, 0, :, m])
                    eigenvectors[i, 0, 0, :, m] *= np.exp(-1j * np.angle(c))
                for i in range(N1):
                    for j in range(1, N2):
                        c = np.dot(np.conj(eigenvectors[i, j-1, 0, :, m]), eigenvectors[i, j, 0, :, m])
                        eigenvectors[i, j, 0, :, m] *= np.exp(-1j * np.angle(c))
                for i in range(N1):
                    for j in range(N2):
                        for k_idx in range(1, N3):
                            c = np.dot(np.conj(eigenvectors[i, j, k_idx-1, :, m]), eigenvectors[i, j, k_idx, :, m])
                            eigenvectors[i, j, k_idx, :, m] *= np.exp(-1j * np.angle(c))
                            
    return k_points, eigenvectors
