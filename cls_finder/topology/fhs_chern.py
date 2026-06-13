import numpy as np
from cls_finder.classify.chern import reciprocal_vectors, _frac_grid

def compute_fhs_chern(H_k, lattice, band_indices, n_x=24, n_y=24):
    """
    Fukui-Hatsugai-Suzuki Chern number calculation for any occupied band subspace.
    
    H_k: MatrixPoly Hamiltonian
    lattice: Lattice object
    band_indices: list of ints, indices of the bands to include in the occupied subspace
    n_x, n_y: BZ grid dimensions
    """
    Q = H_k.rows
    M = len(band_indices)
    if M == 0:
        return {"C": 0, "C_raw": 0.0, "converged": True}

    # Get reciprocal lattice vectors
    prim = lattice.primitive_vectors
    B = reciprocal_vectors(prim)
    
    # Grid in fractional coordinates
    f1 = np.arange(n_x) / n_x
    f2 = np.arange(n_y) / n_y
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")
    frac = np.stack([F1.ravel(), F2.ravel()], axis=1)  # (n_x*n_y, 2)
    k_cart = frac @ B  # (n_x*n_y, 2)
    
    # Evaluate Hamiltonian on the grid
    H_batch = H_k.evaluate_batch(k_cart, prim)  # (n_x*n_y, Q, Q)
    w, v = np.linalg.eigh(H_batch)  # eigenvalues (ascending), eigenvectors
    
    # Reshape eigenvectors to grid: (n_x, n_y, Q, Q)
    v = v.reshape(n_x, n_y, Q, Q)
    w = w.reshape(n_x, n_y, Q)
    
    # Extract eigenvectors for the occupied bands
    # v shape: (n_x, n_y, Q, occupied_bands)
    U = np.zeros((n_x, n_y, Q, M), dtype=complex)
    for i in range(n_x):
        for j in range(n_y):
            # Sort band indices
            w_idx = np.argsort(w[i, j])
            selected_idx = [w_idx[b] for b in band_indices]
            U[i, j] = v[i, j][:, selected_idx]
            
    # Calculate link variable
    def link(a, b):
        # a, b are (Q, M) matrices
        ov = a.conj().T @ b  # (M, M)
        det = np.linalg.det(ov)
        mag = abs(det)
        return det / mag if mag > 1e-300 else 1.0 + 0j

    total = 0.0
    for i in range(n_x):
        ip = (i + 1) % n_x
        for j in range(n_y):
            jp = (j + 1) % n_y
            u1 = link(U[i, j],   U[ip, j])
            u2 = link(U[ip, j],  U[ip, jp])
            u3 = link(U[ip, jp], U[i, jp])
            u4 = link(U[i, jp],  U[i, j])
            total += np.angle(u1 * u2 * u3 * u4)
            
    C_raw = total / (2.0 * np.pi)
    C = int(round(C_raw))
    
    return {
        "C": C,
        "C_raw": float(C_raw),
        "grid": [n_x, n_y],
        "M": M,
        "converged": bool(abs(C_raw - C) < 0.15)
    }
