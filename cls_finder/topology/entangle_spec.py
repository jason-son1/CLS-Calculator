import numpy as np
from cls_finder.classify.chern import reciprocal_vectors
from cls_finder.engineer.hamiltonian import matrixpoly_to_hoppings

def compute_entanglement_spectrum(H_k, lattice, band_indices, N_x=40, n_y=60):
    """
    Computes the Cylinder Entanglement Spectrum (ES) for a given occupied subspace.
    Cylinder is periodic along y (k_y) and open along x.
    
    H_k: MatrixPoly Hamiltonian
    lattice: Lattice object
    band_indices: list of occupied band indices (determines the number of occupied states)
    N_x: number of unit cells in the x direction (cylinder width)
    n_y: number of k_y points to sweep
    """
    Q = H_k.rows
    M = len(band_indices)
    if M == 0 or Q == 0:
        return {"k_y": [], "spectrum": []}

    prim = lattice.primitive_vectors
    B = reciprocal_vectors(prim)
    
    # Extract hoppings from the bulk Hamiltonian MatrixPoly
    hoppings = matrixpoly_to_hoppings(H_k)
    
    # Sweep k_y (fractional coordinates f_y from -0.5 to 0.5)
    f_y_vals = np.linspace(-0.5, 0.5, n_y)
    
    # Region A size (left half)
    N_A = N_x // 2
    dim_A = N_A * Q
    
    # List to store entanglement energies for each k_y
    # Each entry is a list of dim_A floats
    es_data = []
    k_y_list = []
    
    for f_y in f_y_vals:
        # 1. Construct the Cylinder Hamiltonian H_cyl of size (N_x * Q) x (N_x * Q)
        dim_cyl = N_x * Q
        H_cyl = np.zeros((dim_cyl, dim_cyl), dtype=complex)
        
        for x in range(N_x):
            for (alpha, beta, (R_x, R_y)), val in hoppings.items():
                tx = x + R_x
                if 0 <= tx < N_x:
                    row = tx * Q + alpha
                    col = x * Q + beta
                    phase = np.exp(1j * 2.0 * np.pi * f_y * R_y)
                    H_cyl[row, col] += val * phase
                    
        # 2. Diagonalize H_cyl
        w, v = np.linalg.eigh(H_cyl)
        
        # 3. Select occupied states
        # The number of occupied states in the cylinder is N_x * M
        num_occ_cyl = N_x * M
        # Ensure we don't exceed the dimension
        num_occ_cyl = min(num_occ_cyl, dim_cyl)
        
        # Sort by energy and take the lowest num_occ_cyl states
        sort_idx = np.argsort(w)
        occ_vectors = v[:, sort_idx[:num_occ_cyl]]  # (dim_cyl, num_occ_cyl)
        
        # 4. Compute the Correlation Matrix C for Region A
        # C_A is the top-left block of shape (dim_A, dim_A)
        # C_A[i, j] = sum_{n in occ} occ_vectors[i, n] * occ_vectors[j, n].conj()
        # Using matrix multiplication: C_A = occ_vectors_A @ occ_vectors_A.conj().T
        occ_vectors_A = occ_vectors[:dim_A, :]  # (dim_A, num_occ_cyl)
        C_A = occ_vectors_A @ occ_vectors_A.conj().T  # (dim_A, dim_A)
        
        # 5. Diagonalize C_A to get eigenvalues xi
        xi = np.linalg.eigvalsh(C_A)
        
        # 6. Convert xi to entanglement energies epsilon
        # Clip to prevent log(0) or division by zero
        xi_clipped = np.clip(xi, 1e-12, 1.0 - 1e-12)
        epsilon = np.log((1.0 - xi_clipped) / xi_clipped)
        
        # Save results
        es_data.append(epsilon.tolist())
        
        # Convert f_y back to actual k_y coordinate
        k_c_y = (np.array([0, f_y]) @ B)[1]
        k_y_list.append(float(k_c_y))
        
    return {
        "k_y": k_y_list,
        "spectrum": es_data  # list of list of floats, shape (n_y, dim_A)
    }
