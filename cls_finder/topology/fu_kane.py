import numpy as np
from cls_finder.classify.chern import reciprocal_vectors

def compute_fu_kane(H_k, lattice, band_indices, P_matrix=None):
    """
    Computes the Z2 topological invariant (or Chern parity) using the Fu-Kane formula 
    at the 4 TRIM points in 2D, with degenerate subspace projection.
    
    H_k: MatrixPoly Hamiltonian
    lattice: Lattice object
    band_indices: list of occupied band indices
    P_matrix: numpy array of shape (Q, Q) representing the Inversion Parity operator.
              If None, defaults to Identity matrix.
    """
    Q = H_k.rows
    if len(band_indices) == 0 or Q == 0:
        return {"z2": 0, "parity_details": [], "symmetric": True}

    if P_matrix is None:
        P_matrix = np.eye(Q, dtype=complex)
    else:
        P_matrix = np.asarray(P_matrix, dtype=complex)
        
    prim = lattice.primitive_vectors
    B = reciprocal_vectors(prim)
    
    # 1. Define 4 TRIM points in fractional coordinates
    trim_points = {
        "Γ": np.array([0.0, 0.0]),
        "X": np.array([0.5, 0.0]),
        "Y": np.array([0.0, 0.5]),
        "M": np.array([0.5, 0.5])
    }
    
    symmetric = True
    comm_errors = {}
    trim_results = {}
    
    # Check commutation and compute parity eigenvalues for each TRIM point
    for name, frac in trim_points.items():
        k_c = frac @ B
        H_val = H_k.evaluate(k_c, prim)
        
        # Check if [H, P] = 0
        comm = H_val @ P_matrix - P_matrix @ H_val
        comm_norm = np.linalg.norm(comm)
        comm_errors[name] = float(comm_norm)
        
        if comm_norm > 1e-4:
            symmetric = False
            
        # Diagonalize H_val
        w, v = np.linalg.eigh(H_val)
        
        # Select occupied bands
        w_idx = np.argsort(w)
        occ_idx = [w_idx[b] for b in band_indices]
        
        occ_w = w[occ_idx]
        occ_v = v[:, occ_idx]
        
        # Find degenerate energy subspaces within occ_w
        # We group indices by energy proximity (tol = 1e-5)
        groups = []
        visited = set()
        for idx in range(len(occ_idx)):
            if idx in visited:
                continue
            group = [idx]
            for o_idx in range(idx + 1, len(occ_idx)):
                if abs(occ_w[idx] - occ_w[o_idx]) < 1e-5:
                    group.append(o_idx)
            groups.append(group)
            visited.update(group)
            
        # Compute parity eigenvalues for each group
        parity_vals = []
        for group in groups:
            V_sub = occ_v[:, group]  # (Q, D)
            D = len(group)
            
            # Sub-diagonalization of Parity in the degenerate subspace
            P_sub = V_sub.conj().T @ P_matrix @ V_sub  # (D, D)
            p_evals = np.linalg.eigvalsh(P_sub)
            
            # Round parity eigenvalues to nearest +1 or -1
            p_rounded = [int(np.sign(val.real)) if abs(val) > 0.1 else 1 for val in p_evals]
            
            # Check if this group behaves like a Kramers pair
            # e.g., if D is even and parity values are pairwise identical
            if D % 2 == 0:
                p_sorted = sorted(p_rounded)
                is_kramers = True
                for idx in range(0, D, 2):
                    if p_sorted[idx] != p_sorted[idx+1]:
                        is_kramers = False
                        break
                if is_kramers:
                    # Take one from each pair
                    for idx in range(0, D, 2):
                        parity_vals.append(p_sorted[idx])
                else:
                    # Take all
                    parity_vals.extend(p_rounded)
            else:
                # Take all (spinless case)
                parity_vals.extend(p_rounded)
                
        trim_results[name] = {
            "energies": [float(x) for x in occ_w],
            "parity_eigenvalues": parity_vals,
            "product": int(np.prod(parity_vals))
        }
        
    # Calculate Z2 invariant (or Chern parity)
    # (-1)^nu = product of parity products at all 4 TRIM points
    total_prod = 1
    for name, res in trim_results.items():
        total_prod *= res["product"]
        
    z2 = 1 if total_prod == -1 else 0
    
    details = []
    for name, res in trim_results.items():
        details.append({
            "point": name,
            "parity_vals": res["parity_eigenvalues"],
            "product": res["product"],
            "comm_error": comm_errors[name]
        })
        
    return {
        "z2": z2,
        "parity_details": details,
        "symmetric": symmetric,
        "comm_errors": comm_errors
    }
