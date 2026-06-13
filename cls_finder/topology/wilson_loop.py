import numpy as np
from cls_finder.classify.chern import reciprocal_vectors

def compute_wilson_loop(H_k, lattice, band_indices, n_x=40, n_y=40):
    """
    Computes the Wilson Loop (WCC flow) for a given occupied subspace. Sweep is performed
    along k_x for each k_y.
    
    H_k: MatrixPoly Hamiltonian
    lattice: Lattice object
    band_indices: list of ints, indices of occupied bands
    n_x: discretization points along k_x (path length)
    n_y: discretization points along k_y (sweep parameter)
    """
    Q = H_k.rows
    M = len(band_indices)
    if M == 0:
        return {"wcc": [], "k_y": [], "chern": 0, "z2": 0}

    prim = lattice.primitive_vectors
    B = reciprocal_vectors(prim)
    
    # We sweep k_y from -pi to pi (fractional coordinates -0.5 to 0.5)
    # k_y points
    f_y_vals = np.linspace(-0.5, 0.5, n_y, endpoint=False)
    
    # Store WCCs for each k_y
    # wcc_flow shape: (n_y, M)
    wcc_flow = np.zeros((n_y, M))
    
    # Eigenvectors of Wilson loop for sorting
    prev_evecs = None
    
    # To keep track of continuous paths
    wcc_tracks = [[] for _ in range(M)]
    k_y_list = []

    for y_idx, f_y in enumerate(f_y_vals):
        # Build 1D path in k_x from -0.5 to 0.5 (fractional)
        f_x_vals = np.linspace(-0.5, 0.5, n_x + 1)  # n_x intervals, n_x + 1 points
        
        # Evaluate eigenvectors along the path
        # To ensure PBC: U[n_x] is copied from U[0]
        U = []
        for f_x in f_x_vals[:-1]:
            k_frac = np.array([f_x, f_y])
            k_c = k_frac @ B
            H_val = H_k.evaluate(k_c, prim)
            w, v = np.linalg.eigh(H_val)
            
            # Select occupied eigenvectors
            w_idx = np.argsort(w)
            sel_idx = [w_idx[b] for b in band_indices]
            U.append(v[:, sel_idx])
            
        # Enforce PBC
        U.append(U[0])
        
        # Compute link matrices and enforce Unitarity via SVD
        links = []
        for i in range(n_x):
            M_link = U[i].conj().T @ U[i+1]  # (M, M)
            # SVD: M = U_svd * S * V_dagger
            u_svd, s_svd, vh_svd = np.linalg.svd(M_link)
            # M_unitary = U_svd * V_dagger
            M_unitary = u_svd @ vh_svd
            links.append(M_unitary)
            
        # Wilson loop matrix W = link[n_x-1] @ ... @ link[0]
        W = np.eye(M, dtype=complex)
        for link_mat in links:
            W = link_mat @ W
            
        # Diagonalize W to get eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(W)
        
        # Calculate WCC phase angles in [-0.5, 0.5]
        phases = np.angle(eigenvals) / (2.0 * np.pi)
        
        # Sort current eigenvalues/vectors by phases to start
        sort_idx = np.argsort(phases)
        phases = phases[sort_idx]
        eigenvecs = eigenvecs[:, sort_idx]
        
        # Maximum Overlap Sorting to align with previous step
        if prev_evecs is not None:
            # Calculate overlap matrix O[a, b] = |prev_evecs[a]^\dagger curr_evecs[b]|
            O = np.abs(prev_evecs.conj().T @ eigenvecs)
            
            # Greedy matching
            available_curr = list(range(M))
            matched_idx = []
            for a in range(M):
                best_b = max(available_curr, key=lambda b: O[a, b])
                available_curr.remove(best_b)
                matched_idx.append(best_b)
                
            # Reorder phases and eigenvectors
            phases = phases[matched_idx]
            eigenvecs = eigenvecs[:, matched_idx]
            
        # Save eigenvectors for next step
        prev_evecs = eigenvecs
        
        # Save to tracks
        for a in range(M):
            wcc_tracks[a].append(float(phases[a]))
            
        # Convert f_y back to actual k_y coordinate along the reciprocal vector
        # Using Cartesian k_y value
        k_c_y = (np.array([0, f_y]) @ B)[1]
        k_y_list.append(float(k_c_y))
        
    # Calculate Chern number from WCC winding
    # For each track, check how many times it wraps across the boundary.
    # We unwrap the tracks first to see the total displacement.
    chern = 0.0
    for track in wcc_tracks:
        unwrapped = np.unwrap(2 * np.pi * np.array(track)) / (2 * np.pi)
        diff = unwrapped[-1] - unwrapped[0]
        chern += round(diff)
        
    # Calculate Z2 invariant from reference line crossing if TRS is preserved
    # We can estimate Z2 by drawing a reference line (e.g. at nu = 0.0) and counting crossings
    # For a spin-half Time-Reversal invariant system, WCCs come in pairs.
    # Here we can count crossings of nu = 0.0 by WCC flow for 0 <= k_y <= pi.
    # Since we swept k_y from -pi to pi (fractional -0.5 to 0.5), we can look at the half BZ [0, pi] (fractional [0, 0.5])
    # Let's count crossings of the reference line 0.0
    crossings = 0
    ref_line = 0.0
    half_len = n_y // 2
    # k_y indices corresponding to 0 <= f_y < 0.5
    half_indices = [i for i, f in enumerate(f_y_vals) if 0 <= f < 0.5]
    if len(half_indices) > 1:
        for track in wcc_tracks:
            for idx in range(len(half_indices) - 1):
                i1 = half_indices[idx]
                i2 = half_indices[idx+1]
                v1 = track[i1]
                v2 = track[i2]
                
                # Check for crossing of ref_line (modulo 1)
                # To handle modulo 1 crossing, we check if the sign changes after shift
                d1 = (v1 - ref_line + 0.5) % 1.0 - 0.5
                d2 = (v2 - ref_line + 0.5) % 1.0 - 0.5
                if np.sign(d1) != np.sign(d2) and abs(d1 - d2) < 0.5:
                    crossings += 1
                    
    z2 = crossings % 2
    
    # Return serializable dict
    return {
        "k_y": k_y_list,
        "tracks": wcc_tracks,
        "chern": int(round(chern)),
        "z2": int(z2)
    }
