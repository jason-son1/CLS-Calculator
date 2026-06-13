import numpy as np
from cls_finder.core.gpu import eigvalsh_batch

def get_reciprocal_vectors(lattice):
    """
    Compute reciprocal lattice vectors B of shape (d, spatial_dim)
    satisfying B @ A.T = 2 * pi * I_d.
    """
    A = lattice.primitive_vectors  # shape (d, spatial_dim)
    # B = 2 * pi * (A @ A.T)^-1 @ A
    A_AT = A @ A.T
    B = 2.0 * np.pi * np.linalg.solve(A_AT, A)
    return B

def compute_bz_grid(lattice, grid_size):
    """
    Generate a uniform grid of physical k-points and fractional coordinates.
    grid_size: list of int of length lattice.dimension
    Returns:
      k_points: array of shape (N_total, spatial_dim)
      fractional_points: array of shape (N_total, d)
    """
    d = lattice.dimension
    if len(grid_size) != d:
        raise ValueError(f"Grid size length must match lattice dimension: {len(grid_size)} vs {d}")
        
    # Generate fractional coordinates xi_l in [-0.5, 0.5)
    xi_axes = []
    for N in grid_size:
        xi_axes.append(np.linspace(-0.5, 0.5, N, endpoint=False))
        
    # Meshgrid
    xi_grids = np.meshgrid(*xi_axes, indexing='ij')
    # Reshape to (N_total, d)
    fractional_points = np.stack([grid.flatten() for grid in xi_grids], axis=-1)
    
    # Convert to physical k-points: k = xi @ B
    B = get_reciprocal_vectors(lattice)
    k_points = fractional_points @ B
    
    return k_points, fractional_points

def compute_bands(H_k, lattice, grid_size):
    """
    Sample BZ and diagonalize H(k).
    Returns:
      k_points: array of shape (N_total, spatial_dim)
      energies: array of shape (N_total, Q)
    """
    k_points, _ = compute_bz_grid(lattice, grid_size)
    H_batch = H_k.evaluate_batch(k_points, lattice.primitive_vectors)  # (N, Q, Q)
    energies = eigvalsh_batch(H_batch)                                 # (N, Q)
    return k_points, energies

def detect_flat_bands(H_k, lattice, grid_size, flat_tol=1e-5):
    """
    Detect flat bands in the system.
    Returns a list of dicts:
    [
      {
        "band_index": int,
        "energy": float,
        "degenerate_indices": list[int]
      }, ...
    ]
    """
    k_points, energies = compute_bands(H_k, lattice, grid_size)
    Q = H_k.rows
    
    flat_bands = []
    
    # Compute mean and variance for each band
    means = np.mean(energies, axis=0)
    vars_val = np.var(energies, axis=0)
    
    detected_indices = []
    for n in range(Q):
        if vars_val[n] < flat_tol:
            detected_indices.append(n)
            
    # Group degenerate flat bands
    # If the energy difference is small, they are degenerate
    grouped_indices = []
    visited = set()
    
    for n in detected_indices:
        if n in visited:
            continue
        # Find all other flat bands with the same energy
        deg_group = [n]
        visited.add(n)
        for m in detected_indices:
            if m != n and m not in visited:
                if abs(means[n] - means[m]) < 1e-3:
                    deg_group.append(m)
                    visited.add(m)
        grouped_indices.append(deg_group)
        
    for group in grouped_indices:
        for idx in group:
            flat_bands.append({
                "band_index": idx,
                "energy": float(means[idx]),
                "degenerate_indices": group
            })
            
    return flat_bands


def detect_nearly_flat_bands(energies, flat_tol=1e-5, nearly_flat_ratio=0.05):
    """
    Find bands that are NOT truly flat (var >= flat_tol) but have a small
    bandwidth relative to the total spectrum — a signature of flat bands
    whose flatness was broken by a perturbation.

    Parameters
    ----------
    energies         : (N_k, Q) ndarray — pre-computed band energies
    flat_tol         : variance threshold for *true* flat bands (same as
                       detect_flat_bands) — those bands are excluded here
    nearly_flat_ratio: bandwidth_n / total_bandwidth < this → nearly flat
                       default 0.05 means the band spans < 5% of spectrum

    Returns
    -------
    list of dicts: {band_index, energy, bandwidth, flatness_ratio, variance}
    sorted by flatness_ratio ascending (flattest first).
    """
    Q = energies.shape[1]
    vars_val = np.var(energies, axis=0)
    flat_mask = vars_val < flat_tol

    e_min_all = float(energies.min())
    e_max_all = float(energies.max())
    total_bw = e_max_all - e_min_all

    if total_bw < 1e-12:
        return []

    result = []
    for n in range(Q):
        if flat_mask[n]:
            continue  # already a true flat band
        band = energies[:, n]
        bw = float(band.max() - band.min())
        ratio = bw / total_bw
        if ratio < nearly_flat_ratio:
            result.append({
                "band_index": n,
                "energy": float(np.mean(band)),
                "bandwidth": float(bw),
                "flatness_ratio": float(ratio),
                "variance": float(vars_val[n]),
            })

    result.sort(key=lambda d: d["flatness_ratio"])
    return result


def detect_high_symmetry_and_k_path(lattice, H_k=None):
    """
    Automatically analyzes the lattice geometry and optionally the Hamiltonian
    to identify High Symmetry Points (HSP) and recommend a closed k-space path.
    If H_k is provided and has flat band touchings, it dynamically appends
    the touching/singularity points to the path.
    """
    d = lattice.dimension
    A = np.array(lattice.primitive_vectors, dtype=float)
    
    hs_pts = {}
    path_labels = []
    
    if d == 1:
        hs_pts = {"Γ": [0.0], "X": [0.5], "-X": [-0.5]}
        path_labels = ["-X", "Γ", "X"]
        
    elif d == 2:
        # Determine 2D Bravais lattice type
        v1, v2 = A[0], A[1]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        ca = np.dot(v1, v2) / (l1 * l2)
        ang = np.degrees(np.arccos(np.clip(ca, -1.0, 1.0)))
        
        is_square = abs(l1 - l2) < 1e-3 and abs(ang - 90.0) < 1.0
        is_hex = (abs(ang - 60.0) < 1.0 or abs(ang - 120.0) < 1.0) and abs(l1 - l2) < 1e-3
        is_rect = not is_square and abs(ang - 90.0) < 1.0
        
        if is_hex:
            # Hexagonal BZ: Gamma, M, K
            hs_pts = {
                "Γ": [0.0, 0.0],
                "M": [0.5, 0.0],
                "K": [2.0/3.0, 1.0/3.0],
                "K'": [1.0/3.0, 2.0/3.0]
            }
            path_labels = ["Γ", "M", "K", "K'", "Γ"]
        elif is_square:
            # Square BZ: Gamma, X, M
            hs_pts = {
                "Γ": [0.0, 0.0],
                "X": [0.5, 0.0],
                "Y": [0.0, 0.5],
                "M": [0.5, 0.5]
            }
            path_labels = ["Γ", "X", "M", "Γ"]
        elif is_rect:
            # Rectangular BZ: Gamma, X, S, Y
            hs_pts = {
                "Γ": [0.0, 0.0],
                "X": [0.5, 0.0],
                "Y": [0.0, 0.5],
                "S": [0.5, 0.5]
            }
            path_labels = ["Γ", "X", "S", "Y", "Γ"]
        else:
            # Oblique BZ: generic oblique path
            hs_pts = {
                "Γ": [0.0, 0.0],
                "X": [0.5, 0.0],
                "Y": [0.0, 0.5],
                "M": [0.5, 0.5]
            }
            path_labels = ["Γ", "X", "M", "Y", "Γ"]
            
    elif d == 3:
        # Determine 3D Bravais lattice type
        l1 = np.linalg.norm(A[0])
        l2 = np.linalg.norm(A[1])
        l3 = np.linalg.norm(A[2])
        ang_12 = np.degrees(np.arccos(np.clip(np.dot(A[0], A[1]) / (l1 * l2), -1.0, 1.0)))
        ang_23 = np.degrees(np.arccos(np.clip(np.dot(A[1], A[2]) / (l2 * l3), -1.0, 1.0)))
        ang_31 = np.degrees(np.arccos(np.clip(np.dot(A[2], A[0]) / (l3 * l1), -1.0, 1.0)))
        
        is_fcc = (abs(l1 - l2) < 1e-3 and abs(l2 - l3) < 1e-3 and
                  abs(ang_12 - 60.0) < 2.0 and abs(ang_23 - 60.0) < 2.0 and abs(ang_31 - 60.0) < 2.0)
                  
        is_bcc = (abs(l1 - l2) < 1e-3 and abs(l2 - l3) < 1e-3 and
                  abs(ang_12 - 109.47) < 2.0 and abs(ang_23 - 109.47) < 2.0 and abs(ang_31 - 109.47) < 2.0)
                  
        if is_fcc:
            hs_pts = {
                "Γ": [0.0, 0.0, 0.0],
                "X": [0.5, 0.5, 0.0],
                "W": [0.5, 0.75, 0.25],
                "L": [0.5, 0.5, 0.5],
                "U": [0.625, 0.625, 0.25],
                "K": [0.625, 0.625, 0.25]
            }
            path_labels = ["Γ", "X", "W", "L", "Γ"]
        elif is_bcc:
            hs_pts = {
                "Γ": [0.0, 0.0, 0.0],
                "H": [0.5, -0.5, 0.5],
                "P": [0.25, 0.25, 0.25],
                "N": [0.0, 0.0, 0.5],
                "G": [0.0, 0.0, 0.0]
            }
            path_labels = ["Γ", "H", "P", "N", "Γ"]
        else:
            # Simple Cubic or general 3D
            hs_pts = {
                "Γ": [0.0, 0.0, 0.0],
                "X": [0.5, 0.0, 0.0],
                "Y": [0.0, 0.5, 0.0],
                "Z": [0.0, 0.0, 0.5],
                "M": [0.5, 0.5, 0.0],
                "R": [0.5, 0.5, 0.5]
            }
            path_labels = ["Γ", "X", "M", "R", "Γ"]

    # Optional: Automatically scan for flat band touching points (singularities)
    # and append them as high-symmetry points to the recommended path.
    if H_k is not None:
        try:
            # Find flat bands first
            flat_bands = detect_flat_bands(H_k, lattice, [24]*d, flat_tol=1e-4)
            if flat_bands:
                eps0 = flat_bands[0]["energy"]
                # Run a quick singularity check to identify touching points in BZ
                # We can import classify_singularity locally to avoid circular dependencies
                from cls_finder.classify.singularity import classify_singularity
                from cls_finder.eigen.eigenstate import extract_eigenstate_analytical
                import sympy
                
                symbols = [sympy.symbols('x1'), sympy.symbols('x2'), sympy.symbols('x3')][:d]
                w_k_list = extract_eigenstate_analytical(H_k, eps0, symbols)
                if w_k_list:
                    sc = classify_singularity(w_k_list, lattice, H_k, eps0, [24]*d)
                    if sc["singular"] and sc["k0_list"]:
                        # Convert k0 physical coordinates back to fractional coordinates
                        B = get_reciprocal_vectors(lattice)
                        B_inv = np.linalg.pinv(B)
                        
                        # Add up to 3 unique touching points to hs_pts and recommended path
                        added_count = 0
                        for idx, k0 in enumerate(sc["k0_list"]):
                            frac_k0 = k0 @ B_inv
                            # Wrap to [-0.5, 0.5)
                            frac_k0_wrapped = (frac_k0 + 0.5) % 1.0 - 0.5
                            
                            # Give a unique label, e.g., "K0_1"
                            label = f"K0_{added_count + 1}"
                            hs_pts[label] = [float(x) for x in frac_k0_wrapped]
                            
                            # Insert into the k-path before the final Gamma point (or at the end)
                            if len(path_labels) > 1:
                                path_labels.insert(-1, label)
                            else:
                                path_labels.append(label)
                                
                            added_count += 1
                            if added_count >= 3:
                                break
        except Exception:
            # Fail silently to make sure the general auto-detection does not crash
            pass
            
    return hs_pts, path_labels

