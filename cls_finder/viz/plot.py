import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
from mpl_toolkits.mplot3d import Axes3D
from cls_finder.band.bands import get_reciprocal_vectors

def get_symmetry_path(lattice):
    """
    Get standard high-symmetry path points in fractional coordinates.
    Returns: points (list of array), labels (list of str)
    """
    d = lattice.dimension
    A = np.array(lattice.primitive_vectors, dtype=float)
    if d == 1:
        return [np.array([-0.5]), np.array([0.5])], [r'$-\pi$', r'$\pi$']
    elif d == 2:
        v1 = A[0]
        v2 = A[1]
        l1 = np.linalg.norm(v1)
        l2 = np.linalg.norm(v2)
        ca = np.dot(v1, v2) / (l1 * l2)
        ang = np.degrees(np.arccos(np.clip(ca, -1.0, 1.0)))
        
        is_square = abs(l1 - l2) < 1e-3 and abs(ang - 90.0) < 1.0
        is_hex = (abs(ang - 60.0) < 1.0 or abs(ang - 120.0) < 1.0) and abs(l1 - l2) < 1e-3
        is_rect = not is_square and abs(ang - 90.0) < 1.0
        
        if is_hex:
            return ([np.array([0.0, 0.0]), np.array([0.5, 0.0]),
                     np.array([2.0/3.0, 1.0/3.0]), np.array([0.0, 0.0])],
                    [r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
        elif is_square:
            return ([np.array([0.0, 0.0]), np.array([0.5, 0.0]),
                     np.array([0.5, 0.5]), np.array([0.0, 0.0])],
                    [r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$'])
        elif is_rect:
            return ([np.array([0.0, 0.0]), np.array([0.5, 0.0]),
                     np.array([0.5, 0.5]), np.array([0.0, 0.5]),
                     np.array([0.0, 0.0])],
                    [r'$\Gamma$', r'$X$', r'$S$', r'$Y$', r'$\Gamma$'])
        else:
            return ([np.array([0.0, 0.0]), np.array([0.5, 0.0]),
                     np.array([0.5, 0.5]), np.array([0.0, 0.5]),
                     np.array([0.0, 0.0])],
                    [r'$\Gamma$', r'$X$', r'$M$', r'$Y$', r'$\Gamma$'])
    else:
        # d == 3
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
            return ([np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.0]),
                     np.array([0.5, 0.75, 0.25]), np.array([0.5, 0.5, 0.5]),
                     np.array([0.0, 0.0, 0.0])],
                    [r'$\Gamma$', r'$X$', r'$W$', r'$L$', r'$\Gamma$'])
        elif is_bcc:
            return ([np.array([0.0, 0.0, 0.0]), np.array([0.5, -0.5, 0.5]),
                     np.array([0.25, 0.25, 0.25]), np.array([0.0, 0.0, 0.5]),
                     np.array([0.0, 0.0, 0.0])],
                    [r'$\Gamma$', r'$H$', r'$P$', r'$N$', r'$\Gamma$'])
        else:
            return ([np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]),
                     np.array([0.5, 0.5, 0.0]), np.array([0.5, 0.5, 0.5]),
                     np.array([0.0, 0.0, 0.0])],
                    [r'$\Gamma$', r'$X$', r'$M$', r'$R$', r'$\Gamma$'])

def generate_path_k_points(lattice, points, num_points_per_seg=40):
    """
    Interpolate points along the symmetry path and convert to physical k-points.
    """
    B = get_reciprocal_vectors(lattice)
    frac_path = []
    seg_ticks = [0]
    
    for i in range(len(points) - 1):
        p_start = points[i]
        p_end = points[i + 1]
        for t in np.linspace(0, 1, num_points_per_seg, endpoint=False):
            frac_path.append(p_start + t * (p_end - p_start))
        seg_ticks.append(len(frac_path))
        
    # Append the last point
    frac_path.append(points[-1])
    frac_path = np.array(frac_path)
    
    k_points = frac_path @ B
    return k_points, frac_path, seg_ticks

def plot_bands(H_k, lattice, filepath):
    """
    Plots the band structure along the high-symmetry path.
    """
    Q = H_k.rows
    
    # 1. Generate path
    points, labels = get_symmetry_path(lattice)
    k_points, frac_path, seg_ticks = generate_path_k_points(lattice, points)
    
    # Print reciprocal space info to stdout
    B = get_reciprocal_vectors(lattice)
    print("\n[역격자 및 고대칭점 정보 (Reciprocal Space Info)]")
    print("--------------------------------------------------")
    print("역격자 벡터 (Reciprocal Lattice Vectors B):")
    for idx, b_vec in enumerate(B):
        b_str = ", ".join(f"{x:.6f}" for x in b_vec)
        b_str_2pi = ", ".join(f"{x/(2.0*np.pi):.6f}" for x in b_vec)
        print(f"  b_{idx+1} = [{b_str}]  (in units of 2pi: [{b_str_2pi}])")
        
    print("\n고대칭점 경로 및 좌표 (High Symmetry Points):")
    for p, label in zip(points, labels):
        plain_label = label.replace(r'$\Gamma$', 'Gamma').replace(r'$\pi$', 'pi').replace('$', '').replace('\\', '')
        cart_coord = p @ B
        cart_str = ", ".join(f"{x:.6f}" for x in cart_coord)
        cart_str_2pi = ", ".join(f"{x/(2.0*np.pi):.6f}" for x in cart_coord)
        frac_str = ", ".join(f"{x:.6f}" for x in p)
        print(f"  {plain_label:<5} : Fractional = [{frac_str}] | Cartesian = [{cart_str}] (units of 2pi: [{cart_str_2pi}])")
    print("--------------------------------------------------\n")
    
    # 2. Diagonalize along the path
    energies_path = np.zeros((len(k_points), Q))
    for idx, k in enumerate(k_points):
        H_num = H_k.evaluate(k, lattice.primitive_vectors)
        evals = np.linalg.eigvalsh(H_num)
        energies_path[idx, :] = evals
        
    # 3. Plotting
    plt.figure(figsize=(7, 5))
    x_axis = np.arange(len(k_points))
    
    # Find flat bands along the path to highlight them
    flat_tol = 1e-4
    for n in range(Q):
        band = energies_path[:, n]
        is_flat = np.var(band) < flat_tol
        color = 'red' if is_flat else 'gray'
        lw = 2.5 if is_flat else 1.0
        alpha = 1.0 if is_flat else 0.7
        label = "Flat Band" if is_flat and n == np.argmax([np.var(energies_path[:, i]) < flat_tol for i in range(Q)]) else None
        plt.plot(x_axis, band, color=color, linewidth=lw, alpha=alpha, label=label)
        
    # Add vertical lines at high-symmetry points
    for tick in seg_ticks:
        plt.axvline(x=tick, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        
    plt.xticks(seg_ticks, labels, fontsize=12)
    plt.ylabel('Energy', fontsize=12)
    plt.title('Band Structure (Flat Bands in Red)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

def get_orbital_color_and_style(lattice, q, val):
    """
    Get color and border styling for a CLS amplitude on orbital q.
    Returns: (fill_color, edgecolor, linestyle, linewidth)
    """
    # Predefined set of beautiful, high-contrast base colors (hex)
    SUBLATTICE_COLORS = [
        '#E6194B', # Red
        '#4363D8', # Blue
        '#3CB44B', # Green
        '#FFE119', # Yellow
        '#911EB4', # Purple
        '#46F0F0', # Cyan
        '#F032E6', # Magenta
        '#BCFD4C', # Lime
        '#FABEBE', # Pink
        '#008080', # Teal
        '#E6BEFF', # Lavender
        '#9A6324', # Brown
        '#FFFAC8', # Beige
        '#800000', # Maroon
        '#AAFFC3', # Mint
        '#808000', # Olive
        '#FFD8B1', # Apricot
        '#000075', # Navy
    ]
    
    # Get sublattice index and orbital index within the sublattice
    sub_idx = 0
    sub_orb_idx = 0
    if hasattr(lattice, 'orbitals') and q < len(lattice.orbitals):
        sub_idx = lattice.orbitals[q].get("sublattice_index", 0)
        sub_orb_idx = lattice.orbitals[q].get("orbital_index_in_sublattice", 0)
    else:
        # Fallback if lattice is not as expected
        sub_idx = q
        
    # Choose base color for the sublattice
    base_hex = SUBLATTICE_COLORS[sub_idx % len(SUBLATTICE_COLORS)]
    rgb = mcolors.to_rgb(base_hex)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Adjust hue slightly if there are multiple orbitals on the same sublattice
    if sub_orb_idx > 0:
        h = (h + sub_orb_idx * 0.15) % 1.0
        
    # Adjust lightness and saturation based on the sign/phase of val
    is_real = abs(val.imag) < 1e-4
    
    if is_real:
        if val.real > 0:
            # Positive: vibrant, saturated
            l_new = max(0.2, min(0.65, l))
            s_new = min(1.0, s * 1.1)
            linestyle = '-'
            linewidth = 1.5
            edgecolor = 'black'
        else:
            # Negative: lighter, pastel, dashed/dotted border
            l_new = min(0.9, l + (1.0 - l) * 0.5)
            s_new = s * 0.6
            linestyle = '--'
            linewidth = 1.5
            edgecolor = 'black'
    else:
        # Complex: intermediate lightness, mixed/distinct style
        l_new = min(0.75, l + (1.0 - l) * 0.25)
        s_new = s * 0.8
        linestyle = ':'
        linewidth = 2.0
        edgecolor = '#444444'
        
    rgb_new = colorsys.hls_to_rgb(h, l_new, s_new)
    fill_color = mcolors.to_hex(rgb_new)
    
    return fill_color, edgecolor, linestyle, linewidth

def plot_lattice_cls(lattice, H_k, A_0_R, filepath, plot_range=2):
    """
    Plots the real-space lattice with hopping bonds and the CLS amplitude overlay.
    A_0_R: dict mapping q -> {m_tuple: complex_coef}
    """
    d = lattice.dimension
    Q = lattice.num_orbitals
    
    # Detect Fourier convention
    is_convention_ii = False
    for r in range(H_k.rows):
        for c in range(H_k.cols):
            for exp_tuple in H_k.data[r][c].coefs.keys():
                if any(abs(x - round(x)) > 1e-9 for x in exp_tuple):
                    is_convention_ii = True
                    break
            if is_convention_ii: break
        if is_convention_ii: break
        
    if not is_convention_ii and A_0_R:
        for q in A_0_R:
            for exp in A_0_R[q].keys():
                if any(abs(x - round(x)) > 1e-9 for x in exp):
                    is_convention_ii = True
                    break
            if is_convention_ii: break

    # Extract hoppings from MatrixPoly H_k
    hoppings = []
    for r in range(H_k.rows):
        for c in range(H_k.cols):
            for exp, coef in H_k.data[r][c].coefs.items():
                if abs(coef) > 1e-6:
                    hoppings.append({
                        "i": r,
                        "j": c,
                        "R": exp,
                        "t": coef
                    })
                    
    # Generate cell coordinates to plot
    import itertools
    r_list = list(range(-plot_range, plot_range + 1))
    cell_coords = list(itertools.product(r_list, repeat=d))
    
    # Build coordinates of all sites
    site_positions = {}
    for cell in cell_coords:
        for q in range(Q):
            pos = lattice.get_cartesian_position(cell, q)
            site_positions[(cell, q)] = pos
            
    # Set up matplotlib figure
    fig = plt.figure(figsize=(8, 8))
    
    if d == 1:
        ax = fig.add_subplot(111)
        # Plot sites
        for (cell, q), pos in site_positions.items():
            ax.scatter(pos[0], 0.0, color='gray', s=40, alpha=0.5, zorder=2)
            
        # Draw hoppings
        for cell in cell_coords:
            for hop in hoppings:
                ri = hop["i"]
                ci = hop["j"]
                exp_hop = hop["R"]
                if is_convention_ii:
                    tau_ri = lattice.orbitals[ri]["position"]
                    tau_ci = lattice.orbitals[ci]["position"]
                    ec_raw = tuple(cell[l] + exp_hop[l] - tau_ri[l] + tau_ci[l] for l in range(d))
                    ec = tuple(int(round(x)) for x in ec_raw)
                else:
                    ec = tuple(cell[l] + exp_hop[l] for l in range(d))
                if (cell, ci) in site_positions and (ec, ri) in site_positions:
                    p1 = site_positions[(cell, ci)]
                    p2 = site_positions[(ec, ri)]
                    if np.allclose(p1, p2): continue
                    ax.plot([p1[0], p2[0]], [0.0, 0.0], color='lightgray', linestyle='-', linewidth=1.0, zorder=1)
                    
        # Overlay CLS
        max_amp = 1e-12
        for q in range(Q):
            if q in A_0_R:
                for val in A_0_R[q].values():
                    max_amp = max(max_amp, abs(val))
                    
        for q in range(Q):
            if q not in A_0_R: continue
            tau_q = lattice.orbitals[q]["position"]
            for exp, val in A_0_R[q].items():
                if is_convention_ii:
                    cell = tuple(int(round(x - t)) for x, t in zip(exp, tau_q))
                else:
                    cell = tuple(int(round(x)) for x in exp)
                if (cell, q) in site_positions:
                    pos = site_positions[(cell, q)]
                    size = (abs(val) / max_amp) * 300 + 50
                    color, edgecolor, linestyle, linewidth = get_orbital_color_and_style(lattice, q, val)
                    ax.scatter(pos[0], 0.0, facecolor=color, s=size, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, zorder=3)
                    ax.text(pos[0], 0.1, f"{val.real:.2f}", ha='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
                    
        ax.set_ylim(-1, 1)
        ax.set_xlabel('x', fontsize=12)
        ax.get_yaxis().set_visible(False)
        
    elif d == 2:
        ax = fig.add_subplot(111)
        # Plot sites
        for (cell, q), pos in site_positions.items():
            ax.scatter(pos[0], pos[1], color='gray', s=40, alpha=0.5, zorder=2)
            
        # Draw hoppings
        for cell in cell_coords:
            for hop in hoppings:
                ri = hop["i"]
                ci = hop["j"]
                exp_hop = hop["R"]
                if is_convention_ii:
                    tau_ri = lattice.orbitals[ri]["position"]
                    tau_ci = lattice.orbitals[ci]["position"]
                    ec_raw = tuple(cell[l] + exp_hop[l] - tau_ri[l] + tau_ci[l] for l in range(d))
                    ec = tuple(int(round(x)) for x in ec_raw)
                else:
                    ec = tuple(cell[l] + exp_hop[l] for l in range(d))
                if (cell, ci) in site_positions and (ec, ri) in site_positions:
                    p1 = site_positions[(cell, ci)]
                    p2 = site_positions[(ec, ri)]
                    if np.allclose(p1, p2): continue
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='lightgray', linestyle='-', linewidth=1.0, zorder=1)
                    
        # Overlay CLS
        max_amp = 1e-12
        for q in range(Q):
            if q in A_0_R:
                for val in A_0_R[q].values():
                    max_amp = max(max_amp, abs(val))
                    
        for q in range(Q):
            if q not in A_0_R: continue
            tau_q = lattice.orbitals[q]["position"]
            for exp, val in A_0_R[q].items():
                if is_convention_ii:
                    cell = tuple(int(round(x - t)) for x, t in zip(exp, tau_q))
                else:
                    cell = tuple(int(round(x)) for x in exp)
                if (cell, q) in site_positions:
                    pos = site_positions[(cell, q)]
                    size = (abs(val) / max_amp) * 300 + 50
                    color, edgecolor, linestyle, linewidth = get_orbital_color_and_style(lattice, q, val)
                    ax.scatter(pos[0], pos[1], facecolor=color, s=size, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, zorder=3)
                    label_str = f"{val.real:.2f}" if abs(val.imag) < 1e-4 else f"{val.real:.1f}+{val.imag:.1f}i"
                    ax.text(pos[0], pos[1] + 0.1, label_str, ha='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
                    
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal')
        
    elif d == 3:
        ax = fig.add_subplot(projection='3d')
        # Plot sites
        xs, ys, zs = [], [], []
        for (cell, q), pos in site_positions.items():
            xs.append(pos[0])
            ys.append(pos[1])
            zs.append(pos[2])
        ax.scatter(xs, ys, zs, color='gray', s=20, alpha=0.3, zorder=2)
        
        # Draw hoppings
        for cell in cell_coords:
            for hop in hoppings:
                ri = hop["i"]
                ci = hop["j"]
                exp_hop = hop["R"]
                if is_convention_ii:
                    tau_ri = lattice.orbitals[ri]["position"]
                    tau_ci = lattice.orbitals[ci]["position"]
                    ec_raw = tuple(cell[l] + exp_hop[l] - tau_ri[l] + tau_ci[l] for l in range(d))
                    ec = tuple(int(round(x)) for x in ec_raw)
                else:
                    ec = tuple(cell[l] + exp_hop[l] for l in range(d))
                if (cell, ci) in site_positions and (ec, ri) in site_positions:
                    p1 = site_positions[(cell, ci)]
                    p2 = site_positions[(ec, ri)]
                    if np.allclose(p1, p2): continue
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
                    
        # Overlay CLS
        max_amp = 1e-12
        for q in range(Q):
            if q in A_0_R:
                for val in A_0_R[q].values():
                    max_amp = max(max_amp, abs(val))
                    
        for q in range(Q):
            if q not in A_0_R: continue
            tau_q = lattice.orbitals[q]["position"]
            for exp, val in A_0_R[q].items():
                if is_convention_ii:
                    cell = tuple(int(round(x - t)) for x, t in zip(exp, tau_q))
                else:
                    cell = tuple(int(round(x)) for x in exp)
                if (cell, q) in site_positions:
                    pos = site_positions[(cell, q)]
                    size = (abs(val) / max_amp) * 150 + 20
                    color, edgecolor, linestyle, linewidth = get_orbital_color_and_style(lattice, q, val)
                    ax.scatter(pos[0], pos[1], pos[2], facecolor=color, s=size, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle, zorder=3)
                    
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_zlabel('z', fontsize=12)
        
    # Add sublattice/orbital mapping legend
    from matplotlib.patches import Patch
    legend_elements = []
    if A_0_R:
        present_q = sorted(list(A_0_R.keys()))
        for q in present_q:
            if q >= len(lattice.orbitals): continue
            sub_idx = lattice.orbitals[q]["sublattice_index"]
            label = lattice.orbitals[q]["label"]
            col, _, _, _ = get_orbital_color_and_style(lattice, q, 1.0)
            legend_elements.append(Patch(facecolor=col, edgecolor='black', label=f'Orbital {q} ({label}, Sub {sub_idx})'))
            
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', title='Sublattice / Orbital Map', fontsize=9, title_fontsize=10)

    plt.title('Real-Space CLS / NLS Amplitude Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()
