"""
CLS Finder Web Bridge
Runs inside Pyodide. Wraps the analysis pipeline and returns JSON data
for JavaScript to render (Plotly charts instead of matplotlib PNGs).
"""
import json, traceback, itertools
import numpy as np
from cls_finder.core.gpu import eigvalsh_batch

# ─── JSON serializer ──────────────────────────────────────────────────────────
def _ser(obj):
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, np.bool_):        return bool(obj)
    if isinstance(obj, np.integer):      return int(obj)
    if isinstance(obj, np.floating):     return float(obj)
    if isinstance(obj, (np.complexfloating, complex)):
        r, i = float(obj.real), float(obj.imag)
        return r if abs(i) < 1e-12 else {"re": r, "im": i}
    raise TypeError(f"Cannot serialize {type(obj)}")

# ─── Symmetry path helpers (mirror of viz/plot.py, no matplotlib) ─────────────
def _recip(lattice):
    A = lattice.primitive_vectors
    return 2.0 * np.pi * np.linalg.solve(A @ A.T, A)

def _sym_path(lattice, H_k=None):
    from cls_finder.band.bands import detect_high_symmetry_and_k_path
    hs_pts, path_labels = detect_high_symmetry_and_k_path(lattice, H_k)
    pts = [np.array(hs_pts[label]) for label in path_labels]
    return pts, path_labels

def _parse_k_path(k_path_str, lattice, k_points_override=None):
    """
    Parse custom k-path string into coordinates and labels.
    Supported inputs: "Γ - X - M - Γ" or "[0,0] - [0.5,0] - [0.5,0.5] - [0,0]"
    """
    import re
    d = lattice.dimension
    
    # Split path by standard separators (-, ->, →, comma, space) but ignore separators inside brackets
    normalized_str = k_path_str.replace("->", "-").replace("→", "-")
    tokens = []
    current_token = []
    bracket_depth = 0
    for c in normalized_str:
        if c in '[(':
            bracket_depth += 1
            current_token.append(c)
        elif c in '])':
            bracket_depth = max(0, bracket_depth - 1)
            current_token.append(c)
        elif bracket_depth == 0 and c in '-, \t\n':
            tok = "".join(current_token).strip()
            if tok:
                tokens.append(tok)
            current_token = []
        else:
            current_token.append(c)
    tok = "".join(current_token).strip()
    if tok:
        tokens.append(tok)
        
    if not tokens:
        raise ValueError("k-경로가 비어 있습니다.")
        
    A = np.array(lattice.primitive_vectors, dtype=float)
    is_hex = False
    is_fcc = False
    is_bcc = False
    
    if d == 2:
        v1, v2 = A[0], A[1]
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        ca = np.dot(v1, v2) / (l1 * l2)
        ang = np.degrees(np.arccos(np.clip(ca, -1.0, 1.0)))
        is_hex = (abs(ang - 60.0) < 1.0 or abs(ang - 120.0) < 1.0) and abs(l1 - l2) < 1e-3
    elif d == 3:
        l1, l2, l3 = np.linalg.norm(A[0]), np.linalg.norm(A[1]), np.linalg.norm(A[2])
        ang_12 = np.degrees(np.arccos(np.clip(np.dot(A[0], A[1]) / (l1 * l2), -1.0, 1.0)))
        ang_23 = np.degrees(np.arccos(np.clip(np.dot(A[1], A[2]) / (l2 * l3), -1.0, 1.0)))
        ang_31 = np.degrees(np.arccos(np.clip(np.dot(A[2], A[0]) / (l3 * l1), -1.0, 1.0)))
        is_fcc = (abs(l1 - l2) < 1e-3 and abs(l2 - l3) < 1e-3 and
                  abs(ang_12 - 60.0) < 2.0 and abs(ang_23 - 60.0) < 2.0 and abs(ang_31 - 60.0) < 2.0)
        is_bcc = (abs(l1 - l2) < 1e-3 and abs(l2 - l3) < 1e-3 and
                  abs(ang_12 - 109.47) < 2.0 and abs(ang_23 - 109.47) < 2.0 and abs(ang_31 - 109.47) < 2.0)

    # Standard labels database
    standard_pts = {}
    if d == 1:
        standard_pts = {
            "Γ": [0.0], "G": [0.0], "GAMMA": [0.0],
            "X": [0.5], "X+": [0.5], "X-": [-0.5],
            "π": [0.5], "PI": [0.5], "-π": [-0.5], "-PI": [-0.5]
        }
    elif d == 2:
        standard_pts = {
            "Γ": [0.0, 0.0], "G": [0.0, 0.0], "GAMMA": [0.0, 0.0],
            "X": [0.5, 0.0], "Y": [0.0, 0.5],
            "M": [0.5, 0.5], "S": [0.5, 0.5],
            "K": [2.0/3.0, 1.0/3.0] if is_hex else [0.5, 0.5],
            "K'": [1.0/3.0, 2.0/3.0] if is_hex else [0.5, 0.5]
        }
    elif d == 3:
        if is_fcc:
            standard_pts = {
                "Γ": [0.0, 0.0, 0.0], "G": [0.0, 0.0, 0.0], "GAMMA": [0.0, 0.0, 0.0],
                "X": [0.5, 0.5, 0.0], "W": [0.5, 0.75, 0.25], "L": [0.5, 0.5, 0.5]
            }
        elif is_bcc:
            standard_pts = {
                "Γ": [0.0, 0.0, 0.0], "G": [0.0, 0.0, 0.0], "GAMMA": [0.0, 0.0, 0.0],
                "H": [0.5, -0.5, 0.5], "P": [0.25, 0.25, 0.25], "N": [0.0, 0.0, 0.5]
            }
        else:
            standard_pts = {
                "Γ": [0.0, 0.0, 0.0], "G": [0.0, 0.0, 0.0], "GAMMA": [0.0, 0.0, 0.0],
                "X": [0.5, 0.0, 0.0], "Y": [0.0, 0.5, 0.0], "Z": [0.0, 0.0, 0.5],
                "M": [0.5, 0.5, 0.0], "R": [0.5, 0.5, 0.5]
            }
            
    if k_points_override:
        for k, v in k_points_override.items():
            standard_pts[k] = v
            
    pts, labels = [], []
    for tok in tokens:
        coord_match = re.match(r'^[\[\(]([^\]\)]+)[\]\)]$', tok)
        if coord_match:
            try:
                vals = [float(x.strip()) for x in coord_match.group(1).split(',')]
                if len(vals) != d:
                    raise ValueError(f"좌표 차원 불일치: {tok} (차원: {d} 필요)")
                pts.append(np.array(vals))
                labels.append(tok)
            except Exception as e:
                raise ValueError(f"좌표 파싱 실패: {tok} - {str(e)}")
        else:
            key = tok.upper()
            match_key = next((k for k in standard_pts if k.upper() == key), None)
            if match_key:
                pts.append(np.array(standard_pts[match_key]))
                labels.append(tok)
            else:
                if tok == 'Γ' or tok == 'G' or tok == 'g' or tok == 'gamma':
                    pts.append(np.zeros(d))
                    labels.append('Γ')
                else:
                    raise ValueError(f"알 수 없는 고대칭점 또는 형식 오류: '{tok}'")
                    
    return pts, labels

def _gen_path(lattice, pts, n=50):
    B = _recip(lattice)
    frac, ticks = [], [0]
    for i in range(len(pts) - 1):
        p0, p1 = pts[i], pts[i+1]
        for t in np.linspace(0, 1, n, endpoint=False):
            frac.append(p0 + t*(p1-p0))
        ticks.append(len(frac))
    frac.append(pts[-1])
    return np.array(frac) @ B, ticks

# ─── Build Plotly data for CLS lattice visualization ─────────────────────────
def _cls_plot_data(lattice, H_k, A_0_R, plot_range=3):
    d, Q = lattice.dimension, lattice.num_orbitals

    # 1. Detect non-integer exponents.
    # Mode A (hopping list) always has integer exponents.
    # Mode B symbolic input can have Cartesian k-coefficients as exponents,
    # which are non-integer for non-orthogonal lattices.
    is_convention_ii = False
    for r in range(H_k.rows):
        for c in range(H_k.cols):
            for exp in H_k.data[r][c].coefs.keys():
                if any(abs(x - round(x)) > 1e-9 for x in exp):
                    is_convention_ii = True
                    break
            if is_convention_ii:
                break
        if is_convention_ii:
            break

    if not is_convention_ii and A_0_R:
        for q in A_0_R:
            for exp in A_0_R[q].keys():
                if any(abs(x - round(x)) > 1e-9 for x in exp):
                    is_convention_ii = True
                    break
            if is_convention_ii:
                break

    # Precompute lattice inverse for Cartesian ↔ fractional conversion
    A_mat = np.array(lattice.primitive_vectors, dtype=float)   # (d, spatial_dim)
    if is_convention_ii:
        A_inv = (np.linalg.inv(A_mat)
                 if A_mat.shape[0] == A_mat.shape[1]
                 else np.linalg.pinv(A_mat))                   # (spatial_dim, d)

        def _exp_to_cell(exp_val, orb_pos_frac):
            """Mode B Cartesian: exp = site Cartesian position.
            Subtract orbital offset (in Cartesian) then convert to fractional unit cell."""
            tau_cart = np.array(orb_pos_frac, dtype=float) @ A_mat
            r_cart   = np.array(exp_val, dtype=float) - tau_cart
            r_frac   = r_cart @ A_inv
            return tuple(int(round(x)) for x in r_frac)

        def _cell_to_key(cell_int, orb_pos_frac):
            """Reverse: integer cell → expected Cartesian key in A_0_R."""
            tau_cart = np.array(orb_pos_frac, dtype=float) @ A_mat
            return tuple(np.array(cell_int, dtype=float) @ A_mat + tau_cart)

        def _hop_to_ec(cell_int, exp_val, tau_ri_frac, tau_ci_frac):
            """Mode B: hopping exponent → integer target cell."""
            tau_ri_c = np.array(tau_ri_frac, dtype=float) @ A_mat
            tau_ci_c = np.array(tau_ci_frac, dtype=float) @ A_mat
            r_hop    = np.array(exp_val, dtype=float) - tau_ri_c + tau_ci_c
            r_frac   = r_hop @ A_inv
            return tuple(int(round(cell_int[l] + r_frac[l])) for l in range(d))
    else:
        def _exp_to_cell(exp_val, _orb_pos):
            return tuple(int(round(x)) for x in exp_val)

        def _cell_to_key(cell_int, _orb_pos):
            return cell_int

        def _hop_to_ec(cell_int, exp_val, _tau_ri, _tau_ci):
            return tuple(cell_int[l] + int(round(exp_val[l])) for l in range(d))

    # 2. Build Cell Grid
    cls_cells = set()
    # Precompute integer-cell → amplitude map to avoid float-tolerance issues.
    # _exp_to_cell converts any exponent (integer or Cartesian float) to an integer unit-cell
    # tuple, making the lookup exact regardless of floating-point drift in exponents.
    amp_int_map = {}
    if A_0_R:
        for q_idx in A_0_R:
            tau_q = lattice.orbitals[q_idx]["position"]
            amp_int_map[q_idx] = {}
            for exp, coef in A_0_R[q_idx].items():
                cell_int = _exp_to_cell(exp, tau_q)
                amp_int_map[q_idx][cell_int] = coef
                cls_cells.add(cell_int)

    r_max = min(max(plot_range,
                    max((max(abs(c) for c in cell)
                         for cell in cls_cells), default=0) + 1), 5)

    rng = list(range(-r_max, r_max + 1))

    if d == 1:
        cells = [(r,) for r in rng]
    elif d == 2:
        cells = list(itertools.product(rng, repeat=2))
    else:
        rng_3d = list(range(-min(r_max, 2), min(r_max, 2) + 1))
        cells = list(itertools.product(rng_3d, repeat=3))

    # Build sublattice info from Lattice class
    n_sub = lattice.sublattice_count
    is_multi_orbital = lattice.is_multi_orbital

    sites, site_map = [], {}
    for cell in cells:
        for q in range(Q):
            pos = lattice.get_cartesian_position(cell, q)
            # Find amplitude by exact integer cell lookup (robust to float drift)
            amp = None
            if q in amp_int_map:
                amp = amp_int_map[q].get(tuple(cell))

            orb = lattice.orbitals[q]
            sub_idx = orb.get("sublattice_index", q)
            sub = lattice.sublattices[sub_idx] if sub_idx < len(lattice.sublattices) else None
            # Build sublattice label from first orbital in sublattice
            sub_label = ""
            if sub:
                first_orb_in_sub = sub["orbital_indices"][0]
                sub_label = lattice.orbitals[first_orb_in_sub]["label"]

            s = {"cell": list(cell), "orbital": q,
                 "label": orb["label"],
                 "sublattice": sub_idx,
                 "sublattice_label": sub_label,
                 "orbital_in_sublattice": orb.get("orbital_index_in_sublattice", 0),
                 "x": float(pos[0]),
                 "y": float(pos[1]) if len(pos) >= 2 else 0.0,
                 "z": float(pos[2]) if len(pos) >= 3 else 0.0,
                 "amplitude": None, "amp_re": None, "amp_im": None,
                 "is_cls": False,
                 # Integer cell stored for robust amplitude patching from repr candidates
                 "int_cell": list(cell)}
            if amp is not None:
                c = complex(amp)
                s.update(amplitude=float(abs(c)),
                         amp_re=float(c.real), amp_im=float(c.imag),
                         is_cls=True)
            site_map[(tuple(cell), q)] = len(sites)
            sites.append(s)

    # Group overlapping orbital positions (multi-orbital sites)
    # The frontend now renders overlapping orbitals concentrically, so no physical offset is applied.
    sublattice_centers = []
    sublattice_links = []

    bonds, bond_set = [], set()
    for ri in range(H_k.rows):
        for ci in range(H_k.cols):
            for exp, coef in H_k.data[ri][ci].coefs.items():
                if abs(coef) < 1e-8: continue
                tau_ri = lattice.orbitals[ri]["position"]
                tau_ci = lattice.orbitals[ci]["position"]
                
                for cell in cells:
                    ec = _hop_to_ec(cell, exp, tau_ri, tau_ci)
                        
                    idx1 = site_map.get((cell, ci))
                    idx2 = site_map.get((ec, ri))
                    if idx1 is None or idx2 is None: continue
                    if idx1 == idx2: continue
                    key = tuple(sorted([idx1, idx2]))
                    if key in bond_set: continue
                    bond_set.add(key)
                    p1, p2 = sites[idx1], sites[idx2]
                    bonds.append({"x0":p1["x"],"y0":p1["y"],"z0":p1["z"],
                                  "x1":p2["x"],"y1":p2["y"],"z1":p2["z"],
                                  "t": float(abs(coef))})

    # Build sublattice info summary for frontend
    sublattice_info = []
    for sub in lattice.sublattices:
        orb_labels = [lattice.orbitals[oi]["label"] for oi in sub["orbital_indices"]]
        sublattice_info.append({
            "index": sub["index"],
            "position": sub["position"].tolist(),
            "orbital_indices": sub["orbital_indices"],
            "orbital_labels": orb_labels,
            "n_orbitals": len(sub["orbital_indices"]),
        })

    return {
        "sites": sites,
        "bonds": bonds,
        "dimension": d,
        "primitive_vectors": np.array(lattice.primitive_vectors).tolist(),
        "sublattices": sublattice_centers,
        "sublattice_links": sublattice_links,
        "sublattice_info": sublattice_info,
        "sublattice_count": n_sub,
        "is_multi_orbital": is_multi_orbital,
    }

def _bz_data(lattice, k_path_str=None, k_points_override=None, H_k=None):
    d = lattice.dimension
    B = _recip(lattice)
    
    if k_path_str:
        try:
            pts, labels = _parse_k_path(k_path_str, lattice, k_points_override)
        except Exception:
            pts, labels = _sym_path(lattice, H_k)
    else:
        pts, labels = _sym_path(lattice, H_k)
        
    if k_points_override:
        for i, label in enumerate(labels):
            match_key = next((k for k in k_points_override if k.upper() == label.upper()), None)
            if match_key:
                pts[i] = np.array(k_points_override[match_key])
    
    if d == 1:
        val = abs(B[0, 0]) / 2.0
        sym_coords = []
        for p, label in zip(pts, labels):
            coord = p @ B
            sym_coords.append({"label": label, "x": float(coord[0])})
        # Add Gamma
        sym_coords.append({"label": "Γ", "x": 0.0})
        return {
            "dimension": 1,
            "vertices": [[-val], [val]],
            "sym_points": sym_coords,
            "recip_vectors": B.tolist()
        }
        
    elif d == 2:
        b1, b2 = B[0], B[1]
        points = []
        for m in [-2, -1, 0, 1, 2]:
            for n in [-2, -1, 0, 1, 2]:
                points.append(m * b1 + n * b2)
        points = np.array(points)
        origin_idx = 12
        
        try:
            from scipy.spatial import Voronoi
            vor = Voronoi(points)
            region_idx = vor.point_region[origin_idx]
            vert_idxs = [i for i in vor.regions[region_idx] if i >= 0]
            region_vertices = vor.vertices[vert_idxs]
            angles = np.arctan2(region_vertices[:, 1], region_vertices[:, 0])
            sort_idx = np.argsort(angles)
            bz_polygon = region_vertices[sort_idx].tolist()
            bz_polygon.append(bz_polygon[0])
        except Exception:
            bz_polygon = [
                (-0.5*b1 - 0.5*b2).tolist(),
                (0.5*b1 - 0.5*b2).tolist(),
                (0.5*b1 + 0.5*b2).tolist(),
                (-0.5*b1 + 0.5*b2).tolist(),
                (-0.5*b1 - 0.5*b2).tolist()
            ]
            
        sym_coords = []
        for p, label in zip(pts, labels):
            coord = p @ B
            sym_coords.append({"label": label, "x": float(coord[0]), "y": float(coord[1])})
            
        return {
            "dimension": 2,
            "vertices": bz_polygon,
            "sym_points": sym_coords,
            "recip_vectors": B.tolist()
        }
        
    else: # d == 3
        b1, b2, b3 = B[0], B[1], B[2]
        points = []
        origin_idx = 13
        idx = 0
        for m in [-1, 0, 1]:
            for n in [-1, 0, 1]:
                for o in [-1, 0, 1]:
                    points.append(m*b1 + n*b2 + o*b3)
                    if m == 0 and n == 0 and o == 0:
                        origin_idx = idx
                    idx += 1
        points = np.array(points)
        
        try:
            from scipy.spatial import Voronoi
            vor = Voronoi(points)
            faces = []
            for ridge_idx, p_pair in enumerate(vor.ridge_points):
                if origin_idx in p_pair:
                    vert_idxs = vor.ridge_vertices[ridge_idx]
                    if all(v >= 0 for v in vert_idxs):
                        face_verts = vor.vertices[vert_idxs].tolist()
                        if len(face_verts) > 0:
                            face_verts.append(face_verts[0])
                            faces.append(face_verts)
        except Exception:
            faces = []
            corners = []
            for dx in [-0.5, 0.5]:
                for dy in [-0.5, 0.5]:
                    for dz in [-0.5, 0.5]:
                        corners.append(dx*b1 + dy*b2 + dz*b3)
            face_indices = [
                [0, 1, 3, 2], [4, 5, 7, 6],
                [0, 1, 5, 4], [2, 3, 7, 6],
                [0, 2, 6, 4], [1, 3, 7, 5]
            ]
            for f_idx in face_indices:
                face_verts = [corners[i].tolist() for i in f_idx]
                face_verts.append(face_verts[0])
                faces.append(face_verts)
                
        sym_coords = []
        for p, label in zip(pts, labels):
            coord = p @ B
            sym_coords.append({"label": label, "x": float(coord[0]), "y": float(coord[1]), "z": float(coord[2])})
            
        return {
            "dimension": 3,
            "faces": faces,
            "sym_points": sym_coords,
            "recip_vectors": B.tolist()
        }


# ─── Serialize A_0_R (orbital→cell→amplitude) ─────────────────────────────────
def _ser_amp(A_0_R, orb_labels, lattice=None):
    """Serialize amplitudes for the JS frontend.
    Keys are converted to integer unit-cell coordinates.
    For Mode B Cartesian models (non-integer exponents), uses the
    primitive-vector inverse to get the correct fractional unit cell.
    """
    # Detect whether any key is non-integer
    is_noninteger = any(
        any(abs(x - round(x)) > 1e-9 for x in exp)
        for q in A_0_R for exp in A_0_R[q].keys()
    ) if A_0_R else False

    if is_noninteger and lattice is not None:
        A_mat = np.array(lattice.primitive_vectors, dtype=float)
        A_inv = (np.linalg.inv(A_mat)
                 if A_mat.shape[0] == A_mat.shape[1]
                 else np.linalg.pinv(A_mat))
        def to_int_cell(exp, orb_pos_frac):
            tau_cart = np.array(orb_pos_frac, dtype=float) @ A_mat
            r_frac   = (np.array(exp, dtype=float) - tau_cart) @ A_inv
            return [int(round(x)) for x in r_frac]
    else:
        def to_int_cell(exp, _orb_pos):
            return [int(round(x)) for x in exp]

    out = {}
    for q, amp_dict in A_0_R.items():
        orb_pos = lattice.orbitals[q]["position"] if lattice is not None else [0]*2
        out[str(q)] = {"label": orb_labels[q] if q < len(orb_labels) else str(q),
                       "amplitudes": []}
        for cell, coef in amp_dict.items():
            c = complex(coef)
            int_cell = to_int_cell(cell, orb_pos)
            out[str(q)]["amplitudes"].append(
                {"cell": int_cell, "re": float(c.real),
                 "im": float(c.imag), "abs": float(abs(c))})
    return out

def validate_gauge(H_k, eps0, x_k_min, lattice, k_points):
    """
    Checks if x_k_min is an eigenvector of H(k) at eps0.
    Returns: (bool, message)
    """
    k_arr = np.asarray(list(k_points) if not isinstance(k_points, np.ndarray) else k_points, dtype=float)
    if len(k_arr) == 0:
        return True, "검증 k-점 없음"
    H_batch = H_k.evaluate_batch(k_arr, lattice.primitive_vectors)        # (K, Q, Q)
    x_batch = np.stack(
        [poly.evaluate_batch(k_arr, lattice.primitive_vectors) for poly in x_k_min],
        axis=1)                                                            # (K, Q)
    lhs = np.einsum('kij,kj->ki', H_batch, x_batch)                      # (K, Q)
    diff = np.linalg.norm(lhs - eps0 * x_batch, axis=1)                  # (K,)
    norm_x = np.linalg.norm(x_batch, axis=1)                             # (K,)
    valid = norm_x > 1e-12
    rel_errs = np.where(valid, diff / np.where(valid, norm_x, 1.0), 0.0)
    max_error = float(np.max(rel_errs))
    if max_error < 1e-5:
        return True, f"수치 검증 통과 (최대 상대 오차: {max_error:.4e})"
    else:
        return False, f"검증 실패 (최대 상대 오차: {max_error:.4e})"

# ─── Public: models list ──────────────────────────────────────────────────────
def get_models_list():
    return json.dumps([
        {"id":"zigzag",       "name":"Zigzag Chain",      "dim":1,"Q":2,
         "desc":"1D 사슬, Q=2, 비특이형, E=\u22122  |  k-path: \u2212\u03c0\u2192\u03c0"},
        {"id":"kagome",       "name":"Kagome NN",         "dim":2,"Q":3,
         "desc":"2D 육각격자, Q=3, 특이형 k\u2080=(0,0), E=\u22122  |  \u0393\u2192M\u2192K\u2192\u0393"},
        {"id":"bilayer",      "name":"Bilayer Square",    "dim":2,"Q":2,
         "desc":"2D 정방격자, Q=2, 비특이형, E=2  |  \u0393\u2192X\u2192M\u2192\u0393"},
        {"id":"lieb",         "name":"Lieb",              "dim":2,"Q":3,
         "desc":"2D 정방격자, Q=3, 특이형 k\u2080=(\u03c0,\u03c0), E=0  |  \u0393\u2192X\u2192M\u2192\u0393"},
        {"id":"modified_lieb","name":"Modified Lieb",     "dim":2,"Q":3,
         "desc":"2D 정방격자, Q=3, 특이형 k\u2080=(0,0), E=0  |  \u0393\u2192X\u2192M\u2192\u0393"},
        {"id":"checker1",     "name":"Checkerboard I",   "dim":2,"Q":2,
         "desc":"2D 정방격자, Q=2, 특이형 k\u2080=(0,0), E=0  |  \u0393\u2192X\u2192M\u2192\u0393"},
        {"id":"checker2",     "name":"Checkerboard II",  "dim":2,"Q":2,
         "desc":"2D 정방격자, Q=2, 특이형 k\u2080=(\u03c0,\u03c0), E=0  |  \u0393\u2192X\u2192M\u2192\u0393"},
        {"id":"checker3",     "name":"Checkerboard III", "dim":2,"Q":2,
         "desc":"2D 정방격자, Q=2, 비특이형, E=0  |  \u0393\u2192X\u2192M\u2192\u0393"},
        {"id":"honeycomb",    "name":"Honeycomb",        "dim":2,"Q":2,
         "desc":"2D 육각격자, Q=2, 비특이형, E=0  |  \u0393\u2192M\u2192K\u2192\u0393"},
        {"id":"cubic3d",      "name":"Cubic 3D",         "dim":3,"Q":3,
         "desc":"3D 단순입방, Q=3, 특이형, E=0  |  \u0393\u2192X\u2192M\u2192R\u2192\u0393"},
        {"id":"kagome3",      "name":"Kagome-3",         "dim":2,"Q":3,
         "desc":"2D 육각격자, Q=3, 이중축퇴 비특이형, E=\u22122  |  \u0393\u2192M\u2192K\u2192\u0393"},
        {"id":"fb5trig",      "name":"Flat-5 (trig)",    "dim":2,"Q":5,
         "desc":"허브3+림2, Q=5, 비특이형, E=0 (M=1) · Schur 환원  |  Γ→X→M→Γ"},
        {"id":"fb5sqrt3",     "name":"Flat-5 (sqrt3)",   "dim":2,"Q":5,
         "desc":"허브3+림2, Q=5, √3 위상, E=0 (M=1) · 무리수 GCD  |  Γ→X→M→Γ"},
        {"id":"fb10deg",      "name":"Flat-10 (deg)",    "dim":2,"Q":10,
         "desc":"허브6+림4, Q=10, √3 위상, 이중축퇴 E=0 (M=2) · Schur 환원(안정 4×4)  |  Γ→X→M→Γ"},
        {"id":"chern3",       "name":"Chern C=3 Flatband", "dim":2,"Q":7,
         "desc":"이분 골격 C=3 단일 FB, Q=7, E=0 (M=1) · C6/C3 대칭 및 NNN  |  Γ→K→M→Γ"},
    ])

# ─── Public: get preset spec ─────────────────────────────────────────────────
def get_model_spec(model_id):
    from cls_finder.models.library import (
        zigzag_chain, kagome_nn, bilayer_square, lieb, modified_lieb,
        checkerboard_1, checkerboard_2, checkerboard_3, honeycomb_flat,
        cubic_3D, kagome_3, flatband_5_trig, flatband_5_sqrt3,
        flatband_10_sqrt3_deg, chern_3_flatband)
    m = {"zigzag":zigzag_chain,"kagome":kagome_nn,"bilayer":bilayer_square,
         "lieb":lieb,"modified_lieb":modified_lieb,"checker1":checkerboard_1,
         "checker2":checkerboard_2,"checker3":checkerboard_3,
         "honeycomb":honeycomb_flat,"cubic3d":cubic_3D,"kagome3":kagome_3,
         "fb5trig":flatband_5_trig,"fb5sqrt3":flatband_5_sqrt3,
         "fb10deg":flatband_10_sqrt3_deg,"chern3":chern_3_flatband}
    if model_id not in m:
        return json.dumps({"error": f"Unknown model: {model_id}"})
    return json.dumps(m[model_id]())

# ─── Public: Robust Boundary Mode (RBM) ───────────────────────────────────────
def _extract_band_cls(lattice, H_k, flat_tol, band_index=None, grid=None):
    """Pick a flat band, return (eps0, M, A_0_R, singular, k0_list, band_index)."""
    import sympy
    from cls_finder.band.bands import detect_flat_bands
    from cls_finder.eigen.eigenstate import extract_eigenstate_analytical
    from cls_finder.classify.singularity import classify_singularity
    from cls_finder.cls.analytic import select_cls_basis

    d = lattice.dimension
    grid = grid or ([24] * d)
    fbs = detect_flat_bands(H_k, lattice, grid, flat_tol)
    if not fbs:
        raise ValueError("이 모델에는 평탄 밴드가 없습니다.")
    uniq, seen = [], set()
    for fb in fbs:
        if fb["band_index"] not in seen:
            uniq.append(fb); seen.add(fb["band_index"])
    fb = (next((f for f in uniq if f["band_index"] == band_index), uniq[0])
          if band_index is not None else uniq[0])
    eps0 = fb["energy"]; deg = fb["degenerate_indices"]; M = len(deg)
    syms = sympy.symbols('x1 x2 x3')[:d]

    singular, k0_list = None, []
    try:
        w = extract_eigenstate_analytical(H_k, eps0, syms)
        sc = classify_singularity(w, lattice, H_k, eps0, grid)
        singular, k0_list = sc["singular"], sc["k0_list"]
    except Exception:
        pass
    A_0_R = select_cls_basis(H_k, eps0, syms, M, lattice=lattice)[0][1]
    return eps0, M, A_0_R, singular, k0_list, fb["band_index"]


def compute_rbm(spec_json, Nx, Ny=None, Nz=None, band_index=None,
                defect_cell=None, k0_override=None):
    """
    Robust Boundary Mode for a chosen flat band on an open-boundary finite
    lattice. Returns JSON: real-space boundary-mode sites (x,y,z, amp, phase),
    the bulk-cancellation verdict, skin-depth profiles per axis, and metadata.

    Nx/Ny/Nz: open-boundary system size (Ny/Nz default to Nx for d>=2/3).
    defect_cell: optional [n1,...] — omit the CLS copy centered there (robustness
                 control: singular bands stay locally robust, nonsingular shatter).
    """
    try:
        from cls_finder.io.parser import parse_input
        from cls_finder.rbm import compute_rbm as _compute_rbm
        from cls_finder.rbm.boundary_mode import (boundary_mode_with_defect,
                                                  verify_bulk_cancellation,
                                                  boundary_mode_sites)
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        d = lattice.dimension
        flat_tol = spec.get("options", {}).get("flat_tol", 1e-4)
        grid = spec.get("options", {}).get("k_grid", [24] * d)

        Nx = int(Nx)
        size = [Nx]
        if d >= 2:
            size.append(int(Ny) if Ny else Nx)
        if d >= 3:
            size.append(int(Nz) if Nz else Nx)

        eps0, M, A_0_R, singular, k0_list, bidx = _extract_band_cls(
            lattice, H_k, flat_tol, band_index, grid)
        k0 = (np.array(k0_override, dtype=float) if k0_override is not None
              else (k0_list[0] if k0_list else None))

        out = _compute_rbm(lattice, A_0_R, size, k_singularity=k0, singular=singular)

        payload = {
            "band_index": int(bidx), "energy": float(eps0), "M": int(M),
            "singular": (None if singular is None else bool(singular)),
            "k_singularity": (None if k0 is None else [float(x) for x in np.ravel(k0)]),
            "system_size": out["system_size"],
            "support_radius": out["support_radius"],
            "validation": out["validation"],
            "skin_depth": {str(ax): out["skin_depth"][ax] for ax in out["skin_depth"]},
            "sites": out["sites"],
            "n_nonzero_sites": out["n_nonzero_sites"],
            "max_amplitude": out["max_amplitude"],
            "dimension": d,
        }

        if defect_cell is not None:
            dres = boundary_mode_with_defect(lattice, A_0_R, size, k_singularity=k0,
                                             omit_centers=[tuple(int(x) for x in defect_cell)])
            dval = verify_bulk_cancellation(dres, tol=1e-10)
            dsites = boundary_mode_sites(lattice, dres, amp_tol=1e-9)
            payload["defect"] = {
                "cell": [int(x) for x in defect_cell],
                "validation": dval,
                "sites": dsites,
                "n_nonzero_sites": sum(1 for s in dsites if s["nonzero"]),
            }
        return json.dumps(payload, default=_ser)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc()})


def compute_chern(spec_json, band_index=None, grid_n=24, scan_n=120):
    """
    Chern number of a flat band: robust FHS lattice field strength + the
    finite-Fourier (Rhim-Yang) structural analysis (common zeros, projector
    continuity, local winding). Returns JSON with the number, the structural
    verdict, band isolation, and an explanation.
    """
    try:
        from cls_finder.io.parser import parse_input
        from cls_finder.cls.analytic import select_cls_basis
        from cls_finder.classify.chern import analyze_flat_band_chern
        import sympy
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        d = lattice.dimension
        if d != 2:
            return json.dumps({"error": "Chern 수 계산은 2D 격자 모델만 지원합니다."})
        flat_tol = spec.get("options", {}).get("flat_tol", 1e-4)
        grid = spec.get("options", {}).get("k_grid", [24, 24])

        eps0, M, _A, singular, k0_list, bidx = _extract_band_cls(
            lattice, H_k, flat_tol, band_index, grid)
        syms = sympy.symbols('x1 x2')
        try:
            x_k = select_cls_basis(H_k, eps0, list(syms), M, lattice=lattice)[0][0]
        except Exception:
            x_k = None
        orb_labels = [o["label"] for o in lattice.orbitals]
        plot_n = int(spec.get("options", {}).get("plot_n", 60))
        out = analyze_flat_band_chern(H_k, lattice, eps0, M, x_k=x_k,
                                      symbols=orb_labels, grid_n=grid_n, scan_n=scan_n, n_grid=plot_n)
        out["band_index"] = int(bidx)
        out["singular"] = (None if singular is None else bool(singular))
        return json.dumps(out, default=_ser)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}",
                           "trace": traceback.format_exc()})


def _resolve_band_indices(lattice, H_k, spec, band_indices=None, flat_band_index=None):
    if band_indices is not None and len(band_indices) > 0:
        return [int(x) for x in band_indices]
    # Fallback to flat band detection
    flat_tol = spec.get("options", {}).get("flat_tol", 1e-4)
    grid = spec.get("options", {}).get("k_grid", [24, 24])
    try:
        from cls_finder.band.bands import detect_flat_bands
        fbs = detect_flat_bands(H_k, lattice, grid, flat_tol)
        if fbs:
            uniq = []
            seen = set()
            for fb in fbs:
                if fb["band_index"] not in seen:
                    uniq.append(fb)
                    seen.add(fb["band_index"])
            fb = (next((f for f in uniq if f["band_index"] == flat_band_index), uniq[0])
                  if flat_band_index is not None else uniq[0])
            return [int(x) for x in fb["degenerate_indices"]]
    except Exception:
        pass
    # Absolute fallback: just occupy the lowest band
    return [0]


def compute_wilson_loop_api(spec_json, band_indices=None, flat_band_index=None, n_x=40, n_y=40):
    try:
        from cls_finder.io.parser import parse_input
        from cls_finder.topology.wilson_loop import compute_wilson_loop
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        d = lattice.dimension
        if d != 2:
            return json.dumps({"error": "Wilson Loop 계산은 2D 격자 모델만 지원합니다."})
        resolved_indices = _resolve_band_indices(lattice, H_k, spec, band_indices, flat_band_index)
        out = compute_wilson_loop(H_k, lattice, resolved_indices, n_x=int(n_x), n_y=int(n_y))
        out["band_indices"] = resolved_indices
        return json.dumps(out, default=_ser)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})


def compute_entanglement_spectrum_api(spec_json, band_indices=None, flat_band_index=None, N_x=40, n_y=60):
    try:
        from cls_finder.io.parser import parse_input
        from cls_finder.topology.entangle_spec import compute_entanglement_spectrum
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        d = lattice.dimension
        if d != 2:
            return json.dumps({"error": "Entanglement Spectrum 계산은 2D 격자 모델만 지원합니다."})
        resolved_indices = _resolve_band_indices(lattice, H_k, spec, band_indices, flat_band_index)
        out = compute_entanglement_spectrum(H_k, lattice, resolved_indices, N_x=int(N_x), n_y=int(n_y))
        out["band_indices"] = resolved_indices
        return json.dumps(out, default=_ser)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})


def compute_fu_kane_api(spec_json, band_indices=None, flat_band_index=None, P_matrix_list=None):
    try:
        from cls_finder.io.parser import parse_input
        from cls_finder.topology.fu_kane import compute_fu_kane
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        d = lattice.dimension
        if d != 2:
            return json.dumps({"error": "Fu-Kane Formula 계산은 2D 격자 모델만 지원합니다."})
        resolved_indices = _resolve_band_indices(lattice, H_k, spec, band_indices, flat_band_index)
        
        Q = H_k.rows
        P_matrix = None
        if P_matrix_list is not None:
            arr = np.array(P_matrix_list, dtype=complex)
            if arr.ndim == 1:
                if len(arr) == Q:
                    P_matrix = np.diag(arr)
                else:
                    return json.dumps({"error": f"제공된 Parity 대각 성분의 개수({len(arr)})가 오비탈 개수({Q})와 일치하지 않습니다."})
            elif arr.ndim == 2:
                if arr.shape == (Q, Q):
                    P_matrix = arr
                else:
                    return json.dumps({"error": f"제공된 Parity 행렬 크기 {arr.shape}가 {Q}x{Q}와 일치하지 않습니다."})
        else:
            P_matrix = np.eye(Q, dtype=complex)
            
        out = compute_fu_kane(H_k, lattice, resolved_indices, P_matrix=P_matrix)
        out["band_indices"] = resolved_indices
        return json.dumps(out, default=_ser)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()})


# ─── Public: quick band structure preview ────────────────────────────────────
def get_band_data(spec_json):
    try:
        from cls_finder.io.parser import parse_input
        from cls_finder.band.bands import get_reciprocal_vectors
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        Q = H_k.rows
        flat_tol = spec.get("options", {}).get("flat_tol", 1e-5)
        
        k_path_str = spec.get("options", {}).get("k_path_str", "")
        k_points_override = spec.get("options", {}).get("k_points_override", {})
        if k_path_str:
            pts, labs = _parse_k_path(k_path_str, lattice, k_points_override)
        else:
            pts, labs = _sym_path(lattice, H_k)
            if k_points_override:
                for i, label in enumerate(labs):
                    match_key = next((k for k in k_points_override if k.upper() == label.upper()), None)
                    if match_key:
                        pts[i] = np.array(k_points_override[match_key])
        k_pts, ticks = _gen_path(lattice, pts)
        E = eigvalsh_batch(H_k.evaluate_batch(k_pts, lattice.primitive_vectors))
        flat_E = [float(np.mean(E[:,n])) for n in range(Q)
                  if np.var(E[:,n]) < flat_tol]
                  
        B = get_reciprocal_vectors(lattice)
        recip_vectors = B.tolist()
        sym_coords = []
        for p, label in zip(pts, labs):
            coord = p @ B
            sym_coords.append({"label": label, "frac": p.tolist(), "cart": coord.tolist()})
            
        bz_info = _bz_data(lattice, k_path_str, k_points_override, H_k)
        return json.dumps({"x": list(range(len(k_pts))),
                           "bands": [E[:,n].tolist() for n in range(Q)],
                           "k_ticks": {str(t):l for t,l in zip(ticks,labs)},
                           "flat_energies": flat_E, "n_bands": Q,
                           "reciprocal_vectors": recip_vectors,
                           "high_symmetry_points": sym_coords,
                           "brillouin_zone": bz_info,
                           "error": None}, default=_ser)
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})

# ─── Public: full analysis pipeline ──────────────────────────────────────────
def run_analysis_stream(spec_json):
    from cls_finder.io.parser import parse_input
    from cls_finder.band.bands import detect_flat_bands, detect_nearly_flat_bands
    from cls_finder.cls.analytic import select_cls_basis
    from cls_finder.classify.singularity import classify_singularity
    from cls_finder.cls.gauge_analysis import extract_canonical_minimal_cls
    from cls_finder.cls.numeric import extract_cls_numeric, cross_validate_cls, verify_cls_eigenstate
    from cls_finder.cls.noncontractible import build_noncontractible
    import sympy

    steps = []
    result = {"steps": steps, "band_plot": None, "flat_bands": [], "error": None}

    def log(name, status, msg="", data=None):
        e = {"name": name, "status": status, "message": msg}
        if data is not None: e["data"] = data
        steps.append(e)
        return json.dumps({"type": "log", "step": e}, default=_ser)

    try:
        spec = json.loads(spec_json)
        k_path_str = spec.get("options", {}).get("k_path_str", "")
        k_points_override = spec.get("options", {}).get("k_points_override", {})

        # 1. Parse
        yield log("입력 파싱", "running", "격자와 해밀토니안을 파싱하는 중...")
        lattice, H_k = parse_input(spec)
        Q, dim = H_k.rows, lattice.dimension
        orb_labels = [o["label"] for o in lattice.orbitals]
        yield log("입력 파싱", "success",
            f"{dim}차원 격자 | {Q}개 오비탈: {', '.join(orb_labels)} | {Q}×{Q} H(k)",
            {"dim": dim, "Q": Q, "orbitals": orb_labels})

        # 1b. Lattice classification
        try:
            from cls_finder.core.lattice_classify import classify_lattice
            lattice_info = classify_lattice(lattice, H_k)
            result["lattice_info"] = lattice_info
            
            sub_desc_parts = []
            sub_desc_parts.append(f"브라베 격자: {lattice_info['bravais_type']}")
            if lattice_info.get("known_lattice"):
                sub_desc_parts.append(f"알려진 격자: {lattice_info['known_lattice']}")
            sub_desc_parts.append(f"서브라티스 {lattice_info['sublattice_count']}개")
            if lattice_info.get("is_multi_orbital"):
                sub_desc_parts.append(f"다중 오비탈 ({lattice_info['orbitals_per_sublattice']})")
            if lattice_info.get("is_bipartite") is True:
                sub_desc_parts.append("이분 격자")
            elif lattice_info.get("is_bipartite") is False:
                sub_desc_parts.append("비이분 격자")
            
            yield log("격자 구조 분석", "success",
                " | ".join(sub_desc_parts),
                lattice_info)
        except Exception as e:
            yield log("격자 구조 분석", "warning", f"분류 실패: {e}")
            result["lattice_info"] = None

        # 1c. Reciprocal space info
        try:
            from cls_finder.band.bands import get_reciprocal_vectors
            B = get_reciprocal_vectors(lattice)
            if k_path_str:
                pts_r, labs_r = _parse_k_path(k_path_str, lattice, k_points_override)
            else:
                pts_r, labs_r = _sym_path(lattice, H_k)
                if k_points_override:
                    for i, label in enumerate(labs_r):
                        match_key = next((k for k in k_points_override if k.upper() == label.upper()), None)
                        if match_key:
                            pts_r[i] = np.array(k_points_override[match_key])
            
            recip_info = []
            recip_info.append("역격자 벡터:")
            for idx, b_vec in enumerate(B):
                b_str = ", ".join(f"{x:.4f}" for x in b_vec)
                b_str_2pi = ", ".join(f"{x/(2.0*np.pi):.4f}" for x in b_vec)
                recip_info.append(f"b_{idx+1}=[{b_str}]")
                
            recip_info.append("고대칭점:")
            for p, label in zip(pts_r, labs_r):
                cart_coord = p @ B
                cart_str = ", ".join(f"{x:.4f}" for x in cart_coord)
                frac_str = ", ".join(f"{x:.4f}" for x in p)
                recip_info.append(f"{label}=frac:[{frac_str}]/cart:[{cart_str}]")
                
            yield log("역격자 및 고대칭점 분석", "success",
                " | ".join(recip_info),
                {
                    "reciprocal_vectors": B.tolist(),
                    "high_symmetry_points": [{"label": l, "frac": p.tolist(), "cart": (p @ B).tolist()} for p, l in zip(pts_r, labs_r)]
                })
            bz_info = _bz_data(lattice, k_path_str, k_points_override, H_k)
            result["reciprocal_space"] = {
                "reciprocal_vectors": B.tolist(),
                "high_symmetry_points": [{"label": l, "frac": p.tolist(), "cart": (p @ B).tolist()} for p, l in zip(pts_r, labs_r)],
                "brillouin_zone": bz_info
            }
        except Exception as e:
            yield log("역격자 및 고대칭점 분석", "warning", f"분석 실패: {e}")

        # 2. Band structure
        yield log("밴드 구조", "running", "브릴루앙 구역 샘플링 및 대각화 중...")
        opts = spec.get("options", {})
        grid_size = opts.get("k_grid", [30]*dim)
        if isinstance(grid_size, int): grid_size = [grid_size]*dim
        flat_tol = opts.get("flat_tol", 1e-5)
        nearly_flat_ratio = opts.get("nearly_flat_ratio", 0.05)

        if k_path_str:
            pts, labs = _parse_k_path(k_path_str, lattice, k_points_override)
        else:
            pts, labs = _sym_path(lattice, H_k)
            if k_points_override:
                for i, label in enumerate(labs):
                    match_key = next((k for k in k_points_override if k.upper() == label.upper()), None)
                    if match_key:
                        pts[i] = np.array(k_points_override[match_key])
        k_pts, ticks = _gen_path(lattice, pts)
        E = eigvalsh_batch(H_k.evaluate_batch(k_pts, lattice.primitive_vectors))
        flat_E_all = [float(np.mean(E[:,n])) for n in range(Q)
                      if np.var(E[:,n]) < flat_tol]

        # Detect nearly flat bands using the already-computed path energies
        nearly_flat_bands = detect_nearly_flat_bands(E, flat_tol, nearly_flat_ratio)
        nfb_indices = [nfb["band_index"] for nfb in nearly_flat_bands]

        result["band_plot"] = {
            "x": list(range(len(k_pts))),
            "bands": [E[:,n].tolist() for n in range(Q)],
            "k_ticks": {str(t):l for t,l in zip(ticks,labs)},
            "flat_energies": flat_E_all, "n_bands": Q,
            "nearly_flat_indices": nfb_indices,
        }
        result["nearly_flat_bands"] = nearly_flat_bands

        flat_bands = detect_flat_bands(H_k, lattice, grid_size, flat_tol)
        seen, unique_fb = set(), []
        for fb in flat_bands:
            if fb["band_index"] not in seen:
                unique_fb.append(fb); seen.add(fb["band_index"])

        is_nearly_flat_run = False
        bands_to_process = []
        if unique_fb:
            bands_to_process = unique_fb
            log_msg = f"{len(unique_fb)}개 평탄 밴드 발견 │ 에너지: {[round(f['energy'],4) for f in unique_fb]}"
            if nearly_flat_bands:
                log_msg += f" │ 거의 평탄 {len(nearly_flat_bands)}개: 밴드 {nfb_indices}"
            yield log("밴드 구조", "success", log_msg, {"flat_bands": unique_fb})
        elif nearly_flat_bands:
            is_nearly_flat_run = True
            yield log("밴드 구조", "warning",
                f"완전 평탄 밴드 없음. 거의 평탄 밴드 {len(nearly_flat_bands)}개 발견 "
                f"(perturbation으로 flatness가 파괴된 것으로 의심) │ "
                f"밴드: {nfb_indices} ─ 특이점 검증을 진행합니다.",
                {"nearly_flat_bands": nearly_flat_bands})
            for nfb in nearly_flat_bands:
                bands_to_process.append({
                    "band_index": nfb["band_index"],
                    "energy": nfb["energy"],
                    "degenerate_indices": [nfb["band_index"]],
                    "is_nearly_flat": True
                })
        else:
            yield log("밴드 구조", "info", "평탄 밴드가 발견되지 않았습니다.")
            yield json.dumps({"type": "result", "result": result}, default=_ser)
            return

        x1, x2, x3 = sympy.symbols('x1 x2 x3')
        symbols = [x1, x2, x3][:dim]

        # 3. Process each flat/nearly flat band
        band_results = []
        # Cache the complete CLS basis per degenerate group so an M-fold band
        # shows M distinct, independent CLS (one per band index) in the UI.
        basis_cache = {}
        for i, fb in enumerate(bands_to_process):
            idx_b, eps0, deg = fb["band_index"], fb["energy"], fb["degenerate_indices"]
            is_nearly_flat = fb.get("is_nearly_flat", False)
            if is_nearly_flat:
                tag = f"거의 평탄 밴드 #{i+1} (E_avg={eps0:.4f})"
            else:
                tag = f"밴드 #{i+1} (E={eps0:.4f})"

            br = {"band_index": int(idx_b), "energy": float(eps0),
                  "degenerate_indices": [int(x) for x in deg],
                  "singular": None, "k0_list": [],
                  "cls": None, "nls": [], "cross_check": None,
                  "is_nearly_flat": is_nearly_flat}

            # A. Best eigenvector selection (Minor+Adjugate / Rhim-Yang mixing /
            #    syzygy, non-singular preferred). For an M-fold degenerate band
            #    each band index gets its own member of the complete basis.
            M_deg = len(deg)
            yield log(f"{tag} ─ 고유상태 & CLS", "running",
                "최적 Gauge 선택 및 CLS 계산 중 (비특이·최소 support 기준)..." if not is_nearly_flat
                else "최적 Gauge 선택 및 기호 고유벡터 계산 중 (verify=False)...")

            x_k_min, A_0_R_min = [], {}
            g_id, method_name, is_singular_gauge = "unknown", "Unknown", False

            try:
                grp_key = tuple(sorted(int(x) for x in deg))
                if grp_key not in basis_cache:
                    basis_cache[grp_key] = select_cls_basis(
                        H_k, eps0, symbols, M_deg, lattice, verify=not is_nearly_flat)
                basis = basis_cache[grp_key]
                j = grp_key.index(int(idx_b)) if int(idx_b) in grp_key else 0
                x_k_min, A_0_R_min, gauge_meta = basis[j] if j < len(basis) else basis[-1]
                gauge_id_raw   = gauge_meta["gauge_id"]
                is_singular_gauge = gauge_meta["is_singular"]
                support_raw    = gauge_meta["support_size"]

                if isinstance(gauge_id_raw, int):
                    method_name = f"Minor+Adjugate (가이드: {orb_labels[gauge_id_raw]})"
                    g_id = f"p_{gauge_id_raw}"
                elif str(gauge_id_raw).startswith("syzygy"):
                    method_name = f"Gröbner 시저지 생성원 ({gauge_id_raw})"
                    g_id = str(gauge_id_raw)
                elif str(gauge_id_raw).startswith(("sympy_mix", "sympy_raw")):
                    method_name = f"Rhim-Yang 혼합 ({gauge_id_raw})"
                    g_id = str(gauge_id_raw)
                elif str(gauge_id_raw).startswith("structural"):
                    method_name = f"구조적 Schur 환원 ({gauge_id_raw})"
                    g_id = str(gauge_id_raw)
                else:
                    method_name = f"SymPy 기호 영공간 ({gauge_id_raw})"
                    g_id = str(gauge_id_raw)

                deg_note = f" | 축퇴 {M_deg}중 {j+1}번째 CLS" if M_deg > 1 else ""
                msg_suffix = " | ⚠ 게이지 특이점" if is_singular_gauge else " | ✓ 비특이"
                if is_nearly_flat:
                    msg_prefix = f"{method_name} | 고유벡터 support {support_raw}"
                else:
                    msg_prefix = f"{method_name} | support {support_raw}"
                yield log(f"{tag} ─ 고유상태 & CLS", "success",
                    msg_prefix + msg_suffix + deg_note,
                    {"eigenvector": [repr(p) for p in x_k_min], "gauge": g_id})
            except Exception as e:
                yield log(f"{tag} ─ 고유상태 & CLS", "warning", f"해석적 추출 실패: {e}")

            # B. Singularity classification (reuses eigenvector from step A)
            yield log(f"{tag} ─ 위상 분류", "running",
                "BZ에서 k₀ 특이점을 스캔하는 중...")
            try:
                if x_k_min:
                    sc = classify_singularity([x_k_min], lattice, H_k, eps0, grid_size, degenerate_indices=deg)
                    br["singular"] = bool(sc["singular"])
                    br["k0_list"] = [k0.tolist() for k0 in sc["k0_list"]]
                    cls_name = "특이형 (Singular)" if sc["singular"] else "비특이형 (Non-Singular)"
                    extra = f" │ k₀ 개수: {len(sc['k0_list'])}" if sc["singular"] else ""
                    yield log(f"{tag} ─ 위상 분류", "success", cls_name + extra,
                        {"singular": sc["singular"],
                         "k0_list": [k0.tolist() for k0 in sc["k0_list"]]})
                else:
                    br["singular"] = False
                    yield log(f"{tag} ─ 위상 분류", "info", "고유벡터 없음 → 비특이형으로 가정")
            except Exception as e:
                br["singular"] = False
                yield log(f"{tag} ─ 위상 분류", "warning", f"분류 실패: {e}")

            # C. Canonical CLS + numerical cross-validation
            yield log(f"{tag} ─ CLS 정규화 & 검증", "running",
                "최적 위상 정규화 + 수치 교차검증 중...")
            gauges_list = []

            if x_k_min and A_0_R_min:
                try:
                    # One call: picks best global phase, normalizes, builds site list
                    canonical = extract_canonical_minimal_cls(A_0_R_min, Q, dim, lattice)
                    if canonical:
                        for s in canonical["sites"]:
                            s["label"] = (orb_labels[s["orbital"]]
                                          if s["orbital"] < len(orb_labels)
                                          else str(s["orbital"]))

                    # Build A_0_R in canonical phase for the lattice plot
                    if canonical:
                        theta = np.radians(canonical["global_phase_deg"])
                        f_phase = complex(np.exp(1j * theta))
                        A_0_R_plot = {q: {cell: complex(c) * f_phase
                                          for cell, c in qd.items()}
                                      for q, qd in A_0_R_min.items()}
                        max_abs = max(
                            (abs(complex(v)) for qd in A_0_R_plot.values()
                             for v in qd.values()), default=1.0)
                        if max_abs > 1e-12:
                            A_0_R_plot = {q: {cell: c / max_abs for cell, c in qd.items()}
                                          for q, qd in A_0_R_plot.items()}
                    else:
                        A_0_R_plot = A_0_R_min

                    cls_plot  = _cls_plot_data(lattice, H_k, A_0_R_plot)
                    amp_data  = _ser_amp(A_0_R_plot, orb_labels, lattice)
                    bz_plot   = _bz_data(lattice, k_path_str, k_points_override, H_k)
                    support_size = sum(len(v["amplitudes"]) for v in amp_data.values())

                    gauges_list.append({
                        "gauge_id":    g_id,
                        "method_name": method_name,
                        "x_k_min":     [repr(p) for p in x_k_min],
                        "amplitudes":  amp_data,
                        "support_size": support_size,
                        "singular":    is_singular_gauge,
                        "k0_list":     [],
                        "plot":        cls_plot,
                        "bz_plot":     bz_plot,
                        "canonical":   canonical,
                        "is_primary":  True,
                    })
                except Exception as e:
                    yield log(f"{tag} ─ CLS 정규화 & 검증", "warning", f"정규화 실패: {e}")

                # Numerical cross-validation (not shown as primary gauge)
                try:
                    if is_nearly_flat:
                        br["cross_check"] = {"success": False, "message": "Nearly flat (dispersive) band: strictly localized state is not an exact eigenstate."}
                    elif len(deg) > 1:
                        # Degenerate band: amplitude matching against one numerical
                        # CLS is ill-defined, so verify the eigenstate residual.
                        cross_ok, cross_msg = verify_cls_eigenstate(H_k, lattice, eps0, x_k_min)
                        br["cross_check"] = {"success": cross_ok, "message": cross_msg}
                    else:
                        A_0_R_num = extract_cls_numeric(H_k, lattice, eps0, grid_size)
                        cross_ok, cross_msg = cross_validate_cls(A_0_R_min, A_0_R_num)
                        br["cross_check"] = {"success": cross_ok, "message": cross_msg}
                except Exception as e:
                    br["cross_check"] = {"success": False, "message": str(e)}

            if not gauges_list:
                yield log(f"{tag} ─ CLS", "error", "해석적 고유벡터 도출 실패" if is_nearly_flat else "CLS 도출 실패")
            else:
                br["gauges"]      = gauges_list
                br["minimal_cls"] = gauges_list[0].get("canonical")
                br["cls"] = {
                    "x_k_min":    gauges_list[0]["x_k_min"],
                    "amplitudes": gauges_list[0]["amplitudes"],
                    "plot":       gauges_list[0]["plot"],
                    "bz_plot":    gauges_list[0]["bz_plot"],
                }
                cinfo = gauges_list[0].get("canonical") or {}
                cross_ok = br.get("cross_check", {}).get("success", False)
                realness_pct = f"{cinfo.get('realness', 0)*100:.0f}%" if cinfo else "N/A"
                if is_nearly_flat:
                    yield log(f"{tag} ─ 고유벡터 분석", "success",
                        f"완료 | support {gauges_list[0]['support_size']} | "
                        f"realness {realness_pct} | 수치검증: 우회됨 (Dispersive)")
                else:
                    yield log(f"{tag} ─ CLS 정규화 & 검증", "success",
                        f"완료 | support {gauges_list[0]['support_size']} | "
                        f"realness {realness_pct} | 수치검증: {'✓' if cross_ok else '⚠'}")

            # D'. Chern number of the flat band (FHS + Rhim-Yang structural test)
            if not is_nearly_flat:
                yield log(f"{tag} ─ Chern 수 계산", "running",
                    "FHS 격자장 + 유한 푸리에 공통영점/winding 분석 중...")
                try:
                    from cls_finder.classify.chern import analyze_flat_band_chern
                    plot_n = int(spec.get("options", {}).get("plot_n", 60))
                    ch = analyze_flat_band_chern(
                        H_k, lattice, eps0, max(1, len(deg)),
                        x_k=x_k_min, symbols=orb_labels, grid_n=20, scan_n=80, n_grid=plot_n)
                    br["chern"] = ch
                    iso = ch.get("isolation", {})
                    if not iso.get("isolated", True):
                        msg = (f"C = {ch['chern_number']} (단, 고립 밴드 아님 — 이웃 밴드와 접촉 "
                               f"→ 단일밴드 Chern 미정의)")
                        st = "warning"
                    else:
                        msg = f"C = {ch['chern_number']}"
                        st = "success"
                    yield log(f"{tag} ─ Chern 수 계산", st, msg,
                        {"chern_number": ch["chern_number"],
                         "isolated": iso.get("isolated"),
                         "agreement": ch.get("agreement")})
                except Exception as e:
                    yield log(f"{tag} ─ Chern 수 계산", "warning", f"계산 실패: {e}")

            # E. NLS / NPS (only for singular bands)
            if br["singular"] and br["k0_list"] and not is_nearly_flat:
                yield log(f"{tag} ─ NLS/NPS 구성", "running",
                    f"k₀ = {[round(x,3) for x in br['k0_list'][0]]}에서 "
                    f"비수축 루프/평면 상태 구성 중...")
                try:
                    nls_states = build_noncontractible(
                        H_k, lattice, eps0, np.array(br["k0_list"][0]))
                    for st in nls_states:
                        nls_plot = _cls_plot_data(lattice, H_k, st["amplitudes"])
                        br["nls"].append({
                            "keep_axis": st["keep_axis"],
                            "amplitudes": _ser_amp(st["amplitudes"], orb_labels, lattice),
                            "plot": nls_plot})
                    yield log(f"{tag} ─ NLS/NPS 구성", "success",
                        f"{len(nls_states)}개 비수축 상태 구성 완료")
                except Exception as e:
                    yield log(f"{tag} ─ NLS/NPS 구성", "warning", f"실패: {e}")

            band_results.append(br)

        result["flat_bands"] = band_results
        yield log("분석 완료", "success",
            f"✓ 전체 분석 완료! ({len(band_results)}개 평탄/거의 평탄 밴드 처리)")

    except Exception as e:
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        yield log("오류", "error", str(e))

    yield json.dumps({"type": "result", "result": result}, default=_ser)

def run_analysis(spec_json):
    final_res = None
    for chunk_str in run_analysis_stream(spec_json):
        chunk = json.loads(chunk_str)
        if chunk.get("type") == "result":
            final_res = chunk["result"]
    return json.dumps(final_res, default=_ser)


def get_nanoribbon_data(spec_json, Nx, Nk, selected_bands=None, ky_min=-np.pi, ky_max=np.pi, periodic_dir="y", k_fixed=0.0):
    import json
    import traceback
    try:
        from cls_finder.io.parser import parse_input
        import numpy as np
        import scipy.linalg as la
        
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        Q = H_k.rows
        d = lattice.dimension
        if d != 2:
            return json.dumps({"error": "나노리본 계산은 2D 격자 모델만 지원합니다."})
            
        # Detect convention
        is_convention_ii = False
        for r in range(H_k.rows):
            for c in range(H_k.cols):
                for exp in H_k.data[r][c].coefs.keys():
                    if any(abs(x - round(x)) > 1e-9 for x in exp):
                        is_convention_ii = True
                        break
                if is_convention_ii:
                    break
            if is_convention_ii:
                break
                
        # Determine index of periodic and real-space dimensions
        if periodic_dir == "x":
            p_idx = 0  # periodic is x
            r_idx = 1  # real space is y
        else:
            p_idx = 1  # periodic is y
            r_idx = 0  # real space is x
            
        ky_space = np.linspace(ky_min, ky_max, Nk)
        
        # Collect hoppings
        hoppings = []
        for beta in range(Q):
            pos_beta = lattice.orbitals[beta]["position"]
            for alpha in range(Q):
                pos_alpha = lattice.orbitals[alpha]["position"]
                poly = H_k.data[beta][alpha]
                for exp, coef in poly.coefs.items():
                    if abs(coef) < 1e-10:
                        continue
                    if is_convention_ii:
                        R_trans = int(round(exp[r_idx] - pos_beta[r_idx] + pos_alpha[r_idx]))
                    else:
                        R_trans = int(round(exp[r_idx]))
                    hoppings.append((beta, alpha, R_trans, exp[p_idx], coef))
                    
        # Determine eigenvalue range based on selected bulk bands
        if not selected_bands:
            selected_bands = list(range(Q))
        else:
            selected_bands = sorted(list(set(selected_bands)))
            
        margin = max(5, int(round(0.1 * Nx)))
        min_band = min(selected_bands)
        max_band = max(selected_bands)
        
        lo = max(0, min_band * Nx - margin)
        ui_hi = (max_band + 1) * Nx - 1 + margin
        hi = min(Nx * Q - 1, ui_hi)
        num_selected = hi - lo + 1
        
        energies = np.zeros((Nk, num_selected))
        iprs = np.zeros((Nk, num_selected))
        edge_weights = np.zeros((Nk, num_selected))
        
        # Determine whether to use GPU
        from cls_finder.core.gpu import USE_GPU
        use_gpu_for_this = USE_GPU
        
        if use_gpu_for_this:
            import cupy as cp
            # 1. Vectorized ribbon Hamiltonian construction on GPU
            H_ribbon_batch = cp.zeros((Nk, Nx * Q, Nx * Q), dtype=complex)
            ky_gpu = cp.asarray(ky_space)
            
            for beta, alpha, R_trans, exp_p, coef in hoppings:
                mx_prime_start = max(0, -R_trans)
                mx_prime_end = min(Nx, Nx - R_trans)
                if mx_prime_start >= mx_prime_end:
                    continue
                
                amp = coef * cp.exp(1j * ky_gpu * exp_p)
                mx_prime_indices = np.arange(mx_prime_start, mx_prime_end)
                mx_indices = mx_prime_indices + R_trans
                row_indices = mx_indices * Q + beta
                col_indices = mx_prime_indices * Q + alpha
                H_ribbon_batch[:, row_indices, col_indices] += amp[:, None]
                
            # 2. Batch GPU diagonalization
            evals_gpu_all, evecs_gpu_all = cp.linalg.eigh(H_ribbon_batch)
            
            # 3. Slice and fetch to CPU in a single transfer
            evals_cpu = evals_gpu_all[:, lo:hi+1].get()  # (Nk, num_selected)
            evecs_cpu = evecs_gpu_all[:, :, lo:hi+1].get()  # (Nk, Nx*Q, num_selected)
            
            # 4. Compute IPR and edge weights on CPU
            energies = evals_cpu
            for ik in range(Nk):
                evecs_k = evecs_cpu[ik]  # (Nx*Q, num_selected)
                prob_k = np.abs(evecs_k) ** 2
                
                # IPR
                sum_prob = np.sum(prob_k, axis=0)
                valid_mask = sum_prob > 1e-10
                ipr_val = np.where(valid_mask, np.sum(prob_k ** 2, axis=0) / (sum_prob ** 2), 0.0)
                iprs[ik] = ipr_val
                
                # Edge weight (first 3 and last 3 layers)
                prob_reshaped = prob_k.reshape((Nx, Q, num_selected))
                layer_prob = np.sum(prob_reshaped, axis=1)  # (Nx, num_selected)
                n_edge = min(3, Nx // 2)
                edge_w = np.sum(layer_prob[:n_edge], axis=0) + np.sum(layer_prob[-n_edge:], axis=0)
                edge_weights[ik] = edge_w
        else:
            # 1. Vectorized ribbon Hamiltonian construction on CPU
            H_ribbon_batch = np.zeros((Nk, Nx * Q, Nx * Q), dtype=complex)
            for beta, alpha, R_trans, exp_p, coef in hoppings:
                mx_prime_start = max(0, -R_trans)
                mx_prime_end = min(Nx, Nx - R_trans)
                if mx_prime_start >= mx_prime_end:
                    continue
                
                amp = coef * np.exp(1j * ky_space * exp_p)
                mx_prime_indices = np.arange(mx_prime_start, mx_prime_end)
                mx_indices = mx_prime_indices + R_trans
                row_indices = mx_indices * Q + beta
                col_indices = mx_prime_indices * Q + alpha
                H_ribbon_batch[:, row_indices, col_indices] += amp[:, None]
                
            # 2. Parallel CPU Diagonalization and Vectorized Calculations
            import concurrent.futures
            import os
            
            def _solve_k_cpu(ik):
                try:
                    evals, evecs = la.eigh(H_ribbon_batch[ik], subset_by_index=(lo, hi))
                except TypeError:
                    evals, evecs = la.eigh(H_ribbon_batch[ik])
                    evals = evals[lo:hi+1]
                    evecs = evecs[:, lo:hi+1]
                return ik, evals, evecs

            # eigh releases GIL, enabling true parallelization across cores via thread pools.
            num_cores = os.cpu_count() or 4
            max_workers = min(Nk, num_cores)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_solve_k_cpu, ik) for ik in range(Nk)]
                for fut in concurrent.futures.as_completed(futures):
                    ik, evals, evecs = fut.result()
                    energies[ik] = evals
                    
                    # Vectorized calculations for IPR and edge weights (eliminates slow Python loop)
                    prob_k = np.abs(evecs) ** 2
                    sum_prob = np.sum(prob_k, axis=0)
                    valid_mask = sum_prob > 1e-10
                    ipr_val = np.where(valid_mask, np.sum(prob_k ** 2, axis=0) / (sum_prob ** 2), 0.0)
                    iprs[ik] = ipr_val
                    
                    prob_reshaped = prob_k.reshape((Nx, Q, num_selected))
                    layer_prob = np.sum(prob_reshaped, axis=1)  # (Nx, num_selected)
                    n_edge = min(3, Nx // 2)
                    edge_w = np.sum(layer_prob[:n_edge], axis=0) + np.sum(layer_prob[-n_edge:], axis=0)
                    edge_weights[ik] = edge_w
                
        # Calculate bulk energy bands with the fixed transverse wavevector
        k_pts = np.zeros((Nk, 2))
        if periodic_dir == "x":
            k_pts[:, 0] = ky_space
            k_pts[:, 1] = k_fixed
        else:
            k_pts[:, 0] = k_fixed
            k_pts[:, 1] = ky_space
            
        B = _recip(lattice)
        k_cart = k_pts @ (B / (2.0 * np.pi))
        H_batch = H_k.evaluate_batch(k_cart, lattice.primitive_vectors)
        bulk_energies = np.zeros((Nk, Q))
        for ik in range(Nk):
            bulk_energies[ik] = np.linalg.eigvalsh(H_batch[ik])
            
        return json.dumps({
            "ky_space": ky_space.tolist(),
            "energies": energies.T.tolist(),
            "iprs": iprs.T.tolist(),
            "edge_weights": edge_weights.T.tolist(),
            "bulk_energies": bulk_energies.T.tolist(),
            "lo": lo,
            "error": None
        })
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def get_nanoribbon_state(spec_json, Nx, ky, band_idx, periodic_dir="y"):
    import json
    import traceback
    try:
        from cls_finder.io.parser import parse_input
        import numpy as np
        import scipy.linalg as la
        
        spec = json.loads(spec_json)
        lattice, H_k = parse_input(spec)
        Q = H_k.rows
        d = lattice.dimension
        if d != 2:
            return json.dumps({"error": "나노리본 계산은 2D 격자 모델만 지원합니다."})
            
        # Detect convention
        is_convention_ii = False
        for r in range(H_k.rows):
            for c in range(H_k.cols):
                for exp in H_k.data[r][c].coefs.keys():
                    if any(abs(x - round(x)) > 1e-9 for x in exp):
                        is_convention_ii = True
                        break
                if is_convention_ii:
                    break
            if is_convention_ii:
                break
                
        # Determine index of periodic and real-space dimensions
        if periodic_dir == "x":
            p_idx = 0  # periodic is x
            r_idx = 1  # real space is y
        else:
            p_idx = 1  # periodic is y
            r_idx = 0  # real space is x
            
        # Collect hoppings
        hoppings = []
        for beta in range(Q):
            pos_beta = lattice.orbitals[beta]["position"]
            for alpha in range(Q):
                pos_alpha = lattice.orbitals[alpha]["position"]
                poly = H_k.data[beta][alpha]
                for exp, coef in poly.coefs.items():
                    if abs(coef) < 1e-10:
                        continue
                    if is_convention_ii:
                        R_trans = int(round(exp[r_idx] - pos_beta[r_idx] + pos_alpha[r_idx]))
                    else:
                        R_trans = int(round(exp[r_idx]))
                    hoppings.append((beta, alpha, R_trans, exp[p_idx], coef))
                    
        H_ribbon = np.zeros((Nx * Q, Nx * Q), dtype=complex)
        for beta, alpha, R_trans, exp_p, coef in hoppings:
            mx_prime_start = max(0, -R_trans)
            mx_prime_end = min(Nx, Nx - R_trans)
            if mx_prime_start >= mx_prime_end:
                continue
            
            amp = coef * np.exp(1j * ky * exp_p)
            mx_prime_indices = np.arange(mx_prime_start, mx_prime_end)
            mx_indices = mx_prime_indices + R_trans
            row_indices = mx_indices * Q + beta
            col_indices = mx_prime_indices * Q + alpha
            H_ribbon[row_indices, col_indices] += amp
            
        evals, evecs = la.eigh(H_ribbon)
        
        # Extract the requested state
        if band_idx < 0 or band_idx >= Nx * Q:
            return json.dumps({"error": f"Invalid band index: {band_idx}. Must be between 0 and {Nx*Q - 1}."})
            
        vec = evecs[:, band_idx]
        prob = np.abs(vec) ** 2
        
        # Calculate layer probability density
        layer_prob = np.sum(prob.reshape(Nx, Q), axis=1)
        
        # Calculate site coordinates and their wavefunction amplitudes
        sites_data = []
        vecs = np.array(lattice.primitive_vectors, dtype=float)
        a1 = vecs[0]
        a2 = vecs[1]
        
        for m in range(Nx):
            for q in range(Q):
                pos_q = lattice.orbitals[q]["position"]
                if periodic_dir == "x":
                    wx = pos_q[0] * a1[0] + (m + pos_q[1]) * a2[0]
                    wy = pos_q[0] * a1[1] + (m + pos_q[1]) * a2[1]
                else:
                    wx = (m + pos_q[0]) * a1[0] + pos_q[1] * a2[0]
                    wy = (m + pos_q[0]) * a1[1] + pos_q[1] * a2[1]
                    
                val = vec[m * Q + q]
                sites_data.append({
                    "layer": m,
                    "orbital": q,
                    "label": lattice.orbitals[q]["label"],
                    "x": float(wx),
                    "y": float(wy),
                    "re": float(val.real),
                    "im": float(val.imag),
                    "abs": float(abs(val)),
                    "prob": float(prob[m * Q + q])
                })
                
        ipr = float(np.sum(prob ** 2) / (np.sum(prob) ** 2)) if np.sum(prob) > 1e-10 else 0.0
        n_edge = min(3, Nx // 2)
        edge_w = float(np.sum(layer_prob[:n_edge]) + np.sum(layer_prob[-n_edge:]))
        
        return json.dumps({
            "energy": float(evals[band_idx]),
            "ky": float(ky),
            "band_idx": int(band_idx),
            "ipr": ipr,
            "edge_weight": edge_w,
            "layer_prob": layer_prob.tolist(),
            "sites": sites_data,
            "error": None
        })
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})


def design_flat_band(lattice_spec_json, target_json, E0=0.0, alpha=1.0,
                     dispersive_shape="nn_real", dispersive_strength=0.3,
                     max_retries=8, r_max=None, cls_size=None):
    try:
        from cls_finder.engineer import (
            LatticeSpec,
            SublatticeSpec,
            SingularityTarget,
            DesignTarget,
            design_flat_band as run_design
        )
        
        lattice_spec_data = json.loads(lattice_spec_json)
        target_data = json.loads(target_json)
        
        # Parse LatticeSpec
        prim_vecs = np.array(lattice_spec_data["primitive_vectors"], dtype=float)
        sublattices = [
            SublatticeSpec(label=sub["label"], zeta=float(sub.get("zeta", 0.0)),
                           position=tuple(float(x) for x in sub.get("position", [0.0, 0.0])))
            for sub in lattice_spec_data["sublattices"]
        ]
        lattice_spec = LatticeSpec(primitive_vectors=prim_vecs, sublattices=sublattices)
        
        # Parse DesignTarget
        singularities = [
            SingularityTarget(
                name=s["name"],
                k_frac=tuple(float(x) for x in s["k_frac"]),
                w=int(s["w"])
            )
            for s in target_data["singularities"]
        ]
        target = DesignTarget(C=int(target_data["C"]), singularities=singularities)
        
        # Run the design pipeline
        res = run_design(
            lattice_spec=lattice_spec,
            target=target,
            E0=float(E0),
            alpha=float(alpha),
            dispersive_shape=str(dispersive_shape),
            dispersive_strength=float(dispersive_strength),
            max_retries=int(max_retries),
            r_max=(None if r_max in (None, "", "none") else int(r_max)),
            cls_size=(None if cls_size in (None, "", "none", 0) else int(cls_size)),
            verbose=False
        )

        # Convert hoppings to the format expected by the frontend
        hops_list = []
        for (i, j, R), val in res.hoppings.items():
            if abs(val) < 1e-9:
                continue
            c_val = complex(val)
            hops_list.append({
                "i": int(i),
                "j": int(j),
                "R": [int(R[0]), int(R[1])],
                "t": {"re": float(c_val.real), "im": float(c_val.imag)}
            })
            
        # Serialize LaurentPoly f(k)
        xk_list = []
        for alpha, poly in enumerate(res.x_k):
            poly_terms = []
            for cell, coeff in poly.coefs.items():
                c_coeff = complex(coeff)
                poly_terms.append({
                    "cell": [int(x) for x in cell],
                    "re": float(c_coeff.real),
                    "im": float(c_coeff.imag)
                })
            xk_list.append({
                "sublattice": alpha,
                "label": sublattices[alpha].label,
                "terms": poly_terms,
                "str": str(poly)
            })

        # Return json response
        return json.dumps({
            "success": True,
            "verification": res.verification,
            "hoppings": hops_list,
            "x_k": xk_list,
            "log": res.log,
            "primitive_vectors": prim_vecs.tolist(),
            "orbitals": [
                {"label": sub.label,
                 "position": [float(sub.position[0]), float(sub.position[1])],
                 "sublattice": idx}
                for idx, sub in enumerate(sublattices)
            ],
            "error": None
        }, default=_ser)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()
        })


def _engineer_hops_list(hoppings):
    """[{"i","j","R","t":{"re","im"}}, ...] from a {(i,j,(n,m)): complex} dict
    (cls_finder.engineer hopping-table convention), dropping near-zero terms."""
    hops_list = []
    for (i, j, R), val in hoppings.items():
        if abs(val) < 1e-9:
            continue
        c_val = complex(val)
        hops_list.append({
            "i": int(i),
            "j": int(j),
            "R": [int(R[0]), int(R[1])],
            "t": {"re": float(c_val.real), "im": float(c_val.imag)}
        })
    return hops_list


def _engineer_xk_list(x_k, sublattices):
    """[{"sublattice","label","terms":[{"cell","re","im"}],"str"}, ...] from a
    list[LaurentPoly] (cls_finder.engineer f(k) convention)."""
    xk_list = []
    for alpha, poly in enumerate(x_k):
        poly_terms = []
        for cell, coeff in poly.coefs.items():
            c_coeff = complex(coeff)
            poly_terms.append({
                "cell": [int(x) for x in cell],
                "re": float(c_coeff.real),
                "im": float(c_coeff.imag)
            })
        xk_list.append({
            "sublattice": alpha,
            "label": sublattices[alpha].label,
            "terms": poly_terms,
            "str": str(poly)
        })
    return xk_list


def _engineer_orbitals_json(prim_vecs, sublattices):
    return [
        {"label": sub.label,
         "position": [float(sub.position[0]), float(sub.position[1])],
         "sublattice": idx}
        for idx, sub in enumerate(sublattices)
    ]


def _engineer_parse_lattice_spec(lattice_spec_data):
    from cls_finder.engineer import LatticeSpec, SublatticeSpec
    prim_vecs = np.array(lattice_spec_data["primitive_vectors"], dtype=float)
    sublattices = [
        SublatticeSpec(label=sub["label"], zeta=float(sub.get("zeta", 0.0)),
                       position=tuple(float(x) for x in sub.get("position", [0.0, 0.0])))
        for sub in lattice_spec_data["sublattices"]
    ]
    return LatticeSpec(primitive_vectors=prim_vecs, sublattices=sublattices), prim_vecs, sublattices


def analyze_manual_cls(lattice_spec_json, cls_sites_json, E0=0.0, alpha=1.0,
                        dispersive_shape="nn_real", dispersive_strength=0.3,
                        r_max=None):
    """Manual real-space CLS placement (cls_finder.engineer.manual): build f(k)
    directly from user-clicked sites, construct the always-exactly-flat
    H(k) = E0 I + alpha*H_core(k) + H_core(k) M_tilde(k) H_core(k), auto-discover
    its topology (no pre-specified target), and produce (optionally truncated)
    real-space hoppings -- same response shape as design_flat_band, plus
    valid/trivial/warnings/zeros/chern_report."""
    try:
        from cls_finder.engineer import analyze_manual_cls as run_manual

        lattice_spec_data = json.loads(lattice_spec_json)
        cls_sites_data = json.loads(cls_sites_json)
        lattice_spec, prim_vecs, sublattices = _engineer_parse_lattice_spec(lattice_spec_data)

        cls_sites = [
            {"alpha": int(s["alpha"]), "n": int(s["n"]), "m": int(s["m"]),
             "A": float(s["A"]), "theta": float(s["theta"])}
            for s in cls_sites_data
        ]

        res = run_manual(
            lattice_spec, cls_sites, E0=float(E0), alpha=float(alpha),
            dispersive_shape=str(dispersive_shape),
            dispersive_strength=float(dispersive_strength),
            r_max=(None if r_max in (None, "", "none") else int(r_max)),
        )

        orbitals_json = _engineer_orbitals_json(prim_vecs, sublattices)

        if not res["valid"]:
            return json.dumps({
                "success": True,
                "valid": False,
                "reason": res["reason"],
                "log": res.get("log", []),
                "primitive_vectors": prim_vecs.tolist(),
                "orbitals": orbitals_json,
                "error": None,
            }, default=_ser)

        ift = res["ift"]
        verification = {
            "max_hopping_range": res["max_hopping_range"],
            "natural_hopping_range": res["natural_hopping_range"],
            "truncation_ratio": ift.get("truncation_ratio"),
            "trunc_isolated": res["iso_trunc"].get("isolated"),
            "trunc_numerical_C": res["num_trunc"].get("C"),
            "analytic_C": res["chern_report"]["chern_number"],
            "flat_band_max_dev": res["flat_band_max_dev"],
            "exact_flat": res["exact_flat"],
            "r_max": res["r_max"],
            "r_max_applied": res["r_max_applied"],
            "dispersive_gap_below": res["iso_trunc"].get("gap_below"),
            "dispersive_gap_above": res["iso_trunc"].get("gap_above"),
        }

        return json.dumps({
            "success": True,
            "valid": True,
            "trivial": res["trivial"],
            "warnings": res["warnings"],
            "verification": verification,
            "chern_report": res["chern_report"],
            "zeros": res["zeros"],
            "hoppings": _engineer_hops_list(ift["hoppings"]),
            "x_k": _engineer_xk_list(res["x_k"], sublattices),
            "log": res["log"],
            "primitive_vectors": prim_vecs.tolist(),
            "orbitals": orbitals_json,
            "error": None,
        }, default=_ser)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()
        })


def design_flat_band_explore_stream(lattice_spec_json, target_json, E0=0.0,
                                     dispersive_variants_json=None, offsets_json=None,
                                     rcut_variants_json=None,
                                     max_retries=2, max_candidates=24,
                                     cls_sizes_json=None):
    """Multi-candidate flat-band design explorer (cls_finder.engineer.explore):
    streams one NDJSON {"type":"progress",...} message per (shell_offset0,
    dispersive_shape, dispersive_strength, alpha, R_cut0) attempt as it
    completes, then a final {"type":"done", "ranked":[index,...]} giving the
    deduplicated, best-first ordering of the streamed candidate indices."""
    from cls_finder.engineer import (
        SingularityTarget, DesignTarget, iter_design_attempts, dedupe_and_rank,
    )

    try:
        lattice_spec_data = json.loads(lattice_spec_json)
        target_data = json.loads(target_json)
        lattice_spec, prim_vecs, sublattices = _engineer_parse_lattice_spec(lattice_spec_data)

        singularities = [
            SingularityTarget(
                name=s["name"],
                k_frac=tuple(float(x) for x in s["k_frac"]),
                w=int(s["w"])
            )
            for s in target_data["singularities"]
        ]
        target = DesignTarget(C=int(target_data["C"]), singularities=singularities)

        offsets = [int(x) for x in json.loads(offsets_json)] if offsets_json else list(range(8))
        dispersive_variants = (
            [(str(v[0]), float(v[1]), float(v[2])) for v in json.loads(dispersive_variants_json)]
            if dispersive_variants_json else
            [("nn_real", 0.3, 1.0), ("haldane", 0.3, 1.0), ("combo", 0.25, 1.0), ("none", 0.0, 1.0)]
        )
        rcut_variants = [int(x) for x in json.loads(rcut_variants_json)] if rcut_variants_json else [3]
        # CLS size variants: None (or 0) = auto/minimal shells; a positive int
        # requests a CLS of that spatial extent.
        if cls_sizes_json:
            cls_sizes = [(None if int(x) <= 0 else int(x)) for x in json.loads(cls_sizes_json)]
        else:
            cls_sizes = [None]
        if not cls_sizes:
            cls_sizes = [None]

        def serialize_candidate(cand):
            entry = {
                "index": cand.index, "offset": cand.offset,
                "dispersive_shape": cand.dispersive_shape,
                "dispersive_strength": cand.dispersive_strength,
                "alpha": cand.alpha,
                "R_cut0": cand.R_cut0, "score": cand.score,
                "cls_size": cand.cls_size, "error": cand.error,
            }
            if cand.result is not None:
                res = cand.result
                entry["verification"] = res.verification
                entry["hoppings"] = _engineer_hops_list(res.hoppings)
                entry["x_k"] = _engineer_xk_list(res.x_k, sublattices)
            return entry

        attempts = []
        for cand, idx, total in iter_design_attempts(
                lattice_spec, target, E0=float(E0), offsets=offsets,
                dispersive_variants=dispersive_variants, R_cut0_variants=rcut_variants,
                cls_sizes=cls_sizes,
                max_retries=int(max_retries), max_candidates=int(max_candidates)):
            attempts.append(cand)
            yield json.dumps({"type": "progress", "index": idx, "total": total,
                               "candidate": serialize_candidate(cand)}, default=_ser)

        ranked = dedupe_and_rank(attempts)
        yield json.dumps({
            "type": "done",
            "count": len(attempts),
            "ranked": [c.index for c in ranked],
            "primitive_vectors": prim_vecs.tolist(),
            "orbitals": _engineer_orbitals_json(prim_vecs, sublattices),
        }, default=_ser)

    except Exception as e:
        yield json.dumps({
            "type": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc()
        })
