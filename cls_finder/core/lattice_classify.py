"""
Lattice structure auto-classification module.

Identifies known lattice patterns (Kagome, Lieb, honeycomb, etc.) from
the Lattice object and optional Hamiltonian, providing structured metadata
for visualization and analysis.
"""
import numpy as np
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────
# Known lattice pattern definitions
# ──────────────────────────────────────────────────────────────────────────

_KNOWN_2D_PATTERNS = {
    # key: (bravais_type, sublattice_count, frozenset of rounded sublattice frac positions)
    # Kagome: hexagonal, 3 sublattices at midpoints of primitive vectors
    "kagome": {
        "bravais": "hexagonal",
        "n_sub": 3,
        "positions_set": [
            # Standard Kagome positions (midpoints)
            {(0.5, 0.0), (0.0, 0.5), (0.5, 0.5)},
            # Alternate convention
            {(0.25, 0.0), (0.0, 0.25), (0.25, 0.25)},
        ],
        "description_ko": "카고메 격자",
        "description_en": "Kagome lattice",
        "bipartite": False,
    },
    "honeycomb": {
        "bravais": "hexagonal",
        "n_sub": 2,
        "positions_set": [
            {(0.0, 0.0), (1/3, 1/3)},
            {(0.0, 0.0), (2/3, 1/3)},
            {(1/3, 2/3), (2/3, 1/3)},
            {(0.0, 0.0), (0.5, 0.5)},  # some conventions
        ],
        "description_ko": "벌집 격자 (허니컴)",
        "description_en": "Honeycomb lattice",
        "bipartite": True,
    },
    "lieb": {
        "bravais": "square",
        "n_sub": 3,
        "positions_set": [
            {(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)},
        ],
        "description_ko": "리브 격자 (Lieb)",
        "description_en": "Lieb lattice",
        "bipartite": True,
    },
    "checkerboard": {
        "bravais": "square",
        "n_sub": 2,
        "positions_set": [
            {(0.0, 0.0), (0.5, 0.5)},
        ],
        "description_ko": "체커보드 격자",
        "description_en": "Checkerboard lattice",
        "bipartite": True,
    },
    "bilayer_square": {
        "bravais": "square",
        "n_sub": 2,
        "positions_set": [
            {(0.0, 0.0), (0.5, 0.5)},
            {(0.0, 0.0), (0.5, 0.0)},
        ],
        "description_ko": "이중층 정방 격자",
        "description_en": "Bilayer square lattice",
        "bipartite": True,
    },
    "dice": {
        "bravais": "hexagonal",
        "n_sub": 3,
        "positions_set": [
            {(0.0, 0.0), (1/3, 1/3), (2/3, 2/3)},
        ],
        "description_ko": "다이스 격자 (T₃)",
        "description_en": "Dice (T₃) lattice",
        "bipartite": True,
    },
}


def _round_pos(pos, decimals=4):
    """Round position mod 1 to given decimals for comparison."""
    return tuple(round(x % 1.0, decimals) if abs(x % 1.0) > 1e-6 else 0.0
                 for x in pos)


def _normalize_positions(positions):
    """
    Normalize a set of fractional positions:
    - Shift all positions so the first one is at the origin
    - Apply mod 1 and round
    """
    pos_list = sorted(positions)
    if not pos_list:
        return frozenset()
    offset = np.array(pos_list[0])
    result = set()
    for p in pos_list:
        shifted = np.array(p) - offset
        result.add(_round_pos(shifted))
    return frozenset(result)


def _match_known_pattern(bravais_type, n_sub, sub_positions):
    """
    Try to match sublattice positions against known patterns.
    Returns (pattern_name, pattern_info) or (None, None).
    """
    raw_set = frozenset(_round_pos(p) for p in sub_positions)
    norm_set = _normalize_positions(sub_positions)

    for name, pattern in _KNOWN_2D_PATTERNS.items():
        if pattern["bravais"] != bravais_type:
            continue
        if pattern["n_sub"] != n_sub:
            continue
        for ref_set in pattern["positions_set"]:
            ref_frozen = frozenset(_round_pos(p) for p in ref_set)
            ref_norm = _normalize_positions(ref_set)
            if raw_set == ref_frozen or norm_set == ref_norm:
                return name, pattern

    return None, None


# ──────────────────────────────────────────────────────────────────────────
# Bipartite detection from Hamiltonian
# ──────────────────────────────────────────────────────────────────────────

def _detect_bipartite(lattice, H_k=None):
    """
    Check if the lattice is bipartite:
    - A lattice is bipartite if sublattices can be partitioned into two
      groups A and B such that hopping only connects A↔B (no A↔A or B↔B).

    If H_k is provided, analyze hopping structure.
    Otherwise, use heuristic based on sublattice count and positions.
    """
    n_sub = lattice.sublattice_count

    if n_sub < 2:
        return False, []

    if H_k is None:
        # Without Hamiltonian, only use heuristics
        # 2-sublattice systems are likely bipartite
        if n_sub == 2:
            return True, [[0], [1]]
        return None, []  # Unknown

    # Build sublattice connectivity from Hamiltonian
    # If H_ij(R) != 0 for orbitals i,j on same sublattice, it's not bipartite
    sub_connects = defaultdict(set)  # (sub_i, sub_j) -> set of R vectors

    for ri in range(H_k.rows):
        for ci in range(H_k.cols):
            for exp_tuple, coef in H_k.data[ri][ci].coefs.items():
                if abs(coef) < 1e-10:
                    continue
                sub_ri = lattice.get_sublattice_of(ri)
                sub_ci = lattice.get_sublattice_of(ci)
                # Skip on-site terms (R=0, same orbital)
                if ri == ci and all(e == 0 for e in exp_tuple):
                    continue
                sub_connects[(sub_ri, sub_ci)].add(exp_tuple)

    # Check bipartiteness via graph coloring (BFS)
    adj = defaultdict(set)
    for (si, sj) in sub_connects:
        if si != sj:
            adj[si].add(sj)
            adj[sj].add(si)

    # Check for same-sublattice hopping
    has_same_sub_hopping = any(si == sj for (si, sj) in sub_connects)

    if has_same_sub_hopping:
        return False, []

    # Try 2-coloring
    color = {}
    groups = [[], []]
    is_bipartite = True

    for start in range(n_sub):
        if start in color:
            continue
        # BFS
        queue = [start]
        color[start] = 0
        while queue:
            node = queue.pop(0)
            for neighbor in adj.get(node, []):
                if neighbor not in color:
                    color[neighbor] = 1 - color[node]
                    queue.append(neighbor)
                elif color[neighbor] == color[node]:
                    is_bipartite = False
                    break
            if not is_bipartite:
                break
        if not is_bipartite:
            break

    if is_bipartite:
        for sub_idx, c in color.items():
            groups[c].append(sub_idx)
        return True, groups
    return False, []


# ──────────────────────────────────────────────────────────────────────────
# Coordination number computation
# ──────────────────────────────────────────────────────────────────────────

def _compute_coordination(lattice, H_k):
    """
    Compute coordination number (number of distinct hopping neighbors)
    for each sublattice.
    """
    if H_k is None:
        return [0] * lattice.sublattice_count

    n_sub = lattice.sublattice_count
    coord = [set() for _ in range(n_sub)]

    for ri in range(H_k.rows):
        for ci in range(H_k.cols):
            for exp_tuple, coef in H_k.data[ri][ci].coefs.items():
                if abs(coef) < 1e-10:
                    continue
                # Skip on-site
                if ri == ci and all(e == 0 for e in exp_tuple):
                    continue
                sub_ri = lattice.get_sublattice_of(ri)
                sub_ci = lattice.get_sublattice_of(ci)
                # Neighbor = (target_sublattice, cell_offset)
                coord[sub_ri].add((sub_ci, exp_tuple))

    return [len(s) for s in coord]


# ──────────────────────────────────────────────────────────────────────────
# Main classification function
# ──────────────────────────────────────────────────────────────────────────

def classify_lattice(lattice, H_k=None):
    """
    Perform comprehensive lattice structure classification.

    Parameters:
        lattice: Lattice object
        H_k: MatrixPoly (optional) — Hamiltonian for hopping analysis

    Returns:
        dict with keys:
            bravais_type: str — Bravais lattice type
            known_lattice: str or None — matched known lattice name
            sublattice_count: int
            orbitals_per_sublattice: list[int]
            total_orbitals: int
            is_multi_orbital: bool — any sublattice has >1 orbital
            is_bipartite: bool or None
            bipartite_groups: list[list[int]]
            coordination_numbers: list[int]
            sublattice_positions: list[list[float]]
            sublattice_labels: list[str]
            description_ko: str
            description_en: str
    """
    bravais = lattice.get_bravais_type()
    bravais_type = bravais["type"]

    n_sub = lattice.sublattice_count
    orbs_per_sub = lattice.orbitals_per_sublattice

    # Sublattice positions and labels
    sub_positions = [s["position"].tolist() for s in lattice.sublattices]
    sub_labels = []
    for s in lattice.sublattices:
        # Use the first orbital's label as sublattice label
        first_orb = s["orbital_indices"][0]
        sub_labels.append(lattice.orbitals[first_orb]["label"])

    # Try to match known pattern
    known_name, known_info = None, None
    if lattice.dimension == 2:
        sub_pos_tuples = [tuple(p) for p in sub_positions]
        known_name, known_info = _match_known_pattern(
            bravais_type, n_sub, sub_pos_tuples
        )

    # Bipartite detection
    is_bipartite, bipartite_groups = _detect_bipartite(lattice, H_k)

    # If known pattern provides bipartite info, use it
    if known_info is not None and is_bipartite is None:
        is_bipartite = known_info.get("bipartite", None)

    # Coordination numbers
    coord_nums = _compute_coordination(lattice, H_k)

    # Build descriptions
    desc_parts_ko = []
    desc_parts_en = []

    if known_info:
        desc_parts_ko.append(known_info["description_ko"])
        desc_parts_en.append(known_info["description_en"])
    else:
        bravais_names_ko = {
            "1d_chain": "1D 사슬",
            "hexagonal": "육각격자",
            "square": "정방격자",
            "rectangular": "직사각격자",
            "oblique": "사각격자",
            "fcc": "FCC 격자",
            "bcc": "BCC 격자",
            "simple_cubic": "단순입방격자",
            "general_3d": "일반 3D 격자",
        }
        bravais_names_en = {
            "1d_chain": "1D chain",
            "hexagonal": "Hexagonal",
            "square": "Square",
            "rectangular": "Rectangular",
            "oblique": "Oblique",
            "fcc": "FCC",
            "bcc": "BCC",
            "simple_cubic": "Simple cubic",
            "general_3d": "General 3D",
        }
        desc_parts_ko.append(bravais_names_ko.get(bravais_type, bravais_type))
        desc_parts_en.append(bravais_names_en.get(bravais_type, bravais_type))

    desc_parts_ko.append(f"서브라티스 {n_sub}개")
    desc_parts_en.append(f"{n_sub} sublattice(s)")

    if lattice.is_multi_orbital:
        max_orb = max(orbs_per_sub)
        desc_parts_ko.append(f"다중 오비탈 (최대 {max_orb}개/서브라티스)")
        desc_parts_en.append(f"multi-orbital (max {max_orb}/sublattice)")

    desc_parts_ko.append(f"총 오비탈 {lattice.num_orbitals}개")
    desc_parts_en.append(f"{lattice.num_orbitals} orbital(s) total")

    if is_bipartite is True:
        desc_parts_ko.append("이분 격자 (bipartite)")
        desc_parts_en.append("bipartite")
    elif is_bipartite is False:
        desc_parts_ko.append("비이분 격자")
        desc_parts_en.append("non-bipartite")

    return {
        "bravais_type": bravais_type,
        "known_lattice": known_name,
        "sublattice_count": n_sub,
        "orbitals_per_sublattice": orbs_per_sub,
        "total_orbitals": lattice.num_orbitals,
        "is_multi_orbital": lattice.is_multi_orbital,
        "is_bipartite": is_bipartite,
        "bipartite_groups": bipartite_groups,
        "coordination_numbers": coord_nums,
        "sublattice_positions": sub_positions,
        "sublattice_labels": sub_labels,
        "description_ko": " | ".join(desc_parts_ko),
        "description_en": " | ".join(desc_parts_en),
    }
