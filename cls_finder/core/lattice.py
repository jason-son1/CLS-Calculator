import numpy as np
from collections import defaultdict


class Lattice:
    """
    Manages the crystal lattice structure, dimension, primitive vectors,
    orbital details, and sublattice grouping.

    Key concepts:
      - **Orbital**: A single degree of freedom in the unit cell, identified by
        an index q ∈ [0, Q). Each orbital has a label (e.g., "px", "A") and a
        fractional position within the unit cell.
      - **Sublattice**: A physical site in the unit cell. Multiple orbitals can
        reside on the same sublattice site (e.g., px, py, pz at [0,0,0]).
        The sublattice is identified by an integer index.
      - **Bipartite**: The lattice is bipartite if sublattices can be divided
        into two groups such that hopping only connects different groups.
    """

    # Tolerance for grouping orbitals into the same sublattice by position
    _POS_TOL = 1e-6

    def __init__(self, dimension, primitive_vectors, orbitals):
        """
        dimension: int (1, 2, or 3)
        primitive_vectors: list of list of float, shape (dimension, spatial_dimension)
        orbitals: list of dict, each with 'label' and 'position' (fractional coordinates)
                  Optional: 'sublattice' (int) to explicitly assign sublattice index
        """
        self.dimension = int(dimension)
        self.primitive_vectors = np.array(primitive_vectors, dtype=float)

        # Verify shape of primitive_vectors
        if self.primitive_vectors.shape[0] != self.dimension:
            raise ValueError(
                f"Primitive vectors must have shape (d, spatial_dim), "
                f"got {self.primitive_vectors.shape}"
            )

        self.spatial_dim = self.primitive_vectors.shape[1]

        # Parse orbitals
        self.orbitals = []
        for q, orb in enumerate(orbitals):
            label = orb.get("label", f"O{q+1}")
            pos = np.array(orb["position"], dtype=float)
            if len(pos) != self.dimension:
                raise ValueError(
                    f"Orbital {label} position must be fractional "
                    f"coordinates of length {self.dimension}"
                )
            entry = {
                "index": q,
                "label": label,
                "position": pos,
            }
            # Store explicit sublattice hint if provided
            if "sublattice" in orb:
                entry["_sublattice_hint"] = int(orb["sublattice"])
            self.orbitals.append(entry)

        self.num_orbitals = len(self.orbitals)

        # Build sublattice grouping
        self._build_sublattices()

    # ──────────────────────────────────────────────────────────────────────
    # Sublattice auto-detection / explicit assignment
    # ──────────────────────────────────────────────────────────────────────

    def _build_sublattices(self):
        """
        Group orbitals into sublattices.

        Strategy:
          1. If ALL orbitals have an explicit '_sublattice_hint', use those.
          2. Otherwise, group orbitals by fractional position proximity.
        """
        all_have_hint = all("_sublattice_hint" in orb for orb in self.orbitals)

        if all_have_hint:
            # Use explicit hints
            groups = defaultdict(list)
            for orb in self.orbitals:
                groups[orb["_sublattice_hint"]].append(orb["index"])
            # Re-index sublattices contiguously
            sorted_keys = sorted(groups.keys())
            self._sublattice_map = {}  # orbital_index -> sublattice_index
            self._sublattices = []
            for new_idx, key in enumerate(sorted_keys):
                orbital_indices = groups[key]
                # Use position of first orbital as sublattice position
                pos = self.orbitals[orbital_indices[0]]["position"].copy()
                self._sublattices.append({
                    "index": new_idx,
                    "position": pos,
                    "orbital_indices": sorted(orbital_indices),
                })
                for oi in orbital_indices:
                    self._sublattice_map[oi] = new_idx
        else:
            # Auto-detect by position proximity
            self._sublattice_map = {}
            self._sublattices = []
            for orb in self.orbitals:
                q = orb["index"]
                pos = orb["position"]
                matched = False
                for sub in self._sublattices:
                    if np.allclose(pos, sub["position"], atol=self._POS_TOL):
                        sub["orbital_indices"].append(q)
                        self._sublattice_map[q] = sub["index"]
                        matched = True
                        break
                if not matched:
                    new_idx = len(self._sublattices)
                    self._sublattices.append({
                        "index": new_idx,
                        "position": pos.copy(),
                        "orbital_indices": [q],
                    })
                    self._sublattice_map[q] = new_idx

        # Annotate each orbital with sublattice info
        for orb in self.orbitals:
            q = orb["index"]
            sub_idx = self._sublattice_map[q]
            orb["sublattice_index"] = sub_idx
            sub = self._sublattices[sub_idx]
            orbs_in_sub = sub["orbital_indices"]
            orb["orbital_index_in_sublattice"] = orbs_in_sub.index(q)

    # ──────────────────────────────────────────────────────────────────────
    # Properties
    # ──────────────────────────────────────────────────────────────────────

    @property
    def sublattice_count(self):
        """Number of distinct sublattice sites in the unit cell."""
        return len(self._sublattices)

    @property
    def sublattices(self):
        """
        List of sublattice dicts:
          [{"index": int, "position": ndarray, "orbital_indices": [int, ...]}, ...]
        """
        return self._sublattices

    @property
    def is_multi_orbital(self):
        """True if any sublattice has more than one orbital."""
        return any(len(s["orbital_indices"]) > 1 for s in self._sublattices)

    @property
    def orbitals_per_sublattice(self):
        """List of orbital counts per sublattice."""
        return [len(s["orbital_indices"]) for s in self._sublattices]

    def get_sublattice_of(self, orbital_idx):
        """Return sublattice index for the given orbital."""
        return self._sublattice_map[orbital_idx]

    # ──────────────────────────────────────────────────────────────────────
    # Coordinate helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_cartesian_position(self, cell_indices, orbital_idx):
        """
        Calculate Cartesian coordinates for orbital_idx in the unit cell cell_indices.
        cell_indices: list/tuple of length self.dimension
        orbital_idx: int (0 to num_orbitals-1)
        """
        cell_indices = np.array(cell_indices, dtype=float)
        orb = self.orbitals[orbital_idx]

        # Site position in fractional coordinates: cell_indices + orb_fractional_pos
        frac_pos = cell_indices + orb["position"]

        # Convert to Cartesian: Sum frac_pos[l] * primitive_vectors[l]
        cart_pos = np.zeros(self.spatial_dim)
        for l in range(self.dimension):
            cart_pos += frac_pos[l] * self.primitive_vectors[l]

        return cart_pos

    def get_cartesian_sublattice_position(self, cell_indices, sublattice_idx):
        """
        Calculate Cartesian coordinates for the sublattice site
        (ignoring per-orbital offsets).
        """
        cell_indices = np.array(cell_indices, dtype=float)
        sub = self._sublattices[sublattice_idx]
        frac_pos = cell_indices + sub["position"]
        cart_pos = np.zeros(self.spatial_dim)
        for l in range(self.dimension):
            cart_pos += frac_pos[l] * self.primitive_vectors[l]
        return cart_pos

    # ──────────────────────────────────────────────────────────────────────
    # Lattice geometry helpers
    # ──────────────────────────────────────────────────────────────────────

    def get_bravais_type(self):
        """
        Determine the Bravais lattice type from primitive vectors.
        Returns a dict with 'type' and 'angles', 'lengths'.
        """
        d = self.dimension
        A = self.primitive_vectors

        lengths = [np.linalg.norm(A[i]) for i in range(d)]
        angles = {}
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    cos_a = np.dot(A[i], A[j]) / (lengths[i] * lengths[j])
                    angles[(i, j)] = np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))

        if d == 1:
            return {"type": "1d_chain", "lengths": lengths, "angles": angles}
        elif d == 2:
            ang = angles[(0, 1)]
            l1, l2 = lengths
            eq_len = abs(l1 - l2) < 1e-3
            if eq_len and (abs(ang - 60) < 1.5 or abs(ang - 120) < 1.5):
                btype = "hexagonal"
            elif eq_len and abs(ang - 90) < 1.5:
                btype = "square"
            elif abs(ang - 90) < 1.5:
                btype = "rectangular"
            else:
                btype = "oblique"
            return {"type": btype, "lengths": lengths, "angles": angles}
        else:  # d == 3
            l1, l2, l3 = lengths
            a12, a13, a23 = angles[(0, 1)], angles[(0, 2)], angles[(1, 2)]
            eq_all = abs(l1 - l2) < 1e-3 and abs(l2 - l3) < 1e-3
            if eq_all and all(abs(a - 60) < 2 for a in [a12, a13, a23]):
                btype = "fcc"
            elif eq_all and all(abs(a - 109.47) < 2 for a in [a12, a13, a23]):
                btype = "bcc"
            elif eq_all and all(abs(a - 90) < 2 for a in [a12, a13, a23]):
                btype = "simple_cubic"
            else:
                btype = "general_3d"
            return {"type": btype, "lengths": lengths, "angles": angles}

    def __repr__(self):
        return (
            f"Lattice(d={self.dimension}, orbitals={self.num_orbitals}, "
            f"sublattices={self.sublattice_count}, "
            f"spatial_dim={self.spatial_dim})"
        )
