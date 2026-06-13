"""
Robust Boundary Mode (RBM) — core numerics.

Given a single minimal CLS amplitude tensor A_{0, dR, q} (from CLS_Finder) and an
open-boundary finite lattice of N_1 x ... x N_d unit cells, this places one copy
of the CLS at every unit cell, drops the copies that would spill outside the open
boundary, and sums them with the per-cell phase c_R = exp(-i k0 . R).

For a *singular* flat band the bulk identity  sum_R c_R |chi_R> = 0  holds on the
torus, so after truncation the interior amplitude cancels to ~0 and a finite
amplitude survives only within one CLS-radius of the edge: the Robust Boundary
Mode. For a *nonsingular* band the same construction has no topological
protection — removing a single CLS copy shatters the pattern.

Everything is plain numpy so the module runs unchanged under Pyodide.
"""
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# CLS amplitude -> integer unit-cell offsets
# ──────────────────────────────────────────────────────────────────────────────
def _is_convention_ii(A_0_R):
    """True if any amplitude exponent is non-integer (Cartesian / Convention II)."""
    for q in A_0_R:
        for exp in A_0_R[q]:
            if any(abs(x - round(x)) > 1e-9 for x in exp):
                return True
    return False


def cls_amplitude_to_cells(lattice, A_0_R, tol=1e-12):
    """
    Normalize a CLS amplitude dict {q: {exp: coef}} to integer unit-cell offsets:
    returns {q: {cell_offset_tuple(int): complex_amp}}.

    Convention I (integer exponents, e.g. Kagome/Lieb): the exponent *is* the
    cell offset dR. Convention II (Cartesian exponents): subtract the orbital's
    Cartesian offset and map back to a fractional unit cell, then round. This
    mirrors web/bridge's _exp_to_cell so the RBM agrees with the CLS real-space
    plot.
    """
    d = lattice.dimension
    A_mat = np.array(lattice.primitive_vectors, dtype=float)        # (d, spatial)
    conv_ii = _is_convention_ii(A_0_R)
    if conv_ii:
        A_inv = (np.linalg.inv(A_mat) if A_mat.shape[0] == A_mat.shape[1]
                 else np.linalg.pinv(A_mat))

    out = {}
    for q, terms in A_0_R.items():
        tau = np.array(lattice.orbitals[q]["position"], dtype=float)
        cellmap = {}
        for exp, coef in terms.items():
            c = complex(coef)
            if abs(c) < tol:
                continue
            if conv_ii:
                r_cart = np.array(exp, dtype=float) - tau @ A_mat
                r_frac = r_cart @ A_inv
                cell = tuple(int(round(x)) for x in r_frac)
            else:
                cell = tuple(int(round(x)) for x in exp)
            cellmap[cell] = cellmap.get(cell, 0j) + c
        if cellmap:
            out[q] = cellmap
    return out


def cls_support_radius(cells_by_q):
    """Max Chebyshev radius (in unit cells) of the CLS support around its center.
    Determines how far the boundary mode penetrates (its skin depth)."""
    r = 0
    for q, cellmap in cells_by_q.items():
        for cell in cellmap:
            r = max(r, max(abs(int(c)) for c in cell) if cell else 0)
    return r


# ──────────────────────────────────────────────────────────────────────────────
# Step 1-3: open-boundary translation & accumulation
# ──────────────────────────────────────────────────────────────────────────────
def calculate_boundary_mode(lattice, A_0_R, system_size, k_singularity=None):
    """
    Place a CLS copy at every unit cell of an N_1 x ... x N_d open-boundary
    lattice and sum, dropping copies that fall outside the boundary.

    Parameters
    ----------
    lattice        : Lattice
    A_0_R          : dict {q: {exp: coef}}  — a single minimal CLS amplitude
    system_size    : tuple/list of int, length d
    k_singularity  : Cartesian momentum k0 of the band-crossing singularity (or
                     None / zeros for the gamma-point / nonsingular case). The
                     per-cell phase is c_R = exp(-i k0 . R_cartesian).

    Returns
    -------
    dict with:
      "psi"          : {(cell_tuple, q): complex}   — Psi_Edge (nonzero sites only)
      "cells_by_q"   : normalized integer-offset CLS
      "system_size"  : tuple(int)
      "support_radius": int
      "k_singularity": list(float) or None
    """
    d = lattice.dimension
    N = tuple(int(n) for n in system_size)
    if len(N) != d:
        raise ValueError(f"system_size must have length d={d}, got {len(N)}")

    cells_by_q = cls_amplitude_to_cells(lattice, A_0_R)
    support_radius = cls_support_radius(cells_by_q)

    A_mat = np.array(lattice.primitive_vectors, dtype=float)   # (d, spatial)
    k0 = (np.zeros(A_mat.shape[1]) if k_singularity is None
          else np.array(k_singularity, dtype=float).ravel())
    if k0.shape[0] != A_mat.shape[1]:
        # tolerate a fractional/short k0 by zero-padding/truncating
        k0 = np.resize(k0, A_mat.shape[1])

    # iterate every interior unit cell R = (n_1, ..., n_d)
    ranges = [range(N[l]) for l in range(d)]
    psi = {}
    from itertools import product as iproduct
    for R in iproduct(*ranges):
        R_cart = np.array(R, dtype=float) @ A_mat
        phase = np.exp(-1j * float(np.dot(k0, R_cart)))
        for q, cellmap in cells_by_q.items():
            for dR, amp in cellmap.items():
                target = tuple(R[l] + dR[l] for l in range(d))
                if all(0 <= target[l] < N[l] for l in range(d)):
                    key = (target, q)
                    psi[key] = psi.get(key, 0j) + phase * amp

    return {
        "psi": psi,
        "cells_by_q": cells_by_q,
        "system_size": N,
        "support_radius": int(support_radius),
        "k_singularity": (None if k_singularity is None else [float(x) for x in k0]),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: bulk-cancellation validation
# ──────────────────────────────────────────────────────────────────────────────
def verify_bulk_cancellation(result, tol=1e-10):
    """
    Confirm destructive interference: every site deeper than one CLS-radius from
    *all* boundaries must have ~0 amplitude.

    Returns dict {passed, max_bulk_amp, sum_bulk_amp, n_bulk_sites,
                  n_edge_sites, support_radius}.
    """
    psi = result["psi"]
    N = result["system_size"]
    d = len(N)
    rad = result["support_radius"]

    def _is_deep_bulk(cell):
        return all(rad <= cell[l] <= N[l] - 1 - rad for l in range(d))

    max_bulk = 0.0
    sum_bulk = 0.0
    n_bulk = 0
    n_edge = 0
    for (cell, q), amp in psi.items():
        a = abs(amp)
        if _is_deep_bulk(cell):
            n_bulk += 1
            sum_bulk += a
            max_bulk = max(max_bulk, a)
        elif a > tol:
            n_edge += 1

    # A deep-bulk region must exist for the check to be meaningful.
    has_bulk = all(N[l] - 2 * rad >= 1 for l in range(d))
    passed = bool(has_bulk and max_bulk < tol)
    return {
        "passed": passed,
        "has_bulk_region": bool(has_bulk),
        "max_bulk_amp": float(max_bulk),
        "sum_bulk_amp": float(sum_bulk),
        "n_bulk_sites": int(n_bulk),
        "n_edge_sites": int(n_edge),
        "support_radius": int(rad),
        "tol": tol,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Skin-depth profile
# ──────────────────────────────────────────────────────────────────────────────
def skin_depth_profile(result, axis=0, transverse_bulk=True, amp_tol=1e-12):
    """
    Amplitude penetration profile along `axis`: for each layer index n, the max
    |psi| over the other axes and all orbitals.

    For a *closed* boundary band, naively taking the max over the full transverse
    range always hits the perpendicular edges, giving a flat line. With
    transverse_bulk=True (default) the transverse axes are restricted to the deep
    interior, isolating this axis's two edges — the profile is then the
    step-function the RBM predicts: finite within `support_radius` layers of each
    edge, exactly 0 in between (no exponential tail).

    Returns dict {axis, distance: [int], max_amp: [float], transverse_bulk}.
    """
    psi = result["psi"]
    N = result["system_size"]
    d = len(N)
    rad = result["support_radius"]
    Na = N[axis]
    layer_max = np.zeros(Na)
    for (cell, q), amp in psi.items():
        n = cell[axis]
        if not (0 <= n < Na):
            continue
        if transverse_bulk:
            in_bulk = all(rad < cell[l] < N[l] - 1 - rad
                          for l in range(d) if l != axis)
            if not in_bulk:
                continue
        layer_max[n] = max(layer_max[n], abs(amp))

    return {"axis": int(axis), "distance": list(range(Na)),
            "max_amp": [float(v) for v in layer_max],
            "transverse_bulk": bool(transverse_bulk)}


# ──────────────────────────────────────────────────────────────────────────────
# Real-space sites (for plotting)
# ──────────────────────────────────────────────────────────────────────────────
def boundary_mode_sites(lattice, result, amp_tol=1e-9, include_empty=False):
    """
    Flatten Psi_Edge into a list of site dicts with Cartesian positions, complex
    amplitude, magnitude and phase. By default only nonzero sites are returned
    (the markers); set include_empty=True to also emit zero-amplitude lattice
    sites (the faint background grid).
    """
    psi = result["psi"]
    N = result["system_size"]
    Q = lattice.num_orbitals
    d = lattice.dimension
    from itertools import product as iproduct

    sites = []
    keys = set(psi.keys())
    if include_empty:
        for cell in iproduct(*[range(N[l]) for l in range(d)]):
            for q in range(Q):
                keys.add((cell, q))

    for (cell, q) in keys:
        amp = psi.get((cell, q), 0j)
        a = abs(amp)
        if (not include_empty) and a < amp_tol:
            continue
        pos = lattice.get_cartesian_position(cell, q)
        sites.append({
            "cell": list(cell),
            "orbital": int(q),
            "label": lattice.orbitals[q]["label"],
            "x": float(pos[0]),
            "y": float(pos[1]) if len(pos) >= 2 else 0.0,
            "z": float(pos[2]) if len(pos) >= 3 else 0.0,
            "amp_re": float(amp.real),
            "amp_im": float(amp.imag),
            "abs": float(a),
            "phase": float(np.angle(amp)),
            "nonzero": bool(a >= amp_tol),
        })
    sites.sort(key=lambda s: (s["cell"], s["orbital"]))
    return sites


# ──────────────────────────────────────────────────────────────────────────────
# Defect robustness (Test Case control group)
# ──────────────────────────────────────────────────────────────────────────────
def boundary_mode_with_defect(lattice, A_0_R, system_size, k_singularity=None,
                              omit_centers=()):
    """
    Same as calculate_boundary_mode, but the CLS copies whose *center* cell is in
    `omit_centers` are left out (a boundary/bulk defect).

    For a singular flat band this is local-robust: omitting one interior copy
    only spoils cancellation within one CLS-radius of that cell; the rest of the
    bulk stays 0 and the far boundary band is untouched. For a nonsingular band
    the construction never cancelled to begin with, so it is fragile.
    """
    d = lattice.dimension
    N = tuple(int(n) for n in system_size)
    cells_by_q = cls_amplitude_to_cells(lattice, A_0_R)
    support_radius = cls_support_radius(cells_by_q)
    A_mat = np.array(lattice.primitive_vectors, dtype=float)
    k0 = (np.zeros(A_mat.shape[1]) if k_singularity is None
          else np.resize(np.array(k_singularity, dtype=float).ravel(), A_mat.shape[1]))
    omit = {tuple(int(x) for x in c) for c in omit_centers}

    from itertools import product as iproduct
    psi = {}
    for R in iproduct(*[range(N[l]) for l in range(d)]):
        if R in omit:
            continue
        R_cart = np.array(R, dtype=float) @ A_mat
        phase = np.exp(-1j * float(np.dot(k0, R_cart)))
        for q, cellmap in cells_by_q.items():
            for dR, amp in cellmap.items():
                target = tuple(R[l] + dR[l] for l in range(d))
                if all(0 <= target[l] < N[l] for l in range(d)):
                    key = (target, q)
                    psi[key] = psi.get(key, 0j) + phase * amp
    return {"psi": psi, "cells_by_q": cells_by_q, "system_size": N,
            "support_radius": int(support_radius),
            "k_singularity": (None if k_singularity is None else [float(x) for x in k0])}


# ──────────────────────────────────────────────────────────────────────────────
# High-level bundle
# ──────────────────────────────────────────────────────────────────────────────
def compute_rbm(lattice, A_0_R, system_size, k_singularity=None,
                singular=None, bulk_tol=1e-10, amp_tol=1e-9):
    """
    Full RBM pipeline: translate+sum (Steps 1-3), validate bulk cancellation
    (Step 4), build skin-depth profile and the real-space site list.

    Returns a single dict bundling psi, validation, profiles, sites and metadata
    — convenient for both the matplotlib visualizer and the web bridge.
    """
    result = calculate_boundary_mode(lattice, A_0_R, system_size, k_singularity)
    validation = verify_bulk_cancellation(result, tol=bulk_tol)
    sites = boundary_mode_sites(lattice, result, amp_tol=amp_tol)
    profiles = {ax: skin_depth_profile(result, axis=ax)
                for ax in range(lattice.dimension)}

    max_amp = max((s["abs"] for s in sites), default=0.0)
    return {
        "system_size": list(result["system_size"]),
        "support_radius": result["support_radius"],
        "k_singularity": result["k_singularity"],
        "singular": (None if singular is None else bool(singular)),
        "validation": validation,
        "skin_depth": profiles,
        "sites": sites,
        "n_nonzero_sites": sum(1 for s in sites if s["nonzero"]),
        "max_amplitude": float(max_amp),
        "_result": result,   # raw psi for downstream (robustness/defect tests)
    }
