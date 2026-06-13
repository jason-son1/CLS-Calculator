"""
Analytical CLS representation analysis.

The global-phase gauge freedom  A_q → A_q · e^{iθ}  (θ ∈ [0, 2π))
is treated as a single continuous parameter — phase rotation and
amplitude sign are the same freedom, explored analytically together.

Key principles
--------------
· Max-real θ*         : closed-form  θ* = −arg(Σ A²) / 2
· Discrete π/n grids  : detected via pairwise phase-difference analysis
                        (not by scanning; analytically exact)
· Sign ambiguity      : resolved by  Σ Re(A·e^{iθ}) > 0
                        ("centre-of-mass" convention — no ad-hoc flipping)
"""
import numpy as np


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _collect_amps(A_0_R):
    """Flatten A_0_R → list of non-zero complex amplitudes."""
    out = []
    for q_data in A_0_R.values():
        for coef in q_data.values():
            c = complex(coef)
            if abs(c) > 1e-12:
                out.append(c)
    return out


def _apply_phase(A_0_R, theta):
    """Return new A_0_R with every amplitude multiplied by e^{iθ}."""
    f = complex(np.exp(1j * theta))
    return {q: {cell: complex(coef) * f for cell, coef in qd.items()}
            for q, qd in A_0_R.items()}


def _realness(A_0_R):
    """Σ Re² / Σ |A|²  ∈ [0, 1]."""
    amps = _collect_amps(A_0_R)
    if not amps:
        return 1.0
    total = sum(abs(c) ** 2 for c in amps)
    return sum(c.real ** 2 for c in amps) / total if total > 1e-15 else 1.0


def _phase_pattern(A_0_R, tol=0.06):
    """
    Detect if the relative phases within each orbital are multiples of π/n for n ∈ {1,2,3,4,6}.
    Also checks if the absolute phases can be aligned to the grid.
    """
    # 1. First check within-orbital relative phases (gauge-invariant)
    # We want to find the smallest denom n for which all relative phases in all orbitals are quantized
    active_orbitals = 0
    best_denom = 1
    
    for q, q_data in A_0_R.items():
        amps = [complex(coef) for coef in q_data.values() if abs(coef) > 1e-10]
        if len(amps) < 2:
            if len(amps) == 1:
                active_orbitals += 1
            continue
        active_orbitals += 1
        
        # Find reference (largest magnitude)
        ref_idx = int(np.argmax([abs(c) for c in amps]))
        ref_amp = amps[ref_idx]
        
        # Relative phases
        rel_phases = np.array([np.angle(c / ref_amp) for c in amps])
        
        # Check quantization for this orbital
        orb_best_denom = 6
        for denom in (1, 2, 3, 4, 6):
            unit = np.pi / denom
            quantized = np.round(rel_phases / unit) * unit
            if np.max(np.abs(rel_phases - quantized)) < tol:
                orb_best_denom = denom
                break
        best_denom = max(best_denom, orb_best_denom)
        
    # 2. Check if the absolute phases of the orbitals are also quantized
    # If so, we report the global pattern.
    amps_all = [c for c in _collect_amps(A_0_R) if abs(c) > 1e-10]
    if not amps_all:
        return None
    phases_all = np.array([np.angle(c) for c in amps_all])
    
    global_denom = None
    for denom in (1, 2, 3, 4, 6):
        unit = np.pi / denom
        quantized = np.round(phases_all / unit) * unit
        if np.max(np.abs(phases_all - quantized)) < tol:
            global_denom = denom
            break
            
    if global_denom is not None:
        return 'π' if global_denom == 1 else f'π/{global_denom}'
        
    # If not globally aligned, but relative phases within each orbital are quantized
    if active_orbitals > 0:
        return 'π' if best_denom == 1 else f'π/{best_denom}'
        
    return None


def _symmetry_score(A_0_R):
    """
    Symmetry score calculated per orbital and then averaged.
    For each orbital, we normalize its non-zero amplitudes by its own max magnitude,
    and find the fraction of pairs that are equal (within 2% tolerance).
    This emphasizes that the components of the same orbital/site type are the same component,
    and properly judges their amplitude relationships in real space.
    """
    orbital_scores = []
    for q, q_data in A_0_R.items():
        # Collect non-zero amplitudes for this orbital
        amps = [complex(coef) for coef in q_data.values() if abs(coef) > 1e-12]
        n = len(amps)
        if n == 0:
            continue
        if n == 1:
            orbital_scores.append(1.0)
            continue
            
        mags = [abs(c) for c in amps]
        mx = max(mags)
        if mx < 1e-12:
            orbital_scores.append(1.0)
            continue
            
        norm = [m / mx for m in mags]
        pairs = sum(1 for i in range(n) for j in range(i + 1, n)
                    if abs(norm[i] - norm[j]) < 0.02)
        total_pairs = n * (n - 1) / 2
        orbital_scores.append(pairs / total_pairs)
        
    if not orbital_scores:
        return 1.0
    return float(np.mean(orbital_scores))


def _display_mode(realness, pp):
    if realness > 0.97 or pp == 'π':
        return 'amplitude'
    if pp in ('π/2', 'π/3', 'π/4', 'π/6'):
        return 'phase'
    return 'complex'


def _similar(A1, A2, tol=0.05):
    def mags(A):
        return sorted(abs(c) for c in _collect_amps(A))
    m1, m2 = mags(A1), mags(A2)
    if len(m1) != len(m2):
        return False
    if not m1:
        return True
    scale = max(max(m1), 1e-10)
    return max(abs(a - b) for a, b in zip(m1, m2)) < tol * scale


# ── Analytical core ───────────────────────────────────────────────────────────

def _select_sign(theta, amps):
    """
    Choose between {θ, θ+π} so that  Σ Re(A · e^{iθ}) > 0.

    This is the unified "centre-of-mass" convention: phase rotation and
    amplitude sign are a single parameter θ ∈ [0, 2π), and we fix the
    residual Z₂ ambiguity by requiring the mean amplitude to be positive.

    Fallback when Σ A ≈ 0 (symmetric cancellation): make the amplitude
    with the largest |Re(A·e^{iθ})| positive.
    """
    S = sum(amps)
    if abs(S) < 1e-12:
        re_vals = [abs(c) * np.cos(np.angle(c) + theta) for c in amps]
        best = max(range(len(re_vals)), key=lambda i: abs(re_vals[i]))
        if re_vals[best] < 0:
            theta += np.pi
    else:
        # Re(e^{iθ} · S) = |S| cos(arg(S) + θ)
        if np.cos(float(np.angle(S)) + theta) < 0:
            theta += np.pi
    return float(theta % (2 * np.pi))


def _theta_max_real(A_0_R):
    """
    Closed-form θ ∈ [0, 2π) that maximises Σ |Re(A · e^{iθ})|².

    Derivation
    ----------
    R(θ) = ½ Σ|A|² + ½ Re[e^{2iθ} · Σ A²]

    Maximum when e^{2iθ} · Σ A² is real-positive:
        θ* = −arg(Σ A²) / 2   (unique mod π)

    The two solutions {θ*, θ*+π} have identical |Re|² sums but opposite
    amplitude signs.  The sign is resolved analytically by _select_sign
    (Σ Re > 0), treating rotation and sign as one continuous parameter.
    """
    amps = _collect_amps(A_0_R)
    if not amps:
        return 0.0
    S2 = sum(c ** 2 for c in amps)
    theta = 0.0 if abs(S2) < 1e-15 else float(-np.angle(S2) / 2)
    return _select_sign(theta, amps)


def _discrete_phase_thetas(A_0_R, tol_rad=0.10):
    """
    Analytically detect discrete π/n phase structures and return optimal θ.

    Algorithm (no scanning)
    -----------------------
    For each denominator n ∈ {2, 3, 4, 6}  (grid unit u = π/n):

    1. Compute all pairwise phase differences Δ_qr = φ_q − φ_r.
    2. If  max |Δ_qr − round(Δ_qr/u) · u| > tol_rad  → structure absent.
    3. Otherwise enumerate the 2n grid-aligned rotations in one full 2π period:
           θ_k = k · u − φ_ref ,   k = 0 … 2n−1
       and pick the k maximising  Σ |A_q|² cos²(φ_q + θ_k)  (max realness
       within the discrete grid).
    4. Resolve sign via Σ Re > 0.

    Returns
    -------
    list of (theta: float, label: str)  for each detected structure.
    """
    amps = _collect_amps(A_0_R)
    sig = [c for c in amps if abs(c) > 1e-10]
    if len(sig) < 2:
        return []

    mags = np.array([abs(c) for c in sig])
    phases = np.array([float(np.angle(c)) for c in sig])

    results = []
    for denom in (2, 3, 4, 6):
        unit = np.pi / denom

        # Step 1–2: pairwise difference check
        diffs = (phases[:, None] - phases[None, :]).ravel()
        residuals = np.abs(diffs - np.round(diffs / unit) * unit)
        if residuals.max() > tol_rad:
            continue  # π/n structure not present

        # Step 3: choose best grid-aligned θ (maximise Σ Re²)
        phi_ref = phases[int(np.argmax(mags))]
        best_theta, best_r2 = 0.0, -1.0
        for k in range(2 * denom):
            t = float((k * unit - phi_ref) % (2 * np.pi))
            r2 = float(np.sum((mags * np.cos(phases + t)) ** 2))
            if r2 > best_r2:
                best_r2, best_theta = r2, t

        # Step 4: sign selection
        best_theta = _select_sign(best_theta, sig)
        results.append((best_theta, f'π/{denom}'))

    return results


def _calculate_representation_score(A_0_R, theta):
    """
    Calculate the quality score of a representation rotated by theta.
    Uses the new within-orbital symmetry score and within-orbital phase pattern analysis.
    """
    A_rot = _apply_phase(A_0_R, theta)
    
    # 1. Realness
    r = _realness(A_rot)
    
    # 2. Symmetry score (within-orbital)
    sym = _symmetry_score(A_rot)
    
    # 3. Phase quantization score
    # Check if absolute phases are close to a grid
    amps = [c for c in _collect_amps(A_rot) if abs(c) > 1e-10]
    if not amps:
        quant_score = 1.0
    else:
        phases = np.array([np.angle(c) for c in amps])
        best_dev = 1.0
        for denom in (1, 2, 3, 4, 6):
            unit = np.pi / denom
            quantized = np.round(phases / unit) * unit
            max_dev = np.max(np.abs(phases - quantized))
            if max_dev < best_dev:
                best_dev = max_dev
        # Map deviation to a score in [0, 1]
        quant_score = float(np.exp(-5.0 * best_dev**2))
        
    # Combine scores: 40% symmetry, 30% realness, 30% phase quantization
    # We also add a small bonus (0.1) if it is purely real (realness > 0.99)
    bonus = 0.1 if r > 0.99 else 0.0
    score = sym * 0.4 + r * 0.3 + quant_score * 0.3 + bonus
    return score, r, sym


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_cls_representations(A_0_R, Q, d, max_candidates=4):
    """
    Generate up to *max_candidates* analytically distinct CLS representations
    by exploiting the global-phase gauge freedom A_q → A_q · e^{iθ}.

    Parameters
    ----------
    A_0_R : dict  {q (int) -> {cell (tuple) -> coef (complex)}}
    Q     : int   number of orbitals
    d     : int   spatial dimension
    max_candidates : int

    Returns
    -------
    list of dicts, sorted by quality score (best first).  Each dict:
      label            : str
      description      : str
      A_0_R            : dict  (rotated amplitudes, same format as input)
      display_mode     : str   'amplitude' | 'phase' | 'complex'
      realness         : float – Σ Re² / Σ |A|²
      phase_pattern    : str | None
      global_phase_deg : float – applied rotation in degrees
      score            : float
    """
    if not A_0_R or all(not qd for qd in A_0_R.values()):
        return []

    amps_raw = _collect_amps(A_0_R)
    candidates = []

    # ── 1. Maximally-real (closed-form θ*, analytical sign) ─────────────────
    theta_real = _theta_max_real(A_0_R)
    score_real, r_real, sym_real = _calculate_representation_score(A_0_R, theta_real)
    A_real = _apply_phase(A_0_R, theta_real)
    pp_real = _phase_pattern(A_real)

    candidates.append({
        'label': '실수 최대화',
        'description': (
            f'θ* = {np.degrees(theta_real):.1f}° '
            f'[θ* = −arg(ΣA²)/2, 부호는 ΣRe>0 해석적 결정] '
            f'실수 비율 {r_real * 100:.0f}%, 대칭 {sym_real:.2f}'
        ),
        'A_0_R': A_real,
        'display_mode': _display_mode(r_real, pp_real),
        'realness': r_real,
        'phase_pattern': pp_real,
        'global_phase_deg': float(np.degrees(theta_real)),
        'score': score_real,
    })

    # ── 2. Original CLS (no rotation) ───────────────────────────────────────
    score_raw, r_raw, sym_raw = _calculate_representation_score(A_0_R, 0.0)
    pp_raw = _phase_pattern(A_0_R)

    candidates.append({
        'label': '원본',
        'description': '해석적 방법으로 도출된 원본 CLS (위상 변환 없음)',
        'A_0_R': A_0_R,
        'display_mode': _display_mode(r_raw, pp_raw),
        'realness': r_raw,
        'phase_pattern': pp_raw,
        'global_phase_deg': 0.0,
        'score': score_raw,
    })

    # ── 3. Discrete π/n structures (pairwise phase-diff analysis) ────────────
    seen_pp = {pp_real, pp_raw}
    for theta_disc, pp_disc in _discrete_phase_thetas(A_0_R):
        if pp_disc in seen_pp:
            continue
        score_disc, r_disc, sym_disc = _calculate_representation_score(A_0_R, theta_disc)
        A_disc = _apply_phase(A_0_R, theta_disc)
        seen_pp.add(pp_disc)
        candidates.append({
            'label': f'{pp_disc} 위상',
            'description': (
                f'쌍별 위상차 분석으로 검출된 {pp_disc} 이산 구조 '
                f'(θ = {np.degrees(theta_disc):.1f}°, ΣRe>0 부호 선택)'
            ),
            'A_0_R': A_disc,
            'display_mode': 'phase',
            'realness': r_disc,
            'phase_pattern': pp_disc,
            'global_phase_deg': float(np.degrees(theta_disc)),
            'score': score_disc,
        })
        if len(candidates) >= max_candidates + 1:
            break

    # ── 4. Phase-contrast view (θ_real + π/2, sign-corrected) ───────────────
    if len(candidates) < max_candidates:
        theta_im = _select_sign(theta_real + np.pi / 2, amps_raw)
        score_im, r_im, sym_im = _calculate_representation_score(A_0_R, theta_im)
        A_im = _apply_phase(A_0_R, theta_im)
        pp_im = _phase_pattern(A_im)
        if pp_im not in seen_pp:
            candidates.append({
                'label': '위상 대조',
                'description': (
                    f'θ = {np.degrees(theta_im):.1f}° — 실수 성분 최소화, '
                    f'위상 구조 극대화 (위상 대조 뷰)'
                ),
                'A_0_R': A_im,
                'display_mode': 'phase',
                'realness': r_im,
                'phase_pattern': pp_im,
                'global_phase_deg': float(np.degrees(theta_im)),
                'score': score_im,
            })

    # Sort by score, deduplicate by magnitude pattern
    candidates.sort(key=lambda c: c['score'], reverse=True)
    unique = []
    for c in candidates:
        if not any(_similar(c['A_0_R'], u['A_0_R']) for u in unique):
            unique.append(c)
        if len(unique) >= max_candidates:
            break

    return unique


def extract_canonical_minimal_cls(A_0_R, Q=None, d=None, lattice=None):
    """
    Canonical single representation of a minimized CLS.

    Applies the analytically optimal phase (chosen by the highest-scoring representation
    from representation analysis), normalizes so the maximum site amplitude is 1, and expresses
    every occupied site's cell as a *relative* offset from the reference site
    (the site that ends up with the largest |amplitude|).

    Parameters
    ----------
    A_0_R   : dict {q (int) -> {cell (tuple) -> coef (complex)}}
    Q       : int, number of orbitals (optional)
    d       : int, spatial dimension (optional)
    lattice : Lattice object (optional). When provided and the model uses
              Convention II (non-integer exponents), cell coordinates are
              computed via proper lattice-inverse transformation instead
              of naively rounding the raw exponent.

    Returns
    -------
    dict or None
      sites           : list sorted by distance from reference.
                        Each item: {orbital, cell, rel_cell,
                                    re, im, abs, phase_deg}
                        Caller should add "label" per orbital.
      ref_orbital     : int
      ref_cell        : list[int]
      support_size    : int
      global_phase_deg: float
      display_mode    : 'amplitude' | 'phase' | 'complex'
      realness        : float
      phase_pattern   : str | None
    """
    if not A_0_R or all(not qd for qd in A_0_R.values()):
        return None

    if Q is None:
        Q = len(A_0_R)
    if d is None:
        d = 2
        for qd in A_0_R.values():
            for cell in qd.keys():
                d = len(cell)
                break
            if d is not None:
                break

    # Detect Convention II and build cell-conversion helper
    is_convention_ii = any(
        any(abs(x - round(x)) > 1e-9 for x in exp)
        for qd in A_0_R.values() for exp in qd.keys()
    )

    if is_convention_ii and lattice is not None:
        A_mat = np.array(lattice.primitive_vectors, dtype=float)
        A_inv = (np.linalg.inv(A_mat)
                 if A_mat.shape[0] == A_mat.shape[1]
                 else np.linalg.pinv(A_mat))
        def _to_int_cell(exp_val, q_idx):
            tau_cart = np.array(lattice.orbitals[q_idx]["position"], dtype=float) @ A_mat
            r_frac = (np.array(exp_val, dtype=float) - tau_cart) @ A_inv
            return [int(round(x)) for x in r_frac]
    else:
        def _to_int_cell(exp_val, _q_idx):
            return [int(round(x)) for x in exp_val]

    candidates = analyze_cls_representations(A_0_R, Q, d, max_candidates=1)
    if candidates:
        best_cand = candidates[0]
        A_rot = best_cand['A_0_R']
        theta = np.radians(best_cand['global_phase_deg'])
        display_mode = best_cand['display_mode']
        realness = best_cand['realness']
        phase_pattern = best_cand['phase_pattern']
    else:
        theta = _theta_max_real(A_0_R)
        A_rot = _apply_phase(A_0_R, theta)
        display_mode = _display_mode(_realness(A_rot), _phase_pattern(A_rot))
        realness = float(_realness(A_rot))
        phase_pattern = _phase_pattern(A_rot)

    raw = []
    for q, qd in A_rot.items():
        q_int = int(q)
        for cell, coef in qd.items():
            c = complex(coef)
            if abs(c) > 1e-12:
                raw.append({
                    "orbital": q_int,
                    "cell": _to_int_cell(cell, q_int),
                    "abs": float(abs(c)),
                    "re": float(c.real),
                    "im": float(c.imag),
                    "phase_deg": float(np.degrees(np.angle(c))),
                })

    if not raw:
        return None

    max_abs = max(s["abs"] for s in raw)
    if max_abs > 1e-15:
        for s in raw:
            s["re"] /= max_abs
            s["im"] /= max_abs
            s["abs"] /= max_abs

    # Reference = site with largest |amplitude| (tie-break: lowest orbital index)
    raw.sort(key=lambda s: (-s["abs"], s["orbital"], s["cell"]))
    ref_cell = raw[0]["cell"]

    for s in raw:
        s["rel_cell"] = [s["cell"][i] - ref_cell[i] for i in range(len(ref_cell))]

    # Sort by Euclidean distance from reference, then orbital index
    raw.sort(key=lambda s: (round(sum(x ** 2 for x in s["rel_cell"]) ** 0.5 * 1e6),
                             s["orbital"], s["cell"]))

    return {
        "sites": raw,
        "ref_orbital": raw[0]["orbital"],
        "ref_cell": ref_cell,
        "support_size": len(raw),
        "global_phase_deg": float(np.degrees(theta)),
        "display_mode": display_mode,
        "realness": realness,
        "phase_pattern": phase_pattern,
    }


def canonicalize_amplitudes(A_0_R, Q, d):
    """
    Apply the best canonical global phase rotation and normalize
    so that the maximum amplitude has a magnitude of 1.0.
    Returns: dict in the same format as A_0_R.
    """
    if not A_0_R or all(not qd for qd in A_0_R.values()):
        return A_0_R
        
    candidates = analyze_cls_representations(A_0_R, Q, d, max_candidates=1)
    if candidates:
        A_rot = candidates[0]['A_0_R']
    else:
        theta = _theta_max_real(A_0_R)
        A_rot = _apply_phase(A_0_R, theta)
        
    # Find max amplitude magnitude
    max_abs = 0.0
    for qd in A_rot.values():
        for coef in qd.values():
            max_abs = max(max_abs, abs(coef))
            
    if max_abs > 1e-12:
        return {q: {cell: coef / max_abs for cell, coef in qd.items()}
                for q, qd in A_rot.items()}
    return A_rot

