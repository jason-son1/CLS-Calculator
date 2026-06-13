"""
pipeline.py — design_flat_band(): full Phase 1-5 orchestration.

    LatticeSpec + DesignTarget
        -> Phase1  validate_design                     (spec.py)
        -> Phase2/3 build_cls_design (per singularity) (pairing.py / chiral.py)
        -> Phase4  build_x_k + verify_projector_continuity + analytic_chern
        -> Phase5  build_hamiltonian (NumericHk) + inverse_fourier_transform
        -> DesignResult (cls_designs, x_k, H_k, hoppings, verification log)

FEEDBACK LOOPS (Note B Sec.12 / Rhim-Yang):
  - Phase3/4: if the contour winding at any target singularity does not match
    its target w_i, or the projector is discontinuous there, or the GLOBAL
    analytic_chern(x_k).C != target.C (e.g. an unintended extra common zero
    appeared, ARCHITECTURE.md Sec.0.4), the per-sublattice bond-vector shell
    assignment is rotated (chiral.shell_for_sublattice's `offset`) and Phase
    2-4 is retried, up to `max_retries` times.
  - Phase5: if the R_cut-truncated Hamiltonian H_trunc is gapped (band_isolation)
    but its numerical Chern number does not match target.C, R_cut is increased
    and the IFT is retried, up to `max_rcut_retries` times. chern.py's
    analytic_chern (contour winding of x_k) and numerical_chern (FHS lattice
    Chern of H(k)'s eigenvectors) are independently gauge-invariant but their
    RELATIVE SIGN depends on sign(det(lattice.primitive_vectors)) -- see
    _numerical_sign_convention. That sign is fixed once from the EXACT H(k)
    (always correct: Eq.127 + the Phase3/4 analytic check) and then used to
    ALIGN H_trunc's FHS Chern to target_C's convention for both the retry
    condition and the reported `*_numerical_C` fields, so a sign flip caused
    by IFT truncation breaking the flat band's isolation is reported as a
    genuine MISMATCH (not masked as a "match" by chern.py's own abs()-based
    agreement check, _chern_matches, which remains only as a degenerate
    fallback when even the exact H(k) disagrees with target_C in magnitude).

SCOPE NOTE (S = number of target singularities):
  For S=1 (|target.C|=1 -- the minimal construction of Note A/B), the
  per-sublattice 2-pair closed form (chiral.py) generically yields
  analytic_chern(x_k).C == target.C with no extra common zeros (verified by
  the Phase3/4 checks above; the shell-offset retry is the safety net for
  unlucky/non-generic cases).

  For S>=2 (|target.C|>=2, or w_i's that must cancel to C=0), each CLSDesign
  individually satisfies Condition 1 (Eq.11) ONLY at its OWN k_i; summing
  CLSDesigns (the superposition build_x_k performs) does NOT in general make
  f(k_j)=0 for j!=i hold simultaneously -- realizing |C|>=2 rigorously
  requires either a single higher-order zero (an order-n chiral solver,
  Sum_p A_p e^{iPhi_p} D_p^n = 0) or a JOINTLY solved multi-zero
  interpolation, neither of which this implementation provides (Note B
  itself: "Phase 3 방정식의 난이도는 N과 격자 대칭성에 따라 크게 달라진다").
  design_flat_band() still ATTEMPTS the S>=2 superposition and reports the
  true analytic_chern(x_k).C honestly; if it disagrees with target.C the
  Rhim-Yang feedback loop will exhaust max_retries and `feedback_success`
  will be False -- this is a faithful "design appears contradictory" report,
  not a silent wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from cls_finder.core.laurent import LaurentPoly
from cls_finder.core.matrixpoly import MatrixPoly
from cls_finder.classify import chern as _chern
from cls_finder.engineer.spec import (
    LatticeSpec, DesignTarget, validate_design, with_derived_zetas,
)
from cls_finder.engineer.pairing import CLSDesign
from cls_finder.engineer.chiral import build_cls_design
from cls_finder.engineer.assembly import build_x_k, evaluate_psi, verify_projector_continuity
from cls_finder.engineer.hamiltonian import (
    NumericHk, build_hamiltonian, inverse_fourier_transform, hoppings_to_matrixpoly,
    build_local_flat_hamiltonian,
)


def _matrixpoly_to_hoppings(H: MatrixPoly, tol: float = 1e-9) -> Dict:
    """{(alpha, beta, (n, m)): complex} for the nonzero hoppings of a
    finite-range MatrixPoly (the real-space tight-binding model)."""
    hops: Dict = {}
    for a in range(H.rows):
        for b in range(H.cols):
            for (n, m), val in H.data[a][b].coefs.items():
                if abs(val) > tol:
                    hops[(a, b, (int(n), int(m)))] = complex(val)
    return hops


def _max_hopping_range(hoppings: Dict) -> int:
    return max((max(abs(n), abs(m)) for (_, _, (n, m)) in hoppings), default=0)


def _flatness_from_xk(H, x_k, lattice_spec, E0, n: int = 16) -> float:
    """max_k || H(k) psi(k) - E0 psi(k) || over an n x n BZ grid, with
    psi(k)=f(k)/||f(k)|| from x_k (skipping the common zeros). Works for any
    H exposing .evaluate_batch (NumericHk or MatrixPoly)."""
    try:
        prim = lattice_spec.primitive_vectors
        B = lattice_spec.reciprocal_vectors()
        frac = _chern._frac_grid(n, n)
        k_cart = frac @ B
        F = np.stack([p.evaluate_batch(k_cart, prim) for p in x_k], axis=1)
        norm2 = np.sum(np.abs(F) ** 2, axis=1)
        Hb = H.evaluate_batch(k_cart, prim)
        worst = 0.0
        for i in range(k_cart.shape[0]):
            if norm2[i] < 1e-6:
                continue
            psi = F[i] / np.sqrt(norm2[i])
            worst = max(worst, float(np.linalg.norm(Hb[i] @ psi - E0 * psi)))
        return worst
    except Exception:
        return -1.0


def _chern_matches(numerical_C: int, target_C: int) -> bool:
    """True if a numerical_chern() result `numerical_C` realizes `target_C`,
    ignoring any overall sign convention (abs() match).

    chern.analyze_flat_band_chern's own "agreement" check (chern.py ~line
    610) is `ana["C"]==num["C"] OR abs(ana["C"])==abs(num["C"])`, i.e. it
    already treats a sign flip as agreement -- this mirrors that precedent.

    This is intentionally LOOSE (a |C|=1 target accepts BOTH +1 and -1) and
    is used ONLY as the degenerate fallback in design_flat_band's Phase5 when
    _numerical_sign_convention cannot determine this lattice's FHS sign
    convention from the exact H(k) (sign_conv == 0, see below). In the normal
    case, design_flat_band aligns numerical_chern's sign to target_C's
    convention BEFORE comparing, so a real sign flip (e.g. caused by IFT
    truncation breaking the flat band's isolation) is reported as a genuine
    mismatch rather than masked by abs().
    """
    return numerical_C == target_C or abs(numerical_C) == abs(target_C)


def _numerical_sign_convention(num_full_C: int, target_C: int) -> int:
    """Sign relating chern.numerical_chern's FHS convention to target_C
    (and hence to chern.analytic_chern's contour-winding convention) for
    THIS lattice.

    chern.analytic_chern(x_k) and chern.numerical_chern(H_k) are each
    independently gauge-invariant, but their RELATIVE SIGN depends on
    sign(det(lattice.primitive_vectors)): for some lattices
    numerical_chern(H_k).C == -analytic_chern(x_k).C, for others
    == +analytic_chern(x_k).C (verified empirically for det(prim) = +-1 on
    the square lattice, both w=+1 and w=-1).

    H_k (Eq.127) is exactly flat and Phase3/4 already verified
    analytic_chern(x_k).C == target_C, so num_full_C (FHS on the EXACT H_k)
    must equal +-target_C; whichever sign it is IS this lattice's FHS sign
    convention, and the SAME sign relates numerical_chern(H_trunc).C to
    target_C.

    Returns +1 or -1 normally (target_C == 0 trivially returns +1, since
    +-0 are indistinguishable), or 0 if num_full_C does not match target_C in
    MAGNITUDE either -- a degenerate case (even the exact H(k)'s FHS Chern
    disagrees with the analytically-verified target) where no alignment is
    possible and the caller should fall back to _chern_matches.
    """
    if target_C == 0:
        return 1
    if num_full_C == target_C:
        return 1
    if num_full_C == -target_C:
        return -1
    return 0


def _phase5_success(analytic_match: bool, trunc_isolated: bool,
                    trunc_match: bool) -> bool:
    """Whether the IFT-truncated tight-binding deliverable realizes the
    designed topology.

    CONCEPTUAL BASIS (Note A Sec.8, "Flat band 특이성과의 연결"): a finite-range
    CLS flat band that carries a NONZERO Chern number is necessarily a
    *singular* (band-touching) flat band -- it touches the dispersive sector
    AT the designed common-zero/singularity k_i, where its winding is
    generated. A fully *isolated* (gapped) finite-range flat band is generically
    C=0. Therefore "the truncated band must be gapped/isolated" is the WRONG
    success criterion: requiring it would reject exactly the good topological
    designs.

    Ground truth is the ANALYTIC design: the global contour-winding Chern of
    f(k) equals the target (`analytic_match`, established by Phase 2-4, which
    is independent of M(k) and of any truncation). Given that:

      - if the truncated H is gapped (isolated), the lattice FHS Chern is
        reliable and we additionally require it to confirm the target
        (`trunc_match`);
      - if the truncated H is gapless (touching at the singularity -- the
        EXPECTED case for a genuine topological flat band), the single-band FHS
        Chern is not reliable across the touching, so we accept the design on
        the analytic guarantee alone.
    """
    if not analytic_match:
        return False
    if trunc_isolated:
        return trunc_match
    return True


@dataclass
class DesignResult:
    """The complete output of design_flat_band(): everything Phase 1-5
    produced, plus the verification log."""
    lattice_spec: LatticeSpec
    target: DesignTarget
    cls_designs: List[CLSDesign]
    x_k: List[LaurentPoly]
    H_k: NumericHk
    H_trunc: MatrixPoly
    ift: Dict
    verification: Dict
    log: List[str] = field(default_factory=list)

    @property
    def hoppings(self) -> Dict:
        """{(alpha, beta, (n, m)): t_ab(R)} -- the truncated tight-binding model."""
        return self.ift["hoppings"]

    @property
    def lattice(self):
        """A cls_finder.core.lattice.Lattice for band.bands / viz.plot."""
        return self.lattice_spec.to_lattice()

    def evaluate_psi(self, k_cart: np.ndarray):
        """f(k), psi(k)=f(k)/||f(k)||, ||f(k)|| (Eq. 100) at a Cartesian k."""
        return evaluate_psi(self.x_k, k_cart, self.lattice_spec.primitive_vectors)

    def summary(self) -> str:
        return "\n".join(self.log)


def design_flat_band(lattice_spec: LatticeSpec, target: DesignTarget,
                      E0: float = 0.0, M_k: Optional[MatrixPoly] = None,
                      t: float = 0.3, delta: float = 0.5,
                      n_grid_ift: int = 24, R_cut: int = 3,
                      max_retries: int = 8, max_rcut_retries: int = 4,
                      shell_offset0: int = 0, derive_zeta_gauge: bool = True,
                      local_exact: bool = True, r_max: Optional[int] = None,
                      cls_size: Optional[int] = None,
                      verbose: bool = True) -> DesignResult:
    """Run the full Real-Space CLS Topological Flat Band Engineering pipeline.

    Parameters
    ----------
    lattice_spec : Module 1 lattice geometry (a1, a2, sublattices, zeta's).
    target        : Module 1 topological target (C, [(k_i, w_i)]).
    E0            : flat-band energy (Eq. 127).
    M_k           : optional user-supplied N x N Hermitian MatrixPoly for the
                    dispersive sector; defaults to hamiltonian.default_M_k.
    t, delta      : hopping / on-site-mass scales used by default_M_k if
                    M_k is None (hamiltonian.default_M_k).
    n_grid_ift, R_cut : IFT BZ-grid size and initial hopping-range cutoff.
    max_retries   : Phase3/4 shell-offset feedback retries.
    max_rcut_retries : Phase5 R_cut feedback retries.
    shell_offset0 : starting shell offset (chiral.shell_for_sublattice).
    local_exact   : if True (default), the deliverable tight-binding model is
                    the Rhim-Yang FINITE-RANGE, EXACTLY-FLAT construction
                    H_loc(k) = E0 I + (||f||^2 I - f f^dagger) built directly
                    from x_k (hamiltonian.build_local_flat_hamiltonian) -- so
                    the flat band stays PERFECTLY flat when the model is
                    reloaded into the band/CLS analyser, and touches the
                    dispersive sector only at the designed singularity. If
                    False, the old behaviour is used: the non-local projector
                    H(k)=E0 P+(I-P)M(I-P) is inverse-Fourier-transformed and
                    truncated to |R|<=R_cut (which breaks exact flatness).
    r_max         : optional cap on the deliverable's hopping range (in unit
                    cells), applied ONLY in local_exact mode. None (default)
                    keeps the natural, EXACTLY-flat range set by the CLS
                    geometry. An integer r_max DROPS every hopping with
                    max(|n|,|m|) > r_max -- e.g. r_max=1 forces a
                    nearest-neighbour-only model, r_max=2 allows next-nearest,
                    etc. This is an APPROXIMATION: truncating the exact local
                    model generally breaks perfect flatness, so the resulting
                    flat_band_max_dev (reported honestly) is no longer ~0; use
                    it to trade hopping range against flatness consciously.
    cls_size      : optional target SIZE (spatial extent, in unit cells) of the
                    CLS that builds f(k). None (default) uses the minimal
                    DEFAULT_SHELLS library (nearest-neighbour-ish). A positive
                    integer draws bond-vector shells from
                    chiral.shells_of_size(cls_size), so the CLS reaches that
                    radius and the resulting exactly-flat model has a
                    correspondingly longer (but still finite) hopping range --
                    a direct "make the CLS bigger/smaller" knob, independent of
                    r_max (which instead TRUNCATES an already-built model).
    derive_zeta_gauge : if True (default), the per-sublattice gauge phase
                    zeta^(alpha) is NOT taken from the user input but DERIVED
                    from each orbital's real-space position and the primary
                    singularity, zeta^(alpha) = k_0 . r_alpha (spec.
                    with_derived_zetas; k_0 = target.singularities[0]). This is
                    the physically-determined "site shift" gauge of Note A
                    (zeta is fixed once the sublattice/orbital position is
                    fixed -- it is not a free knob). Per ARCHITECTURE.md Sec.0.1
                    every topological invariant is zeta-invariant, so this only
                    fixes the relative phases of the resulting hoppings t_ab(R)
                    to the standard position-dependent convention.
    verbose       : print the verification log as it is produced.

    Returns
    -------
    DesignResult
    """
    log: List[str] = []

    def emit(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg)

    # ------------------------------------------------------------------ #
    # Phase 1 -- initialization & target validation
    # ------------------------------------------------------------------ #
    validate_design(lattice_spec, target)
    target_C = target.C
    emit(f"[Phase1] N={lattice_spec.N} sublattices, target C = sum(w_i) = "
         f"{target_C}, singularities = "
         f"{[(s.name, s.k_frac, s.w) for s in target.singularities]}")

    # zeta^(alpha) is the physically-determined "site shift" gauge (Note A):
    # once an orbital's position is fixed, zeta = k_0 . r_alpha is fixed too --
    # it is not a free input. Derive it from the primary singularity unless the
    # caller opts out (derive_zeta_gauge=False keeps the LatticeSpec's zetas).
    if derive_zeta_gauge and target.singularities:
        lattice_spec = with_derived_zetas(lattice_spec, target.singularities[0])
        emit("[Phase1] gauge zeta^(alpha) derived from orbital positions and "
             f"the primary singularity '{target.singularities[0].name}': "
             f"{[round(s.zeta, 4) for s in lattice_spec.sublattices]} (rad) "
             "-- topological invariants are zeta-invariant (ARCHITECTURE Sec.0.1)")

    if not target.singularities:
        raise ValueError(
            "design_flat_band requires at least one singularity: f(k) is "
            "assembled from the CLS configurations targeting each k_i "
            "(Phase 2/3), so a target with no singularities has nothing to "
            "construct. (A target.C == 0 with NO topological structure is "
            "outside this algorithm's scope -- see ARCHITECTURE.md Sec. 0.5 "
            "/ spec.DesignTarget.validate.)"
        )
    if len(target.singularities) >= 2:
        emit("[Phase1][NOTE] S>=2 singularities requested: see pipeline.py "
             "module docstring SCOPE NOTE -- realizing |C|>=2 by summing "
             "per-singularity CLS designs is attempted but not guaranteed; "
             "the Phase3/4 feedback loop will report honestly if it fails.")

    lattice = lattice_spec.to_lattice()

    # ------------------------------------------------------------------ #
    # Phase 2/3/4 -- pairing + chiral closed form + assembly, with the
    # shell-offset feedback loop
    # ------------------------------------------------------------------ #
    x_k: List[LaurentPoly] = []
    cls_designs: List[CLSDesign] = []
    continuity: Dict[str, Dict] = {}
    analytic: Dict = {}
    feedback_success = False
    attempt = 0
    # NOTE: a larger cls_size tends to realise a HIGHER |C| (longer CLS bonds
    # wind more), so the first shell pairs may not match a small target C.
    # Matching a small C at a large size is possible but needs many shell
    # retries (slow); that search is left under the user's `max_retries`
    # control rather than forced here. The realised analytic C is always
    # reported honestly.
    eff_max_retries = max_retries
    for attempt in range(eff_max_retries):
        offset = shell_offset0 + attempt
        cls_designs = [build_cls_design(s, lattice_spec, shell_offset=offset,
                                        cls_size=cls_size)
                       for s in target.singularities]

        for design in cls_designs:
            residuals = design.verify_condition1()
            max_res = max((abs(r) for r in residuals), default=0.0)
            if max_res > 1e-8:
                emit(f"[Phase2][WARNING] attempt {attempt} (offset={offset}): "
                     f"singularity '{design.singularity.name}' Condition1 "
                     f"residual {max_res:.2e} > 1e-8")

        x_k = build_x_k(lattice_spec, cls_designs)

        continuity = {}
        all_match = True
        for s in target.singularities:
            k_i = s.k_cartesian(lattice_spec)
            info = verify_projector_continuity(x_k, lattice_spec, k_i)
            continuity[s.name] = info
            loop = info["loop"]
            ok = (loop["winding"] == s.w) and loop["projector_continuous"]
            emit(f"[Phase3/4] attempt {attempt}: '{s.name}' k_frac={s.k_frac} "
                 f"target w={s.w:+d} -> loop_winding={loop['winding']:+d}, "
                 f"continuous={loop['projector_continuous']}, "
                 f"rank_ratio={loop['rank_ratio']:.2e}")
            all_match &= ok

        analytic = _chern.analytic_chern(x_k, lattice)
        emit(f"[Phase4] attempt {attempt}: analytic C={analytic['C']} "
             f"(C_all_zeros={analytic['C_all_zeros']}, "
             f"n_common_zeros={analytic['n_common_zeros']}, "
             f"projector_continuous={analytic['projector_continuous']})")

        if all_match and analytic["projector_continuous"] and analytic["C"] == target_C:
            feedback_success = True
            break

        emit(f"[Phase3/4][WARNING] attempt {attempt}: Rhim-Yang contradiction "
             f"(local design vs. global topology mismatch) -- retrying with "
             f"shell_offset={offset + 1}")

    if not feedback_success:
        emit(f"[Phase3/4][WARNING] exhausted {eff_max_retries} attempts without a "
             f"consistent design (last analytic C={analytic.get('C')}, "
             f"target C={target_C}); proceeding with the last attempt's "
             f"f(k) for Phase 5 (results below may not realize the target "
             f"topology -- see log above)")

    # ------------------------------------------------------------------ #
    # Phase 5 -- non-local reference H_k = E0 P + (I-P) M (I-P) (Eq.127),
    # used only as the analytic-flatness reference + FHS sign convention.
    # ------------------------------------------------------------------ #
    H_k = build_hamiltonian(x_k, lattice_spec, target, E0=E0, M=M_k, t=t, delta=delta)

    iso_full = _chern.band_isolation(H_k, lattice, E0, 1)
    num_full = _chern.numerical_chern(H_k, lattice, E0, 1)
    sign_conv = _numerical_sign_convention(num_full["C"], target_C)
    full_numerical_C = sign_conv * num_full["C"] if sign_conv else num_full["C"]
    full_match = ((full_numerical_C == target_C) if sign_conv
                   else _chern_matches(num_full["C"], target_C))
    emit(f"[Phase5] reference H(k)=E0 P+(I-P)M(I-P): isolated={iso_full['isolated']} "
         f"numerical C={full_numerical_C} (FHS_raw={num_full['C']}, "
         f"sign_convention={sign_conv})")

    # ------------------------------------------------------------------ #
    # Phase 5 deliverable -- FINITE-RANGE, EXACTLY-FLAT tight-binding model.
    # The projector H_k above is exactly flat but NON-local (infinite-range
    # hoppings via the 1/||f||^2 normalisation); inverse-Fourier-truncating it
    # breaks the flatness, so the reloaded model no longer has a clean flat
    # band. Instead we build the Rhim-Yang local construction
    #     H_loc(k) = E0 I + t (||f||^2 I - f f^dagger),
    # which is finite-range AND exactly flat (H_loc f = E0 f identically) AND
    # touches the dispersive sector only at the designed singularity (||f||^2=0)
    # -- where the Chern winding lives (Note A Sec.8). This is the deliverable
    # that survives re-analysis with the flat band intact.
    # ------------------------------------------------------------------ #
    r_max_applied = False
    natural_range = 0
    if local_exact:
        H_full = build_local_flat_hamiltonian(x_k, E0=E0, t=1.0)
        hoppings_full = _matrixpoly_to_hoppings(H_full)
        natural_range = _max_hopping_range(hoppings_full)
        weight_full = sum(abs(v) ** 2 for v in hoppings_full.values())

        if r_max is not None and natural_range > r_max:
            hoppings = {key: v for key, v in hoppings_full.items()
                        if max(abs(key[2][0]), abs(key[2][1])) <= r_max}
            H_trunc = hoppings_to_matrixpoly(hoppings, lattice_spec.N)
            r_max_applied = True
            weight_kept = sum(abs(v) ** 2 for v in hoppings.values())
            trunc_ratio = 1.0 - (weight_kept / weight_full if weight_full > 1e-300 else 1.0)
        else:
            hoppings = hoppings_full
            H_trunc = H_full
            trunc_ratio = 0.0

        max_range = _max_hopping_range(hoppings)
        n_terms = len(hoppings)
        flat_dev = _flatness_from_xk(H_trunc, x_k, lattice_spec, E0)
        iso_trunc = _chern.band_isolation(H_trunc, lattice, E0, 1)
        num_trunc = _chern.numerical_chern(H_trunc, lattice, E0, 1)
        if sign_conv:
            trunc_numerical_C = sign_conv * num_trunc["C"]
            trunc_match = (trunc_numerical_C == target_C)
        else:
            trunc_numerical_C = num_trunc["C"]
            trunc_match = _chern_matches(num_trunc["C"], target_C)
        ift = {"hoppings": hoppings, "R_cut": max_range, "n_grid": n_grid_ift,
               "truncation_ratio": trunc_ratio}
        if r_max_applied:
            emit(f"[Phase5] local model CAPPED to r_max={r_max} (natural range "
                 f"{natural_range}): {n_terms} terms, {trunc_ratio*100:.2f}% of "
                 f"||t||^2 dropped -> flatness max||H psi-E0 psi||={flat_dev:.2e} "
                 "(APPROXIMATE: capping range breaks exact flatness)")
        else:
            emit(f"[Phase5] EXACT local model H_loc(k)=E0 I + (||f||^2 I - f f^dagger): "
                 f"{n_terms} hopping terms, max range {max_range} cells, "
                 f"flatness max||H psi - E0 psi||={flat_dev:.2e} (EXACT, no "
                 "truncation -- flat band survives re-analysis intact)")
        emit(f"[Phase5] H_loc isolated={iso_trunc['isolated']} "
             f"(touching at the singularity is EXPECTED for nonzero C), "
             f"numerical C={trunc_numerical_C} (FHS_raw={num_trunc['C']})")
    else:
        flat_dev = _flatness_from_xk(H_k, x_k, lattice_spec, E0)
        ift = {}
        H_trunc = None
        iso_trunc, num_trunc = {}, {}
        trunc_numerical_C, trunc_match = None, False
        for rattempt in range(max_rcut_retries):
            rc = R_cut + rattempt
            ift = inverse_fourier_transform(H_k, lattice_spec, n_grid=n_grid_ift, R_cut=rc)
            H_trunc = hoppings_to_matrixpoly(ift["hoppings"], lattice_spec.N)
            iso_trunc = _chern.band_isolation(H_trunc, lattice, E0, 1)
            num_trunc = _chern.numerical_chern(H_trunc, lattice, E0, 1)
            if sign_conv:
                trunc_numerical_C = sign_conv * num_trunc["C"]
                trunc_match = (trunc_numerical_C == target_C)
            else:
                trunc_numerical_C = num_trunc["C"]
                trunc_match = _chern_matches(num_trunc["C"], target_C)
            emit(f"[Phase5] IFT R_cut={rc}: truncation_ratio="
                 f"{ift['truncation_ratio']:.2e}, H_trunc isolated="
                 f"{iso_trunc['isolated']}, numerical C_trunc={trunc_numerical_C}")
            if trunc_match or not iso_trunc["isolated"]:
                break

    # ------------------------------------------------------------------ #
    # Verification log
    # ------------------------------------------------------------------ #
    # Ground truth is the analytic design (global contour-winding Chern of
    # f(k) == target, independent of M(k)/truncation). A TOUCHING (non-isolated)
    # truncated band is the EXPECTED outcome for a topological CLS flat band
    # and is accepted on that analytic guarantee; isolation only lets the
    # (otherwise-unreliable) FHS Chern additionally confirm it. See
    # _phase5_success / Note A Sec.8.
    analytic_match = (analytic.get("C") == target_C)
    trunc_singular = not bool(iso_trunc.get("isolated"))
    exact_flat = (flat_dev >= 0.0) and (flat_dev < 1e-6)
    # The deliverable realises the designed topology iff the analytic Chern
    # matches AND the model is (numerically) exactly flat; isolation is NOT
    # required (a nonzero-C finite-range flat band must touch at its
    # singularity -- Note A Sec.8 / _phase5_success).
    phase5_success = (_phase5_success(analytic_match, bool(iso_trunc.get("isolated")),
                                      trunc_match) and exact_flat)
    n_hopping_terms = len(ift.get("hoppings", {}))
    # CLS diagnostics: how many real-space sites the flat-band vector occupies
    # per sublattice, and its spatial extent (Chebyshev radius).
    n_cls_sites = sum(len(p.coefs) for p in x_k)
    cls_extent = max((max(abs(n), abs(m)) for p in x_k for (n, m) in p.coefs),
                     default=0)
    emit(f"[Phase2] CLS: size request={cls_size if cls_size is not None else 'auto(min)'}, "
         f"{n_cls_sites} sites total over {lattice_spec.N} sublattices, "
         f"spatial extent {cls_extent} cell(s)")
    verification = {
        "target_C": target_C,
        "cls_size": cls_size,
        "n_cls_sites": n_cls_sites,
        "cls_extent": cls_extent,
        "analytic_C": analytic.get("C"),
        "analytic_match": analytic_match,
        "analytic": analytic,
        "continuity": continuity,
        "feedback_attempts": attempt + 1,
        "feedback_success": feedback_success,
        "flat_band_max_dev": flat_dev,
        "exact_flat": exact_flat,
        "local_exact": local_exact,
        "r_max": r_max,
        "r_max_applied": r_max_applied,
        "natural_hopping_range": natural_range,
        "n_hopping_terms": n_hopping_terms,
        "max_hopping_range": ift.get("R_cut"),
        "numerical_sign_convention": sign_conv,
        "full_isolated": iso_full["isolated"],
        "full_numerical_C": full_numerical_C,
        "full_numerical_C_raw": num_full["C"],
        "full_chern_match": full_match,
        "trunc_isolated": iso_trunc.get("isolated"),
        "trunc_singular": trunc_singular,
        "trunc_numerical_C": trunc_numerical_C,
        "trunc_numerical_C_raw": num_trunc.get("C"),
        "trunc_chern_match": trunc_match,
        "phase5_success": phase5_success,
        "truncation_ratio": ift.get("truncation_ratio"),
        "R_cut": ift.get("R_cut"),
    }

    full_c_disp = f"{full_numerical_C}" + (f" [FHS_raw={num_full['C']}]" if sign_conv else "")
    trunc_c_disp = f"{trunc_numerical_C}" + (
        f" [FHS_raw={num_trunc.get('C')}]" if sign_conv and num_trunc else "")
    band_kind = ("isolated" if not trunc_singular
                 else "touching/singular (expected for nonzero C)")
    emit("[Verify] " + " | ".join([
        f"C_target={target_C}",
        f"C_analytic={analytic.get('C')} (matches target: {analytic_match})",
        f"C_numerical(full)={full_c_disp}",
        f"C_numerical(trunc,R_cut={ift.get('R_cut')})={trunc_c_disp} "
        f"(matches target: {trunc_match})",
        f"trunc_band={band_kind}",
        f"feedback_success={feedback_success} ({attempt + 1} attempt(s))",
        f"phase5_success={phase5_success}",
    ]))
    for s in target.singularities:
        loop = continuity[s.name]["loop"]
        emit(f"[Verify]   '{s.name}' k_frac={s.k_frac}: target w={s.w:+d}, "
             f"loop_winding={loop['winding']:+d}, "
             f"projector_continuous={loop['projector_continuous']}, "
             f"rank_ratio={loop['rank_ratio']:.2e}")

    return DesignResult(
        lattice_spec=lattice_spec, target=target, cls_designs=cls_designs,
        x_k=x_k, H_k=H_k, H_trunc=H_trunc, ift=ift,
        verification=verification, log=log,
    )
