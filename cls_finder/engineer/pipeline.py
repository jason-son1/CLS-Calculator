"""
pipeline.py — design_flat_band(): full Phase 1-5 orchestration.

    LatticeSpec + DesignTarget
        -> Phase1  validate_design                     (spec.py)
        -> Phase2/3 build_cls_design (per singularity) (pairing.py / chiral.py)
        -> Phase4  build_x_k + verify_projector_continuity + analytic_chern
        -> Phase5  build_engineered_hamiltonian (H_core/alpha/M_tilde)
        -> DesignResult (cls_designs, x_k, H_full, H_trunc, hoppings, verification log)

FEEDBACK LOOP (Note B Sec.12 / Rhim-Yang):
  - Phase3/4: if the contour winding at any target singularity does not match
    its target w_i, or the projector is discontinuous there, or the GLOBAL
    analytic_chern(x_k).C != target.C (e.g. an unintended extra common zero
    appeared, ARCHITECTURE.md Sec.0.4), the per-sublattice bond-vector shell
    assignment is rotated (chiral.shell_for_sublattice's `offset`) and Phase
    2-4 is retried, up to `max_retries` times.

SIGN CONVENTION:
  chern.py's analytic_chern (contour winding of x_k) and numerical_chern (FHS
  lattice Chern of H(k)'s eigenvectors) are each independently gauge-invariant,
  but their RELATIVE SIGN depends on sign(det(lattice.primitive_vectors)) --
  see _numerical_sign_convention. That sign is fixed once from the EXACT,
  untruncated H_full(k) (always correct: H_full f = E0 f for any alpha/M_tilde,
  plus the Phase3/4 analytic check) and then used to ALIGN H_trunc's FHS Chern
  to target_C's convention for the reported `*_numerical_C` fields, so a sign
  flip caused by r_max truncation breaking the flat band's isolation is
  reported as a genuine MISMATCH (not masked as a "match" by chern.py's own
  abs()-based agreement check, _chern_matches, which remains only as a
  degenerate fallback when even H_full(k) disagrees with target_C in
  magnitude).

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
    build_engineered_hamiltonian, matrixpoly_to_hoppings, max_hopping_range,
    flatness_deviation, hoppings_to_matrixpoly,
)
from cls_finder.engineer.dispersive import build_dispersive_M


def _chern_matches(numerical_C: int, target_C: int) -> bool:
    """True if a numerical_chern() result `numerical_C` realizes `target_C`,
    ignoring any overall sign convention (abs() match).

    chern.analyze_flat_band_chern's own "agreement" check (chern.py ~line
    610) is `ana["C"]==num["C"] OR abs(ana["C"])==abs(num["C"])`, i.e. it
    already treats a sign flip as agreement -- this mirrors that precedent.

    This is intentionally LOOSE (a |C|=1 target accepts BOTH +1 and -1) and
    is used ONLY as the degenerate fallback in design_flat_band's Phase5 when
    _numerical_sign_convention cannot determine this lattice's FHS sign
    convention from the exact H_full(k) (sign_conv == 0, see below). In the
    normal case, design_flat_band aligns numerical_chern's sign to target_C's
    convention BEFORE comparing, so a real sign flip (e.g. caused by r_max
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

    H_full(k) is exactly flat for any alpha/M_tilde and Phase3/4 already
    verified analytic_chern(x_k).C == target_C, so num_full_C (FHS on the
    EXACT H_full(k)) must equal +-target_C; whichever sign it is IS this
    lattice's FHS sign convention, and the SAME sign relates
    numerical_chern(H_trunc).C to target_C.

    Returns +1 or -1 normally (target_C == 0 trivially returns +1, since
    +-0 are indistinguishable), or 0 if num_full_C does not match target_C in
    MAGNITUDE either -- a degenerate case (even the exact H_full(k)'s FHS
    Chern disagrees with the analytically-verified target) where no alignment
    is possible and the caller should fall back to _chern_matches.
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
    """Whether the tight-binding deliverable realizes the designed topology.

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
    is independent of M_tilde(k) and of any truncation). Given that:

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
    H_full: MatrixPoly
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
                      E0: float = 0.0, alpha: float = 1.0,
                      dispersive_shape: str = "nn_real",
                      dispersive_strength: float = 0.3,
                      M_tilde: Optional[MatrixPoly] = None,
                      max_retries: int = 8,
                      shell_offset0: int = 0, derive_zeta_gauge: bool = True,
                      r_max: Optional[int] = None,
                      cls_size: Optional[int] = None,
                      verbose: bool = True) -> DesignResult:
    """Run the full Real-Space CLS Topological Flat Band Engineering pipeline.

    Parameters
    ----------
    lattice_spec : Module 1 lattice geometry (a1, a2, sublattices, zeta's).
    target        : Module 1 topological target (C, [(k_i, w_i)]).
    E0            : flat-band energy.
    alpha         : Dispersive-band bandwidth scale -- the coefficient of
                    H_core(k) in H(k) = E0 I + alpha*H_core(k) +
                    H_core(k) M_tilde(k) H_core(k) (Hamiltonian Engineering
                    note, Sec.2).
    dispersive_shape, dispersive_strength :
                    Name (cls_finder.engineer.dispersive.DISPERSIVE_SHAPES)
                    and strength of the M_tilde(k) "dispersive shape" used to
                    dress the N-1 dispersive bands. Ignored if M_tilde is
                    given explicitly.
    M_tilde       : optional user-supplied N x N Hermitian MatrixPoly
                    overriding dispersive_shape/dispersive_strength.
    max_retries   : Phase3/4 shell-offset feedback retries.
    shell_offset0 : starting shell offset (chiral.shell_for_sublattice).
    r_max         : optional cap on the deliverable's hopping range (in unit
                    cells). None (default) keeps the natural, EXACTLY-flat
                    range set by the CLS geometry and (alpha, M_tilde). An
                    integer r_max DROPS every hopping with
                    max(|n|,|m|) > r_max -- e.g. r_max=1 forces a
                    nearest-neighbour-only model, r_max=2 allows next-nearest,
                    etc. This is an APPROXIMATION: truncating the exact model
                    generally breaks perfect flatness, so the resulting
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
    # Phase 5 -- H(k) = E0 I + alpha*H_core(k) + H_core(k) M_tilde(k) H_core(k)
    # (Sec.2 of the Hamiltonian Engineering note). H_core(k) f(k) = 0 for all
    # k and H_core(k_i) = 0 at the designed singularities, so H_full is
    # EXACTLY flat at E0 and touches the dispersive sector only at k_i, for
    # ANY (alpha, M_tilde) -- M_tilde only dresses the N-1 dispersive bands.
    # ------------------------------------------------------------------ #
    M_tilde_resolved = M_tilde if M_tilde is not None else build_dispersive_M(
        lattice_spec, dispersive_shape, dispersive_strength)
    H_full = build_engineered_hamiltonian(x_k, E0=E0, alpha=alpha, M_tilde=M_tilde_resolved)

    hoppings_full = matrixpoly_to_hoppings(H_full)
    natural_range = max_hopping_range(hoppings_full)
    weight_full = sum(abs(v) ** 2 for v in hoppings_full.values())

    iso_full = _chern.band_isolation(H_full, lattice, E0, 1)
    num_full = _chern.numerical_chern(H_full, lattice, E0, 1)
    sign_conv = _numerical_sign_convention(num_full["C"], target_C)
    full_numerical_C = sign_conv * num_full["C"] if sign_conv else num_full["C"]
    full_match = ((full_numerical_C == target_C) if sign_conv
                   else _chern_matches(num_full["C"], target_C))
    emit(f"[Phase5] H(k)=E0 I + alpha*H_core + H_core*M_tilde*H_core "
         f"(alpha={alpha}, dispersive_shape={dispersive_shape!r}, "
         f"dispersive_strength={dispersive_strength}): "
         f"{len(hoppings_full)} hopping terms, natural range {natural_range} "
         f"cells, isolated={iso_full['isolated']}, numerical C={full_numerical_C} "
         f"(FHS_raw={num_full['C']}, sign_convention={sign_conv})")

    r_max_applied = False
    if r_max is not None and natural_range > r_max:
        hoppings = {key: v for key, v in hoppings_full.items()
                    if max(abs(key[2][0]), abs(key[2][1])) <= r_max}
        H_trunc = hoppings_to_matrixpoly(hoppings, lattice_spec.N)
        r_max_applied = True
        weight_kept = sum(abs(v) ** 2 for v in hoppings.values())
        trunc_ratio = 1.0 - (weight_kept / weight_full if weight_full > 1e-300 else 1.0)
    else:
        hoppings, H_trunc, trunc_ratio = hoppings_full, H_full, 0.0

    max_range = max_hopping_range(hoppings)
    n_terms = len(hoppings)
    flat_dev = flatness_deviation(H_trunc, x_k, lattice_spec, E0)
    iso_trunc = _chern.band_isolation(H_trunc, lattice, E0, 1)
    num_trunc = _chern.numerical_chern(H_trunc, lattice, E0, 1)
    if sign_conv:
        trunc_numerical_C = sign_conv * num_trunc["C"]
        trunc_match = (trunc_numerical_C == target_C)
    else:
        trunc_numerical_C = num_trunc["C"]
        trunc_match = _chern_matches(num_trunc["C"], target_C)
    ift = {"hoppings": hoppings, "R_cut": max_range, "truncation_ratio": trunc_ratio}

    if r_max_applied:
        emit(f"[Phase5] CAPPED to r_max={r_max} (natural range {natural_range}): "
             f"{n_terms} terms, {trunc_ratio*100:.2f}% of ||t||^2 dropped -> "
             f"flatness max||H psi-E0 psi||={flat_dev:.2e} (APPROXIMATE: capping "
             "range breaks exact flatness)")
    else:
        emit(f"[Phase5] EXACT model: {n_terms} hopping terms, max range "
             f"{max_range} cells, flatness max||H psi - E0 psi||={flat_dev:.2e} "
             "(EXACT, no truncation -- flat band survives re-analysis intact)")
    emit(f"[Phase5] H_trunc isolated={iso_trunc['isolated']} "
         f"(touching at the singularity is EXPECTED for nonzero C), "
         f"numerical C={trunc_numerical_C} (FHS_raw={num_trunc['C']})")

    # ------------------------------------------------------------------ #
    # Verification log
    # ------------------------------------------------------------------ #
    # Ground truth is the analytic design (global contour-winding Chern of
    # f(k) == target, independent of M_tilde(k)/truncation). A TOUCHING
    # (non-isolated) truncated band is the EXPECTED outcome for a topological
    # CLS flat band and is accepted on that analytic guarantee; isolation only
    # lets the (otherwise-unreliable) FHS Chern additionally confirm it. See
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
        "alpha": alpha,
        "dispersive_shape": dispersive_shape,
        "dispersive_strength": dispersive_strength,
        "dispersive_gap_below": iso_trunc.get("gap_below"),
        "dispersive_gap_above": iso_trunc.get("gap_above"),
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
        x_k=x_k, H_full=H_full, H_trunc=H_trunc, ift=ift,
        verification=verification, log=log,
    )
