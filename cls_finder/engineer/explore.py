"""
Module 7 -- Multi-Candidate Design Explorer.

CLS / flat-band-eigenvector solutions realizing a given (C, w_i) target are
NOT unique: pipeline.design_flat_band() picks ONE (shell-offset, M(k), R_cut)
combination and stops at the first design whose Phase3/4 and Phase5 feedback
loops succeed. This module instead runs the existing, UNMODIFIED
``design_flat_band`` for MANY (shell_offset0, (t, delta), R_cut0) combinations,
collects every result (successes AND failures -- the latter reported as
`error` rather than raised), scores and ranks them, and removes near-duplicate
hopping tables -- giving the user a choice among many valid designs.

``iter_design_attempts`` is a generator yielding each attempt's
``DesignCandidate`` as soon as it is computed (for incremental progress
streaming); ``dedupe_and_rank`` performs the final deduplication/ranking pass;
``explore_designs`` is the convenience wrapper combining both (collect-then-
rank), with an optional ``progress_cb`` for callers that don't need a
generator.

SCOPE NOTE (mirrors pipeline.py's own SCOPE NOTE for
S = len(target.singularities) >= 2): shell_offset0 still applies the SAME
shell rotation to every singularity within one design_flat_band() call
(chiral.shell_for_sublattice's existing per-call `offset`); independent
per-singularity / per-sublattice shell search (the full 8^N space) would
require changing chiral.build_cls_design's signature and is not attempted
here -- a possible future extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from cls_finder.engineer.spec import LatticeSpec, DesignTarget, validate_design
from cls_finder.engineer.pipeline import DesignResult, design_flat_band

DEFAULT_OFFSETS: Sequence[int] = range(8)
DEFAULT_MK_VARIANTS: Sequence[Tuple[float, float]] = ((0.3, 0.5), (0.5, 0.3), (0.2, 0.8))
DEFAULT_R_CUT0_VARIANTS: Sequence[int] = (3,)
DEFAULT_CLS_SIZES: Sequence[Optional[int]] = (None,)


@dataclass
class DesignCandidate:
    """One (shell_offset0, t, delta, R_cut0, cls_size) trial of
    design_flat_band()."""
    index: int
    offset: int
    t: float
    delta: float
    R_cut0: int
    score: float
    cls_size: Optional[int] = None
    result: Optional[DesignResult] = None
    error: Optional[str] = None


def _hoppings_signature(result: DesignResult, decimals: int = 4) -> Tuple:
    """A hashable fingerprint of the truncated hopping table, used to
    deduplicate candidates that converged to the same real-space model."""
    tol = 10 ** (-decimals)
    return tuple(sorted(
        (a, b, n, m, round(val.real, decimals), round(val.imag, decimals))
        for (a, b, (n, m)), val in result.hoppings.items()
        if abs(val) > tol
    ))


def _score(verification: Dict) -> float:
    """Higher is better. Ranking reflects the algorithm's actual goal: realize
    the DESIGNED topology, not produce an isolated band.

    A topological CLS flat band is EXPECTED to touch the dispersive sector at
    its singularity (Note A Sec.8); isolation is therefore NOT a requirement
    and a touching band is NOT penalised. Priority order:
      1. analytic Chern matches the target (the ground truth),
      2. phase5_success (truncated deliverable realises it),
      3. per-singularity feedback consistency,
      4. low truncation weight dropped,
      5. isolation only as a small tiebreak (lets FHS additionally confirm).
    """
    score = 0.0
    if verification.get("analytic_match") or \
            verification.get("analytic_C") == verification.get("target_C"):
        score += 60.0
    if verification.get("phase5_success"):
        score += 100.0
    if verification.get("feedback_success"):
        score += 20.0
    ratio = verification.get("truncation_ratio")
    if ratio is not None:
        score += 5.0 * (1.0 - ratio)
    if verification.get("trunc_isolated"):
        score += 5.0
    return score


def _combos(offsets: Sequence[int], mk_variants: Sequence[Tuple[float, float]],
            R_cut0_variants: Sequence[int], cls_sizes: Sequence[Optional[int]],
            max_candidates: int
            ) -> List[Tuple[int, float, float, int, Optional[int]]]:
    combos = [(offset, t, delta, R_cut0, cls_size)
              for cls_size in cls_sizes
              for offset in offsets
              for (t, delta) in mk_variants
              for R_cut0 in R_cut0_variants]
    return combos[:max_candidates]


def iter_design_attempts(lattice_spec: LatticeSpec, target: DesignTarget,
                         E0: float = 0.0,
                         offsets: Sequence[int] = DEFAULT_OFFSETS,
                         mk_variants: Sequence[Tuple[float, float]] = DEFAULT_MK_VARIANTS,
                         R_cut0_variants: Sequence[int] = DEFAULT_R_CUT0_VARIANTS,
                         cls_sizes: Sequence[Optional[int]] = DEFAULT_CLS_SIZES,
                         n_grid_ift: int = 24, max_retries: int = 2,
                         max_rcut_retries: int = 3, max_candidates: int = 48
                         ) -> Iterator[Tuple[DesignCandidate, int, int]]:
    """Generator: yields ``(candidate, index, total)`` for each
    (shell_offset0, t, delta, R_cut0) combination, one
    ``design_flat_band`` call at a time, BEFORE deduplication/ranking.

    Use this directly for incremental progress streaming; use
    ``explore_designs`` (or ``dedupe_and_rank`` on the collected attempts)
    to get the final ranked, deduplicated list.
    """
    validate_design(lattice_spec, target)
    if not target.singularities:
        raise ValueError(
            "explore_designs requires at least one singularity in `target` "
            "(see design_flat_band)"
        )

    combos = _combos(offsets, mk_variants, R_cut0_variants, cls_sizes, max_candidates)
    total = len(combos)
    for idx, (offset, t, delta, R_cut0, cls_size) in enumerate(combos):
        # In local_exact mode the R_cut0 axis is the hopping-range CAP r_max
        # (0 / non-positive = no cap = the natural exactly-flat range).
        r_max = R_cut0 if (R_cut0 and R_cut0 > 0) else None
        try:
            result = design_flat_band(
                lattice_spec, target, E0=E0, t=t, delta=delta,
                n_grid_ift=n_grid_ift, r_max=r_max, cls_size=cls_size,
                max_retries=max_retries, max_rcut_retries=max_rcut_retries,
                shell_offset0=offset, verbose=False)
            cand = DesignCandidate(idx, offset, t, delta, R_cut0,
                                   _score(result.verification),
                                   cls_size=cls_size, result=result)
        except Exception as e:
            cand = DesignCandidate(idx, offset, t, delta, R_cut0, -1000.0,
                                   cls_size=cls_size,
                                   error=f"{type(e).__name__}: {e}")
        yield cand, idx, total


def dedupe_and_rank(attempts: List[DesignCandidate]) -> List[DesignCandidate]:
    """Deduplicate by hopping-table signature (keeping the best-scoring
    representative of each unique design) and sort best-first by score.
    Failed attempts (``result is None``) are kept, sorted to the bottom."""
    by_sig: Dict[Tuple, DesignCandidate] = {}
    failed: List[DesignCandidate] = []
    for cand in attempts:
        if cand.result is None:
            failed.append(cand)
            continue
        sig = _hoppings_signature(cand.result)
        if sig not in by_sig or cand.score > by_sig[sig].score:
            by_sig[sig] = cand

    deduped = list(by_sig.values()) + failed
    deduped.sort(key=lambda c: c.score, reverse=True)
    return deduped


def explore_designs(lattice_spec: LatticeSpec, target: DesignTarget,
                    E0: float = 0.0,
                    offsets: Sequence[int] = DEFAULT_OFFSETS,
                    mk_variants: Sequence[Tuple[float, float]] = DEFAULT_MK_VARIANTS,
                    R_cut0_variants: Sequence[int] = DEFAULT_R_CUT0_VARIANTS,
                    cls_sizes: Sequence[Optional[int]] = DEFAULT_CLS_SIZES,
                    n_grid_ift: int = 24, max_retries: int = 2,
                    max_rcut_retries: int = 3, max_candidates: int = 48,
                    progress_cb: Optional[Callable[[DesignCandidate, int, int], None]] = None
                    ) -> List[DesignCandidate]:
    """Enumerate offsets x mk_variants x R_cut0_variants (capped at
    max_candidates attempts), run design_flat_band() for each, score,
    deduplicate by hopping-table signature (keeping the best-scoring
    representative), and return candidates ranked best-first.

    progress_cb(candidate, index, total), if given, is called once per
    attempt (before deduplication) -- e.g. to drive a non-generator progress
    UI. For incremental streaming, iterate ``iter_design_attempts`` directly.
    """
    attempts: List[DesignCandidate] = []
    for cand, idx, total in iter_design_attempts(
            lattice_spec, target, E0=E0, offsets=offsets,
            mk_variants=mk_variants, R_cut0_variants=R_cut0_variants,
            cls_sizes=cls_sizes, n_grid_ift=n_grid_ift, max_retries=max_retries,
            max_rcut_retries=max_rcut_retries, max_candidates=max_candidates):
        attempts.append(cand)
        if progress_cb is not None:
            progress_cb(cand, idx, total)

    return dedupe_and_rank(attempts)
