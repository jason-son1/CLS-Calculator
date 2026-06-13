"""
Multi-Candidate Design Explorer (cls_finder.engineer.explore).

Covers:
  - explore_designs runs the existing, unmodified design_flat_band() across
    a small grid of (shell_offset0, (t, delta), R_cut0) combinations for the
    |C|=1 minimal construction, returns DesignCandidates sorted best-first by
    score, with at least one fully-successful (phase5_success) candidate.
  - progress_cb is invoked once per attempt (before deduplication), and
    max_candidates caps the number of attempts.
  - explore_designs re-validates the target up front (same checks as
    design_flat_band's Phase1: sum(w_i)==C and singularities non-empty).
"""
import numpy as np
import pytest

from cls_finder.engineer import (
    SublatticeSpec, LatticeSpec, SingularityTarget, DesignTarget,
    DesignCandidate, explore_designs,
)

_PRIM = np.array([[1.0, 0.0], [0.0, 1.0]])


def _square_lattice():
    return LatticeSpec(primitive_vectors=_PRIM,
                        sublattices=[SublatticeSpec('A'), SublatticeSpec('B')])


def test_explore_designs_returns_ranked_nonempty_candidates():
    lat = _square_lattice()
    target = DesignTarget(C=1, singularities=[SingularityTarget('Gamma', 'Gamma', 1)])
    candidates = explore_designs(
        lat, target, offsets=range(2), mk_variants=((0.3, 0.5), (0.5, 0.3)),
        R_cut0_variants=(3,), max_candidates=4)

    assert len(candidates) >= 1
    assert all(isinstance(c, DesignCandidate) for c in candidates)

    scores = [c.score for c in candidates]
    assert scores == sorted(scores, reverse=True)

    assert any(c.result is not None and c.result.verification.get('phase5_success')
               for c in candidates)


def test_explore_designs_progress_cb_and_max_candidates():
    lat = _square_lattice()
    target = DesignTarget(C=1, singularities=[SingularityTarget('Gamma', 'Gamma', 1)])

    calls = []

    def cb(cand, idx, total):
        calls.append((idx, total))

    candidates = explore_designs(
        lat, target, offsets=range(8), mk_variants=((0.3, 0.5),),
        R_cut0_variants=(3,), max_candidates=3, progress_cb=cb)

    assert len(calls) == 3
    assert all(total == 3 for _, total in calls)
    # every attempt is either a successful result or a recorded error, never both
    for c in candidates:
        assert (c.result is None) != (c.error is None)


def test_explore_designs_rejects_inconsistent_target():
    lat = _square_lattice()
    target = DesignTarget(C=1, singularities=[SingularityTarget('Gamma', 'Gamma', -1)])
    with pytest.raises(ValueError):
        explore_designs(lat, target, offsets=range(1), mk_variants=((0.3, 0.5),))


def test_explore_designs_rejects_empty_singularities():
    lat = _square_lattice()
    target = DesignTarget(C=0, singularities=[])
    with pytest.raises(ValueError):
        explore_designs(lat, target, offsets=range(1), mk_variants=((0.3, 0.5),))


def test_explore_designs_with_cls_sizes():
    lat = _square_lattice()
    target = DesignTarget(C=1, singularities=[SingularityTarget('Gamma', 'Gamma', 1)])
    candidates = explore_designs(
        lat, target, offsets=range(1), mk_variants=((0.3, 0.5),),
        R_cut0_variants=(3,), cls_sizes=(None, 1), max_candidates=4
    )
    assert len(candidates) >= 1
    cls_sizes_found = {c.cls_size for c in candidates}
    assert None in cls_sizes_found or 1 in cls_sizes_found

