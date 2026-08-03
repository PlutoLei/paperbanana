"""Tests for _select_final_iteration (argmax final-image selection)."""

from paperbanana.core.pipeline import _select_final_iteration
from paperbanana.core.types import CritiqueResult, IterationRecord


def _rec(n: int, score):
    critique = None
    if score is not None:
        critique = CritiqueResult(critic_suggestions=[], score=score)
    return IterationRecord(
        iteration=n, description="d", image_path=f"iter_{n}.png", critique=critique
    )


def test_picks_highest_score():
    its = [_rec(1, 9.5), _rec(2, 8.0), _rec(3, 9.0)]
    assert _select_final_iteration(its) is its[0]


def test_ties_prefer_latest():
    its = [_rec(1, 8.5), _rec(2, 8.5), _rec(3, 8.0)]
    assert _select_final_iteration(its) is its[1]


def test_no_scores_falls_back_to_last():
    its = [_rec(1, None), _rec(2, None)]
    assert _select_final_iteration(its) is its[-1]


def test_unscored_iterations_skipped():
    its = [_rec(1, 7.0), _rec(2, None), _rec(3, 6.5)]
    assert _select_final_iteration(its) is its[0]


def test_single_iteration():
    its = [_rec(1, 8.5)]
    assert _select_final_iteration(its) is its[0]
