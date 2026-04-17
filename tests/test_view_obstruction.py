"""Tests for ObstructionAccumulator."""

import numpy as np

from bachman_cortex.checks.view_obstruction import (
    ObstructionAccumulator,
    ObstructionThresholds,
    is_frame_obstructed,
)


def _solid_black(h=80, w=80) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _textured(h=80, w=80) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_black_frame_is_obstructed():
    assert is_frame_obstructed(_solid_black()) is True


def test_high_entropy_frame_is_not_obstructed():
    assert is_frame_obstructed(_textured()) is False


def test_accumulator_tracks_per_frame_booleans_and_ratio():
    acc = ObstructionAccumulator()
    for i in range(3):
        acc.process_frame(_solid_black(), i * 2)
    for i in range(7):
        acc.process_frame(_textured(), 6 + i * 2)
    r = acc.finalize()
    assert r.obstructed_array[:3] == [True, True, True]
    assert not any(r.obstructed_array[3:])
    assert r.obstructed_ratio == 0.3
    assert r.sample_indices == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


def test_empty_accumulator_reports_zero():
    r = ObstructionAccumulator().finalize()
    assert r.obstructed_array == []
    assert r.obstructed_ratio == 0.0
