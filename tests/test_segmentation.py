"""Tests for segmentation utility."""

from __future__ import annotations

import math

from bachman_cortex.segmentation import (
    Run,
    group_runs,
    merge_short_runs,
    segment_angle_value,
    segment_confidence_value,
    segment_contact_value,
)


# 10 samples at 10 Hz → 1 sample/0.1s, native_fps = 10.
SAMPLE_PERIOD_S = 0.1
NATIVE_FPS = 10.0


def _samples_to_second(idx: int) -> float:
    return idx / NATIVE_FPS


# ── group_runs ─────────────────────────────────────────────────────────────

def test_group_runs_basic_segmentation():
    values = [True, True, False, False, False, True, True, True, False, True]
    idx = list(range(10))
    runs = group_runs(values, idx)
    assert [(r.state, r.start_sample, r.end_sample) for r in runs] == [
        (True, 0, 2), (False, 2, 5), (True, 5, 8),
        (False, 8, 9), (True, 9, 10),
    ]
    assert runs[0].raw_pass_frame_indices == (0, 1)
    assert runs[1].raw_pass_frame_indices == ()


def test_group_runs_custom_condition():
    vals = [0.2, 0.8, 0.9, 0.1, 0.95]
    runs = group_runs(vals, list(range(5)), condition=lambda v: v >= 0.5)
    assert [r.state for r in runs] == [False, True, False, True]


def test_group_runs_mismatched_lengths_raises():
    import pytest
    with pytest.raises(ValueError):
        group_runs([True, False], [0])


# ── merge_short_runs ──────────────────────────────────────────────────────

def test_middle_short_run_absorbs_into_preceding_neighbour():
    # 5 pass, 2 fail, 3 pass. fail-run duration = 0.3s (2 samples + period).
    # min_duration 0.5s → absorb into preceding pass run (flipping to pass).
    values = ([True] * 5) + ([False] * 2) + ([True] * 3)
    runs = group_runs(values, list(range(10)))
    merged = merge_short_runs(
        runs,
        min_duration_s=0.5,
        sample_to_second=_samples_to_second,
        sample_period_s=SAMPLE_PERIOD_S,
    )
    # Runs: [pass 0..7] [pass 7..10]? No — the second pass run wasn't short,
    # so the absorption results in two adjacent pass runs in the output,
    # which is fine (single pass, no cascade).
    states = [r.state for r in merged]
    assert states[0] is True
    assert all(s is True for s in states)


def test_leading_short_run_absorbs_into_following():
    # 2 fail then 8 pass. fail is 0.3s < 0.5s → absorb into following pass.
    values = ([False] * 2) + ([True] * 8)
    runs = group_runs(values, list(range(10)))
    merged = merge_short_runs(
        runs,
        min_duration_s=0.5,
        sample_to_second=_samples_to_second,
        sample_period_s=SAMPLE_PERIOD_S,
    )
    assert len(merged) == 1
    assert merged[0].state is True
    # Pass-frames set excludes the absorbed fail-frames.
    assert merged[0].raw_pass_frame_indices == tuple(range(2, 10))


def test_whole_video_single_short_run_kept_as_is():
    values = [True, True]
    runs = group_runs(values, [0, 1])
    merged = merge_short_runs(
        runs,
        min_duration_s=5.0,
        sample_to_second=_samples_to_second,
        sample_period_s=SAMPLE_PERIOD_S,
    )
    assert len(merged) == 1
    assert merged[0].state is True


def test_trailing_short_run_absorbs_into_preceding_merged_state():
    # Pass 5, fail 5 (0.5s — at the boundary, NOT short), trailing pass 1
    # (0.1s — short). Expected: pass run survives, fail run survives,
    # trailing short pass absorbs into the preceding fail (flipping state).
    values = ([True] * 5) + ([False] * 5) + [True]
    idx = list(range(11))
    runs = group_runs(values, idx)
    merged = merge_short_runs(
        runs,
        min_duration_s=0.5,
        sample_to_second=_samples_to_second,
        sample_period_s=SAMPLE_PERIOD_S,
    )
    states = [r.state for r in merged]
    assert states == [True, False]
    assert merged[-1].end_sample == 11
    # The absorbed trailing pass-frame (index 10) must NOT count as a
    # qualifying pass-frame in the merged fail-run.
    assert 10 not in merged[-1].raw_pass_frame_indices


def test_no_cascade_after_merge():
    # 2 pass, 5 fail, 2 pass, 1 fail. After first pass: the leading pass
    # (0.2s) absorbs into fail → fail 0..7. Then the trailing pass (0.2s)
    # would absorb into fail → fail 0..9. Then the trailing fail (0.1s)
    # would itself be "short" but there's no cascade, so it absorbs into
    # the preceding fail and the combined run keeps fail state.
    values = ([True] * 2) + ([False] * 5) + ([True] * 2) + [False]
    runs = group_runs(values, list(range(10)))
    merged = merge_short_runs(
        runs,
        min_duration_s=0.5,
        sample_to_second=_samples_to_second,
        sample_period_s=SAMPLE_PERIOD_S,
    )
    # With a single left-to-right pass:
    #   first pass (0.2s) absorbs into following fail → fail 0..7
    #   trailing pass (0.2s) absorbs into preceding fail → fail 0..9
    #   trailing fail (0.1s) absorbs into preceding fail → fail 0..10
    assert len(merged) == 1
    assert merged[0].state is False
    # Pass-frames are excluded from the merged fail run.
    assert merged[0].raw_pass_frame_indices == ()


def test_empty_input_returns_empty():
    assert merge_short_runs(
        [],
        min_duration_s=1.0,
        sample_to_second=_samples_to_second,
        sample_period_s=SAMPLE_PERIOD_S,
    ) == []


# ── Value helpers ─────────────────────────────────────────────────────────

def test_segment_confidence_excludes_absorbed_fail_frames():
    # merged pass-run spanning sample 0..10 (frames 0..9) but only 0..2
    # and 5..7 originally qualified (confidence reported) — we check that
    # conf 0.99 in an absorbed fail frame (index 3) is NOT picked up.
    run = Run(
        state=True,
        start_sample=0, end_sample=10,
        raw_frame_indices=tuple(range(10)),
        raw_pass_frame_indices=(0, 1, 2, 5, 6, 7),
    )
    conf = {0: 0.8, 1: 0.82, 2: 0.9,
            3: 0.99,               # this is in an absorbed fail gap
            5: 0.85, 6: 0.88, 7: 0.87}
    assert segment_confidence_value(run, conf) == 0.9


def test_segment_angle_mean_over_all_frames():
    run = Run(
        state=True,
        start_sample=0, end_sample=3,
        raw_frame_indices=(0, 1, 2),
        raw_pass_frame_indices=(0, 2),
    )
    angles = {0: 10.0, 1: 20.0, 2: 30.0}
    assert segment_angle_value(run, angles) == 20.0


def test_segment_angle_ignores_nan():
    run = Run(
        state=False,
        start_sample=0, end_sample=3,
        raw_frame_indices=(0, 1, 2),
        raw_pass_frame_indices=(),
    )
    angles = {0: float("nan"), 1: 40.0, 2: float("nan")}
    assert segment_angle_value(run, angles) == 40.0


def test_segment_contact_most_common():
    run = Run(
        state=True,
        start_sample=0, end_sample=4,
        raw_frame_indices=(0, 1, 2, 3),
        raw_pass_frame_indices=(0, 1, 2),
    )
    contacts = {0: "P", 1: "P", 2: "F", 3: None}
    assert segment_contact_value(run, contacts) == "P"


def test_segment_contact_none_when_all_none():
    run = Run(
        state=False,
        start_sample=0, end_sample=2,
        raw_frame_indices=(0, 1),
        raw_pass_frame_indices=(),
    )
    assert segment_contact_value(run, {0: None, 1: None}) is None
