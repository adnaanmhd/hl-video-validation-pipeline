"""Unit tests for MotionAnalyzer.finalize_whole_video.

Fabricated frames + direct injection of analyzer state let us verify:
- per-frame jitter broadcast across the whole second,
- frozen runs materialise into the per-native-frame boolean mask.
"""

from __future__ import annotations

import numpy as np

from bachman_cortex.checks.motion_analysis import (
    FrozenThresholds,
    MotionAnalyzer,
    StabilityThresholds,
)


def _make_analyzer(native_fps: float, total_frames: int) -> MotionAnalyzer:
    return MotionAnalyzer(
        native_fps=native_fps,
        total_frames=total_frames,
        target_fps=native_fps,   # process every frame (skip=1)
    )


def test_per_frame_jitter_broadcasts_to_every_frame_in_the_second():
    a = _make_analyzer(native_fps=10.0, total_frames=30)
    # 3 seconds, inject crafted translations that score > 0 only in sec 1.
    a.per_second_trans = {0: [0.0] * 5, 1: [50.0] * 5, 2: [0.0] * 5}
    a.per_second_rot = {0: [0.0] * 5, 1: [1.0] * 5, 2: [0.0] * 5}

    # Disable the high-pass filter's smoothing effect — window 1 is a no-op.
    r = a.finalize_whole_video(
        stability=StabilityThresholds(highpass_window_sec=0.01),
    )

    # Sec 0 and 2 have zero motion → per-frame jitter 0.0. Sec 1 is the
    # shaky second → score > 0 broadcast across its 10 native frames.
    assert all(r.per_frame_jitter[i] == 0.0 for i in range(0, 10))
    assert all(
        r.per_frame_jitter[i] is not None and r.per_frame_jitter[i] > 0.0
        for i in range(10, 20)
    )
    assert all(r.per_frame_jitter[i] == 0.0 for i in range(20, 30))
    assert r.shaky_seconds == [1]


def test_frozen_mask_spans_detected_run_in_native_frames():
    a = _make_analyzer(native_fps=10.0, total_frames=60)
    # 50-frame freeze from idx 5..54 inclusive. skip=1, max_consecutive=30
    # → eff_max=30 → 50 > 30 triggers fail + a frozen run.
    a.sampled = [
        (i, 0.0 if 5 <= i < 55 else 10.0, 0.0 if 5 <= i < 55 else 1.0)
        for i in range(60)
    ]
    # Supply some per-second entries so the stability finalise doesn't
    # choke on the empty path.
    a.per_second_trans = {s: [10.0] * 5 for s in range(6)}
    a.per_second_rot = {s: [1.0] * 5 for s in range(6)}

    r = a.finalize_whole_video(frozen=FrozenThresholds(max_consecutive=30))

    assert not r.frozen_pass
    # Pre-run: no frozen flag. Inside the run: True. Post-run: False.
    assert not any(r.per_frame_frozen[:5])
    assert all(r.per_frame_frozen[5:55])
    assert not any(r.per_frame_frozen[55:])


def test_short_frozen_run_does_not_fail_or_flag_frames():
    a = _make_analyzer(native_fps=10.0, total_frames=30)
    # 10-frame freeze — below max_consecutive=30 → pass.
    a.sampled = [
        (i, 0.0 if 5 <= i < 15 else 10.0, 0.0 if 5 <= i < 15 else 1.0)
        for i in range(30)
    ]
    a.per_second_trans = {s: [10.0] * 5 for s in range(3)}
    a.per_second_rot = {s: [1.0] * 5 for s in range(3)}

    r = a.finalize_whole_video(frozen=FrozenThresholds(max_consecutive=30))
    assert r.frozen_pass
    # Short runs are not written into the per-frame mask.
    assert not any(r.per_frame_frozen)


def test_empty_video_yields_passes_with_null_jitter():
    a = _make_analyzer(native_fps=30.0, total_frames=0)
    r = a.finalize_whole_video()
    assert r.stability_pass is True
    assert r.frozen_pass is True
    assert r.per_frame_jitter == []
    assert r.per_frame_frozen == []
