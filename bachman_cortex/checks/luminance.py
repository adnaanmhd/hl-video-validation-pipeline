"""Luminance + flicker accumulator.

Per-frame: bucket each sampled frame into {dead_black, too_dark, usable,
blown_out} and hold onto mean-luminance for the rolling flicker window.
Finalize: fail if fewer than `good_frame_ratio` of sampled frames were
usable-and-non-flickering.

`luminance_class` encoding (stored in the parquet):
    0 = dead_black   (mean < dead_black_max)
    1 = too_dark     (dead_black_max <= mean < too_dark_max)
    2 = usable       (too_dark_max <= mean <= blown_out_min)
    3 = blown_out    (mean > blown_out_min)

`luminance_flicker` is True for each frame inside any
`flicker_window`-sized rolling window whose stddev exceeds
`flicker_stddev_threshold`.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LuminanceThresholds:
    good_frame_ratio: float = 0.50
    dead_black_max: float = 15.0
    too_dark_max: float = 30.0
    blown_out_min: float = 230.0
    flicker_window: int = 10
    flicker_stddev_threshold: float = 30.0


@dataclass
class LuminanceFinalizeResult:
    pass_fail: bool
    detected: str
    class_array: list[int]         # per-sampled-frame 0..3
    flicker_array: list[bool]      # per-sampled-frame
    sample_indices: list[int]      # native frame idx of each sample
    good_ratio: float


def _classify(mean: float, th: LuminanceThresholds) -> int:
    if mean < th.dead_black_max:
        return 0
    if mean < th.too_dark_max:
        return 1
    if mean > th.blown_out_min:
        return 3
    return 2


@dataclass
class LuminanceAccumulator:
    """Consume 360p frames at luminance cadence; finalise at end of video.

    The accumulator is agnostic to total frame count — it grows only
    with the number of samples actually processed.
    """

    thresholds: LuminanceThresholds = field(default_factory=LuminanceThresholds)

    _means: list[float] = field(init=False, default_factory=list)
    _classes: list[int] = field(init=False, default_factory=list)
    _sample_indices: list[int] = field(init=False, default_factory=list)

    def process_frame(self, frame_360p: np.ndarray, frame_idx: int) -> None:
        """Add a sampled frame. Caller is responsible for the cadence gate."""
        gray = cv2.cvtColor(frame_360p, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        self._means.append(mean)
        self._classes.append(_classify(mean, self.thresholds))
        self._sample_indices.append(frame_idx)

    def finalize(self) -> LuminanceFinalizeResult:
        """Derive pass/fail + per-sample class + flicker arrays."""
        th = self.thresholds
        n = len(self._means)
        if n == 0:
            return LuminanceFinalizeResult(
                pass_fail=False,
                detected="no_samples",
                class_array=[],
                flicker_array=[],
                sample_indices=[],
                good_ratio=0.0,
            )

        flicker = [False] * n
        if n >= th.flicker_window:
            means = np.asarray(self._means)
            win = th.flicker_window
            # Rolling stddev via cumulative sums: O(n).
            cumsum = np.concatenate(([0.0], np.cumsum(means)))
            cumsum_sq = np.concatenate(([0.0], np.cumsum(means ** 2)))
            # For each window starting at k, mean = (cumsum[k+w]-cumsum[k])/w.
            for start in range(n - win + 1):
                s = cumsum[start + win] - cumsum[start]
                ss = cumsum_sq[start + win] - cumsum_sq[start]
                mean_w = s / win
                var_w = max(0.0, ss / win - mean_w ** 2)
                if np.sqrt(var_w) > th.flicker_stddev_threshold:
                    for k in range(start, start + win):
                        flicker[k] = True

        good = sum(
            1 for i in range(n)
            if self._classes[i] == 2 and not flicker[i]
        )
        ratio = good / n
        passes = ratio >= th.good_frame_ratio
        detected = f"good_ratio={ratio:.3f}"

        return LuminanceFinalizeResult(
            pass_fail=passes,
            detected=detected,
            class_array=list(self._classes),
            flicker_array=flicker,
            sample_indices=list(self._sample_indices),
            good_ratio=round(ratio, 4),
        )
