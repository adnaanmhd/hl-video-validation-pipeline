"""Per-frame obstruction accumulator ("obstructed" metric).

Signals (all on the central 80% crop): low spatial variance, low
brightness, low Canny edge density, dominant-colour-channel uniformity.
A frame is obstructed when >=2 of the 4 signals trigger. This metric is
per-frame boolean — the scoring engine builds segments at report time
from the full per-frame array.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObstructionThresholds:
    variance_threshold: float = 200.0
    brightness_threshold: float = 25.0
    edge_density_threshold: float = 0.02
    color_uniformity_threshold: float = 0.80
    min_signals: int = 2


@dataclass
class ObstructionFinalizeResult:
    obstructed_array: list[bool]    # per-sampled-frame
    sample_indices: list[int]
    obstructed_ratio: float


def is_frame_obstructed(
    frame_bgr: np.ndarray,
    th: ObstructionThresholds | None = None,
) -> bool:
    t = th or ObstructionThresholds()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    y1, y2 = h // 10, 9 * h // 10
    x1, x2 = w // 10, 9 * w // 10
    center_gray = gray[y1:y2, x1:x2]
    center_bgr = frame_bgr[y1:y2, x1:x2]

    signals = 0
    if np.var(center_gray.astype(np.float32)) < t.variance_threshold:
        signals += 1
    if float(np.mean(center_gray)) < t.brightness_threshold:
        signals += 1
    edges = cv2.Canny(center_gray, 50, 150)
    if (float(np.sum(edges > 0)) / edges.size) < t.edge_density_threshold:
        signals += 1
    for ch in range(3):
        hist = cv2.calcHist([center_bgr], [ch], None, [32], [0, 256])
        dominant = float(hist.max()) / float(hist.sum())
        if dominant > t.color_uniformity_threshold:
            signals += 1
            break

    return signals >= t.min_signals


@dataclass
class ObstructionAccumulator:
    thresholds: ObstructionThresholds = field(default_factory=ObstructionThresholds)

    _obstructed: list[bool] = field(init=False, default_factory=list)
    _sample_indices: list[int] = field(init=False, default_factory=list)

    def process_frame(self, frame_720p: np.ndarray, frame_idx: int) -> None:
        self._obstructed.append(is_frame_obstructed(frame_720p, self.thresholds))
        self._sample_indices.append(frame_idx)

    def finalize(self) -> ObstructionFinalizeResult:
        n = len(self._obstructed)
        ratio = (sum(self._obstructed) / n) if n else 0.0
        return ObstructionFinalizeResult(
            obstructed_array=list(self._obstructed),
            sample_indices=list(self._sample_indices),
            obstructed_ratio=round(ratio, 4),
        )
