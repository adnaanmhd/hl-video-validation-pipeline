"""Luminance & blur quality check with per-frame classification and segment analysis.

Implements the decision table from checks.md:
- Dead black / too dark / blown out -> reject
- Low light noise zone -> raw Tenengrad thresholds
- Normal range / soft overexposed -> normalized Tenengrad thresholds

Video-level: accept if (accept + review) frames >= 80% of total.
"""

import cv2
import numpy as np
from dataclasses import dataclass

from ml_checks.checks.check_results import CheckResult


@dataclass
class LuminanceBlurConfig:
    """Thresholds for the luminance/blur decision table."""
    # Luminance zones (0-255)
    lum_hard_dark: float = 20.0
    lum_soft_dark: float = 40.0
    lum_noise_zone: float = 70.0
    lum_soft_overexp: float = 210.0
    lum_hard_overexp: float = 235.0

    # Normalized Tenengrad (mean(G^2) / mean(I)^2)
    tn_hard_reject: float = 0.04
    tn_soft_reject: float = 0.10
    tn_sharp: float = 0.30

    # Raw Tenengrad fallback (noise zone only)
    tr_hard_reject: float = 80.0
    tr_soft_reject: float = 200.0

    # Video-level
    min_good_ratio: float = 0.80

    # Brightness stability
    max_brightness_std: float = 60.0


@dataclass
class FrameMetrics:
    """Per-frame luminance and blur metrics."""
    frame_idx: int
    mean_luminance: float
    tenengrad_raw: float
    tenengrad_norm: float
    label: str      # accept, review, reject
    reject_reason: str


def compute_frame_metrics(frame: np.ndarray, frame_idx: int) -> FrameMetrics:
    """Compute luminance and Tenengrad metrics for a single frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    g_squared = gx ** 2 + gy ** 2

    mean_i = float(np.mean(gray))
    mean_g2 = float(np.mean(g_squared))

    tn_norm = mean_g2 / (mean_i ** 2 + 1e-6)
    tn_raw = mean_g2

    return FrameMetrics(
        frame_idx=frame_idx,
        mean_luminance=mean_i,
        tenengrad_raw=tn_raw,
        tenengrad_norm=tn_norm,
        label="",
        reject_reason="",
    )


def classify_frame(m: FrameMetrics, cfg: LuminanceBlurConfig) -> FrameMetrics:
    """Classify a frame using the luminance/blur decision table."""
    lum = m.mean_luminance
    tn = m.tenengrad_norm
    tr = m.tenengrad_raw

    # Dead black
    if lum < cfg.lum_hard_dark:
        m.label, m.reject_reason = "reject", "dead_black"
        return m

    # Too dark
    if lum < cfg.lum_soft_dark:
        m.label, m.reject_reason = "reject", "too_dark"
        return m

    # Blown out
    if lum > cfg.lum_hard_overexp:
        m.label, m.reject_reason = "reject", "blown_out"
        return m

    # Low light / noise zone: normalized Tenengrad unreliable, use raw
    if lum < cfg.lum_noise_zone:
        if tr < cfg.tr_hard_reject:
            m.label, m.reject_reason = "reject", "blur_lowlight"
        elif tr < cfg.tr_soft_reject:
            m.label = "review"
        else:
            m.label = "accept"
        return m

    # Soft overexposed zone
    if lum > cfg.lum_soft_overexp:
        if tn < cfg.tn_hard_reject:
            m.label, m.reject_reason = "reject", "blur_overexposed"
        elif tn < cfg.tn_soft_reject:
            m.label = "review"
        else:
            m.label = "accept"
        return m

    # Normal range (70-210): use normalized Tenengrad
    if tn < cfg.tn_hard_reject:
        m.label, m.reject_reason = "reject", "blur"
    elif tn < cfg.tn_soft_reject:
        m.label = "review"
    else:
        m.label = "accept"

    return m


@dataclass
class Segment:
    """A contiguous run of good or bad frames."""
    start: int
    end: int        # exclusive
    kind: str       # "good" or "bad"

    @property
    def length(self) -> int:
        return self.end - self.start


def find_segments(labels: list[str]) -> list[Segment]:
    """Collapse per-frame labels into contiguous good/bad segments.

    "review" counts as good for segmentation purposes.
    """
    if not labels:
        return []

    is_good = [label != "reject" for label in labels]
    segments = []
    i = 0
    n = len(is_good)

    while i < n:
        j = i + 1
        while j < n and is_good[j] == is_good[i]:
            j += 1
        kind = "good" if is_good[i] else "bad"
        segments.append(Segment(start=i, end=j, kind=kind))
        i = j

    return segments


def check_luminance_blur(
    frames: list[np.ndarray],
    config: LuminanceBlurConfig | None = None,
) -> CheckResult:
    """Run luminance/blur classification on all frames with segment analysis.

    Pass if (accept + review) frames >= min_good_ratio of total frames.
    """
    cfg = config or LuminanceBlurConfig()

    if not frames:
        return CheckResult(status="fail", metric_value=0.0, confidence=1.0,
                           details={"error": "no frames"})

    # Classify each frame
    all_metrics = []
    for i, frame in enumerate(frames):
        m = compute_frame_metrics(frame, i)
        m = classify_frame(m, cfg)
        all_metrics.append(m)

    labels = [m.label for m in all_metrics]
    accept_count = sum(1 for l in labels if l == "accept")
    review_count = sum(1 for l in labels if l == "review")
    reject_count = sum(1 for l in labels if l == "reject")
    good_ratio = (accept_count + review_count) / len(labels)

    # Brightness stability: std dev of per-frame mean luminance
    per_frame_luminances = [m.mean_luminance for m in all_metrics]
    brightness_std = float(np.std(per_frame_luminances))
    brightness_stable = brightness_std <= cfg.max_brightness_std

    # Segment analysis
    segments = find_segments(labels)

    # Video-level decision: must pass both good-ratio and brightness stability
    if good_ratio >= cfg.min_good_ratio and brightness_stable:
        status = "pass"
    else:
        status = "fail"

    # Rejection reason breakdown
    reject_reasons = {}
    for m in all_metrics:
        if m.reject_reason:
            reject_reasons[m.reject_reason] = reject_reasons.get(m.reject_reason, 0) + 1

    return CheckResult(
        status=status,
        metric_value=round(good_ratio, 4),
        confidence=1.0,
        details={
            "accept_frames": accept_count,
            "review_frames": review_count,
            "reject_frames": reject_count,
            "total_frames": len(frames),
            "good_ratio": round(good_ratio, 4),
            "min_good_ratio": cfg.min_good_ratio,
            "brightness_std": round(brightness_std, 2),
            "max_brightness_std": cfg.max_brightness_std,
            "brightness_stable": brightness_stable,
            "reject_reasons": reject_reasons,
            "segments": [
                {"start": s.start, "end": s.end, "kind": s.kind, "length": s.length}
                for s in segments
            ],
        },
    )
