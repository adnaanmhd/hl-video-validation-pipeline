"""Frame-level quality checks.

Operates on sampled frames (same as ML checks). Three checks:
average brightness, brightness stability, and near-black frame detection.
"""

import cv2
import numpy as np

from bachman_cortex.checks.check_results import CheckResult


def check_average_brightness(
    frames: list[np.ndarray],
    min_brightness: float = 40.0,
) -> CheckResult:
    """Check that mean grayscale intensity across all frames >= threshold.

    Args:
        frames: List of BGR frames.
        min_brightness: Minimum acceptable mean brightness (0-255).
    """
    if not frames:
        return CheckResult(status="fail", metric_value=0.0, confidence=1.0,
                           details={"error": "no frames"})

    per_frame_means = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        per_frame_means.append(float(np.mean(gray)))

    overall_mean = float(np.mean(per_frame_means))
    passes = overall_mean >= min_brightness

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(overall_mean, 2),
        confidence=1.0,
        details={
            "mean_brightness": round(overall_mean, 2),
            "min_brightness": min_brightness,
            "per_frame_min": round(min(per_frame_means), 2),
            "per_frame_max": round(max(per_frame_means), 2),
            "total_frames": len(frames),
        },
    )


def check_brightness_stability(
    frames: list[np.ndarray],
    max_std_dev: float = 60.0,
) -> CheckResult:
    """Check that brightness std dev across frames <= threshold.

    Args:
        frames: List of BGR frames.
        max_std_dev: Maximum acceptable std dev of per-frame mean brightness.
    """
    if not frames:
        return CheckResult(status="fail", metric_value=0.0, confidence=1.0,
                           details={"error": "no frames"})

    per_frame_means = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        per_frame_means.append(float(np.mean(gray)))

    std_dev = float(np.std(per_frame_means))
    passes = std_dev <= max_std_dev

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(std_dev, 2),
        confidence=1.0,
        details={
            "brightness_std_dev": round(std_dev, 2),
            "max_std_dev": max_std_dev,
            "mean_brightness": round(float(np.mean(per_frame_means)), 2),
            "total_frames": len(frames),
        },
    )


def check_near_black_frames(
    frames: list[np.ndarray],
    min_pixel_mean: float = 10.0,
) -> CheckResult:
    """Check that all frames have mean pixel value >= threshold.

    Args:
        frames: List of BGR frames.
        min_pixel_mean: Minimum mean pixel value per frame.
    """
    if not frames:
        return CheckResult(status="fail", metric_value=0.0, confidence=1.0,
                           details={"error": "no frames"})

    black_frame_indices = []
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < min_pixel_mean:
            black_frame_indices.append(i)

    clean_ratio = 1.0 - len(black_frame_indices) / len(frames)
    passes = len(black_frame_indices) == 0

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(clean_ratio, 4),
        confidence=1.0,
        details={
            "near_black_frames": len(black_frame_indices),
            "total_frames": len(frames),
            "min_pixel_mean": min_pixel_mean,
            "black_frame_indices": black_frame_indices[:20],
        },
    )
