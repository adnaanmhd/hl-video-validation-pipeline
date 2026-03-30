"""Check 6: View Obstruction.

Criterion: <= 10% frames obstructed.
Purpose: Detect lens coverage/blockage (finger over lens, fogged lens, cap).
"""

import cv2
import numpy as np
from ml_checks.checks.check_results import CheckResult


def is_frame_obstructed(
    frame_bgr: np.ndarray,
    variance_threshold: float = 200.0,
    brightness_threshold: float = 25.0,
    edge_density_threshold: float = 0.02,
    color_uniformity_threshold: float = 0.80,
    min_signals: int = 2,
) -> bool:
    """Check if a single frame is obstructed using multi-signal heuristic.

    Combines 4 signals on the central 80% of the frame:
    1. Low spatial variance (homogeneous region)
    2. Low brightness
    3. Low edge density
    4. Color channel uniformity

    Returns True if >= min_signals signals trigger.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Central 80% crop
    y1, y2 = h // 10, 9 * h // 10
    x1, x2 = w // 10, 9 * w // 10
    center_gray = gray[y1:y2, x1:x2]
    center_bgr = frame_bgr[y1:y2, x1:x2]

    signals_triggered = 0

    # Signal 1: Low spatial variance
    variance = np.var(center_gray.astype(np.float32))
    if variance < variance_threshold:
        signals_triggered += 1

    # Signal 2: Low brightness
    brightness = np.mean(center_gray)
    if brightness < brightness_threshold:
        signals_triggered += 1

    # Signal 3: Low edge density
    edges = cv2.Canny(center_gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if edge_density < edge_density_threshold:
        signals_triggered += 1

    # Signal 4: Color channel uniformity (dominant color bin > threshold)
    for ch in range(3):
        hist = cv2.calcHist([center_bgr], [ch], None, [32], [0, 256])
        dominant_ratio = hist.max() / hist.sum()
        if dominant_ratio > color_uniformity_threshold:
            signals_triggered += 1
            break

    return signals_triggered >= min_signals


def check_view_obstruction(
    frames: list[np.ndarray],
    max_obstructed_ratio: float = 0.10,
) -> CheckResult:
    """Check that the camera view is not obstructed in too many frames.

    Args:
        frames: List of BGR frames.
        max_obstructed_ratio: Max fraction of frames allowed to be obstructed.

    Returns:
        CheckResult.
    """
    total_frames = len(frames)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    obstructed_count = 0
    for frame in frames:
        if is_frame_obstructed(frame):
            obstructed_count += 1

    obstructed_ratio = obstructed_count / total_frames
    clear_ratio = 1.0 - obstructed_ratio
    status = "pass" if obstructed_ratio <= max_obstructed_ratio else "fail"

    # Confidence
    if obstructed_ratio <= max_obstructed_ratio:
        margin = max_obstructed_ratio - obstructed_ratio
        confidence = min(1.0, 0.7 + margin * 3)
    else:
        excess = obstructed_ratio - max_obstructed_ratio
        confidence = max(0.0, 0.5 - excess * 3)

    return CheckResult(
        status=status,
        metric_value=round(clear_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "obstructed_frames": obstructed_count,
            "total_frames": total_frames,
            "obstructed_ratio": round(obstructed_ratio, 4),
            "max_allowed_ratio": max_obstructed_ratio,
        },
    )
