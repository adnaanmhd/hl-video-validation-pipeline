"""Check 7: POV-Hand Angle.

Criterion: Angle between camera POV center and detected hands < 40 degrees
           in >= 80% of sampled frames.
Purpose: Ensure hands are in the working area of the frame (relevant for robot learning).
"""

import math
import numpy as np
from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.models.hand_detector import HandDetection


def compute_hand_angle(
    hand_bbox: np.ndarray,
    frame_width: int,
    frame_height: int,
    diagonal_fov_degrees: float = 90.0,
) -> float:
    """Compute angle from frame center to hand bbox center.

    Args:
        hand_bbox: [x1, y1, x2, y2] bounding box.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        diagonal_fov_degrees: Assumed camera diagonal field of view.

    Returns:
        Angle in degrees from frame center to hand center.
    """
    # Hand center
    hx = (hand_bbox[0] + hand_bbox[2]) / 2
    hy = (hand_bbox[1] + hand_bbox[3]) / 2

    # Frame center
    cx = frame_width / 2
    cy = frame_height / 2

    # Pixel distance from center
    dx = hx - cx
    dy = hy - cy
    pixel_dist = math.sqrt(dx ** 2 + dy ** 2)

    # Frame diagonal half-length
    half_diagonal = math.sqrt(cx ** 2 + cy ** 2)

    # Normalized distance (0 = center, 1 = corner)
    normalized_dist = pixel_dist / half_diagonal

    # Map to angle using FOV
    angle = normalized_dist * (diagonal_fov_degrees / 2)

    return angle


def check_pov_hand_angle(
    per_frame_hands: list[list[HandDetection]],
    frame_dims: tuple[int, int],  # (height, width)
    angle_threshold: float = 40.0,
    pass_rate_threshold: float = 0.80,
    diagonal_fov_degrees: float = 90.0,
) -> CheckResult:
    """Check that detected hands are within the working area of the frame.

    For each frame, computes the angle from frame center to each detected hand.
    The frame passes if ALL detected hands are within the angle threshold.
    Frames with no hands are counted as failures.

    Args:
        per_frame_hands: Hand detections per frame.
        frame_dims: (height, width).
        angle_threshold: Max angle in degrees from center.
        pass_rate_threshold: Fraction of frames that must pass.
        diagonal_fov_degrees: Assumed camera diagonal FOV.

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_hands)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    frame_h, frame_w = frame_dims
    passing_frames = 0
    all_angles = []

    for hands in per_frame_hands:
        if not hands:
            # No hands detected = frame fails
            continue

        frame_passes = True
        for hand in hands:
            angle = compute_hand_angle(
                hand.bbox, frame_w, frame_h, diagonal_fov_degrees
            )
            all_angles.append(angle)
            if angle >= angle_threshold:
                frame_passes = False

        if frame_passes:
            passing_frames += 1

    pass_ratio = passing_frames / total_frames
    status = "pass" if pass_ratio >= pass_rate_threshold else "fail"

    # Confidence
    if pass_ratio >= pass_rate_threshold:
        margin = pass_ratio - pass_rate_threshold
        confidence = min(1.0, 0.7 + margin * 3)
    else:
        deficit = pass_rate_threshold - pass_ratio
        confidence = max(0.0, 0.5 - deficit * 2)

    return CheckResult(
        status=status,
        metric_value=round(pass_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "passing_frames": passing_frames,
            "total_frames": total_frames,
            "angle_threshold": angle_threshold,
            "pass_rate_threshold": pass_rate_threshold,
            "diagonal_fov_degrees": diagonal_fov_degrees,
            "mean_angle": round(np.mean(all_angles), 2) if all_angles else None,
            "max_angle": round(max(all_angles), 2) if all_angles else None,
        },
    )
