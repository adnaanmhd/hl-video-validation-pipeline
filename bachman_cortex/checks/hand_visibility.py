"""Check 3: Hand Visibility.

Criterion: Both hands detected (>= 0.7 confidence) and fully within the
frame (bbox at least 2 px from every edge) in >= 90% of frames.
Purpose: Ensure wearer's hands are completely visible for robot learning.
"""

import numpy as np

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.models.hand_detector import HandDetection, HandSide


def _hand_in_frame(bbox: np.ndarray, frame_w: int, frame_h: int,
                   margin: int) -> bool:
    """Return True if the hand bbox has at least *margin* px clearance from
    every frame edge."""
    x1, y1, x2, y2 = bbox
    return (x1 > margin and y1 > margin
            and x2 < frame_w - margin and y2 < frame_h - margin)


def check_hand_visibility(
    per_frame_hands: list[list[HandDetection]],
    frame_dims: tuple[int, int],
    confidence_threshold: float = 0.7,
    pass_rate_threshold: float = 0.90,
    frame_margin: int = 2,
) -> CheckResult:
    """Check that both hands are fully visible in most frames.

    A hand counts as "fully visible" when it is detected with sufficient
    confidence AND its bounding box is at least *frame_margin* pixels away
    from every frame edge (i.e. not clipped).

    Args:
        per_frame_hands: Hand detections per frame.
        frame_dims: (height, width) of each frame.
        confidence_threshold: Min confidence for a hand to count as detected.
        pass_rate_threshold: Fraction of frames requiring both hands fully visible.
        frame_margin: Min pixel clearance from frame edges (default 2).

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_hands)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    frame_h, frame_w = frame_dims

    both_hands_frames = 0
    both_hands_by_side = 0
    any_hand_frames = 0
    clipped_frames = 0

    for hands in per_frame_hands:
        confident_hands = [h for h in hands if h.confidence >= confidence_threshold]
        in_frame_hands = [h for h in confident_hands
                          if _hand_in_frame(h.bbox, frame_w, frame_h, frame_margin)]

        has_left = any(h.side == HandSide.LEFT for h in in_frame_hands)
        has_right = any(h.side == HandSide.RIGHT for h in in_frame_hands)

        # Primary: 2+ confident, fully-in-frame hand detections.
        # The Hands23 model's left/right classifier can mislabel both hands
        # as the same side, so we count distinct detections instead.
        if len(in_frame_hands) >= 2:
            both_hands_frames += 1
        # Secondary: track side-based count for diagnostics
        if has_left and has_right:
            both_hands_by_side += 1
        if in_frame_hands:
            any_hand_frames += 1
        # Track frames where confidence was fine but bbox was clipped
        if len(confident_hands) >= 2 and len(in_frame_hands) < 2:
            clipped_frames += 1

    both_ratio = both_hands_frames / total_frames
    status = "pass" if both_ratio >= pass_rate_threshold else "fail"

    # Confidence: based on distance from threshold
    if both_ratio >= pass_rate_threshold:
        ratio_margin = both_ratio - pass_rate_threshold
        confidence = min(1.0, 0.7 + ratio_margin * 3)
    else:
        deficit = pass_rate_threshold - both_ratio
        confidence = max(0.0, 0.5 - deficit * 2)

    return CheckResult(
        status=status,
        metric_value=round(both_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "both_hands_frames": both_hands_frames,
            "both_hands_by_side_label": both_hands_by_side,
            "any_hand_frames": any_hand_frames,
            "clipped_frames": clipped_frames,
            "total_frames": total_frames,
            "confidence_threshold": confidence_threshold,
            "pass_rate_threshold": pass_rate_threshold,
            "frame_margin_px": frame_margin,
        },
    )
