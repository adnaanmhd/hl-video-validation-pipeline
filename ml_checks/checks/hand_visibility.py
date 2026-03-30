"""Check 3: Hand Visibility.

Criterion: Both hands detected (>= 0.7 confidence) in >= 90% of frames.
Purpose: Ensure wearer's hands are visible for robot learning.
"""

from ml_checks.checks.check_results import CheckResult
from ml_checks.models.hand_detector import HandDetection, HandSide


def check_hand_visibility(
    per_frame_hands: list[list[HandDetection]],
    confidence_threshold: float = 0.7,
    pass_rate_threshold: float = 0.90,
) -> CheckResult:
    """Check that both hands are visible in most frames.

    Args:
        per_frame_hands: Hand detections per frame.
        confidence_threshold: Min confidence for a hand to count as "detected."
        pass_rate_threshold: Fraction of frames requiring both hands visible.

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_hands)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    both_hands_frames = 0
    both_hands_by_side = 0
    any_hand_frames = 0

    for hands in per_frame_hands:
        confident_hands = [h for h in hands if h.confidence >= confidence_threshold]

        has_left = any(h.side == HandSide.LEFT for h in confident_hands)
        has_right = any(h.side == HandSide.RIGHT for h in confident_hands)

        # Primary: 2+ confident hand detections (regardless of side label).
        # The 100DOH model's left/right classifier can mislabel both hands
        # as the same side, so we count distinct detections instead.
        if len(confident_hands) >= 2:
            both_hands_frames += 1
        # Secondary: track side-based count for diagnostics
        if has_left and has_right:
            both_hands_by_side += 1
        if confident_hands:
            any_hand_frames += 1

    both_ratio = both_hands_frames / total_frames
    status = "pass" if both_ratio >= pass_rate_threshold else "fail"

    # Confidence: based on distance from threshold
    if both_ratio >= pass_rate_threshold:
        margin = both_ratio - pass_rate_threshold
        confidence = min(1.0, 0.7 + margin * 3)
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
            "total_frames": total_frames,
            "confidence_threshold": confidence_threshold,
            "pass_rate_threshold": pass_rate_threshold,
        },
    )
