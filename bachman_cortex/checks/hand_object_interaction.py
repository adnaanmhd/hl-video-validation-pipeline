"""Check 4: Hand-Object Interaction.

Criterion: Interaction detected in >= 70% of frames.
Purpose: Verify actual task manipulation is captured (critical for robot training).
"""

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.models.hand_detector import HandDetection, ContactState


def check_hand_object_interaction(
    per_frame_hands: list[list[HandDetection]],
    pass_rate_threshold: float = 0.70,
) -> CheckResult:
    """Check that hands are interacting with objects in most frames.

    Uses the contact_state from hand detection:
    - 100DOH backend: Contact state is learned (P=portable, F=stationary)
    - MediaPipe backend: Contact state is estimated via proximity heuristic

    Args:
        per_frame_hands: Hand detections per frame (with contact_state populated).
        pass_rate_threshold: Fraction of frames requiring interaction.

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_hands)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    INTERACTION_STATES = {ContactState.PORTABLE_OBJ, ContactState.STATIONARY_OBJ}
    interaction_frames = 0

    for hands in per_frame_hands:
        has_interaction = any(
            h.contact_state in INTERACTION_STATES for h in hands
        )
        if has_interaction:
            interaction_frames += 1

    interaction_ratio = interaction_frames / total_frames
    status = "pass" if interaction_ratio >= pass_rate_threshold else "fail"

    # Confidence
    if interaction_ratio >= pass_rate_threshold:
        margin = interaction_ratio - pass_rate_threshold
        confidence = min(1.0, 0.7 + margin * 3)
    else:
        deficit = pass_rate_threshold - interaction_ratio
        confidence = max(0.0, 0.5 - deficit * 2)

    return CheckResult(
        status=status,
        metric_value=round(interaction_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "interaction_frames": interaction_frames,
            "total_frames": total_frames,
            "pass_rate_threshold": pass_rate_threshold,
        },
    )
