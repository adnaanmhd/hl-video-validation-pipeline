"""Check 4: Hand-Object Interaction.

Criterion: Interaction detected in >= 70% of frames.
Purpose: Verify actual task manipulation is captured (critical for robot training).
"""

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.models.hand_detector import HandDetection, ContactState


def check_hand_object_interaction(
    per_frame_hands: list[list[HandDetection]],
    pass_rate_threshold: float = 0.70,
    accept_threshold: float = 0.70,
    review_threshold: float = 0.50,
    timestamps: list[float] | None = None,
) -> CheckResult:
    """Check that hands are interacting with objects in most frames.

    A frame counts as an interaction frame when any hand is in an interaction
    contact state (PORTABLE_OBJ / STATIONARY_OBJ) with contact-state confidence
    >= review_threshold. Frames where the best interaction's contact-state
    confidence is in [review_threshold, accept_threshold) are flagged for
    review but still count as interaction frames (review counts as accept).

    Args:
        per_frame_hands: Hand detections per frame (with contact_state populated).
        pass_rate_threshold: Fraction of frames requiring interaction.
        accept_threshold: Contact-state confidence for a clean pass.
        review_threshold: Lower bound for the review band.
        timestamps: Per-frame absolute timestamps (sec). Required for review
            tracking; if None, review_frames will be empty.

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_hands)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    INTERACTION_STATES = {ContactState.PORTABLE_OBJ, ContactState.STATIONARY_OBJ}
    interaction_frames = 0
    review_frames: list[dict] = []

    for idx, hands in enumerate(per_frame_hands):
        best_conf = 0.0
        best_state = None
        for h in hands:
            if h.contact_state not in INTERACTION_STATES:
                continue
            if h.contact_state_confidence >= review_threshold and \
                    h.contact_state_confidence > best_conf:
                best_conf = h.contact_state_confidence
                best_state = h.contact_state

        if best_state is not None:
            interaction_frames += 1
            if best_conf < accept_threshold:
                if timestamps is not None and idx < len(timestamps):
                    review_frames.append({
                        "timestamp_sec": timestamps[idx],
                        "confidence": best_conf,
                        "contact_state": best_state.name.lower(),
                    })

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
            "accept_threshold": accept_threshold,
            "review_threshold": review_threshold,
            "review_frames": review_frames,
        },
    )
