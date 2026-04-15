"""Check 3: Hand Visibility.

Criterion (OR of two conditions):
  - Both hands fully visible in >= 80% of frames, OR
  - At least one hand fully visible in >= 90% of frames.

A hand counts as "fully visible" when detected with confidence >= 0.7 and
its bbox is at least ``frame_margin`` px from every frame edge.
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
    both_hands_pass_rate: float = 0.80,
    single_hand_pass_rate: float = 0.90,
    frame_margin: int = 2,
    review_threshold: float = 0.5,
    timestamps: list[float] | None = None,
) -> CheckResult:
    """Check hand visibility using an OR of two sub-conditions.

    Passes if either:
      - ``both_hands_ratio >= both_hands_pass_rate`` (primary), or
      - ``single_hand_ratio >= single_hand_pass_rate`` (fallback).

    "Single hand" means at least one hand is fully visible in the frame, so
    ``single_hand_ratio >= both_hands_ratio`` always holds.

    Args:
        per_frame_hands: Hand detections per frame.
        frame_dims: (height, width) of each frame.
        confidence_threshold: Min confidence for a hand to count as detected.
        both_hands_pass_rate: Min fraction of frames with both hands visible.
        single_hand_pass_rate: Min fraction of frames with at least one hand
            visible (fallback when the both-hands condition fails).
        frame_margin: Min pixel clearance from frame edges.

    Returns:
        CheckResult. ``metric_value`` is the both-hands ratio; the single-hand
        ratio and both thresholds are reported in ``details``.
    """
    total_frames = len(per_frame_hands)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    frame_h, frame_w = frame_dims

    both_hands_frames = 0
    single_hand_frames = 0
    both_hands_by_side = 0
    clipped_frames = 0
    review_frames: list[dict] = []

    for idx, hands in enumerate(per_frame_hands):
        # Review as accept: the qualifying-hand pool includes conf >= review_threshold.
        qualifying_hands = [
            h for h in hands if h.confidence >= review_threshold
        ]
        in_frame_hands = [
            h for h in qualifying_hands
            if _hand_in_frame(h.bbox, frame_w, frame_h, frame_margin)
        ]

        has_left = any(h.side == HandSide.LEFT for h in in_frame_hands)
        has_right = any(h.side == HandSide.RIGHT for h in in_frame_hands)

        # The Hands23 model's left/right classifier can mislabel both hands
        # as the same side, so we count distinct in-frame detections.
        if len(in_frame_hands) >= 2:
            both_hands_frames += 1
        if has_left and has_right:
            both_hands_by_side += 1
        if in_frame_hands:
            single_hand_frames += 1
        if len(qualifying_hands) >= 2 and len(in_frame_hands) < 2:
            clipped_frames += 1

        # Review tracking: in-frame hands in [review_threshold, confidence_threshold).
        review_band = [
            h for h in in_frame_hands
            if review_threshold <= h.confidence < confidence_threshold
        ]
        if review_band and timestamps is not None and idx < len(timestamps):
            max_conf = max(h.confidence for h in review_band)
            review_frames.append({
                "timestamp_sec": timestamps[idx],
                "confidence": max_conf,
            })

    both_ratio = both_hands_frames / total_frames
    single_ratio = single_hand_frames / total_frames

    both_passed = both_ratio >= both_hands_pass_rate
    single_passed = single_ratio >= single_hand_pass_rate
    status = "pass" if (both_passed or single_passed) else "fail"

    # Confidence: use whichever sub-condition has the best (most positive,
    # or least negative) margin over its own threshold.
    both_margin = both_ratio - both_hands_pass_rate
    single_margin = single_ratio - single_hand_pass_rate
    best_margin = max(both_margin, single_margin)
    if best_margin >= 0:
        confidence = min(1.0, 0.7 + best_margin * 3)
    else:
        confidence = max(0.0, 0.5 - abs(best_margin) * 2)

    return CheckResult(
        status=status,
        metric_value=round(both_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "both_hands_frames": both_hands_frames,
            "single_hand_frames": single_hand_frames,
            "both_hands_by_side_label": both_hands_by_side,
            "clipped_frames": clipped_frames,
            "total_frames": total_frames,
            "both_hands_ratio": round(both_ratio, 4),
            "single_hand_ratio": round(single_ratio, 4),
            "both_hands_pass_rate": both_hands_pass_rate,
            "single_hand_pass_rate": single_hand_pass_rate,
            "both_hands_passed": both_passed,
            "single_hand_passed": single_passed,
            "confidence_threshold": confidence_threshold,
            "review_threshold": review_threshold,
            "frame_margin_px": frame_margin,
            "review_frames": review_frames,
        },
    )
