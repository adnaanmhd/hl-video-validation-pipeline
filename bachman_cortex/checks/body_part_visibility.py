"""Check: Body Part Visibility.

Criterion: >= 90% of frames show ONLY hands/forearms (up to elbows) of the
camera wearer — no other body parts like shoulders, torso, legs, or feet.

Uses YOLO11m-pose keypoint detection to identify which body parts are visible.
"""

import numpy as np
from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.models.yolo_pose_detector import (
    PoseDetection,
    ALLOWED_WEARER_KEYPOINTS,
    KEYPOINT_NAMES,
)
from bachman_cortex.models.hand_detector import HandDetection


def _is_wearer_pose(
    pose: PoseDetection,
    hand_detections: list[HandDetection],
    frame_height: int,
    frame_width: int,
    hand_overlap_threshold: float = 0.3,
) -> bool:
    """Check if a pose detection belongs to the camera wearer.

    Same heuristics as participants check: bottom-anchored or overlapping
    with detected hands.
    """
    px1, py1, px2, py2 = pose.bbox
    person_center_x = (px1 + px2) / 2
    person_bottom = py2

    # Bottom-center anchor
    if (person_bottom > frame_height * 0.85
            and frame_width * 0.2 < person_center_x < frame_width * 0.8):
        return True

    # Overlap with detected hands
    person_area = max((px2 - px1) * (py2 - py1), 1)
    for hand in hand_detections:
        hx1, hy1, hx2, hy2 = hand.bbox
        overlap_x1 = max(px1, hx1)
        overlap_y1 = max(py1, hy1)
        overlap_x2 = min(px2, hx2)
        overlap_y2 = min(py2, hy2)
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
            if overlap_area / person_area > hand_overlap_threshold:
                return True

    return False


def _get_disallowed_keypoints(
    pose: PoseDetection,
    keypoint_conf_threshold: float = 0.5,
) -> list[str]:
    """Return names of visible keypoints that are NOT allowed for the wearer."""
    disallowed = []
    for idx in range(17):
        if idx in ALLOWED_WEARER_KEYPOINTS:
            continue
        conf = pose.keypoints[idx, 2]
        if conf >= keypoint_conf_threshold:
            disallowed.append(KEYPOINT_NAMES[idx])
    return disallowed


def check_body_part_visibility(
    per_frame_poses: list[list[PoseDetection]],
    per_frame_hands: list[list[HandDetection]],
    frame_dims: tuple[int, int],  # (height, width)
    pass_rate_threshold: float = 0.90,
    pose_conf_threshold: float = 0.4,
    keypoint_conf_threshold: float = 0.5,
) -> CheckResult:
    """Check that only hands/forearms of the wearer are visible.

    For each frame, finds pose detections that belong to the camera wearer
    and checks whether any body keypoints beyond wrists and elbows are
    detected with sufficient confidence.

    Args:
        per_frame_poses: YOLO pose detections per frame.
        per_frame_hands: Hand detections per frame (for wearer identification).
        frame_dims: (height, width) of the video frames.
        pass_rate_threshold: Fraction of frames that must be clean.
        pose_conf_threshold: Min confidence for a pose detection to be considered.
        keypoint_conf_threshold: Min confidence for a keypoint to count as visible.

    Returns:
        CheckResult with pass/fail status.
    """
    total_frames = len(per_frame_poses)
    if total_frames == 0:
        return CheckResult(
            status="fail", metric_value=0.0, confidence=0.0,
            details={"error": "no frames"},
        )

    frame_h, frame_w = frame_dims
    clean_frames = 0
    flagged_parts_summary: dict[str, int] = {}

    for i in range(total_frames):
        poses = per_frame_poses[i]
        hands = per_frame_hands[i] if i < len(per_frame_hands) else []

        frame_has_disallowed = False

        for pose in poses:
            if pose.confidence < pose_conf_threshold:
                continue
            if not _is_wearer_pose(pose, hands, frame_h, frame_w):
                continue

            disallowed = _get_disallowed_keypoints(pose, keypoint_conf_threshold)
            if disallowed:
                frame_has_disallowed = True
                for part in disallowed:
                    flagged_parts_summary[part] = flagged_parts_summary.get(part, 0) + 1
                break

        if not frame_has_disallowed:
            clean_frames += 1

    clean_ratio = clean_frames / total_frames

    status = "pass" if clean_ratio >= pass_rate_threshold else "fail"

    if clean_ratio >= pass_rate_threshold:
        margin = clean_ratio - pass_rate_threshold
        confidence = min(1.0, 0.7 + margin * 6)
    else:
        deficit = pass_rate_threshold - clean_ratio
        confidence = max(0.0, 0.5 - deficit * 5)

    return CheckResult(
        status=status,
        metric_value=round(clean_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "clean_frames": clean_frames,
            "total_frames": total_frames,
            "pass_rate_threshold": pass_rate_threshold,
            "flagged_body_parts": flagged_parts_summary,
        },
    )
