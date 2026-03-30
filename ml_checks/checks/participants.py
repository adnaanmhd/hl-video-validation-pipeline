"""Check 2: Participants.

Criterion: 0 other persons (face or body parts) in >= 95% of frames.
Purpose: Verify solo task performance. Camera wearer is invisible.
"""

import numpy as np
from ml_checks.checks.check_results import CheckResult
from ml_checks.models.yolo_detector import Detection
from ml_checks.models.scrfd_detector import FaceDetection
from ml_checks.models.hand_detector import HandDetection


def _is_wearer_body_part(
    person_bbox: np.ndarray,
    hand_detections: list[HandDetection],
    frame_height: int,
    frame_width: int,
    hand_overlap_threshold: float = 0.3,
) -> bool:
    """Check if a person detection is likely the camera wearer's own body.

    Heuristic: the wearer's arms/hands appear at the bottom-center of
    egocentric frames. A "person" detection that overlaps with detected
    hands or is anchored at the bottom of the frame is likely the wearer.
    """
    px1, py1, px2, py2 = person_bbox
    person_center_x = (px1 + px2) / 2
    person_bottom = py2
    person_height = py2 - py1

    # Check 1: person detection anchored at bottom of frame
    if person_bottom > frame_height * 0.85 and person_center_x > frame_width * 0.2 and person_center_x < frame_width * 0.8:
        return True

    # Check 2: person detection overlaps significantly with detected hands
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


def check_participants(
    per_frame_persons: list[list[Detection]],
    per_frame_faces: list[list[FaceDetection]],
    per_frame_hands: list[list[HandDetection]],
    frame_dims: tuple[int, int],  # (height, width)
    pass_rate_threshold: float = 0.95,
    person_conf_threshold: float = 0.4,
    face_conf_threshold: float = 0.5,
    min_person_height_ratio: float = 0.15,
) -> CheckResult:
    """Check that no other person appears in the video.

    Combines YOLO person detections and SCRFD face detections.
    Filters out the camera wearer's own body parts.

    Args:
        per_frame_persons: YOLO person detections per frame.
        per_frame_faces: SCRFD face detections per frame.
        per_frame_hands: Hand detections per frame (for wearer filtering).
        frame_dims: (height, width) of the video frames.
        pass_rate_threshold: Fraction of frames that must be clear of other people.
        person_conf_threshold: Min YOLO confidence for person detection.
        face_conf_threshold: Min SCRFD confidence for face detection.
        min_person_height_ratio: Min person bbox height as fraction of frame height.

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_persons)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    frame_h, frame_w = frame_dims
    min_person_height = frame_h * min_person_height_ratio
    clean_frames = 0

    for i in range(total_frames):
        persons = per_frame_persons[i]
        faces = per_frame_faces[i]
        hands = per_frame_hands[i] if i < len(per_frame_hands) else []

        other_person_detected = False

        # Check YOLO person detections
        for p in persons:
            if p.confidence < person_conf_threshold:
                continue
            bbox_height = p.bbox[3] - p.bbox[1]
            if bbox_height < min_person_height:
                continue
            if _is_wearer_body_part(p.bbox, hands, frame_h, frame_w):
                continue
            other_person_detected = True
            break

        # Check SCRFD face detections (a face = another person present)
        if not other_person_detected:
            for f in faces:
                if f.confidence >= face_conf_threshold:
                    other_person_detected = True
                    break

        if not other_person_detected:
            clean_frames += 1

    clean_ratio = clean_frames / total_frames
    status = "pass" if clean_ratio >= pass_rate_threshold else "fail"

    # Confidence: based on distance from threshold
    if clean_ratio >= pass_rate_threshold:
        margin = clean_ratio - pass_rate_threshold
        confidence = min(1.0, 0.7 + margin * 6)  # scale 0-0.05 margin to 0.7-1.0
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
        },
    )
