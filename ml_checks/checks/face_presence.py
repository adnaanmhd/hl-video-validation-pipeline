"""Check 1: Face Presence.

Criterion: Face detection confidence < 0.8 in ALL sampled frames.
Purpose: Privacy — ensure other people's faces aren't prominently captured.
"""

from ml_checks.checks.check_results import CheckResult
from ml_checks.models.scrfd_detector import FaceDetection


def check_face_presence(
    per_frame_faces: list[list[FaceDetection]],
    confidence_threshold: float = 0.8,
) -> CheckResult:
    """Check that no face is detected with high confidence in any frame.

    Args:
        per_frame_faces: List of face detection results per frame.
        confidence_threshold: Max allowed face confidence.

    Returns:
        CheckResult with status, metric_value (fraction of frames with no prominent face),
        and confidence score.
    """
    total_frames = len(per_frame_faces)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    frames_with_prominent_face = 0
    max_face_conf_seen = 0.0

    for faces in per_frame_faces:
        max_conf = max((f.confidence for f in faces), default=0.0)
        max_face_conf_seen = max(max_face_conf_seen, max_conf)
        if max_conf >= confidence_threshold:
            frames_with_prominent_face += 1

    # Metric: fraction of frames with NO prominent face (higher = better)
    clean_ratio = 1.0 - (frames_with_prominent_face / total_frames)

    # Pass: ALL frames must have no prominent face
    status = "pass" if frames_with_prominent_face == 0 else "fail"

    # Confidence: based on how far the worst-case face conf is from threshold
    if max_face_conf_seen < confidence_threshold:
        # No face exceeded threshold — confidence based on margin
        margin = confidence_threshold - max_face_conf_seen
        confidence = min(1.0, 0.5 + margin)
    else:
        # Faces exceeded threshold — low confidence proportional to violation severity
        confidence = max(0.0, 1.0 - (frames_with_prominent_face / total_frames))

    return CheckResult(
        status=status,
        metric_value=round(clean_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "frames_with_prominent_face": frames_with_prominent_face,
            "total_frames": total_frames,
            "max_face_confidence_seen": round(max_face_conf_seen, 4),
            "threshold": confidence_threshold,
        },
    )
