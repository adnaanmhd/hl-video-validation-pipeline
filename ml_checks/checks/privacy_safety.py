"""Check 5: Privacy Safety.

Criterion: Sensitive objects detected = 0 in ALL frames.
Purpose: Data privacy compliance.
"""

from ml_checks.checks.check_results import CheckResult
from ml_checks.models.yolo_detector import Detection
from ml_checks.models.grounding_dino_detector import ZeroShotDetection


def check_privacy_safety(
    per_frame_yolo_sensitive: list[list[Detection]],
    per_frame_gdino: list[list[ZeroShotDetection]] | None = None,
    yolo_conf_threshold: float = 0.6,
    gdino_conf_threshold: float = 0.3,
) -> CheckResult:
    """Check that no sensitive objects appear in any frame.

    Two-stage approach:
    1. YOLO11m detects COCO sensitive classes (tv, laptop, cell_phone)
    2. Grounding DINO (if provided) detects fine-grained items (credit cards, IDs, documents)

    Args:
        per_frame_yolo_sensitive: YOLO sensitive object detections per frame.
        per_frame_gdino: Grounding DINO detections per frame (only on flagged frames, or None).
        yolo_conf_threshold: Min YOLO confidence for sensitive object.
        gdino_conf_threshold: Min Grounding DINO confidence.

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_yolo_sensitive)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    frames_with_sensitive = 0
    all_detections_info = []

    for i, yolo_dets in enumerate(per_frame_yolo_sensitive):
        frame_has_sensitive = False

        # Check YOLO detections
        high_conf_yolo = [d for d in yolo_dets if d.confidence >= yolo_conf_threshold]
        if high_conf_yolo:
            frame_has_sensitive = True
            for d in high_conf_yolo:
                all_detections_info.append({
                    "frame": i,
                    "source": "yolo",
                    "class": d.class_name,
                    "confidence": round(d.confidence, 3),
                })

        # Check Grounding DINO detections (if available for this frame)
        if per_frame_gdino and i < len(per_frame_gdino):
            gdino_dets = per_frame_gdino[i]
            high_conf_gdino = [d for d in gdino_dets if d.confidence >= gdino_conf_threshold]
            if high_conf_gdino:
                frame_has_sensitive = True
                for d in high_conf_gdino:
                    all_detections_info.append({
                        "frame": i,
                        "source": "grounding_dino",
                        "label": d.label,
                        "confidence": round(d.confidence, 3),
                    })

        if frame_has_sensitive:
            frames_with_sensitive += 1

    # Metric: fraction of CLEAN frames (no sensitive objects)
    clean_ratio = 1.0 - (frames_with_sensitive / total_frames)

    # Pass: ALL frames must be clean (zero tolerance)
    status = "pass" if frames_with_sensitive == 0 else "fail"

    # Confidence
    if frames_with_sensitive == 0:
        confidence = 0.9  # High but not 1.0 (zero-shot detection could miss items)
    else:
        confidence = max(0.0, 1.0 - (frames_with_sensitive / total_frames))

    return CheckResult(
        status=status,
        metric_value=round(clean_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "frames_with_sensitive_objects": frames_with_sensitive,
            "total_frames": total_frames,
            "detections": all_detections_info[:20],  # Cap for readability
        },
    )
