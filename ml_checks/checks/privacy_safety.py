"""Check 5: Privacy Safety.

Criterion: Sensitive objects detected = 0 in ALL frames.
Purpose: Data privacy compliance.
"""

from ml_checks.checks.check_results import CheckResult
from ml_checks.models.grounding_dino_detector import ZeroShotDetection


def _seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    total_secs = int(seconds)
    h = total_secs // 3600
    m = (total_secs % 3600) // 60
    s = total_secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def check_privacy_safety(
    per_frame_gdino: list[list[ZeroShotDetection]],
    gdino_conf_threshold: float = 0.3,
    frame_timestamps_sec: list[float] | None = None,
) -> CheckResult:
    """Check that no sensitive objects appear in any frame.

    Uses Grounding DINO zero-shot detection on every frame to find
    sensitive items (credit cards, ID cards, documents, screens).

    Args:
        per_frame_gdino: Grounding DINO detections per frame.
        gdino_conf_threshold: Min Grounding DINO confidence.
        frame_timestamps_sec: Timestamp in seconds for each frame. Used to report
            detection times in HH:MM:SS format.

    Returns:
        CheckResult.
    """
    total_frames = len(per_frame_gdino)
    if total_frames == 0:
        return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                           details={"error": "no frames"})

    frames_with_sensitive = 0
    all_detections_info = []
    # Grouped summary: timestamp -> list of detected objects
    sensitive_timestamps: list[dict] = []

    for i, gdino_dets in enumerate(per_frame_gdino):
        ts_sec = frame_timestamps_sec[i] if frame_timestamps_sec else i

        high_conf = [d for d in gdino_dets if d.confidence >= gdino_conf_threshold]
        if high_conf:
            frames_with_sensitive += 1
            frame_objects: list[str] = []
            for d in high_conf:
                all_detections_info.append({
                    "frame": i,
                    "timestamp": _seconds_to_hhmmss(ts_sec),
                    "label": d.label,
                    "confidence": round(d.confidence, 3),
                })
                frame_objects.append(d.label)
            sensitive_timestamps.append({
                "timestamp": _seconds_to_hhmmss(ts_sec),
                "objects": frame_objects,
            })

    # Metric: fraction of CLEAN frames (no sensitive objects)
    clean_ratio = 1.0 - (frames_with_sensitive / total_frames)

    # Pass: ALL frames must be clean (zero tolerance)
    status = "pass" if frames_with_sensitive == 0 else "fail"

    # Confidence
    if frames_with_sensitive == 0:
        confidence = 0.9  # High but not 1.0 (zero-shot detection could miss items)
    else:
        confidence = max(0.0, 1.0 - (frames_with_sensitive / total_frames))

    # Log timestamps to stdout
    if sensitive_timestamps:
        print(f"\n  Privacy safety: sensitive objects detected at "
              f"{len(sensitive_timestamps)} timestamp(s):")
        for entry in sensitive_timestamps:
            objs = ", ".join(entry["objects"])
            print(f"    [{entry['timestamp']}] {objs}")

    return CheckResult(
        status=status,
        metric_value=round(clean_ratio, 4),
        confidence=round(confidence, 4),
        details={
            "frames_with_sensitive_objects": frames_with_sensitive,
            "total_frames": total_frames,
            "sensitive_timestamps": sensitive_timestamps,
            "detections": all_detections_info,
        },
    )
