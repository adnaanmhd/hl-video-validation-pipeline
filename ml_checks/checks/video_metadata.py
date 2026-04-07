"""Video metadata acceptance checks.

Six deterministic checks on container format, codec, resolution,
frame rate, duration, and orientation. All use confidence=1.0.
"""

from ml_checks.checks.check_results import CheckResult


def check_format(metadata: dict) -> CheckResult:
    """Check that container is MP4 (MPEG-4)."""
    fmt = metadata["container_format"]
    # ffprobe reports MP4 as "mov,mp4,m4a,3gp,3g2,mj2"
    is_mp4 = "mp4" in fmt.lower()
    return CheckResult(
        status="pass" if is_mp4 else "fail",
        metric_value=1.0 if is_mp4 else 0.0,
        confidence=1.0,
        details={"container_format": fmt, "expected": "mp4"},
    )


def check_encoding(metadata: dict) -> CheckResult:
    """Check that video codec is H.264."""
    codec = metadata["video_codec"]
    is_h264 = codec.lower() == "h264"
    return CheckResult(
        status="pass" if is_h264 else "fail",
        metric_value=1.0 if is_h264 else 0.0,
        confidence=1.0,
        details={"video_codec": codec, "expected": "h264"},
    )


def check_resolution(metadata: dict) -> CheckResult:
    """Check that resolution is at least 1920x1080."""
    w, h = metadata["width"], metadata["height"]
    passes = w >= 1920 and h >= 1080
    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=1.0 if passes else 0.0,
        confidence=1.0,
        details={
            "width": w, "height": h,
            "min_width": 1920, "min_height": 1080,
        },
    )


def check_frame_rate(metadata: dict) -> CheckResult:
    """Check that frame rate is at least 28 FPS."""
    fps = metadata["fps"]
    passes = fps >= 28.0
    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=1.0 if passes else 0.0,
        confidence=1.0,
        details={"fps": fps, "min_fps": 28.0},
    )


def check_duration(metadata: dict) -> CheckResult:
    """Check that duration is at least 180 seconds."""
    dur = metadata["duration_s"]
    passes = dur >= 180.0
    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=1.0 if passes else 0.0,
        confidence=1.0,
        details={"duration_s": dur, "min_duration_s": 180.0},
    )


def check_orientation(metadata: dict) -> CheckResult:
    """Check that rotation is 0 or 180 degrees and video is landscape (width > height)."""
    rotation = metadata["rotation"]
    w, h = metadata["width"], metadata["height"]
    passes = rotation in (0, 180) and w > h
    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=1.0 if passes else 0.0,
        confidence=1.0,
        details={
            "rotation": rotation, "width": w, "height": h,
            "expected_rotation": "0 or 180", "expected": "width > height",
        },
    )


def run_all_metadata_checks(metadata: dict) -> dict[str, CheckResult]:
    """Run all 6 metadata checks and return results keyed by name."""
    return {
        "meta_format": check_format(metadata),
        "meta_encoding": check_encoding(metadata),
        "meta_resolution": check_resolution(metadata),
        "meta_frame_rate": check_frame_rate(metadata),
        "meta_duration": check_duration(metadata),
        "meta_orientation": check_orientation(metadata),
    }
