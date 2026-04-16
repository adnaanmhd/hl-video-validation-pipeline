"""Video metadata acceptance checks.

Six deterministic checks on container format, codec, resolution,
frame rate, duration, and orientation. All use confidence=1.0.
"""

from bachman_cortex.checks.check_results import CheckResult


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
    """Check that video codec is H.264 or HEVC (H.265)."""
    codec = metadata["video_codec"]
    normalized = codec.lower()
    accepted = normalized in ("h264", "hevc", "h265")
    return CheckResult(
        status="pass" if accepted else "fail",
        metric_value=1.0 if accepted else 0.0,
        confidence=1.0,
        details={"video_codec": codec, "expected": "h264 or hevc"},
    )


def check_resolution(metadata: dict) -> CheckResult:
    """Check that displayed resolution is at least 1920x1080.

    Applies the rotation tag first: for rotation 90 or 270, the stored
    (width, height) are swapped on display.
    """
    w, h = metadata["width"], metadata["height"]
    rotation = metadata["rotation"]
    if rotation in (90, 270):
        disp_w, disp_h = h, w
    else:
        disp_w, disp_h = w, h
    passes = disp_w >= 1920 and disp_h >= 1080
    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=1.0 if passes else 0.0,
        confidence=1.0,
        details={
            "width": w, "height": h,
            "rotation": rotation,
            "displayed_width": disp_w, "displayed_height": disp_h,
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
    """Check that duration is at least 10 seconds."""
    dur = metadata["duration_s"]
    passes = dur >= 10.0
    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=1.0 if passes else 0.0,
        confidence=1.0,
        details={"duration_s": dur, "min_duration_s": 10.0},
    )


def check_orientation(metadata: dict) -> CheckResult:
    """Check that rotation is 0, 90, or 270 and displayed video is landscape.

    Upside-down (rotation=180) is rejected. After applying rotation,
    displayed_width must be greater than displayed_height.
    """
    rotation = metadata["rotation"]
    w, h = metadata["width"], metadata["height"]
    if rotation in (90, 270):
        disp_w, disp_h = h, w
    else:
        disp_w, disp_h = w, h
    passes = rotation in (0, 90, 270) and disp_w > disp_h
    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=1.0 if passes else 0.0,
        confidence=1.0,
        details={
            "rotation": rotation, "width": w, "height": h,
            "displayed_width": disp_w, "displayed_height": disp_h,
            "expected_rotation": "0, 90, or 270",
            "expected": "displayed_width > displayed_height",
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
