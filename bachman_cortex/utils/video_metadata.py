"""Video metadata extraction via FFprobe.

Single ffprobe call to get container format, codec, resolution, FPS,
duration, rotation, and file size.
"""

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path


def get_video_metadata(video_path: str | Path) -> dict:
    """Extract video metadata using ffprobe.

    Returns:
        Dict with keys: container_format, video_codec, width, height,
        fps, duration_s, rotation, file_size_mb.
    """
    video_path = str(video_path)

    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path,
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr}")

    data = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    fmt = data.get("format", {})

    # Container format
    container_format = fmt.get("format_name", "unknown")

    # Video codec
    video_codec = video_stream.get("codec_name", "unknown")

    # Resolution
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    # FPS -- prefer avg_frame_rate, fall back to r_frame_rate
    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "0/1")
    try:
        fps = float(Fraction(fps_str))
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    # Duration
    duration_s = float(fmt.get("duration", video_stream.get("duration", 0)))

    # Rotation -- check tags and side_data_list
    rotation = 0
    tags = video_stream.get("tags", {})
    if "rotate" in tags:
        rotation = abs(int(tags["rotate"]))
    else:
        for sd in video_stream.get("side_data_list", []):
            if "rotation" in sd:
                rotation = abs(int(sd["rotation"]))
                break

    # File size
    file_size_mb = round(os.path.getsize(video_path) / 1024 / 1024, 1)

    return {
        "container_format": container_format,
        "video_codec": video_codec,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "duration_s": round(duration_s, 2),
        "rotation": rotation,
        "file_size_mb": file_size_mb,
    }
