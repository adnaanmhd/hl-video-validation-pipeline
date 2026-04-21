"""Video metadata extraction via FFprobe.

One ffprobe call surfaces the fields the gated checks need (container,
codec, resolution, FPS, duration, rotation, file size) and the fields
the new metadata observations consume (bitrate, b-frames, colour
space, vendor tags, side-data). A second cheap packet-level call
computes the average GOP size.
"""

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


def get_video_metadata(video_path: str | Path) -> dict:
    """Extract video metadata using ffprobe.

    Returns a dict keyed by the fields below. Fields prefixed with
    `_raw_` expose the untouched ffprobe sub-trees so downstream
    vendor-registry detectors can inspect tags / side_data / other
    streams without re-running ffprobe.
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

    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    fmt = data.get("format", {})

    container_format = fmt.get("format_name", "unknown")
    video_codec = video_stream.get("codec_name", "unknown")
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))

    fps_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate", "0/1")
    try:
        fps = float(Fraction(fps_str))
    except (ValueError, ZeroDivisionError):
        fps = 0.0

    duration_s = float(fmt.get("duration", video_stream.get("duration", 0)))

    rotation = 0
    tags = video_stream.get("tags", {})
    if "rotate" in tags:
        rotation = abs(int(tags["rotate"]))
    else:
        for sd in video_stream.get("side_data_list", []):
            if "rotation" in sd:
                rotation = abs(int(sd["rotation"]))
                break

    file_size_mb = round(os.path.getsize(video_path) / 1024 / 1024, 1)

    # ── Observation fields ─────────────────────────────────────────────
    bitrate_bps_raw = video_stream.get("bit_rate")
    try:
        bitrate_bps: int | None = int(bitrate_bps_raw) if bitrate_bps_raw is not None else None
    except (TypeError, ValueError):
        bitrate_bps = None

    has_b_frames_raw = video_stream.get("has_b_frames")
    try:
        has_b_frames: int | None = int(has_b_frames_raw) if has_b_frames_raw is not None else None
    except (TypeError, ValueError):
        has_b_frames = None

    pix_fmt = video_stream.get("pix_fmt", "")

    bpr_raw = video_stream.get("bits_per_raw_sample")
    try:
        bits_per_raw_sample: int | None = int(bpr_raw) if bpr_raw is not None else None
    except (TypeError, ValueError):
        bits_per_raw_sample = None

    return {
        "container_format": container_format,
        "video_codec": video_codec,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "duration_s": round(duration_s, 2),
        "rotation": rotation,
        "file_size_mb": file_size_mb,

        # observation source fields
        "bitrate_bps": bitrate_bps,
        "has_b_frames": has_b_frames,
        "pix_fmt": pix_fmt,
        "bits_per_raw_sample": bits_per_raw_sample,
        "color_transfer": video_stream.get("color_transfer", ""),
        "color_primaries": video_stream.get("color_primaries", ""),
        "color_space": video_stream.get("color_space", ""),
        "codec_tag_string": video_stream.get("codec_tag_string", ""),

        # raw ffprobe subtrees for vendor-registry detectors
        "_raw_video_stream": video_stream,
        "_raw_streams": data.get("streams", []),
        "_raw_format": fmt,
    }


def get_avg_gop(video_path: str | Path) -> float | None:
    """Average GOP size via packet-level keyframe scan (no decode).

    Returns `total_packets / keyframe_count`. `None` when we cannot
    determine it (no keyframes reported, ffprobe failure). Cost is
    about 4 s for a 1-hour clip — packet headers only, no decoding.
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-select_streams", "v:0",
            "-show_entries", "packet=flags",
            "-of", "csv=p=0",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None

    total = 0
    keyframes = 0
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        # ffprobe emits "K_" for keyframes, "__" for non-key.
        if line.startswith("K"):
            keyframes += 1

    if keyframes == 0 or total == 0:
        return None
    return total / keyframes


# ── Helpers used by the vendor registry ────────────────────────────────────

def collect_tag_surface(meta: dict) -> dict[str, Any]:
    """Flatten the ffprobe output into the surfaces a vendor detector cares about.

    Keys:
      - `video_tags`      : tags dict on the video stream
      - `format_tags`     : tags dict on the format block
      - `all_stream_tags` : list of dicts (one per stream) — includes CAMM, GPMD, etc.
      - `all_stream_handler_names` : list of str (handler_name per stream, empty string if absent)
      - `side_data_types` : list of str — side_data_type values from the video stream
      - `codec_tag_string`: str
      - `encoder`         : str (from video stream tags, else format tags, else "")
    """
    raw_video = meta.get("_raw_video_stream", {}) or {}
    raw_fmt = meta.get("_raw_format", {}) or {}
    raw_streams = meta.get("_raw_streams", []) or []

    video_tags = raw_video.get("tags", {}) or {}
    format_tags = raw_fmt.get("tags", {}) or {}

    all_stream_tags = [(s.get("tags", {}) or {}) for s in raw_streams]
    all_stream_handler_names = [
        (s.get("tags", {}) or {}).get("handler_name", "") for s in raw_streams
    ]
    side_data_types = [
        sd.get("side_data_type", "")
        for sd in (raw_video.get("side_data_list") or [])
    ]

    encoder = video_tags.get("encoder") or format_tags.get("encoder") or ""

    return {
        "video_tags": video_tags,
        "format_tags": format_tags,
        "all_stream_tags": all_stream_tags,
        "all_stream_handler_names": all_stream_handler_names,
        "side_data_types": side_data_types,
        "codec_tag_string": meta.get("codec_tag_string", ""),
        "encoder": encoder,
    }
