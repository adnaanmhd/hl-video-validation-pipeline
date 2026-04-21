"""Non-gating metadata observations extracted from an ffprobe read.

Seven fields — bitrate, GOP, colour depth, B-frames, HDR,
stabilization, FOV — computed from the same ffprobe output the gated
checks already consume, plus one extra cheap packet-level scan for
GOP. No decode, no model inference.

Rules are documented in `checks.md`; this module is the implementation.
Each primitive is pure and unit-testable.
"""

from __future__ import annotations

import re
from typing import Any

from bachman_cortex.data_types import MetadataObservations
from bachman_cortex.utils.gpmd import detect_gpmd_stream
from bachman_cortex.utils.video_metadata import collect_tag_surface


# ── Primitive extractors ───────────────────────────────────────────────────

def compute_bitrate_mbps(bitrate_bps: int | None) -> float | None:
    """Video-stream bitrate in megabits per second, rounded to 2 decimals."""
    if bitrate_bps is None or bitrate_bps <= 0:
        return None
    return round(bitrate_bps / 1_000_000.0, 2)


_PIX_FMT_DEPTH_RE = re.compile(r"p(\d+)(le|be)?$")


def compute_color_depth_bits(
    pix_fmt: str,
    bits_per_raw_sample: int | None,
) -> int | None:
    """Colour depth in bits.

    Priority:
      1. `bits_per_raw_sample` when ffprobe reports it (most authoritative).
      2. Regex on `pix_fmt`: `yuv420p10le` → 10, `yuv420p` → 8, etc.
      3. `None` when neither resolves.
    """
    if bits_per_raw_sample is not None and bits_per_raw_sample > 0:
        return bits_per_raw_sample
    if not pix_fmt:
        return None
    m = _PIX_FMT_DEPTH_RE.search(pix_fmt)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    # Formats without an explicit `pNN` suffix (e.g. `yuv420p`, `yuvj420p`,
    # `nv12`) are 8-bit by convention.
    if pix_fmt.startswith(("yuv", "yuvj", "nv", "rgb", "bgr", "gray")):
        return 8
    return None


def compute_b_frames(has_b_frames: int | None) -> str:
    """Y/N from ffprobe's `has_b_frames` (>0 → present)."""
    if has_b_frames is None:
        return "N"
    return "Y" if has_b_frames > 0 else "N"


_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
_DOLBY_VISION_SIDE_DATA = {
    "dovi configuration record",
    "dovi rpu data",
    "dolby vision configuration record",
}
_DOLBY_VISION_CODEC_TAGS = {"dvhe", "dvav", "dvh1", "dva1"}


def compute_hdr(
    color_transfer: str,
    color_primaries: str,
    codec_tag_string: str,
    side_data_types: list[str],
) -> str:
    """ON iff any of:
    - color_transfer is PQ (`smpte2084`) or HLG (`arib-std-b67`)
    - side_data_list contains a Dolby Vision record
    - codec_tag_string names a Dolby Vision profile
    Else OFF. Colour primaries alone (e.g. `bt2020`) are **not** enough.
    """
    ct = (color_transfer or "").lower().strip()
    if ct in _HDR_TRANSFERS:
        return "ON"
    codec = (codec_tag_string or "").lower().strip()
    if codec in _DOLBY_VISION_CODEC_TAGS:
        return "ON"
    for sd in side_data_types or []:
        if (sd or "").lower().strip() in _DOLBY_VISION_SIDE_DATA:
            return "ON"
    return "OFF"


# ── Vendor registry: stabilization ─────────────────────────────────────────

def detect_stabilization(tag_surface: dict[str, Any], raw_streams: list[dict]) -> str:
    """Return "Y" / "N" / "Unknown" per the vendor registry.

    Detectors are evaluated in priority order; first positive match wins.
    The ordering favours post-stabilization tools (an explicit
    affirmative signal) over baked-in camera encoders.

    Absence of any signal is reported as `Unknown` — not `N` — because
    most containers do not carry a "stabilization: off" marker.
    """
    encoder = (tag_surface.get("encoder", "") or "").lower()
    video_tags = tag_surface.get("video_tags", {}) or {}
    format_tags = tag_surface.get("format_tags", {}) or {}
    all_stream_tags = tag_surface.get("all_stream_tags", []) or []
    all_handler_names = [
        (h or "").lower() for h in tag_surface.get("all_stream_handler_names", []) or []
    ]

    # 1. Post-stabilization tools named in the encoder string.
    if re.search(r"\b(gyroflow|reelsteady|hypersmooth)\b", encoder):
        return "Y"

    # 2. GoPro encoder — HyperSmooth is baked into the pixels when used.
    #    Presence of the GoPro AVC/HEVC encoder is a reasonable proxy;
    #    the GPMD stream carries the precise HyperSmooth state for
    #    future refinement.
    if "gopro" in encoder:
        return "Y"
    if detect_gpmd_stream(raw_streams).present:
        return "Y"

    # 3. Google CAMM track present → motion captured, typically paired with EIS.
    if any("camm" in h for h in all_handler_names):
        return "Y"

    # 4. Samsung — presence of `smta` or `svss` makernote atom.
    #    ffprobe exposes these as tags on the format block or stream tags.
    if _has_samsung_stabilization_atom(format_tags, video_tags, all_stream_tags):
        return "Y"

    # 5. DJI — encoder / handler / vendor tags.
    if "dji" in encoder:
        return "Y"
    if any("dji" in h for h in all_handler_names):
        return "Y"

    # 6. Apple iPhone — software tag present. Can't tell from the container
    #    whether EIS/OIS was active during capture, so report Unknown.
    if format_tags.get("com.apple.quicktime.software") \
            or video_tags.get("com.apple.quicktime.software"):
        return "Unknown"

    return "Unknown"


def _has_samsung_stabilization_atom(
    format_tags: dict,
    video_tags: dict,
    all_stream_tags: list[dict],
) -> bool:
    """Samsung `smta` / `svss` Makernote atom presence.

    ffprobe surfaces these as either top-level keys inside a tag dict,
    or as prefixes on custom key-value keys (observed in the wild as
    `com.samsung.smta` or bare `smta`/`svss`).
    """
    def _dict_has(d: dict) -> bool:
        if not d:
            return False
        for k in d.keys():
            key = (k or "").lower()
            if key in ("smta", "svss"):
                return True
            if "smta" in key or "svss" in key or "samsung" in key:
                return True
        return False

    if _dict_has(format_tags) or _dict_has(video_tags):
        return True
    return any(_dict_has(t) for t in (all_stream_tags or []))


# ── Vendor registry: FOV ───────────────────────────────────────────────────

# QuickTime tag for iPhone 35mm-equivalent focal length. Lives under
# `meta.ilst.<key>` in the MOV container and is exposed by ffprobe
# with this flat key.
_APPLE_FOCAL_35MM_KEYS = (
    "com.apple.quicktime.focal.length.35mmequiv",
    "com.apple.quicktime.focallength35mmequiv",
)


def detect_fov(
    tag_surface: dict[str, Any],
    raw_streams: list[dict],
) -> str:
    """Return a raw vendor FOV label (string) or "Unknown".

    Registry order (first match wins):

    1. GoPro — GPMD stream present. Without the KLV parser we cannot
       read the actual lens preset yet, so report "GoPro-embedded" so
       the report reflects that richer data exists and is pending the
       IMU-extraction work that will unlock it.
    2. DJI — `handler_name` / encoder matches; real `udta` parser not
       yet wired, return "DJI-embedded".
    3. Apple — derive degrees from the iPhone 35mm-equivalent focal
       length tag when present.
    4. Nothing found → "Unknown".
    """
    encoder = (tag_surface.get("encoder", "") or "").lower()
    video_tags = tag_surface.get("video_tags", {}) or {}
    format_tags = tag_surface.get("format_tags", {}) or {}
    all_handler_names = [
        (h or "").lower() for h in tag_surface.get("all_stream_handler_names", []) or []
    ]

    if detect_gpmd_stream(raw_streams).present or "gopro" in encoder:
        return "GoPro-embedded"

    if "dji" in encoder or any("dji" in h for h in all_handler_names):
        return "DJI-embedded"

    # Apple iPhone — look for the 35mm-equiv focal length in both flat and
    # normalised-lowercase forms.
    for container in (video_tags, format_tags):
        for candidate_key in _APPLE_FOCAL_35MM_KEYS:
            for real_key, real_val in container.items():
                if (real_key or "").lower() == candidate_key:
                    deg = _fov_from_35mm_equiv(real_val)
                    if deg is not None:
                        return f"~{deg:.0f}°"

    return "Unknown"


def _fov_from_35mm_equiv(val: Any) -> float | None:
    """Horizontal FOV (degrees) from a 35mm-equivalent focal length.

    35mm frame width is 36 mm; HFOV = 2 * atan(36 / (2 * f)), in degrees.
    `val` is typically a string like "28" or "28 mm".
    """
    if val is None:
        return None
    import math
    s = str(val).strip().split()[0]
    try:
        f = float(s)
    except ValueError:
        return None
    if f <= 0:
        return None
    return math.degrees(2.0 * math.atan(36.0 / (2.0 * f)))


# ── Top-level builder ──────────────────────────────────────────────────────

def build_observations(meta: dict, avg_gop: float | None) -> MetadataObservations:
    """Assemble the full `MetadataObservations` from an ffprobe meta dict
    and a precomputed GOP average.
    """
    tag_surface = collect_tag_surface(meta)
    raw_streams = meta.get("_raw_streams", []) or []
    side_data_types: list[str] = tag_surface.get("side_data_types", []) or []

    return MetadataObservations(
        bitrate_mbps=compute_bitrate_mbps(meta.get("bitrate_bps")),
        gop=(round(avg_gop, 1) if avg_gop is not None else None),
        color_depth_bits=compute_color_depth_bits(
            meta.get("pix_fmt", ""),
            meta.get("bits_per_raw_sample"),
        ),
        b_frames=compute_b_frames(meta.get("has_b_frames")),
        hdr=compute_hdr(
            color_transfer=meta.get("color_transfer", ""),
            color_primaries=meta.get("color_primaries", ""),
            codec_tag_string=meta.get("codec_tag_string", ""),
            side_data_types=side_data_types,
        ),
        stabilization=detect_stabilization(tag_surface, raw_streams),
        fov=detect_fov(tag_surface, raw_streams),
    )
