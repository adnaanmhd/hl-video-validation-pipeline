"""GoPro GPMD timed-metadata stream — parser stub.

GoPro clips carry a private timed-metadata stream (`handler_name` equal
to "GoPro MET" or the handler type is `meta` with a `gpmd` codec tag)
that encodes HyperSmooth state, lens preset, gyro, accelerometer, GPS
and more in the KLV-style GPMF format.

Only the presence-detection half lives here right now. The real KLV
decoder lands in the follow-up IMU-extraction work; this module is
the seam it will plug into, so the observations pipeline doesn't
need to change when richer parsing arrives.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GpmdInfo:
    """What the stub can tell us before a real KLV parser lands."""
    present: bool
    stream_index: int | None           # index into the ffprobe streams list, or None
    handler_name: str                  # raw handler_name of the detected stream
    lens_label: str | None = None      # filled by the real parser later
    hypersmooth_state: str | None = None  # filled by the real parser later


_GPMD_HANDLER_NEEDLES = ("gopro met", "gpmd")


def detect_gpmd_stream(raw_streams: list[dict]) -> GpmdInfo:
    """Return presence info for the GoPro GPMD stream.

    Detection rule: any stream whose `tags.handler_name` contains
    "GoPro MET" (case-insensitive) or whose `codec_tag_string` is
    `gpmd`. Both signals mean the GPMF-encoded metadata is available
    on this file.

    `stream_index` is the flat index in `data["streams"]`, suitable
    for `-map 0:<index>` when the real parser lands.
    """
    for i, s in enumerate(raw_streams or []):
        handler = ((s.get("tags") or {}).get("handler_name", "") or "").lower()
        codec_tag = (s.get("codec_tag_string", "") or "").lower()
        if any(needle in handler for needle in _GPMD_HANDLER_NEEDLES) \
                or codec_tag == "gpmd":
            return GpmdInfo(
                present=True,
                stream_index=i,
                handler_name=handler,
            )
    return GpmdInfo(present=False, stream_index=None, handler_name="")
