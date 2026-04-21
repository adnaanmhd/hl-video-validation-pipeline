"""Data types for the scoring engine.

Schema matches SCORING_ENGINE_PLAN.md §4. Reports are assembled from these
dataclasses; parquet is a separate artefact (see `per_frame_store.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


# ── Metadata stage ────────────────────────────────────────────────────────

@dataclass
class MetadataCheckResult:
    check: str          # "format" | "encoding" | "resolution" | "frame_rate" | "duration" | "orientation"
    status: str         # "pass" | "fail"
    accepted: str
    detected: str


@dataclass
class MetadataObservations:
    """Non-gating metadata readings. All seven fields populated for every
    video (including metadata-failed ones) — they come from the same
    ffprobe call the gated checks consume. Unknown values use the
    sentinels listed below rather than `None`, so the report table
    never renders a bare dash for a field we can in principle compute.

    - `bitrate_mbps`: video-stream bitrate in Mbps (float, 2 decimals). `None` only if ffprobe did not emit `bit_rate` on the stream (rare).
    - `gop`: average GOP size from packet-level keyframe scan (float, 1 decimal). `None` when unresolvable (no keyframes reported).
    - `color_depth_bits`: 8/10/12 inferred from `bits_per_raw_sample` with `pix_fmt` as fallback. `None` if neither source resolves.
    - `b_frames`: "Y" / "N" derived from `has_b_frames`.
    - `hdr`: "ON" / "OFF" per the transfer-function rule (see `checks.md`).
    - `stabilization`: "Y" / "N" / "Unknown" per the vendor registry.
    - `fov`: raw vendor label (e.g. "Wide", "~78°", "GoPro-embedded") or "Unknown".
    """
    bitrate_mbps: float | None
    gop: float | None
    color_depth_bits: int | None
    b_frames: str
    hdr: str
    stabilization: str
    fov: str


# ── Technical stage ───────────────────────────────────────────────────────

@dataclass
class TechnicalCheckResult:
    check: str          # "stability" | "frozen" | "luminance" | "pixelation"
    status: str         # "pass" | "fail" | "skipped"
    accepted: str
    detected: str
    skipped: bool = False


# ── Quality stage ─────────────────────────────────────────────────────────

@dataclass
class QualitySegment:
    """Half-open segment [start_s, end_s) of a single quality metric.

    `value` type varies per metric (see §1 of plan):
      - confidence-based metrics: float in [0.0, 1.0]
      - hand_angle: float (degrees) or NaN if no hands
      - hand_obj_interaction: str (contact char) or None
      - obstructed: bool
    """
    start_s: float
    end_s: float
    duration_s: float
    value: Union[float, str, bool, None]
    value_label: str    # "confidence" | "angle" | "contact_state" | "obstructed"


@dataclass
class QualityMetricResult:
    metric: str         # see §4 of plan
    percent_frames: float
    segments: list[QualitySegment]
    skipped: bool = False


# ── Aggregates for batch stats ────────────────────────────────────────────

@dataclass
class CheckStats:
    pass_count: int
    pass_duration_s: float
    fail_count: int
    fail_duration_s: float
    skipped_count: int = 0


@dataclass
class QualityStats:
    mean_percent: float
    median_percent: float
    min_percent: float
    max_percent: float


# ── Error reporting ───────────────────────────────────────────────────────

@dataclass
class ProcessingErrorReport:
    video_path: str
    video_name: str
    error_reason: str   # "audio_only" | "decode_failed" | "corrupt" | "metadata_probe_failed" | ...


# ── Top-level reports ─────────────────────────────────────────────────────

@dataclass
class VideoScoreReport:
    video_path: str
    video_name: str
    generated_at: str                # ISO 8601 UTC
    processing_wall_time_s: float
    duration_s: float

    metadata_checks: list[MetadataCheckResult] = field(default_factory=list)
    metadata_observations: MetadataObservations | None = None
    technical_checks: list[TechnicalCheckResult] = field(default_factory=list)
    quality_metrics: list[QualityMetricResult] = field(default_factory=list)

    technical_skipped: bool = False
    quality_skipped: bool = False


@dataclass
class BatchScoreReport:
    generated_at: str
    video_count: int
    total_duration_s: float
    total_wall_time_s: float

    metadata_check_stats: dict[str, CheckStats] = field(default_factory=dict)
    technical_check_stats: dict[str, CheckStats] = field(default_factory=dict)
    quality_metric_stats: dict[str, QualityStats] = field(default_factory=dict)

    videos: list[VideoScoreReport] = field(default_factory=list)
    errors: list[ProcessingErrorReport] = field(default_factory=list)


# ── Canonical metric/check names ──────────────────────────────────────────

METADATA_CHECKS: tuple[str, ...] = (
    "format", "encoding", "resolution", "frame_rate", "duration", "orientation",
)

TECHNICAL_CHECKS: tuple[str, ...] = (
    "luminance", "stability", "frozen", "pixelation",
)

QUALITY_METRICS: tuple[str, ...] = (
    "both_hands_visibility",
    "single_hand_visibility",
    "hand_obj_interaction",
    "hand_angle",
    "participants",
    "obstructed",
)

# Value labels keyed by quality metric name
QUALITY_VALUE_LABELS: dict[str, str] = {
    "both_hands_visibility": "confidence",
    "single_hand_visibility": "confidence",
    "hand_obj_interaction": "contact_state",
    "hand_angle": "angle",
    "participants": "confidence",
    "obstructed": "obstructed",
}

# Metadata observations — canonical order for report rows and CSV columns.
METADATA_OBSERVATIONS: tuple[str, ...] = (
    "bitrate_mbps",
    "gop",
    "color_depth_bits",
    "b_frames",
    "hdr",
    "stabilization",
    "fov",
)

# Which observations are numeric (aggregated as mean/median/min/max in batch
# MD) vs categorical (aggregated as a distribution histogram).
METADATA_OBSERVATION_NUMERIC: tuple[str, ...] = (
    "bitrate_mbps", "gop", "color_depth_bits",
)
METADATA_OBSERVATION_CATEGORICAL: tuple[str, ...] = (
    "b_frames", "hdr", "stabilization", "fov",
)
