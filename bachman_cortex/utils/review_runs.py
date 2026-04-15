"""Collapse per-frame review records into contiguous time ranges.

Review frames are per-frame borderline detections (confidence/metric in a
"review band"). For reporting, they're grouped into contiguous runs and
only runs strictly longer than ``min_duration_s`` are kept — shorter runs
are treated as noise and dropped (promoted to pass).
"""

from dataclasses import dataclass
from typing import Iterable


@dataclass
class ReviewRun:
    """A contiguous run of review frames for a single check."""
    start_sec: float
    end_sec: float  # exclusive: last_ts + frame_interval
    frame_count: int
    # Per-metric-key aggregates: {key: (mean, min, max)}
    aggregates: dict[str, tuple[float, float, float]]

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


def collapse_review_runs(
    frames: list[dict],
    frame_interval: float,
    min_duration_s: float = 2.0,
    value_keys: Iterable[str] = (),
) -> list[ReviewRun]:
    """Collapse per-frame review records into contiguous runs > min_duration_s.

    Args:
        frames: Per-frame records. Each dict must have ``timestamp_sec`` and
            one entry per key in ``value_keys``.
        frame_interval: Expected gap between consecutive sampled frames (s).
            Used both for grouping and to compute run end times.
        min_duration_s: Minimum run duration (strict ``>``). Runs with
            duration <= this threshold are dropped.
        value_keys: Numeric metric keys to aggregate per run.

    Returns:
        Runs sorted by start_sec, each strictly longer than min_duration_s.
    """
    if not frames:
        return []

    # Sort by timestamp; tolerate unsorted input.
    sorted_frames = sorted(frames, key=lambda f: f["timestamp_sec"])
    value_keys = tuple(value_keys)

    # Gap tolerance: two frames are "contiguous" if separated by ~one interval.
    gap_tolerance = frame_interval * 1.5

    runs: list[ReviewRun] = []
    current: list[dict] = [sorted_frames[0]]

    for f in sorted_frames[1:]:
        gap = f["timestamp_sec"] - current[-1]["timestamp_sec"]
        if gap <= gap_tolerance:
            current.append(f)
        else:
            runs.append(_build_run(current, frame_interval, value_keys))
            current = [f]

    runs.append(_build_run(current, frame_interval, value_keys))

    return [r for r in runs if r.duration > min_duration_s]


def _build_run(
    frames: list[dict],
    frame_interval: float,
    value_keys: tuple[str, ...],
) -> ReviewRun:
    start = frames[0]["timestamp_sec"]
    end = frames[-1]["timestamp_sec"] + frame_interval

    aggregates: dict[str, tuple[float, float, float]] = {}
    for k in value_keys:
        vals = [f[k] for f in frames if k in f and f[k] is not None]
        if vals:
            aggregates[k] = (sum(vals) / len(vals), min(vals), max(vals))

    return ReviewRun(
        start_sec=start,
        end_sec=end,
        frame_count=len(frames),
        aggregates=aggregates,
    )
