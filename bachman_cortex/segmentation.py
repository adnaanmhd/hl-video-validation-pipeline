"""Pure-function segmentation utilities for report rendering.

`group_runs` collapses a per-frame array into contiguous runs of
"the condition held". `merge_short_runs` implements the §1 absorption
rule: any run shorter than `min_duration_s` flips state and folds into
its preceding neighbour (first run folds into its following neighbour
instead). Single left-to-right pass — no cascade.

Segment value semantics (§1):
  - Conf-based metrics: max over *qualifying* (condition-true) frames
    in the original un-merged window; absorbed-fail frames do NOT
    contribute. Falls back to 0.0 when the final merged segment holds
    no qualifying frames.
  - Hand angle: mean over ALL frames in the final merged segment.
  - Hand-object contact char: most common char in the final merged
    segment.
  - Obstructed: bool equal to the merged segment's state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import math


@dataclass(frozen=True)
class Run:
    """A contiguous stretch of per-frame samples with a single `state`.

    `start_sample` and `end_sample` are sample-array indices, half-open.
    `raw_frame_indices` holds the NATIVE frame indices for the samples
    in this run so downstream code can convert to seconds using
    `native_fps` without needing the sample-index → frame-index map
    again.
    """
    state: bool
    start_sample: int        # inclusive
    end_sample: int          # exclusive
    raw_frame_indices: tuple[int, ...]
    raw_pass_frame_indices: tuple[int, ...]   # subset where condition was True


def group_runs(
    values: Sequence[Any],
    frame_indices: Sequence[int],
    *,
    condition: Callable[[Any], bool] = bool,
) -> list[Run]:
    """Collapse per-sample values into runs via `condition(value)`.

    `values` and `frame_indices` must have the same length. Runs are
    contiguous, state-homogeneous, and cover the full input.
    """
    if len(values) != len(frame_indices):
        raise ValueError(
            f"group_runs: values ({len(values)}) and frame_indices "
            f"({len(frame_indices)}) must be the same length"
        )
    if not values:
        return []

    runs: list[Run] = []
    states = [bool(condition(v)) for v in values]
    i = 0
    n = len(values)
    while i < n:
        state = states[i]
        j = i + 1
        while j < n and states[j] == state:
            j += 1
        raw = tuple(frame_indices[i:j])
        pass_raw = raw if state else ()
        runs.append(Run(
            state=state,
            start_sample=i,
            end_sample=j,
            raw_frame_indices=raw,
            raw_pass_frame_indices=pass_raw,
        ))
        i = j
    return runs


def _run_duration_s(run: Run, sample_to_second: Callable[[int], float]) -> float:
    """Duration of a run in seconds, using the sample → second mapping."""
    if not run.raw_frame_indices:
        return 0.0
    # Use the seconds of the first and last sample, plus one sample's
    # worth of spacing to approximate the half-open interval.
    start_s = sample_to_second(run.raw_frame_indices[0])
    last_s = sample_to_second(run.raw_frame_indices[-1])
    # Sample spacing: estimate from the next sample position; if only
    # one sample, use a small positive delta so merge thresholds still
    # apply consistently.
    return max(last_s - start_s, 0.0)


def merge_short_runs(
    runs: list[Run],
    min_duration_s: float,
    *,
    sample_to_second: Callable[[int], float],
    sample_period_s: float,
) -> list[Run]:
    """Absorb runs shorter than `min_duration_s` into their neighbour.

    Rule (plan §1):
      - A run whose duration is < `min_duration_s` absorbs into the
        PRECEDING run (flipping state so the combined stretch carries
        the neighbour's state).
      - If the short run is the first in the list, it absorbs into the
        FOLLOWING run.
      - Single left-to-right pass. No cascade re-merging: once a short
        run has been folded in, its merged parent keeps its state even
        if the combined window would now itself be short.
      - Trailing short run after a merge is left as-is (boundary artefact).

    Duration is inclusive-of-one-sample: `last_sample - first_sample +
    sample_period_s`. This matches the plan's intuition that a single-
    sample run still has non-zero duration.
    """
    if not runs:
        return []
    if min_duration_s <= 0:
        return list(runs)

    def _dur(run: Run) -> float:
        return _run_duration_s(run, sample_to_second) + sample_period_s

    merged: list[Run] = []
    i = 0
    n = len(runs)
    while i < n:
        run = runs[i]
        if _dur(run) < min_duration_s:
            if not merged:
                # First run is short → fold into the following run.
                if i + 1 < n:
                    follower = runs[i + 1]
                    combined_raw = run.raw_frame_indices + follower.raw_frame_indices
                    # Pass-frames in the absorbed short run do not count.
                    combined_pass = follower.raw_pass_frame_indices
                    merged.append(Run(
                        state=follower.state,
                        start_sample=run.start_sample,
                        end_sample=follower.end_sample,
                        raw_frame_indices=combined_raw,
                        raw_pass_frame_indices=combined_pass,
                    ))
                    i += 2
                    continue
                else:
                    # Whole video is a single short run → keep as-is.
                    merged.append(run)
                    i += 1
                    continue
            else:
                # Fold short run into the preceding (flips state).
                prev = merged.pop()
                combined_raw = prev.raw_frame_indices + run.raw_frame_indices
                # Absorbed fail-run's pass-frames set is empty (or, if
                # we're absorbing a pass-run into a fail-run, the
                # absorbed pass-frames still do NOT count — plan §1).
                combined_pass = prev.raw_pass_frame_indices
                merged.append(Run(
                    state=prev.state,
                    start_sample=prev.start_sample,
                    end_sample=run.end_sample,
                    raw_frame_indices=combined_raw,
                    raw_pass_frame_indices=combined_pass,
                ))
                i += 1
                continue
        else:
            merged.append(run)
            i += 1

    return merged


# ── Value helpers ──────────────────────────────────────────────────────────

def segment_confidence_value(
    run: Run,
    conf_by_frame: dict[int, float],
) -> float:
    """Max confidence over qualifying (non-absorbed) pass frames.

    Falls back to 0.0 when the merged run contains no qualifying
    pass-frames (e.g., it's a fail-run).
    """
    if not run.raw_pass_frame_indices:
        return 0.0
    return float(max(
        conf_by_frame.get(idx, 0.0) for idx in run.raw_pass_frame_indices
    ))


def segment_angle_value(
    run: Run,
    angle_by_frame: dict[int, float],
) -> float:
    """Mean angle across all (merged-or-not) frames in the segment.

    Skips NaNs. Returns NaN when every frame is NaN (zero hands in the
    entire window).
    """
    vals = [angle_by_frame.get(i) for i in run.raw_frame_indices]
    clean = [v for v in vals if v is not None and not math.isnan(v)]
    if not clean:
        return float("nan")
    return sum(clean) / len(clean)


def segment_contact_value(
    run: Run,
    contact_by_frame: dict[int, str | None],
) -> str | None:
    """Most common contact char across the merged segment (nulls skipped)."""
    chars = [
        contact_by_frame.get(i) for i in run.raw_frame_indices
    ]
    clean = [c for c in chars if c is not None]
    if not clean:
        return None
    return Counter(clean).most_common(1)[0][0]
