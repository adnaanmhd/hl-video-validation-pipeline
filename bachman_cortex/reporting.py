"""Report generation for the validation & processing pipeline.

Generates per-video timeline reports and batch-level summary reports.
No clip files are produced — reports describe usable/unusable/rejected
time ranges with failing reasons.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.data_types import (
    VideoProcessingResult,
    TimeSegment,
    CheckFrameResults,
    SegmentValidationResult,
)
from bachman_cortex.pipeline import PipelineConfig
from bachman_cortex.utils.review_runs import ReviewRun, collapse_review_runs


_FACE_REVIEW_LO = 0.5
_FACE_REVIEW_HI = 0.8
_REVIEW_MIN_DURATION_S = 2.0


# ── Timeline entry ───────────────────────────────────────────────────────


@dataclass
class TimelineEntry:
    """One row in the unified chronological timeline."""
    start_sec: float
    end_sec: float
    category: str  # "USABLE" | "UNUSABLE" | "REJECTED"
    reasons: list[str]

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


# ── Accepted / Observed value helpers ────────────────────────────────────


def _meta_accepted_observed(name: str, r: CheckResult) -> tuple[str, str]:
    """Return (accepted_value, observed_value) for a metadata check."""
    d = r.details or {}
    if name == "meta_format":
        return "mp4", d.get("container_format", "?")
    elif name == "meta_encoding":
        return d.get("expected", "h264"), d.get("video_codec", "?")
    elif name == "meta_resolution":
        return (
            f">={d.get('min_width', 1920)}x{d.get('min_height', 1080)}",
            f"{d.get('width', '?')}x{d.get('height', '?')}",
        )
    elif name == "meta_frame_rate":
        return f">={d.get('min_fps', 28.0)} FPS", f"{d.get('fps', '?')} FPS"
    elif name == "meta_duration":
        return f">={d.get('min_duration_s', 119.0)}s", f"{d.get('duration_s', '?')}s"
    elif name == "meta_orientation":
        return (
            f"rotation {d.get('expected_rotation', '0 or 180')}, landscape",
            f"rotation={d.get('rotation', '?')}, {d.get('width', '?')}x{d.get('height', '?')}",
        )
    return "-", "-"


def _segment_accepted_observed(name: str, r: CheckResult) -> tuple[str, str]:
    """Return (accepted_value, observed_value) for a Phase 2 segment check."""
    d = r.details or {}
    metric = r.metric_value

    if name == "luminance_blur":
        threshold = d.get("min_good_ratio", "?")
        return f">={_fmt_pct(threshold)}", _fmt_pct(metric)
    elif name == "ml_hand_visibility":
        both_th = d.get("both_hands_pass_rate", "?")
        single_th = d.get("single_hand_pass_rate", "?")
        both_obs = d.get("both_hands_ratio", metric)
        single_obs = d.get("single_hand_ratio", "?")
        accepted = f"both>={_fmt_pct(both_th)} OR single>={_fmt_pct(single_th)}"
        observed = f"both={_fmt_pct(both_obs)}, single={_fmt_pct(single_obs)}"
        return accepted, observed
    elif name == "ml_hand_object_interaction":
        threshold = d.get("pass_rate_threshold", "?")
        return f">={_fmt_pct(threshold)}", _fmt_pct(metric)
    elif name == "ml_view_obstruction":
        max_ratio = d.get("max_allowed_ratio", "?")
        obstructed = d.get("obstructed_ratio", 1.0 - metric)
        return f"obstructed <={_fmt_pct(max_ratio)}", f"obstructed {_fmt_pct(obstructed)}"
    elif name == "ml_pov_hand_angle":
        threshold = d.get("pass_rate_threshold", "?")
        return f">={_fmt_pct(threshold)}", _fmt_pct(metric)
    elif name == "ml_face_presence":
        threshold = d.get("threshold", "?")
        frames_with = d.get("frames_with_prominent_face", 0)
        total = d.get("total_frames", 0)
        max_conf = d.get("max_face_confidence_seen", 0.0)
        accepted = f"0 frames with face conf >={threshold}"
        observed = f"{frames_with}/{total} frames, max conf {max_conf:.2f}"
        return accepted, observed
    elif name == "motion_camera_stability":
        threshold = d.get("shaky_score_threshold", "?")
        return f"<={_fmt_pct(threshold)}", _fmt_pct(metric)
    elif name == "motion_frozen_segments":
        max_consec = d.get("max_consecutive_native", "?")
        longest = d.get("longest_frozen_run_native_est", metric)
        return f"<{max_consec} consecutive frozen", f"{longest:.1f} longest run"

    return "-", f"{metric:.4f}"


# ── Formatting helpers ───────────────────────────────────────────────────


def _fmt_pct(value) -> str:
    """Format a 0-1 ratio as a percentage string."""
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    return str(value)


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _fmt_duration(seconds: float) -> str:
    """Format a duration as '123.4s (2m 3s)' or '45.6s' for short values."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    if h:
        return f"{seconds:.1f}s ({h}h {m}m {s:.0f}s)"
    return f"{seconds:.1f}s ({m}m {s:.0f}s)"


def _segments_overlap(a_start: float, a_end: float, b: TimeSegment) -> bool:
    return a_start < b.end_sec and b.start_sec < a_end


def _count_bad_frames_in_window(
    cfr: CheckFrameResults, start_sec: float, end_sec: float,
) -> tuple[int, int]:
    """Return (bad_frame_count, total_frame_count) within the time window."""
    bad = 0
    total = 0
    for fl in cfr.frame_labels:
        if start_sec <= fl.timestamp_sec < end_sec:
            total += 1
            if not fl.passed:
                bad += 1
    return bad, total


# ── Timeline construction ────────────────────────────────────────────────


def _phase1_bad_reasons(
    seg_start: float,
    seg_end: float,
    phase1_check_results: list[CheckFrameResults],
) -> list[str]:
    """For a merged Phase 1 bad segment, list the failing checks with counts."""
    reasons: list[str] = []
    for cfr in phase1_check_results:
        if not any(_segments_overlap(seg_start, seg_end, bs)
                   for bs in cfr.bad_segments):
            continue
        bad, total = _count_bad_frames_in_window(cfr, seg_start, seg_end)
        if total > 0:
            reasons.append(f"{cfr.check_name} ({bad}/{total} frames failed)")
        else:
            reasons.append(cfr.check_name)
    return reasons or ["phase1_unknown"]


def _segment_reject_reasons(sr: SegmentValidationResult) -> list[str]:
    """For a Phase 2 rejected segment, list failing checks + observed metrics."""
    reasons: list[str] = []
    for name in sr.failing_checks:
        r = sr.check_results.get(name)
        if r is None:
            reasons.append(name)
            continue
        accepted, observed = _segment_accepted_observed(name, r)
        reasons.append(f"{name} (accepted {accepted}; observed {observed})")
    return reasons or ["phase2_unknown"]


def _metadata_reject_reasons(
    meta_results: dict[str, CheckResult],
) -> list[str]:
    """List failing metadata checks with accepted/observed values."""
    reasons: list[str] = []
    for name, r in meta_results.items():
        if r.status != "fail":
            continue
        accepted, observed = _meta_accepted_observed(name, r)
        reasons.append(f"{name} (accepted {accepted}; observed {observed})")
    return reasons or ["metadata_unknown"]


def _build_timeline(result: VideoProcessingResult) -> list[TimelineEntry]:
    """Build the unified chronological timeline for a successful video.

    Does not handle metadata-fail videos; caller must branch for those.
    """
    entries: list[TimelineEntry] = []

    # USABLE — segments that passed Phase 2
    for seg in result.usable_segments:
        entries.append(TimelineEntry(
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            category="USABLE",
            reasons=[],
        ))

    # UNUSABLE — Phase 1 bad segments
    for seg in result.phase1_bad_segments:
        entries.append(TimelineEntry(
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            category="UNUSABLE",
            reasons=_phase1_bad_reasons(
                seg.start_sec, seg.end_sec, result.phase1_check_frame_results,
            ),
        ))

    # UNUSABLE — Phase 1 discarded (clean gaps too short for Phase 2)
    for seg in result.phase1_discarded_segments:
        entries.append(TimelineEntry(
            start_sec=seg.start_sec,
            end_sec=seg.end_sec,
            category="UNUSABLE",
            reasons=["segment_too_short"],
        ))

    # REJECTED — Phase 2 validation failures
    for sr in result.segment_results:
        if sr.passed:
            continue
        entries.append(TimelineEntry(
            start_sec=sr.segment.start_sec,
            end_sec=sr.segment.end_sec,
            category="REJECTED",
            reasons=_segment_reject_reasons(sr),
        ))

    entries.sort(key=lambda e: e.start_sec)
    return entries


def _timeline_table(entries: list[TimelineEntry]) -> list[str]:
    """Render a unified timeline as a markdown table."""
    if not entries:
        return ["_No timeline entries._", ""]

    lines = [
        "| # | Start | End | Duration | Category | Reasons |",
        "|---|---|---|---|---|---|",
    ]
    for i, e in enumerate(entries, 1):
        reasons = "; ".join(e.reasons) if e.reasons else "-"
        lines.append(
            f"| {i} | {_fmt_ts(e.start_sec)} | {_fmt_ts(e.end_sec)} | "
            f"{e.duration:.3f}s | **{e.category}** | {reasons} |"
        )
    lines.append("")
    return lines


# ── Review frames (per-video) ────────────────────────────────────────────


def _collect_face_review_frames(
    result: VideoProcessingResult,
    frame_interval: float,
) -> list[dict]:
    """Face-presence review frames from Phase 1 across the whole video."""
    face_cfr = next(
        (c for c in result.phase1_check_frame_results
         if c.check_name == "ml_face_presence"),
        None,
    )
    if face_cfr is None:
        return []

    records: list[dict] = []
    for fl in face_cfr.frame_labels:
        conf = fl.confidence
        if not (_FACE_REVIEW_LO <= conf < _FACE_REVIEW_HI):
            continue
        records.append({
            "timestamp_sec": fl.timestamp_sec,
            "confidence": conf,
        })
    return records


def _collect_segment_review_frames(
    result: VideoProcessingResult,
    check_name: str,
) -> list[dict]:
    """Aggregate per-segment review_frames from all Phase 2 segments."""
    records: list[dict] = []
    for sr in result.segment_results:
        cr = (sr.check_results or {}).get(check_name)
        if cr is None:
            continue
        records.extend((cr.details or {}).get("review_frames", []))
    return records


def _fmt_triple(agg: tuple[float, float, float] | None, decimals: int = 2) -> str:
    if agg is None:
        return "-"
    mean, lo, hi = agg
    return f"{mean:.{decimals}f} / {lo:.{decimals}f} / {hi:.{decimals}f}"


def _review_run_table(
    runs: list[ReviewRun],
    columns: list[tuple[str, str, int]],
) -> list[str]:
    """Render a review-runs table.

    columns: list of (header_label, aggregate_key, decimals) for value columns.
    """
    headers = ["Start", "End", "Duration", "Frames"] + [c[0] for c in columns]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in runs:
        row = [
            _fmt_ts(r.start_sec),
            _fmt_ts(r.end_sec),
            f"{r.duration:.3f}s",
            str(r.frame_count),
        ]
        for _label, key, decimals in columns:
            row.append(_fmt_triple(r.aggregates.get(key), decimals))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _review_section(
    result: VideoProcessingResult,
    frame_interval: float,
) -> list[str]:
    """Render the 'Frames Flagged for Review' section."""
    lines: list[str] = []

    specs = [
        (
            "ml_face_presence",
            _collect_face_review_frames(result, frame_interval),
            [("Conf (mean / min / max)", "confidence", 2)],
        ),
        (
            "ml_hand_visibility",
            _collect_segment_review_frames(result, "ml_hand_visibility"),
            [("Conf (mean / min / max)", "confidence", 2)],
        ),
        (
            "ml_hand_object_interaction",
            _collect_segment_review_frames(result, "ml_hand_object_interaction"),
            [("Conf (mean / min / max)", "confidence", 2)],
        ),
        (
            "luminance_blur",
            _collect_segment_review_frames(result, "luminance_blur"),
            [
                ("Mean Luminance (mean / min / max)", "mean_luminance", 2),
                ("Norm. Tenengrad (mean / min / max)", "tenengrad_norm", 4),
                ("Raw Tenengrad (mean / min / max)", "tenengrad_raw", 2),
            ],
        ),
    ]

    all_runs: dict[str, list[ReviewRun]] = {}
    any_runs = False
    for name, frames, columns in specs:
        value_keys = [c[1] for c in columns]
        runs = collapse_review_runs(
            frames,
            frame_interval=frame_interval,
            min_duration_s=_REVIEW_MIN_DURATION_S,
            value_keys=value_keys,
        )
        all_runs[name] = runs
        if runs:
            any_runs = True

    if not any_runs:
        return lines

    lines.append("---")
    lines.append("")
    lines.append("## Frames Flagged for Review")
    lines.append("")
    lines.append(
        f"Contiguous review runs strictly longer than {_REVIEW_MIN_DURATION_S:g}s "
        "per check. These frames **count as accept** for pass/fail; "
        "this section is for manual QA only."
    )
    lines.append("")

    for name, _frames, columns in specs:
        runs = all_runs[name]
        if not runs:
            continue
        total_dur = sum(r.duration for r in runs)
        lines.append(f"### {name} — {len(runs)} range(s), {total_dur:.1f}s")
        lines.append("")
        lines.extend(_review_run_table(runs, columns))

    return lines


# ── Per-video report ─────────────────────────────────────────────────────


def write_video_report(
    result: VideoProcessingResult,
    output_dir: Path,
    config: PipelineConfig | None = None,
) -> Path:
    """Write a per-video markdown report with a unified timeline view."""
    report_path = output_dir / "report.md"
    lines: list[str] = []

    lines.append(f"# Video Report: {result.video_name}")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Source:** `{result.video_path}`  ")
    lines.append(f"**Processing time:** {result.processing_time_sec:.1f}s")
    lines.append("")

    # ── Summary ──────────────────────────────────────────────────────
    unusable_p1 = sum(s.duration for s in result.phase1_bad_segments)
    unusable_short = sum(s.duration for s in result.phase1_discarded_segments)
    rejected_dur = sum(
        sr.segment.duration for sr in result.segment_results if not sr.passed
    )

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Original duration | {result.original_duration_sec:.1f}s |")
    lines.append(f"| Usable | {result.usable_duration_sec:.1f}s |")
    lines.append(f"| Unusable (Phase 1 filter) | {unusable_p1:.1f}s |")
    lines.append(f"| Unusable (segment too short) | {unusable_short:.1f}s |")
    lines.append(f"| Rejected (Phase 2 validation) | {rejected_dur:.1f}s |")
    lines.append(f"| **Yield** | **{result.yield_ratio:.1%}** |")
    lines.append(f"| Metadata passed | {'Yes' if result.metadata_passed else 'No'} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Phase 0: Metadata ────────────────────────────────────────────
    lines.append("## Metadata")
    lines.append("")
    meta = result.metadata
    if meta:
        lines.append(
            f"**Resolution:** {meta.get('width', '?')}x{meta.get('height', '?')} | "
            f"**FPS:** {meta.get('fps', '?')} | "
            f"**Duration:** {meta.get('duration_s', '?')}s | "
            f"**Codec:** {meta.get('video_codec', '?')} | "
            f"**Size:** {meta.get('file_size_mb', '?')} MB"
        )
        lines.append("")

    lines.append("| Check | Status | Accepted | Observed |")
    lines.append("|---|---|---|---|")
    for name, r in result.metadata_results.items():
        accepted, observed = _meta_accepted_observed(name, r)
        lines.append(f"| {name} | **{r.status.upper()}** | {accepted} | {observed} |")
    lines.append("")

    # ── Timeline ─────────────────────────────────────────────────────
    lines.append("---")
    lines.append("")
    lines.append("## Timeline")
    lines.append("")

    if not result.metadata_passed:
        # Whole-video REJECTED at the metadata gate.
        dur = result.original_duration_sec or 0.0
        entries = [TimelineEntry(
            start_sec=0.0,
            end_sec=dur,
            category="REJECTED",
            reasons=_metadata_reject_reasons(result.metadata_results),
        )]
    else:
        entries = _build_timeline(result)

    lines.extend(_timeline_table(entries))

    # ── Frames Flagged for Review ────────────────────────────────────
    if config is not None and result.metadata_passed:
        frame_interval = 1.0 / config.sampling_fps if config.sampling_fps else 1.0
        lines.extend(_review_section(result, frame_interval))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ── Batch report ─────────────────────────────────────────────────────────


def write_batch_report(
    all_results: list[VideoProcessingResult],
    output_dir: Path,
    config: PipelineConfig,
    wall_clock_sec: float | None = None,
) -> Path:
    """Write a batch-level summary report with topline stats."""
    report_path = output_dir / "batch_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    # Topline stats
    total_original = sum(r.original_duration_sec for r in all_results)
    total_usable = sum(r.usable_duration_sec for r in all_results)
    total_unusable = sum(r.unusable_duration_sec for r in all_results)
    total_yield = total_usable / total_original if total_original > 0 else 0.0
    total_errors = sum(1 for r in all_results if r.error)
    total_processing_sec = sum(r.processing_time_sec for r in all_results)

    lines.append("# Batch Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}  ")
    lines.append(f"**Videos processed:** {len(all_results)}  ")
    lines.append(f"**Sampling FPS:** {config.sampling_fps}  ")
    lines.append(f"**Min segment duration:** {config.min_checkable_segment_sec}s  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Topline stats table
    lines.append("## Topline Stats")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Total videos | {len(all_results)} |")
    lines.append(f"| Total original duration | {total_original:.1f}s ({total_original/60:.1f} min) |")
    lines.append(f"| Total usable footage | {total_usable:.1f}s ({total_usable/60:.1f} min) |")
    lines.append(f"| Total unusable footage | {total_unusable:.1f}s ({total_unusable/60:.1f} min) |")
    lines.append(f"| **Total yield** | **{total_yield:.1%}** |")
    if total_errors:
        lines.append(f"| Errors | {total_errors} |")
    lines.append(f"| Total processing time (sum) | {_fmt_duration(total_processing_sec)} |")
    if wall_clock_sec is not None:
        lines.append(f"| **Wall-clock time** | **{_fmt_duration(wall_clock_sec)}** |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-video summary table
    lines.append("## Per-Video Summary")
    lines.append("")
    lines.append(
        "| # | Video | Duration | Usable | Unusable | Yield | Status | Time |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for idx, r in enumerate(all_results, 1):
        if r.error:
            lines.append(
                f"| {idx} | {r.video_name} | {r.original_duration_sec:.1f}s | "
                f"- | - | - | ERROR | - |"
            )
            continue

        if not r.metadata_passed:
            status = "META FAIL"
        elif not r.prefiltered_segments:
            status = "P1 REJECT"
        elif r.usable_segments:
            status = "HAS USABLE"
        else:
            status = "ALL REJECT"

        lines.append(
            f"| {idx} | {r.video_name} | {r.original_duration_sec:.1f}s | "
            f"{r.usable_duration_sec:.1f}s | {r.unusable_duration_sec:.1f}s | "
            f"{r.yield_ratio:.1%} | {status} | {r.processing_time_sec:.1f}s |"
        )

    lines.append("")

    # ── Face Presence — per-segment breakdown ────────────────────────
    face_rows: list[tuple[int, str, int, int, int, float, float]] = []
    for idx, r in enumerate(all_results, 1):
        for sr in r.segment_results or []:
            fp = (sr.check_results or {}).get("ml_face_presence")
            if not fp or fp.status != "fail":
                continue
            d = fp.details or {}
            face_rows.append((
                idx,
                r.video_name,
                sr.segment.segment_idx,
                d.get("frames_with_prominent_face", 0),
                d.get("total_frames", 0),
                d.get("max_face_confidence_seen", 0.0),
                d.get("threshold", 0.0),
            ))

    if face_rows:
        lines.append("---")
        lines.append("")
        lines.append("## Face Presence — Failing Segments")
        lines.append("")
        lines.append("Segments rejected because one or more sampled frames contain "
                     "a face above the confidence threshold.")
        lines.append("")
        lines.append(
            "| # | Video | Segment | Frames w/ Face | Total Frames | Max Face Conf | Threshold |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for (idx, vname, seg_idx, frames_with, total, max_conf, threshold) in face_rows:
            lines.append(
                f"| {idx} | {vname} | {seg_idx} | {frames_with} | {total} | "
                f"{max_conf:.2f} | {threshold} |"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
