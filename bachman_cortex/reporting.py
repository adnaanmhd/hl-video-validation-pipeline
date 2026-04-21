"""Per-video + batch report writers.

Per-video:    `report.md`, `{video_name}.json`, `{video_name}.parquet`
Batch:        `batch_report.md`, `batch_results.json`, `batch_results.csv`

The dataclasses in `data_types.py` are the single source of truth — this
module is pure formatting. Parquet is written separately by the
`PerFrameStore` handed back from the engine.

Output dir layout (§5):

    results/run_NNN/
        {video_name}/
            report.md
            {video_name}.json
            {video_name}.parquet     # omitted if metadata failed
        batch_report.md
        batch_results.json
        batch_results.csv
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from bachman_cortex.data_types import (
    BatchScoreReport,
    CheckStats,
    METADATA_CHECKS,
    METADATA_OBSERVATIONS,
    METADATA_OBSERVATION_CATEGORICAL,
    METADATA_OBSERVATION_NUMERIC,
    MetadataCheckResult,
    MetadataObservations,
    QUALITY_METRICS,
    QUALITY_VALUE_LABELS,
    QualityMetricResult,
    QualitySegment,
    QualityStats,
    TECHNICAL_CHECKS,
    TechnicalCheckResult,
    VideoScoreReport,
)
from bachman_cortex.per_frame_store import PerFrameStore


_STATUS_DISPLAY = {
    "pass": "PASS", "fail": "FAIL", "skipped": "SKIPPED",
}


def _fmt_status(status: str) -> str:
    return _STATUS_DISPLAY.get(status.lower(), status.upper())


def _fmt_value(value: Any) -> str:
    """Segment value formatting per plan §6:
    floats → 2 decimals, strings/bools → as-is, NaN → 'NaN'.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.2f}"
    if isinstance(value, int):
        return str(value)
    return str(value)


# ── Output layout ──────────────────────────────────────────────────────────

def allocate_run_dir(out_root: str | Path, max_attempts: int = 100) -> Path:
    """Create `run_NNN` under `out_root` via atomic mkdir with bump-on-collision.

    NNN is zero-padded to 3 digits up to 999, then auto-extends.
    """
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    existing = [p.name for p in root.iterdir() if p.is_dir()
                and p.name.startswith("run_")]
    start = 1
    for name in existing:
        suffix = name.removeprefix("run_")
        if suffix.isdigit():
            start = max(start, int(suffix) + 1)

    for attempt in range(max_attempts):
        n = start + attempt
        width = max(3, len(str(n)))
        candidate = root / f"run_{n:0{width}d}"
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(
        f"Could not allocate a fresh run_ directory under {root} "
        f"after {max_attempts} attempts"
    )


# ── Per-video JSON ─────────────────────────────────────────────────────────

def _video_report_to_dict(report: VideoScoreReport) -> dict[str, Any]:
    d = dataclasses.asdict(report)
    # Rewrite segment values so NaN serialises as the string "NaN"
    # rather than invalid JSON.
    for m in d.get("quality_metrics", []):
        for seg in m.get("segments", []):
            if isinstance(seg["value"], float) and math.isnan(seg["value"]):
                seg["value"] = "NaN"
    return d


def _write_video_json(report: VideoScoreReport, path: Path) -> None:
    path.write_text(json.dumps(_video_report_to_dict(report), indent=2))


# ── Per-video markdown ─────────────────────────────────────────────────────

def _render_metadata_table(checks: list[MetadataCheckResult]) -> str:
    lines = ["| Check | Status | Accepted | Detected |",
             "|---|---|---|---|"]
    for c in checks:
        lines.append(
            f"| {c.check} | {_fmt_status(c.status)} | {c.accepted} | {c.detected} |"
        )
    return "\n".join(lines)


def _fmt_observation_value(field: str, value: Any) -> str:
    """Human-readable cell for a single observation field."""
    if value is None:
        return "Unknown" if field in METADATA_OBSERVATION_CATEGORICAL else "-"
    if field == "bitrate_mbps" and isinstance(value, (int, float)):
        return f"{float(value):.2f} Mbps"
    if field == "gop" and isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    if field == "color_depth_bits" and isinstance(value, int):
        return f"{value}-bit"
    return str(value)


def _csv_observation_value(field: str, value: Any) -> str:
    """CSV-safe cell for a single observation field (no unit suffixes)."""
    if value is None:
        return "Unknown" if field in METADATA_OBSERVATION_CATEGORICAL else "-"
    if field == "bitrate_mbps" and isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    if field == "gop" and isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    return str(value)


def _render_observations_table(obs: MetadataObservations | None) -> str:
    lines = ["| Field | Value |", "|---|---|"]
    if obs is None:
        for f in METADATA_OBSERVATIONS:
            lines.append(f"| {f} | - |")
        return "\n".join(lines)
    for f in METADATA_OBSERVATIONS:
        lines.append(f"| {f} | {_fmt_observation_value(f, getattr(obs, f))} |")
    return "\n".join(lines)


def _render_technical_table(checks: list[TechnicalCheckResult]) -> str:
    lines = ["| Check | Status | Accepted | Detected |",
             "|---|---|---|---|"]
    for c in checks:
        if c.skipped or c.status == "skipped":
            lines.append(f"| {c.check} | SKIPPED | - | - |")
        else:
            lines.append(
                f"| {c.check} | {_fmt_status(c.status)} | {c.accepted} | {c.detected} |"
            )
    return "\n".join(lines)


def _render_quality_section(metrics: list[QualityMetricResult]) -> str:
    parts: list[str] = []
    for m in metrics:
        parts.append(f"### {m.metric}")
        if m.skipped:
            parts.append("SKIPPED")
            parts.append("")
            continue
        parts.append(f"- percent_frames: {m.percent_frames:.2f}%")
        if not m.segments:
            parts.append("- segments: (none)")
            parts.append("")
            continue
        label = QUALITY_VALUE_LABELS[m.metric]
        parts.append("")
        parts.append(f"| start_s | end_s | duration_s | {label} |")
        parts.append("|---|---|---|---|")
        for seg in m.segments:
            parts.append(
                f"| {seg.start_s:.3f} | {seg.end_s:.3f} | "
                f"{seg.duration_s:.3f} | {_fmt_value(seg.value)} |"
            )
        parts.append("")
    return "\n".join(parts)


def _render_video_markdown(report: VideoScoreReport) -> str:
    head = [
        f"# {report.video_name}",
        "",
        f"- generated_at: {report.generated_at}",
        f"- duration_s: {report.duration_s:.2f}",
        f"- processing_wall_time_s: {report.processing_wall_time_s:.2f}",
        "",
        "## Metadata",
        "",
        _render_metadata_table(report.metadata_checks),
        "",
        "## Metadata observations",
        "",
        _render_observations_table(report.metadata_observations),
        "",
        "## Technical",
        "",
        _render_technical_table(report.technical_checks),
        "",
        "## Quality",
        "",
        _render_quality_section(report.quality_metrics),
    ]
    return "\n".join(head).rstrip() + "\n"


# ── Public: per-video writer ──────────────────────────────────────────────

def write_video_report(
    report: VideoScoreReport,
    out_dir: str | Path,
    per_frame_store: PerFrameStore | None = None,
) -> dict[str, Path]:
    """Write `report.md`, `{video_name}.json`, and (if applicable) parquet.

    Returns the map of artefact name → absolute path actually written.
    Parquet is skipped when the metadata stage failed (no decode).
    """
    video_dir = Path(out_dir) / Path(report.video_name).stem
    video_dir.mkdir(parents=True, exist_ok=True)

    md_path = video_dir / "report.md"
    md_path.write_text(_render_video_markdown(report))

    json_path = video_dir / f"{Path(report.video_name).stem}.json"
    _write_video_json(report, json_path)

    paths = {"md": md_path, "json": json_path}
    if per_frame_store is not None and len(per_frame_store):
        parquet_path = video_dir / f"{Path(report.video_name).stem}.parquet"
        per_frame_store.flush(parquet_path)
        paths["parquet"] = parquet_path
    return paths


# ── Batch: markdown + JSON + CSV ──────────────────────────────────────────

def _render_batch_markdown(batch: BatchScoreReport) -> str:
    lines = [
        "# Batch Report",
        "",
        f"- generated_at: {batch.generated_at}",
        f"- video_count: {batch.video_count}",
        f"- total_duration_s: {batch.total_duration_s:.2f}",
        f"- total_wall_time_s: {batch.total_wall_time_s:.2f}",
        "",
    ]

    lines.extend([
        "## Metadata check stats",
        "",
        "| Check | Pass | Pass duration (s) | Fail | Fail duration (s) | Skipped |",
        "|---|---|---|---|---|---|",
    ])
    for c in METADATA_CHECKS:
        stats = batch.metadata_check_stats.get(c, CheckStats(0, 0.0, 0, 0.0))
        lines.append(
            f"| {c} | {stats.pass_count} | {stats.pass_duration_s:.2f} | "
            f"{stats.fail_count} | {stats.fail_duration_s:.2f} | "
            f"{stats.skipped_count} |"
        )

    lines.extend([
        "",
        "## Technical check stats",
        "",
        "| Check | Pass | Pass duration (s) | Fail | Fail duration (s) | Skipped |",
        "|---|---|---|---|---|---|",
    ])
    for c in TECHNICAL_CHECKS:
        stats = batch.technical_check_stats.get(c, CheckStats(0, 0.0, 0, 0.0))
        lines.append(
            f"| {c} | {stats.pass_count} | {stats.pass_duration_s:.2f} | "
            f"{stats.fail_count} | {stats.fail_duration_s:.2f} | "
            f"{stats.skipped_count} |"
        )

    lines.extend([
        "",
        "## Quality metric stats (% frames, across non-skipped videos)",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|---|---|---|---|---|",
    ])
    for m in QUALITY_METRICS:
        qs = batch.quality_metric_stats.get(m)
        if qs is None:
            lines.append(f"| {m} | - | - | - | - |")
        else:
            lines.append(
                f"| {m} | {qs.mean_percent:.2f} | {qs.median_percent:.2f} | "
                f"{qs.min_percent:.2f} | {qs.max_percent:.2f} |"
            )

    lines.extend(_render_observations_aggregate(batch.videos))

    if batch.errors:
        lines.extend([
            "",
            "## Errors",
            "",
            "| Video | Reason |",
            "|---|---|",
        ])
        for e in batch.errors:
            lines.append(f"| {e.video_name} | {e.error_reason} |")

    lines.extend([
        "",
        "## Videos",
        "",
        "| Video | Duration (s) | Metadata | Technical | Quality |",
        "|---|---|---|---|---|",
    ])
    for v in batch.videos:
        meta_status = ("FAIL"
                       if any(c.status == "fail" for c in v.metadata_checks)
                       else "PASS")
        if v.technical_skipped:
            tech_status = "SKIPPED"
        else:
            tech_status = ("FAIL"
                           if any(c.status == "fail" for c in v.technical_checks)
                           else "PASS")
        qual_status = "SKIPPED" if v.quality_skipped else "OK"
        lines.append(
            f"| {v.video_name} | {v.duration_s:.2f} | "
            f"{meta_status} | {tech_status} | {qual_status} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_observations_aggregate(videos: list[VideoScoreReport]) -> list[str]:
    """Batch-level aggregate: mean/median/min/max for numeric fields,
    distribution histogram for categorical fields. Skips sections when
    no data is available (e.g. an empty batch).
    """
    if not videos:
        return []

    numeric: dict[str, list[float]] = {f: [] for f in METADATA_OBSERVATION_NUMERIC}
    categorical: dict[str, dict[str, int]] = {
        f: {} for f in METADATA_OBSERVATION_CATEGORICAL
    }

    for v in videos:
        obs = v.metadata_observations
        if obs is None:
            continue
        for f in METADATA_OBSERVATION_NUMERIC:
            val = getattr(obs, f, None)
            if val is None:
                continue
            try:
                numeric[f].append(float(val))
            except (TypeError, ValueError):
                continue
        for f in METADATA_OBSERVATION_CATEGORICAL:
            val = getattr(obs, f, None)
            key = "Unknown" if val is None else str(val)
            categorical[f][key] = categorical[f].get(key, 0) + 1

    out: list[str] = ["", "## Metadata observations (aggregate)", ""]

    any_numeric = any(numeric[f] for f in METADATA_OBSERVATION_NUMERIC)
    if any_numeric:
        out.extend([
            "### Numeric",
            "",
            "| Field | Mean | Median | Min | Max |",
            "|---|---|---|---|---|",
        ])
        for f in METADATA_OBSERVATION_NUMERIC:
            vals = numeric[f]
            if not vals:
                out.append(f"| {f} | - | - | - | - |")
                continue
            precision = {"gop": 1, "color_depth_bits": 0}.get(f, 2)
            fmt = f"%.{precision}f"
            out.append(
                f"| {f} | {fmt % statistics.fmean(vals)} | "
                f"{fmt % statistics.median(vals)} | "
                f"{fmt % min(vals)} | {fmt % max(vals)} |"
            )

    any_categorical = any(categorical[f] for f in METADATA_OBSERVATION_CATEGORICAL)
    if any_categorical:
        if any_numeric:
            out.append("")
        out.extend([
            "### Categorical",
            "",
            "| Field | Distribution |",
            "|---|---|",
        ])
        for f in METADATA_OBSERVATION_CATEGORICAL:
            hist = categorical[f]
            if not hist:
                out.append(f"| {f} | - |")
                continue
            ordered = sorted(hist.items(), key=lambda kv: (-kv[1], kv[0]))
            body = ", ".join(f"{label}: {count}" for label, count in ordered)
            out.append(f"| {f} | {body} |")

    return out


def _batch_report_to_dict(batch: BatchScoreReport) -> dict[str, Any]:
    d = dataclasses.asdict(batch)
    for video in d.get("videos", []):
        for m in video.get("quality_metrics", []):
            for seg in m.get("segments", []):
                if isinstance(seg["value"], float) and math.isnan(seg["value"]):
                    seg["value"] = "NaN"
    return d


def _render_batch_csv(batch: BatchScoreReport) -> str:
    """Row per video; two columns per check (status + value) + quality % columns."""
    rows: list[list[str]] = []
    header = ["video_name", "duration_s"]
    for c in METADATA_CHECKS:
        header.extend([f"meta_{c}_status", f"meta_{c}_value"])
    for c in TECHNICAL_CHECKS:
        header.extend([f"tech_{c}_status", f"tech_{c}_value"])
    for m in QUALITY_METRICS:
        header.append(f"quality_{m}_pct")
    for f in METADATA_OBSERVATIONS:
        header.append(f"meta_{f}")
    rows.append(header)

    for v in batch.videos:
        row = [v.video_name, f"{v.duration_s:.2f}"]
        meta_by = {c.check: c for c in v.metadata_checks}
        for c in METADATA_CHECKS:
            mc = meta_by.get(c)
            if mc is None:
                row.extend(["-", "-"])
            else:
                row.extend([_fmt_status(mc.status), mc.detected])
        tech_by = {t.check: t for t in v.technical_checks}
        for c in TECHNICAL_CHECKS:
            tc = tech_by.get(c)
            if tc is None:
                row.extend(["-", "-"])
            elif tc.skipped or tc.status == "skipped":
                row.extend(["SKIPPED", "-"])
            else:
                row.extend([_fmt_status(tc.status), tc.detected])
        qual_by = {q.metric: q for q in v.quality_metrics}
        for m in QUALITY_METRICS:
            q = qual_by.get(m)
            if q is None or q.skipped:
                row.append("SKIPPED")
            else:
                row.append(f"{q.percent_frames:.2f}")
        for f in METADATA_OBSERVATIONS:
            row.append(_csv_observation_value(
                f, getattr(v.metadata_observations, f, None)
                if v.metadata_observations is not None else None
            ))
        rows.append(row)

    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue()


def write_batch_report(batch: BatchScoreReport, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "batch_report.md"
    md_path.write_text(_render_batch_markdown(batch))

    json_path = out / "batch_results.json"
    json_path.write_text(json.dumps(_batch_report_to_dict(batch), indent=2))

    csv_path = out / "batch_results.csv"
    csv_path.write_text(_render_batch_csv(batch))

    return {"md": md_path, "json": json_path, "csv": csv_path}


# ── Batch stats aggregation ───────────────────────────────────────────────

def aggregate_batch_stats(
    videos: list[VideoScoreReport],
) -> tuple[dict[str, CheckStats], dict[str, CheckStats], dict[str, QualityStats]]:
    """Fold per-video results into batch-level aggregates."""

    meta_stats: dict[str, CheckStats] = {
        c: CheckStats(0, 0.0, 0, 0.0) for c in METADATA_CHECKS
    }
    tech_stats: dict[str, CheckStats] = {
        c: CheckStats(0, 0.0, 0, 0.0) for c in TECHNICAL_CHECKS
    }
    quality_percents: dict[str, list[float]] = {m: [] for m in QUALITY_METRICS}

    for v in videos:
        for c in v.metadata_checks:
            slot = meta_stats[c.check]
            if c.status == "pass":
                slot.pass_count += 1
                slot.pass_duration_s += v.duration_s
            else:
                slot.fail_count += 1
                slot.fail_duration_s += v.duration_s
        for t in v.technical_checks:
            slot = tech_stats[t.check]
            if t.skipped or t.status == "skipped":
                slot.skipped_count += 1
            elif t.status == "pass":
                slot.pass_count += 1
                slot.pass_duration_s += v.duration_s
            else:
                slot.fail_count += 1
                slot.fail_duration_s += v.duration_s
        for q in v.quality_metrics:
            if not q.skipped:
                quality_percents[q.metric].append(q.percent_frames)

    quality_stats: dict[str, QualityStats] = {}
    for m in QUALITY_METRICS:
        vals = quality_percents[m]
        if not vals:
            continue
        quality_stats[m] = QualityStats(
            mean_percent=round(statistics.fmean(vals), 2),
            median_percent=round(statistics.median(vals), 2),
            min_percent=round(min(vals), 2),
            max_percent=round(max(vals), 2),
        )
    return meta_stats, tech_stats, quality_stats
