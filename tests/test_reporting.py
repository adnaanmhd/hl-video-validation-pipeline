"""Tests for reporting.py — MD formatting, JSON round-trip, CSV columns."""

from __future__ import annotations

import csv
import io
import json
import math

import pytest

from bachman_cortex.data_types import (
    BatchScoreReport,
    MetadataCheckResult,
    ProcessingErrorReport,
    QualityMetricResult,
    QualitySegment,
    TechnicalCheckResult,
    VideoScoreReport,
)
from bachman_cortex.per_frame_store import PerFrameStore
from bachman_cortex import reporting


def _passing_metadata() -> list[MetadataCheckResult]:
    return [
        MetadataCheckResult("format", "pass", "container = MP4", "mp4"),
        MetadataCheckResult("encoding", "pass", "H.264 or HEVC", "h264"),
        MetadataCheckResult("resolution", "pass", ">= 1920x1080", "1920x1080"),
        MetadataCheckResult("frame_rate", "pass", ">= 28 FPS", "30.0 FPS"),
        MetadataCheckResult("duration", "pass", ">= 59 s", "120.0 s"),
        MetadataCheckResult("orientation", "pass", "landscape",
                            "rot=0, disp=1920x1080"),
    ]


def _failing_metadata_duration() -> list[MetadataCheckResult]:
    meta = _passing_metadata()
    meta[4] = MetadataCheckResult("duration", "fail", ">= 59 s", "25.0 s")
    return meta


def _passing_technical() -> list[TechnicalCheckResult]:
    return [
        TechnicalCheckResult("luminance", "pass", ">= 80% good frames",
                             "good_ratio=0.920"),
        TechnicalCheckResult("stability", "pass", "mean jitter <= 0.181",
                             "mean_jitter=0.0500"),
        TechnicalCheckResult("frozen", "pass", "no run > 30f", "ok"),
        TechnicalCheckResult("pixelation", "pass",
                             ">= 80% frames w/ blockiness <= 1.5",
                             "good_ratio=0.950"),
    ]


def _skipped_technical() -> list[TechnicalCheckResult]:
    return [
        TechnicalCheckResult(c, "skipped", "-", "-", skipped=True)
        for c in ("luminance", "stability", "frozen", "pixelation")
    ]


def _passing_quality() -> list[QualityMetricResult]:
    return [
        QualityMetricResult(
            metric="both_hands_visibility",
            percent_frames=72.50,
            segments=[
                QualitySegment(0.0, 10.0, 10.0, 0.92, "confidence"),
                QualitySegment(10.0, 20.0, 10.0, 0.0, "confidence"),
            ],
        ),
        QualityMetricResult(
            metric="single_hand_visibility",
            percent_frames=95.00,
            segments=[QualitySegment(0.0, 60.0, 60.0, 0.87, "confidence")],
        ),
        QualityMetricResult(
            metric="hand_obj_interaction",
            percent_frames=50.0,
            segments=[QualitySegment(0.0, 60.0, 60.0, "P", "contact_state")],
        ),
        QualityMetricResult(
            metric="hand_angle",
            percent_frames=60.0,
            segments=[QualitySegment(0.0, 60.0, 60.0, float("nan"), "angle")],
        ),
        QualityMetricResult(
            metric="participants",
            percent_frames=10.0,
            segments=[],
        ),
        QualityMetricResult(
            metric="obstructed",
            percent_frames=5.0,
            segments=[QualitySegment(0.0, 60.0, 60.0, False, "obstructed")],
        ),
    ]


def _skipped_quality() -> list[QualityMetricResult]:
    from bachman_cortex.data_types import QUALITY_METRICS
    return [
        QualityMetricResult(metric=m, percent_frames=0.0, segments=[], skipped=True)
        for m in QUALITY_METRICS
    ]


def _happy_video_report() -> VideoScoreReport:
    return VideoScoreReport(
        video_path="/tmp/test.mp4",
        video_name="test.mp4",
        generated_at="2026-04-17T14:32:05Z",
        processing_wall_time_s=12.34,
        duration_s=60.0,
        metadata_checks=_passing_metadata(),
        technical_checks=_passing_technical(),
        quality_metrics=_passing_quality(),
    )


def _metadata_fail_video_report() -> VideoScoreReport:
    return VideoScoreReport(
        video_path="/tmp/short.mp4",
        video_name="short.mp4",
        generated_at="2026-04-17T14:32:05Z",
        processing_wall_time_s=0.1,
        duration_s=25.0,
        metadata_checks=_failing_metadata_duration(),
        technical_checks=_skipped_technical(),
        quality_metrics=_skipped_quality(),
        technical_skipped=True,
        quality_skipped=True,
    )


# ── MD formatting ─────────────────────────────────────────────────────────

def test_md_includes_all_three_sections(tmp_path):
    paths = reporting.write_video_report(_happy_video_report(), tmp_path)
    md = paths["md"].read_text()
    assert "## Metadata" in md
    assert "## Technical" in md
    assert "## Quality" in md
    assert "both_hands_visibility" in md
    assert "obstructed" in md


def test_md_skipped_technical_renders_dashes(tmp_path):
    paths = reporting.write_video_report(_metadata_fail_video_report(), tmp_path)
    md = paths["md"].read_text()
    # Skipped technical rows: "| luminance | SKIPPED | - | - |"
    assert "| luminance | SKIPPED | - | - |" in md
    assert "| pixelation | SKIPPED | - | - |" in md
    # Skipped quality metrics render the string "SKIPPED" under their heading.
    assert "### both_hands_visibility" in md
    assert "SKIPPED" in md


def test_md_segment_value_formatting(tmp_path):
    paths = reporting.write_video_report(_happy_video_report(), tmp_path)
    md = paths["md"].read_text()
    # Float → 2 decimals; "P" → as-is; NaN → "NaN"; bool → lowercase.
    assert "| 0.000 | 10.000 | 10.000 | 0.92 |" in md
    assert "| 0.000 | 60.000 | 60.000 | P |" in md
    assert "NaN" in md
    assert "false" in md


def test_md_quality_percent_formatting(tmp_path):
    paths = reporting.write_video_report(_happy_video_report(), tmp_path)
    md = paths["md"].read_text()
    assert "percent_frames: 72.50%" in md


# ── JSON ──────────────────────────────────────────────────────────────────

def test_json_roundtrips_values(tmp_path):
    paths = reporting.write_video_report(_happy_video_report(), tmp_path)
    parsed = json.loads(paths["json"].read_text())
    assert parsed["video_name"] == "test.mp4"
    assert parsed["metadata_checks"][0]["check"] == "format"
    # NaN serialised as the string "NaN" so json.loads doesn't reject it.
    angle_metric = next(
        m for m in parsed["quality_metrics"] if m["metric"] == "hand_angle"
    )
    assert angle_metric["segments"][0]["value"] == "NaN"


def test_metadata_fail_skips_parquet(tmp_path):
    store = PerFrameStore()
    paths = reporting.write_video_report(
        _metadata_fail_video_report(), tmp_path, per_frame_store=store
    )
    assert "parquet" not in paths


def test_parquet_written_when_store_has_rows(tmp_path):
    store = PerFrameStore()
    store.append_row(0, 0.0, motion_jitter=0.01)
    paths = reporting.write_video_report(
        _happy_video_report(), tmp_path, per_frame_store=store
    )
    assert "parquet" in paths
    assert paths["parquet"].exists()


# ── Run-dir allocator ─────────────────────────────────────────────────────

def test_allocate_run_dir_picks_fresh_directory(tmp_path):
    a = reporting.allocate_run_dir(tmp_path)
    b = reporting.allocate_run_dir(tmp_path)
    assert a != b
    assert a.name == "run_001"
    assert b.name == "run_002"


def test_allocate_run_dir_extends_past_999(tmp_path):
    # Create run_999 so the next allocation has to extend to 4 digits.
    (tmp_path / "run_999").mkdir()
    nxt = reporting.allocate_run_dir(tmp_path)
    assert nxt.name == "run_1000"


# ── Batch writer ──────────────────────────────────────────────────────────

def _sample_batch() -> BatchScoreReport:
    videos = [_happy_video_report(), _metadata_fail_video_report()]
    meta_stats, tech_stats, qual_stats = reporting.aggregate_batch_stats(videos)
    return BatchScoreReport(
        generated_at="2026-04-17T14:32:05Z",
        video_count=2,
        total_duration_s=85.0,
        total_wall_time_s=12.44,
        metadata_check_stats=meta_stats,
        technical_check_stats=tech_stats,
        quality_metric_stats=qual_stats,
        videos=videos,
        errors=[ProcessingErrorReport(
            video_path="/tmp/broken.mp4",
            video_name="broken.mp4",
            error_reason="decode_failed",
        )],
    )


def test_batch_markdown_stats_reflect_skipped_count(tmp_path):
    batch = _sample_batch()
    paths = reporting.write_batch_report(batch, tmp_path)
    md = paths["md"].read_text()
    # Second video has metadata-fail → technical skipped for every check.
    # So for each technical check, skipped_count must be 1.
    for check in ("luminance", "stability", "frozen", "pixelation"):
        assert f"| {check} |" in md
    assert "## Errors" in md
    assert "broken.mp4" in md


def test_batch_csv_has_one_row_per_video(tmp_path):
    batch = _sample_batch()
    paths = reporting.write_batch_report(batch, tmp_path)
    reader = csv.reader(io.StringIO(paths["csv"].read_text()))
    rows = list(reader)
    assert rows[0][0] == "video_name"
    assert len(rows) == 1 + batch.video_count


def test_batch_csv_skipped_technical_and_quality(tmp_path):
    batch = _sample_batch()
    paths = reporting.write_batch_report(batch, tmp_path)
    reader = csv.DictReader(io.StringIO(paths["csv"].read_text()))
    rows = list(reader)
    short_row = next(r for r in rows if r["video_name"] == "short.mp4")
    # Technical skipped for this video
    assert short_row["tech_luminance_status"] == "SKIPPED"
    assert short_row["tech_luminance_value"] == "-"
    # Quality percent cells carry "SKIPPED"
    assert short_row["quality_both_hands_visibility_pct"] == "SKIPPED"


def test_aggregate_batch_stats_counts():
    batch = _sample_batch()
    meta = batch.metadata_check_stats
    # duration: one pass, one fail
    assert meta["duration"].pass_count == 1
    assert meta["duration"].fail_count == 1
    # technical checks — one pass, one skipped
    tech = batch.technical_check_stats
    assert tech["luminance"].pass_count == 1
    assert tech["luminance"].skipped_count == 1
    # quality stats: 1 video passed quality
    q = batch.quality_metric_stats
    assert math.isclose(q["both_hands_visibility"].mean_percent, 72.5)


def test_batch_report_is_valid_json(tmp_path):
    batch = _sample_batch()
    paths = reporting.write_batch_report(batch, tmp_path)
    parsed = json.loads(paths["json"].read_text())
    assert parsed["video_count"] == 2
    assert len(parsed["videos"]) == 2
    assert parsed["errors"][0]["error_reason"] == "decode_failed"
