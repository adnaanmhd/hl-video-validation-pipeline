"""Scoring engine: single-video and batch orchestrator.

Runs the three-stage gated flow for a single video in one decode pass:
  metadata checks → technical checks → quality metrics.

On metadata fail: no decode, no parquet; technical + quality rows
rendered as SKIPPED.
On technical fail: decode still runs (single pass always runs all
accumulators per §1), quality is rendered as SKIPPED in JSON/MD but
parquet keeps the raw per-frame columns.

See SCORING_ENGINE_PLAN.md §2 for the runtime flow diagram.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.checks.hand_visibility import (
    HandsAccumulator,
    HandsFinalizeResult,
    HandsThresholds,
)
from bachman_cortex.checks.luminance import (
    LuminanceAccumulator,
    LuminanceFinalizeResult,
    LuminanceThresholds,
)
from bachman_cortex.checks.motion_analysis import (
    FrozenThresholds,
    MotionAnalyzer,
    MotionFinalizeResult,
    StabilityThresholds,
)
from bachman_cortex.checks.participants import (
    ParticipantsAccumulator,
    ParticipantsFinalizeResult,
    ParticipantsThresholds,
)
from bachman_cortex.checks.pixelation import (
    PixelationAccumulator,
    PixelationFinalizeResult,
    PixelationThresholds,
)
from bachman_cortex.checks.video_metadata import run_all_metadata_checks
from bachman_cortex.checks.view_obstruction import (
    ObstructionAccumulator,
    ObstructionFinalizeResult,
    ObstructionThresholds,
)
from bachman_cortex.config import Config
from bachman_cortex.data_types import (
    METADATA_CHECKS,
    MetadataCheckResult,
    ProcessingErrorReport,
    QUALITY_METRICS,
    QUALITY_VALUE_LABELS,
    QualityMetricResult,
    QualitySegment,
    TECHNICAL_CHECKS,
    TechnicalCheckResult,
    VideoScoreReport,
)
from bachman_cortex.per_frame_store import PerFrameStore
from bachman_cortex.segmentation import (
    group_runs,
    merge_short_runs,
    segment_angle_value,
    segment_confidence_value,
    segment_contact_value,
)
from bachman_cortex.utils.frame_extractor import iter_native_frames
from bachman_cortex.utils.metadata_observations import build_observations
from bachman_cortex.utils.video_metadata import get_avg_gop, get_video_metadata


# ── Metadata formatters ────────────────────────────────────────────────────

def _fmt_metadata_checks(
    raw: dict[str, CheckResult],
    min_duration_s: float,
    min_fps: float,
    min_w: int,
    min_h: int,
) -> list[MetadataCheckResult]:
    """Convert the dict of CheckResults into MetadataCheckResult entries."""
    by_name = {
        "format": (raw["meta_format"],
                   "container = MP4"),
        "encoding": (raw["meta_encoding"],
                     "H.264 or HEVC"),
        "resolution": (raw["meta_resolution"],
                       f">= {min_w}x{min_h}"),
        "frame_rate": (raw["meta_frame_rate"],
                       f">= {min_fps:g} FPS"),
        "duration": (raw["meta_duration"],
                     f">= {min_duration_s:g} s"),
        "orientation": (raw["meta_orientation"],
                        "rotation in (0, 90, 270) + landscape"),
    }

    out: list[MetadataCheckResult] = []
    for name in METADATA_CHECKS:
        check, accepted = by_name[name]
        detected = _fmt_metadata_detected(name, check.details or {})
        out.append(MetadataCheckResult(
            check=name,
            status=check.status,
            accepted=accepted,
            detected=detected,
        ))
    return out


def _fmt_metadata_detected(name: str, details: dict) -> str:
    if name == "format":
        return details.get("container_format", "?")
    if name == "encoding":
        return details.get("video_codec", "?")
    if name == "resolution":
        return f"{details.get('displayed_width', '?')}x{details.get('displayed_height', '?')}"
    if name == "frame_rate":
        return f"{details.get('fps', '?')} FPS"
    if name == "duration":
        return f"{details.get('duration_s', '?')} s"
    if name == "orientation":
        return (f"rot={details.get('rotation')}, "
                f"disp={details.get('displayed_width')}x{details.get('displayed_height')}")
    return "?"


# ── Per-metric quality-segment builders ────────────────────────────────────

def _build_segments(
    pass_array: list[bool],
    sample_indices: list[int],
    *,
    native_fps: float,
    duration_s: float,
    merge_threshold_s: float,
    value_fn,
) -> tuple[list[QualitySegment], float]:
    """Group pass/fail per-sample → merge short runs → build QualitySegments.

    Returns (segments, percent_frames_from_unmerged_array).
    """
    total = len(pass_array)
    percent = (100.0 * sum(1 for v in pass_array if v) / total) if total else 0.0

    if total == 0:
        return [], 0.0

    # Sample period (seconds): use the mean spacing; fall back to 1/native_fps.
    if len(sample_indices) >= 2:
        diffs = [
            (sample_indices[i + 1] - sample_indices[i]) / native_fps
            for i in range(len(sample_indices) - 1)
        ]
        sample_period_s = sum(diffs) / len(diffs)
    else:
        sample_period_s = 1.0 / native_fps if native_fps > 0 else 0.0

    def sample_to_second(idx: int) -> float:
        return idx / native_fps if native_fps > 0 else 0.0

    runs = group_runs(pass_array, sample_indices)
    merged = merge_short_runs(
        runs,
        min_duration_s=merge_threshold_s,
        sample_to_second=sample_to_second,
        sample_period_s=sample_period_s,
    )

    segments: list[QualitySegment] = []
    for i, run in enumerate(merged):
        if not run.raw_frame_indices:
            continue
        start_s = sample_to_second(run.raw_frame_indices[0])
        # End of segment = start of next segment, or end-of-video for the last one.
        if i + 1 < len(merged) and merged[i + 1].raw_frame_indices:
            end_s = sample_to_second(merged[i + 1].raw_frame_indices[0])
        else:
            end_s = duration_s
        end_s = max(end_s, start_s + sample_period_s)

        segments.append(QualitySegment(
            start_s=round(start_s, 3),
            end_s=round(end_s, 3),
            duration_s=round(end_s - start_s, 3),
            value=value_fn(run),
            value_label="",   # filled by caller
        ))
    return segments, percent


def _build_quality_metric(
    name: str,
    pass_array: list[bool],
    sample_indices: list[int],
    value_fn,
    *,
    native_fps: float,
    duration_s: float,
    merge_threshold_s: float,
) -> QualityMetricResult:
    label = QUALITY_VALUE_LABELS[name]
    segments, percent = _build_segments(
        pass_array=pass_array,
        sample_indices=sample_indices,
        native_fps=native_fps,
        duration_s=duration_s,
        merge_threshold_s=merge_threshold_s,
        value_fn=value_fn,
    )
    for seg in segments:
        seg.value_label = label
    return QualityMetricResult(
        metric=name,
        percent_frames=round(percent, 2),
        segments=segments,
        skipped=False,
    )


# ── Main engine ────────────────────────────────────────────────────────────

@dataclass
class ScoringEngine:
    """Single-video orchestrator. Load models once, score many videos."""

    config: Config = field(default_factory=Config)
    hand_detector_repo: str | None = None
    scrfd_root: str | None = None
    yolo_model: str | None = None

    # Models — lazy-loaded in __post_init__ if not provided.
    hands_detector: Any = None
    scrfd_detector: Any = None
    yolo_detector: Any = None

    def __post_init__(self) -> None:
        if self.hands_detector is None:
            from bachman_cortex.models.hand_detector import HandObjectDetectorHands23
            kwargs = {}
            if self.hand_detector_repo:
                kwargs["repo_dir"] = self.hand_detector_repo
            self.hands_detector = HandObjectDetectorHands23(**kwargs)
        if self.scrfd_detector is None:
            from bachman_cortex.models.scrfd_detector import SCRFDDetector
            kwargs = {}
            if self.scrfd_root:
                kwargs["root"] = self.scrfd_root
            self.scrfd_detector = SCRFDDetector(**kwargs)
        if self.yolo_detector is None:
            from bachman_cortex.models.yolo_detector import YOLODetector
            model_path = self.yolo_model or "yolo11m.pt"
            self.yolo_detector = YOLODetector(model_path=model_path)

        # Warmup all three models on a dummy frame so the first real
        # video doesn't pay the autotune cost.
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            self.hands_detector.detect(dummy)
        except Exception:
            pass
        try:
            self.scrfd_detector.detect(dummy)
        except Exception:
            pass
        try:
            self.yolo_detector.detect(dummy)
        except Exception:
            pass

    # ── Public: score one video ────────────────────────────────────────

    def score_video(
        self,
        video_path: str | Path,
        per_frame_store: PerFrameStore | None = None,
    ) -> tuple[VideoScoreReport, PerFrameStore | None]:
        """Score one video. Returns (report, store-or-None-if-metadata-failed)."""
        cfg = self.config
        video_path = Path(video_path)
        t_start = time.perf_counter()
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        meta = get_video_metadata(video_path)
        duration_s = float(meta["duration_s"])

        # Observations are populated for every video (including metadata
        # failures) — they come from the same ffprobe call plus one cheap
        # packet-level GOP scan.
        avg_gop = get_avg_gop(video_path)
        observations = build_observations(meta, avg_gop)

        raw_meta = run_all_metadata_checks(meta)
        metadata_checks = _fmt_metadata_checks(
            raw_meta,
            min_duration_s=cfg.metadata.min_duration_s,
            min_fps=cfg.metadata.min_fps,
            min_w=cfg.metadata.min_width,
            min_h=cfg.metadata.min_height,
        )
        metadata_failed = any(c.status == "fail" for c in metadata_checks)

        if metadata_failed:
            return VideoScoreReport(
                video_path=str(video_path),
                video_name=video_path.name,
                generated_at=generated_at,
                processing_wall_time_s=round(time.perf_counter() - t_start, 3),
                duration_s=duration_s,
                metadata_checks=metadata_checks,
                metadata_observations=observations,
                technical_checks=[
                    TechnicalCheckResult(check=c, status="skipped",
                                         accepted="-", detected="-", skipped=True)
                    for c in TECHNICAL_CHECKS
                ],
                quality_metrics=[
                    QualityMetricResult(metric=m, percent_frames=0.0,
                                        segments=[], skipped=True)
                    for m in QUALITY_METRICS
                ],
                technical_skipped=True,
                quality_skipped=True,
            ), None

        # ── Single decode pass ─────────────────────────────────────────
        info, gen = iter_native_frames(video_path)
        native_fps = info.native_fps
        total_frames = info.total_frames

        store = per_frame_store if per_frame_store is not None else PerFrameStore()
        # Holders for quality signals populated during decode.
        # Kept lean — accumulators own the canonical state.
        motion = MotionAnalyzer(
            native_fps=native_fps,
            total_frames=total_frames,
            target_fps=cfg.cadences.motion_fps,
        )
        lum = LuminanceAccumulator(
            thresholds=LuminanceThresholds(
                good_frame_ratio=cfg.technical.luminance.good_frame_ratio,
                dead_black_max=cfg.technical.luminance.dead_black_max,
                too_dark_max=cfg.technical.luminance.too_dark_max,
                blown_out_min=cfg.technical.luminance.blown_out_min,
                flicker_window=cfg.technical.luminance.flicker_window,
                flicker_stddev_threshold=cfg.technical.luminance.flicker_stddev_threshold,
            ),
        )
        pix = PixelationAccumulator(
            thresholds=PixelationThresholds(
                good_frame_ratio=cfg.technical.pixelation.good_frame_ratio,
                max_blockiness_ratio=cfg.technical.pixelation.max_blockiness_ratio,
                block_size=cfg.technical.pixelation.block_size,
            ),
        )
        obs = ObstructionAccumulator()
        hands_acc = HandsAccumulator(
            thresholds=HandsThresholds(
                conf_threshold=cfg.quality.hands.hands23_conf,
                max_angle_deg=cfg.quality.angle.max_degrees,
            ),
        )
        parts_acc = ParticipantsAccumulator(
            thresholds=ParticipantsThresholds(
                yolo_conf=cfg.quality.participants.yolo_conf,
                scrfd_conf=cfg.quality.participants.scrfd_conf,
                extra_hand_conf=cfg.quality.participants.extra_hand_conf,
                min_bbox_height_frac=cfg.quality.participants.min_bbox_height_frac,
            ),
        )

        # Cadence skips — frame i ticks cadence C iff i % skip_C == 0.
        def _skip(target_fps: float) -> int:
            return max(1, round(native_fps / target_fps)) if target_fps > 0 else 1_000_000_000

        motion_skip = _skip(cfg.cadences.motion_fps)
        lum_skip = _skip(cfg.cadences.luminance_fps)
        pix_skip = _skip(cfg.cadences.pixelation_fps)
        obs_skip = pix_skip             # obstruction lives on the same cadence family
        qual_skip = _skip(cfg.cadences.quality_fps)

        for frame_idx, frame in gen:
            if frame_idx % motion_skip == 0:
                # MotionAnalyzer does its own 0.5x downscale on gray.
                motion.process_frame(frame, frame_idx)
            if frame_idx % lum_skip == 0:
                # Downscale 720p → 360p for luminance (in-place 0.5x).
                small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_AREA)
                lum.process_frame(small, frame_idx)
            if frame_idx % pix_skip == 0:
                pix.process_frame(frame, frame_idx)
            if frame_idx % obs_skip == 0:
                obs.process_frame(frame, frame_idx)
            if frame_idx % qual_skip == 0:
                h, w = frame.shape[:2]
                hands, _ = self.hands_detector.detect(frame)
                faces = self.scrfd_detector.detect(frame)
                yolo_det = self.yolo_detector.detect(frame)
                persons = [d for d in yolo_det
                           if d.class_id == 0 and d.confidence >= cfg.quality.participants.yolo_conf]
                hands_acc.process_frame(hands, frame_idx, frame_wh=(w, h))
                parts_acc.process_frame(
                    yolo_persons=persons,
                    scrfd_faces=faces,
                    hands=hands,
                    frame_idx=frame_idx,
                    frame_wh=(w, h),
                )

        # ── Finalize ──────────────────────────────────────────────────
        motion_res = motion.finalize_whole_video(
            stability=StabilityThresholds(
                shaky_score_threshold=cfg.technical.stability.shaky_score_threshold,
                trans_threshold=cfg.technical.stability.trans_threshold,
                jump_threshold=cfg.technical.stability.jump_threshold,
                rot_threshold=cfg.technical.stability.rot_threshold,
                variance_threshold=cfg.technical.stability.variance_threshold,
                w_trans=cfg.technical.stability.w_trans,
                w_var=cfg.technical.stability.w_var,
                w_rot=cfg.technical.stability.w_rot,
                w_jump=cfg.technical.stability.w_jump,
                highpass_window_sec=cfg.technical.stability.highpass_window_sec,
            ),
            frozen=FrozenThresholds(
                max_consecutive=cfg.technical.frozen.max_consecutive,
                trans_threshold=cfg.technical.frozen.trans_threshold,
                rot_threshold=cfg.technical.frozen.rot_threshold,
            ),
        )
        lum_res = lum.finalize()
        pix_res = pix.finalize()
        obs_res = obs.finalize()
        hands_res = hands_acc.finalize()
        parts_res = parts_acc.finalize()

        technical_checks = [
            TechnicalCheckResult(
                check="luminance",
                status="pass" if lum_res.pass_fail else "fail",
                accepted=f">= {cfg.technical.luminance.good_frame_ratio:.0%} good frames",
                detected=lum_res.detected,
            ),
            TechnicalCheckResult(
                check="stability",
                status="pass" if motion_res.stability_pass else "fail",
                accepted=f"mean jitter <= {cfg.technical.stability.shaky_score_threshold}",
                detected=motion_res.stability_detected,
            ),
            TechnicalCheckResult(
                check="frozen",
                status="pass" if motion_res.frozen_pass else "fail",
                accepted=f"no run > {cfg.technical.frozen.max_consecutive}f",
                detected=motion_res.frozen_detected,
            ),
            TechnicalCheckResult(
                check="pixelation",
                status="pass" if pix_res.pass_fail else "fail",
                accepted=f">= {cfg.technical.pixelation.good_frame_ratio:.0%} frames w/ blockiness <= {cfg.technical.pixelation.max_blockiness_ratio}",
                detected=pix_res.detected,
            ),
        ]
        technical_failed = any(t.status == "fail" for t in technical_checks)

        # Write the per-frame store (raw truth — always, even on tech fail).
        self._fill_store(
            store,
            total_frames=total_frames,
            native_fps=native_fps,
            motion_res=motion_res,
            lum_res=lum_res,
            pix_res=pix_res,
            obs_res=obs_res,
            hands_res=hands_res,
            parts_res=parts_res,
        )

        # Quality metrics — skipped view when technical failed.
        if technical_failed:
            quality_metrics = [
                QualityMetricResult(metric=m, percent_frames=0.0,
                                    segments=[], skipped=True)
                for m in QUALITY_METRICS
            ]
            quality_skipped = True
        else:
            quality_metrics = self._build_quality_metrics(
                hands_res=hands_res,
                parts_res=parts_res,
                obs_res=obs_res,
                native_fps=native_fps,
                duration_s=duration_s,
                merge_threshold_s=cfg.segmentation.merge_threshold_s,
            )
            quality_skipped = False

        return VideoScoreReport(
            video_path=str(video_path),
            video_name=video_path.name,
            generated_at=generated_at,
            processing_wall_time_s=round(time.perf_counter() - t_start, 3),
            duration_s=duration_s,
            metadata_checks=metadata_checks,
            metadata_observations=observations,
            technical_checks=technical_checks,
            quality_metrics=quality_metrics,
            technical_skipped=False,
            quality_skipped=quality_skipped,
        ), store

    # ── Parquet fill ───────────────────────────────────────────────────

    def _fill_store(
        self,
        store: PerFrameStore,
        *,
        total_frames: int,
        native_fps: float,
        motion_res: MotionFinalizeResult,
        lum_res: LuminanceFinalizeResult,
        pix_res: PixelationFinalizeResult,
        obs_res: ObstructionFinalizeResult,
        hands_res: HandsFinalizeResult,
        parts_res: ParticipantsFinalizeResult,
    ) -> None:
        """Write one parquet row per native frame, null-filling off-cadence cells."""
        lum_by_frame: dict[int, tuple[int, bool]] = {
            lum_res.sample_indices[i]: (lum_res.class_array[i],
                                        lum_res.flicker_array[i])
            for i in range(len(lum_res.sample_indices))
        }
        pix_by_frame: dict[int, float] = {
            pix_res.sample_indices[i]: pix_res.ratio_array[i]
            for i in range(len(pix_res.sample_indices))
        }
        obs_by_frame: dict[int, bool] = {
            obs_res.sample_indices[i]: obs_res.obstructed_array[i]
            for i in range(len(obs_res.sample_indices))
        }
        hands_by_frame: dict[int, dict[str, Any]] = {}
        for i, fi in enumerate(hands_res.sample_indices):
            hands_by_frame[fi] = {
                "both_hands_pass": hands_res.both_hands_pass[i],
                "both_hands_conf": hands_res.both_hands_conf[i],
                "single_hand_pass": hands_res.single_hand_pass[i],
                "single_hand_conf": hands_res.single_hand_conf[i],
                "hand_obj_pass": hands_res.hand_obj_pass[i],
                "hand_obj_contact": hands_res.hand_obj_contact[i],
                "hand_angle_pass": hands_res.hand_angle_pass[i],
                "hand_angle_mean_deg": hands_res.hand_angle_mean_deg[i],
            }
        parts_by_frame: dict[int, dict[str, Any]] = {}
        for i, fi in enumerate(parts_res.sample_indices):
            parts_by_frame[fi] = {
                "participant_pass": parts_res.participant_pass[i],
                "participant_conf": parts_res.participant_conf[i],
                "participant_source": parts_res.participant_source[i],
                "extra_hands_count": parts_res.extra_hands_count[i],
            }

        per_frame_jitter = motion_res.per_frame_jitter
        per_frame_frozen = motion_res.per_frame_frozen

        for i in range(total_frames):
            row: dict[str, Any] = {}
            if i < len(per_frame_jitter) and per_frame_jitter[i] is not None:
                row["motion_jitter"] = float(per_frame_jitter[i])
            if i < len(per_frame_frozen):
                row["frozen_state"] = bool(per_frame_frozen[i])

            if i in lum_by_frame:
                cls, flk = lum_by_frame[i]
                row["luminance_class"] = int(cls)
                row["luminance_flicker"] = bool(flk)

            if i in pix_by_frame:
                row["pixelation_ratio"] = float(pix_by_frame[i])

            if i in obs_by_frame:
                row["obstructed"] = bool(obs_by_frame[i])

            if i in hands_by_frame:
                h = hands_by_frame[i]
                row["both_hands_pass"] = bool(h["both_hands_pass"])
                row["both_hands_conf"] = float(h["both_hands_conf"])
                row["single_hand_pass"] = bool(h["single_hand_pass"])
                row["single_hand_conf"] = float(h["single_hand_conf"])
                row["hand_obj_pass"] = bool(h["hand_obj_pass"])
                row["hand_obj_contact"] = h["hand_obj_contact"]
                row["hand_angle_pass"] = bool(h["hand_angle_pass"])
                row["hand_angle_mean_deg"] = float(h["hand_angle_mean_deg"]) \
                    if not math.isnan(h["hand_angle_mean_deg"]) else float("nan")

            if i in parts_by_frame:
                p = parts_by_frame[i]
                row["participant_pass"] = bool(p["participant_pass"])
                row["participant_conf"] = float(p["participant_conf"])
                row["participant_source"] = p["participant_source"]
                row["extra_hands_count"] = int(p["extra_hands_count"])

            store.append_row(
                frame_idx=i,
                timestamp_s=round(i / native_fps, 6) if native_fps > 0 else 0.0,
                **row,
            )

    # ── Quality metric builders ────────────────────────────────────────

    def _build_quality_metrics(
        self,
        *,
        hands_res: HandsFinalizeResult,
        parts_res: ParticipantsFinalizeResult,
        obs_res: ObstructionFinalizeResult,
        native_fps: float,
        duration_s: float,
        merge_threshold_s: float,
    ) -> list[QualityMetricResult]:
        out: list[QualityMetricResult] = []

        # both_hands_visibility
        both_conf_by_frame = {
            hands_res.sample_indices[i]: float(hands_res.both_hands_conf[i])
            for i in range(len(hands_res.sample_indices))
        }
        out.append(_build_quality_metric(
            "both_hands_visibility",
            hands_res.both_hands_pass,
            hands_res.sample_indices,
            value_fn=lambda run: round(
                segment_confidence_value(run, both_conf_by_frame), 4
            ),
            native_fps=native_fps,
            duration_s=duration_s,
            merge_threshold_s=merge_threshold_s,
        ))

        # single_hand_visibility
        single_conf_by_frame = {
            hands_res.sample_indices[i]: float(hands_res.single_hand_conf[i])
            for i in range(len(hands_res.sample_indices))
        }
        out.append(_build_quality_metric(
            "single_hand_visibility",
            hands_res.single_hand_pass,
            hands_res.sample_indices,
            value_fn=lambda run: round(
                segment_confidence_value(run, single_conf_by_frame), 4
            ),
            native_fps=native_fps,
            duration_s=duration_s,
            merge_threshold_s=merge_threshold_s,
        ))

        # hand_obj_interaction
        contact_by_frame = {
            hands_res.sample_indices[i]: hands_res.hand_obj_contact[i]
            for i in range(len(hands_res.sample_indices))
        }
        out.append(_build_quality_metric(
            "hand_obj_interaction",
            hands_res.hand_obj_pass,
            hands_res.sample_indices,
            value_fn=lambda run: segment_contact_value(run, contact_by_frame),
            native_fps=native_fps,
            duration_s=duration_s,
            merge_threshold_s=merge_threshold_s,
        ))

        # hand_angle
        angle_by_frame = {
            hands_res.sample_indices[i]: hands_res.hand_angle_mean_deg[i]
            for i in range(len(hands_res.sample_indices))
        }
        out.append(_build_quality_metric(
            "hand_angle",
            hands_res.hand_angle_pass,
            hands_res.sample_indices,
            value_fn=lambda run: (
                round(segment_angle_value(run, angle_by_frame), 2)
                if not math.isnan(segment_angle_value(run, angle_by_frame))
                else float("nan")
            ),
            native_fps=native_fps,
            duration_s=duration_s,
            merge_threshold_s=merge_threshold_s,
        ))

        # participants
        parts_conf_by_frame = {
            parts_res.sample_indices[i]: float(parts_res.participant_conf[i])
            for i in range(len(parts_res.sample_indices))
        }
        out.append(_build_quality_metric(
            "participants",
            parts_res.participant_pass,
            parts_res.sample_indices,
            value_fn=lambda run: round(
                segment_confidence_value(run, parts_conf_by_frame), 4
            ),
            native_fps=native_fps,
            duration_s=duration_s,
            merge_threshold_s=merge_threshold_s,
        ))

        # obstructed — value is the segment's state (bool).
        out.append(_build_quality_metric(
            "obstructed",
            obs_res.obstructed_array,
            obs_res.sample_indices,
            value_fn=lambda run: run.state,
            native_fps=native_fps,
            duration_s=duration_s,
            merge_threshold_s=merge_threshold_s,
        ))

        return out
