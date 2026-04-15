"""Validation & Processing Pipeline.

4-phase pipeline that classifies time ranges in egocentric videos:
  Phase 0: Metadata gate (short-circuits on failure)
  Phase 1: Segment filtering (face, participants) — no early stopping
  Phase 2: Segment validation (luminance, motion, hand + face ML checks)
  Phase 3: Yield calculation

Results include per-frame labels, segment timestamps, usable/rejected
segments, and yield metrics. No clip files are produced.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.data_types import (
    TimeSegment,
    FrameLabel,
    CheckFrameResults,
    CheckableSegment,
    SegmentValidationResult,
    VideoProcessingResult,
)
from bachman_cortex.utils.video_metadata import get_video_metadata
from bachman_cortex.checks.video_metadata import run_all_metadata_checks
from bachman_cortex.checks.luminance_blur import check_luminance_blur, LuminanceBlurConfig
from bachman_cortex.checks.motion_analysis import (
    MotionAnalyzer,
    check_motion_combined_from_analyzer,
)
from bachman_cortex.checks.hand_visibility import check_hand_visibility
from bachman_cortex.checks.hand_object_interaction import check_hand_object_interaction
from bachman_cortex.checks.view_obstruction import check_view_obstruction
from bachman_cortex.checks.pov_hand_angle import check_pov_hand_angle
from bachman_cortex.checks.face_presence import check_face_presence
from bachman_cortex.utils.frame_extractor import extract_frames
from bachman_cortex.utils.transcode import maybe_transcode_hevc_to_h264
from bachman_cortex.utils.segment_ops import (
    per_frame_to_bad_segments,
    filter_short_bad_segments,
    merge_bad_segments,
    compute_good_segments,
    filter_checkable_segments,
)
from bachman_cortex.utils.early_stop import (
    eval_face_presence,
    eval_participants,
)

# Batch size for YOLO inference in the ML loop
_BATCH_SIZE = 16


@dataclass
class PipelineConfig:
    """Configuration for the validation & processing pipeline."""
    # Frame sampling
    sampling_fps: float = 1.0
    max_frames: int | None = None

    # Model paths
    scrfd_root: str = "bachman_cortex/models/weights/insightface"
    yolo_model: str = "yolo11s.pt"
    hand_detector_repo: str = "bachman_cortex/models/weights/hands23_detector"

    # ML thresholds
    face_confidence_threshold: float = 0.8
    person_confidence_threshold: float = 0.4
    hand_confidence_threshold: float = 0.7
    hand_pass_rate: float = 0.80  # both-hands fraction threshold
    single_hand_pass_rate: float = 0.90  # fallback: at-least-one-hand fraction
    interaction_pass_rate: float = 0.60
    participant_pass_rate: float = 0.90
    participant_face_threshold: float = 0.5
    participant_min_height_ratio: float = 0.15
    obstruction_max_ratio: float = 0.10
    angle_threshold: float = 40.0
    angle_pass_rate: float = 0.60
    diagonal_fov_degrees: float = 90.0
    hand_frame_margin: int = 0

    # Opt-in preprocessing: lossless HEVC → H.264 conversion before Phase 0.
    transcode_hevc: bool = False

    # Brightness stability threshold
    max_brightness_std: float = 60.0

    # Camera stability (single-pass LK optical flow at 0.5x)
    shaky_score_threshold: float = 0.50
    fast_scale: float = 0.5
    stability_trans_threshold: float = 8.0
    stability_jump_threshold: float = 30.0
    stability_rot_threshold: float = 0.3
    stability_variance_threshold: float = 6.0
    stability_w_trans: float = 0.35
    stability_w_var: float = 0.25
    stability_w_rot: float = 0.20
    stability_w_jump: float = 0.20
    stability_max_corners: int = 300
    stability_lk_win_size: tuple[int, int] = (21, 21)
    stability_lk_max_level: int = 3
    stability_target_fps: float = 30.0

    # Frozen segments
    frozen_max_consecutive: int = 30
    frozen_trans_threshold: float = 0.1
    frozen_rot_threshold: float = 0.001

    # Luminance & blur
    luminance_blur_min_good_ratio: float = 0.70

    # Hands23 input resolution cap
    hands23_max_resolution: int | None = 720

    # Phase 1 frame downscale (long-edge). None disables downscaling.
    # Feeds YOLO / SCRFD / Hands23 with ~2.3x fewer pixels at 720p.
    phase1_long_edge: int | None = 720

    # Thread-pool workers for SCRFD face detection within a batch.
    # InsightFace's FaceAnalysis.get() isn't batch-capable; overlap across
    # frames via a small thread pool.
    scrfd_threads: int = 4

    # Segment filtering
    min_checkable_segment_sec: float = 60.0
    min_bad_segment_sec: float = 2.0


class ValidationProcessingPipeline:
    """4-phase pipeline: metadata gate, segment filtering, segment validation, yield."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._models_loaded = False

    def load_models(self):
        """Load all ML models. Call once at worker startup."""
        print("Loading ML models...")
        t0 = time.perf_counter()

        from bachman_cortex.models.scrfd_detector import SCRFDDetector
        self.face_detector = SCRFDDetector(root=self.config.scrfd_root)
        print("  SCRFD loaded")

        from bachman_cortex.models.yolo_detector import YOLODetector
        self.yolo_detector = YOLODetector(model_path=self.config.yolo_model)
        print(f"  YOLO loaded ({self.config.yolo_model})")

        from bachman_cortex.models.hand_detector import HandObjectDetectorHands23
        self.hand_detector = HandObjectDetectorHands23(
            repo_dir=self.config.hand_detector_repo,
            max_resolution=self.config.hands23_max_resolution,
        )
        print("  Hands23 loaded")

        # Pre-warm all models with a synthetic forward pass so the first
        # real inference doesn't pay ORT autotune / lazy CUDA init cost.
        self._warmup_models()

        elapsed = time.perf_counter() - t0
        print(f"All models loaded in {elapsed:.1f}s")
        self._models_loaded = True

    def _warmup_models(self) -> None:
        """Run one synthetic-frame forward pass through each loaded model."""
        warm_h, warm_w = 720, 1280
        dummy = np.zeros((warm_h, warm_w, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        try:
            self.face_detector.detect(dummy)
        except Exception as e:
            print(f"  warmup SCRFD failed: {e}")
        try:
            self.yolo_detector.detect_batch([dummy])
        except Exception as e:
            print(f"  warmup YOLO failed: {e}")
        try:
            self.hand_detector.detect(dummy)
        except Exception as e:
            print(f"  warmup Hands23 failed: {e}")
        print(f"  models warmed in {time.perf_counter() - t0:.1f}s")

    # ── Main entry point ─────────────────────────────────────────────────

    def process_video(
        self, video_path: str | Path, output_dir: str | Path,
    ) -> VideoProcessingResult:
        """Run the full 4-phase pipeline on a video.

        Args:
            video_path: Path to input video.
            output_dir: Per-video output directory for report + JSON.

        Returns:
            VideoProcessingResult with all phases' data.
        """
        video_path = str(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem
        t_start = time.perf_counter()

        # ── Pre-Phase 0: Optional HEVC → H.264 lossless transcode ────
        transcode_info: dict | None = None
        active_path = video_path
        if self.config.transcode_hevc:
            preprocessed_dir = output_dir.parent / "_preprocessed"
            active_path, transcode_info = maybe_transcode_hevc_to_h264(
                video_path, preprocessed_dir,
            )

        # ── Phase 0: Metadata ────────────────────────────────────────
        print("\n--- Phase 0: Metadata checks ---")
        meta_passed, metadata, meta_results = self._phase0_metadata(active_path)
        original_duration = metadata.get("duration_s", 0.0)

        if not meta_passed:
            print("  Metadata FAILED — rejecting entire video")
            return VideoProcessingResult(
                video_path=video_path,
                video_name=video_name,
                original_duration_sec=original_duration,
                metadata=metadata,
                metadata_passed=False,
                metadata_results=meta_results,
                phase1_check_frame_results=[],
                phase1_bad_segments=[],
                phase1_discarded_segments=[],
                prefiltered_segments=[],
                segment_results=[],
                usable_segments=[],
                rejected_segments=[],
                usable_duration_sec=0.0,
                unusable_duration_sec=original_duration,
                yield_ratio=0.0,
                processing_time_sec=round(time.perf_counter() - t_start, 2),
                transcode_info=transcode_info,
            )

        # ── Extract frames for Phase 1 (tee native stream to motion) ─
        t0 = time.perf_counter()
        probe_cap = cv2.VideoCapture(active_path)
        probe_fps = probe_cap.get(cv2.CAP_PROP_FPS) or 30.0
        probe_total = int(probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        probe_cap.release()

        motion_analyzer = MotionAnalyzer(
            native_fps=probe_fps,
            total_frames=probe_total,
            fast_scale=self.config.fast_scale,
            target_fps=self.config.stability_target_fps,
            max_corners=self.config.stability_max_corners,
            lk_win_size=self.config.stability_lk_win_size,
            lk_max_level=self.config.stability_lk_max_level,
        )

        frames, frame_meta = extract_frames(
            active_path,
            fps=self.config.sampling_fps,
            max_frames=self.config.max_frames,
            motion_analyzer=motion_analyzer,
        )
        frame_h, frame_w = frame_meta["height"], frame_meta["width"]
        print(f"Extracted {len(frames)} frames ({frame_meta['duration_s']}s video) "
              f"in {time.perf_counter() - t0:.2f}s "
              f"[motion: {motion_analyzer.frames_processed} samples]")

        # ── Phase 1: Segment filtering ───────────────────────────────
        print("\n--- Phase 1: Segment filtering (face, participants) ---")
        (
            phase1_check_results,
            bad_segments,
            discarded_segments,
            prefiltered_segments,
            phase1_cache,
        ) = self._phase1_segment_filter(active_path, frames, frame_meta)
        phase1_cache['motion_analyzer'] = motion_analyzer

        if not prefiltered_segments:
            print("  No checkable segments remain — rejecting entire video")
            return VideoProcessingResult(
                video_path=video_path,
                video_name=video_name,
                original_duration_sec=original_duration,
                metadata=metadata,
                metadata_passed=True,
                metadata_results=meta_results,
                phase1_check_frame_results=phase1_check_results,
                phase1_bad_segments=bad_segments,
                phase1_discarded_segments=discarded_segments,
                prefiltered_segments=[],
                segment_results=[],
                usable_segments=[],
                rejected_segments=[],
                usable_duration_sec=0.0,
                unusable_duration_sec=original_duration,
                yield_ratio=0.0,
                processing_time_sec=round(time.perf_counter() - t_start, 2),
                transcode_info=transcode_info,
            )

        print(f"  {len(prefiltered_segments)} pre-filtered segment(s):")
        for seg in prefiltered_segments:
            print(f"    Segment {seg.segment_idx}: {seg.start_sec:.1f}s - {seg.end_sec:.1f}s "
                  f"({seg.duration:.1f}s)")

        # ── Phase 2: Segment validation ──────────────────────────────
        print("\n--- Phase 2: Segment validation ---")
        segment_results = self._phase2_segment_validation(
            active_path, prefiltered_segments, frame_meta, phase1_cache,
        )
        del phase1_cache, frames  # free Phase 1 frames

        usable_segments = [sr.segment for sr in segment_results if sr.passed]
        rejected_segments = [sr.segment for sr in segment_results if not sr.passed]

        for sr in segment_results:
            status = "PASS" if sr.passed else "FAIL"
            print(f"  Segment {sr.segment.segment_idx}: {status}")
            if sr.failing_checks:
                print(f"    Failing: {', '.join(sr.failing_checks)}")

        # ── Phase 3: Yield calculation ───────────────────────────────
        print("\n--- Phase 3: Yield calculation ---")
        usable_dur, unusable_dur, yield_ratio = self._phase3_yield(
            original_duration, segment_results,
        )
        print(f"  Usable:   {usable_dur:.1f}s")
        print(f"  Unusable: {unusable_dur:.1f}s")
        print(f"  Yield:    {yield_ratio:.1%}")

        processing_time = round(time.perf_counter() - t_start, 2)
        print(f"\nTotal processing time: {processing_time:.1f}s")

        return VideoProcessingResult(
            video_path=video_path,
            video_name=video_name,
            original_duration_sec=original_duration,
            metadata=metadata,
            metadata_passed=True,
            metadata_results=meta_results,
            phase1_check_frame_results=phase1_check_results,
            phase1_bad_segments=bad_segments,
            phase1_discarded_segments=discarded_segments,
            prefiltered_segments=prefiltered_segments,
            segment_results=segment_results,
            usable_segments=usable_segments,
            rejected_segments=rejected_segments,
            usable_duration_sec=usable_dur,
            unusable_duration_sec=unusable_dur,
            yield_ratio=yield_ratio,
            processing_time_sec=processing_time,
            transcode_info=transcode_info,
        )

    # ── Phase 0 ──────────────────────────────────────────────────────

    def _phase0_metadata(
        self, video_path: str,
    ) -> tuple[bool, dict, dict[str, CheckResult]]:
        """Run metadata checks. Returns (passed, metadata, results)."""
        metadata = get_video_metadata(video_path)
        results = run_all_metadata_checks(metadata)

        for name, r in results.items():
            print(f"  {name:<25} {r.status:>4}")

        passed = all(r.status != "fail" for r in results.values())
        return passed, metadata, results

    # ── Phase 1 ──────────────────────────────────────────────────────

    def _phase1_segment_filter(
        self,
        video_path: str,
        frames: list[np.ndarray],
        frame_meta: dict,
    ) -> tuple[
        list[CheckFrameResults],
        list[TimeSegment],
        list[TimeSegment],
        list[CheckableSegment],
        dict,
    ]:
        """Run Phase 1 checks on ALL frames, compute segments.

        Returns:
            (check_frame_results, unified_bad_segments,
             discarded_segments, prefiltered_segments, phase1_cache)
        """
        if not self._models_loaded:
            self.load_models()

        total_frames = len(frames)
        frame_timestamps = [i / self.config.sampling_fps for i in range(total_frames)]

        return self._phase1_run_inference(
            video_path, frames, frame_meta, total_frames, frame_timestamps,
        )

    def _phase1_run_inference(
        self,
        video_path: str,
        frames: list[np.ndarray],
        frame_meta: dict,
        total_frames: int,
        frame_timestamps: list[float],
    ) -> tuple[
        list[CheckFrameResults],
        list[TimeSegment],
        list[TimeSegment],
        list[CheckableSegment],
        dict,
    ]:
        native_w, native_h = frame_meta["width"], frame_meta["height"]
        original_duration = frame_meta["duration_s"]

        # Downscale frames once to a shared 720p long-edge target. Feeds
        # YOLO, SCRFD, and Hands23 with ~2.3x fewer pixels than native 1080p
        # and drops the Phase 1 raw-frame cache size proportionally
        # (P1.1 + P1.2).
        target_long = self.config.phase1_long_edge
        if target_long and max(native_h, native_w) > target_long:
            scale = target_long / max(native_h, native_w)
            new_w = int(round(native_w * scale))
            new_h = int(round(native_h * scale))
            frames = [
                cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA)
                for f in frames
            ]
            frame_w, frame_h = new_w, new_h
        else:
            frame_w, frame_h = native_w, native_h

        per_frame_faces = []
        per_frame_yolo_persons = []
        per_frame_hands = []

        t0 = time.perf_counter()

        scrfd_pool = ThreadPoolExecutor(max_workers=self.config.scrfd_threads)
        try:
            for batch_start in range(0, total_frames, _BATCH_SIZE):
                batch_end = min(batch_start + _BATCH_SIZE, total_frames)
                batch_frames = frames[batch_start:batch_end]

                batch_yolo = self.yolo_detector.detect_batch(batch_frames)
                face_futures = [
                    scrfd_pool.submit(self.face_detector.detect, bf)
                    for bf in batch_frames
                ]

                for j, frame in enumerate(batch_frames):
                    faces = face_futures[j].result()
                    per_frame_faces.append(faces)

                    yolo_dets = batch_yolo[j]
                    persons = self.yolo_detector.get_persons(yolo_dets)
                    per_frame_yolo_persons.append(persons)

                    hands, _ = self.hand_detector.detect(frame)
                    per_frame_hands.append(hands)

                    i = batch_start + j
                    if (i + 1) % 50 == 0 or i + 1 == total_frames:
                        print(f"  Phase 1 inference: {i + 1}/{total_frames} frames")
        finally:
            scrfd_pool.shutdown(wait=False)

        # ── Step 2: Evaluate face/participant per frame ──────────────
        face_labels: list[FrameLabel] = []
        participant_labels: list[FrameLabel] = []

        for i in range(total_frames):
            ts = frame_timestamps[i]

            face_passed = eval_face_presence(
                per_frame_faces[i], self.config.face_confidence_threshold,
            )
            max_face_conf = max(
                (f.confidence for f in per_frame_faces[i]), default=0.0,
            )
            face_labels.append(FrameLabel(
                frame_idx=i,
                timestamp_sec=ts,
                passed=face_passed,
                confidence=max_face_conf,
                labels=[f"face:{f.confidence:.2f}" for f in per_frame_faces[i]
                        if f.confidence >= self.config.face_confidence_threshold],
            ))

            participant_passed = eval_participants(
                per_frame_yolo_persons[i],
                per_frame_faces[i],
                per_frame_hands[i],
                frame_h, frame_w,
                self.config.person_confidence_threshold,
                self.config.participant_face_threshold,
                self.config.participant_min_height_ratio,
            )
            participant_labels.append(FrameLabel(
                frame_idx=i,
                timestamp_sec=ts,
                passed=participant_passed,
                confidence=1.0 if participant_passed else 0.0,
                labels=["other_person_detected"] if not participant_passed else None,
            ))

        t_inference = time.perf_counter() - t0
        print(f"  Phase 1 inference complete in {t_inference:.1f}s")

        # ── Step 3: Build per-check bad segments ─────────────────────
        frame_interval = 1.0 / self.config.sampling_fps
        min_bad = self.config.min_bad_segment_sec
        face_bad_raw = per_frame_to_bad_segments(face_labels, frame_interval)
        participant_bad_raw = per_frame_to_bad_segments(participant_labels, frame_interval)
        face_bad = filter_short_bad_segments(face_bad_raw, min_bad)
        participant_bad = filter_short_bad_segments(participant_bad_raw, min_bad)

        check_frame_results = [
            CheckFrameResults("ml_face_presence", face_labels, face_bad),
            CheckFrameResults("ml_participants", participant_labels, participant_bad),
        ]
        segment_lists = [face_bad, participant_bad]

        # Report per-check bad segments
        for cfr in check_frame_results:
            bad_dur = sum(s.duration for s in cfr.bad_segments)
            print(f"  {cfr.check_name}: {len(cfr.bad_segments)} bad segment(s), "
                  f"{bad_dur:.1f}s total")

        # Merge bad segments across all checks, then filter short merged segments
        unified_bad = merge_bad_segments(segment_lists)
        unified_bad = filter_short_bad_segments(unified_bad, min_bad)
        unified_bad_dur = sum(s.duration for s in unified_bad)
        print(f"  Unified bad segments: {len(unified_bad)}, {unified_bad_dur:.1f}s total")

        # Compute good segments and filter by minimum duration
        good_segments = compute_good_segments(original_duration, unified_bad)
        prefiltered_segments, discarded = filter_checkable_segments(
            good_segments, self.config.min_checkable_segment_sec,
        )

        if discarded:
            discarded_dur = sum(s.duration for s in discarded)
            print(f"  Discarded {len(discarded)} segment(s) < "
                  f"{self.config.min_checkable_segment_sec}s ({discarded_dur:.1f}s)")

        phase1_cache = {
            'frames': frames,
            'frame_dims': (frame_h, frame_w),
            'per_frame_hands': per_frame_hands,
            'per_frame_faces': per_frame_faces,
        }

        return check_frame_results, unified_bad, discarded, prefiltered_segments, phase1_cache

    # ── Phase 2 ──────────────────────────────────────────────────────

    def _phase2_segment_validation(
        self,
        video_path: str,
        segments: list[CheckableSegment],
        frame_meta: dict,
        phase1_cache: dict,
    ) -> list[SegmentValidationResult]:
        """Run remaining checks on each checkable segment sequentially."""
        results = []
        with ThreadPoolExecutor(max_workers=1) as motion_executor:
            for seg in segments:
                result = self._phase2_validate_single_segment(
                    video_path, seg, frame_meta, phase1_cache, motion_executor,
                )
                results.append(result)
        return results

    def _phase2_validate_single_segment(
        self,
        video_path: str,
        segment: CheckableSegment,
        frame_meta: dict,
        phase1_cache: dict,
        motion_executor: ThreadPoolExecutor,
    ) -> SegmentValidationResult:
        """Validate a single checkable segment with all Phase 2 checks.

        Reuses frames, hand, face, and person detections cached from Phase 1
        to avoid redundant inference.
        """
        print(f"\n  Validating segment {segment.segment_idx} "
              f"({segment.start_sec:.1f}s - {segment.end_sec:.1f}s, {segment.duration:.1f}s)")

        check_results: dict[str, CheckResult] = {}

        # Reuse frames and hand detections cached from Phase 1
        start_idx = int(segment.start_sec * self.config.sampling_fps)
        end_idx = int(segment.end_sec * self.config.sampling_fps)
        seg_frames = phase1_cache['frames'][start_idx:end_idx]
        per_frame_hands = phase1_cache['per_frame_hands'][start_idx:end_idx]
        per_frame_faces = phase1_cache['per_frame_faces'][start_idx:end_idx]
        seg_timestamps = [
            i / self.config.sampling_fps for i in range(start_idx, end_idx)
        ]

        # Bboxes in the phase1 cache are in Phase-1 coordinates (720p
        # long-edge by default; see phase1_long_edge). Use those dims so
        # the check functions' dim-relative thresholds match.
        seg_h, seg_w = phase1_cache.get(
            'frame_dims', (frame_meta["height"], frame_meta["width"])
        )

        print(f"    {len(seg_frames)} frames for segment (cached from Phase 1)")

        if not seg_frames:
            return SegmentValidationResult(
                segment=segment, passed=False,
                check_results={},
                failing_checks=["no_frames_extracted"],
            )

        # Reuse the per-video MotionAnalyzer populated during single-pass
        # decode (no per-segment re-open). The derivation is CPU-cheap; run it
        # inline rather than on the background executor.
        analyzer: MotionAnalyzer = phase1_cache['motion_analyzer']

        def _run_motion():
            return check_motion_combined_from_analyzer(
                analyzer,
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                shaky_score_threshold=self.config.shaky_score_threshold,
                trans_threshold=self.config.stability_trans_threshold,
                jump_threshold=self.config.stability_jump_threshold,
                rot_threshold=self.config.stability_rot_threshold,
                variance_threshold=self.config.stability_variance_threshold,
                w_trans=self.config.stability_w_trans,
                w_var=self.config.stability_w_var,
                w_rot=self.config.stability_w_rot,
                w_jump=self.config.stability_w_jump,
                frozen_max_consecutive=self.config.frozen_max_consecutive,
                frozen_trans_threshold=self.config.frozen_trans_threshold,
                frozen_rot_threshold=self.config.frozen_rot_threshold,
            )

        motion_future = motion_executor.submit(_run_motion)

        # Luminance & blur
        lb_config = LuminanceBlurConfig(
            min_good_ratio=self.config.luminance_blur_min_good_ratio,
            max_brightness_std=self.config.max_brightness_std,
        )
        check_results["luminance_blur"] = check_luminance_blur(
            seg_frames, config=lb_config, timestamps=seg_timestamps,
        )

        # Hand visibility: both-hands >=80% OR single-hand >=90%
        check_results["ml_hand_visibility"] = check_hand_visibility(
            per_frame_hands,
            frame_dims=(seg_h, seg_w),
            confidence_threshold=self.config.hand_confidence_threshold,
            both_hands_pass_rate=self.config.hand_pass_rate,
            single_hand_pass_rate=self.config.single_hand_pass_rate,
            frame_margin=self.config.hand_frame_margin,
            timestamps=seg_timestamps,
        )

        # Hand-object interaction (Phase 2 threshold: 0.60)
        check_results["ml_hand_object_interaction"] = check_hand_object_interaction(
            per_frame_hands,
            pass_rate_threshold=self.config.interaction_pass_rate,
            timestamps=seg_timestamps,
        )

        # View obstruction
        check_results["ml_view_obstruction"] = check_view_obstruction(
            seg_frames,
            max_obstructed_ratio=self.config.obstruction_max_ratio,
        )

        # POV hand angle
        check_results["ml_pov_hand_angle"] = check_pov_hand_angle(
            per_frame_hands,
            frame_dims=(seg_h, seg_w),
            angle_threshold=self.config.angle_threshold,
            pass_rate_threshold=self.config.angle_pass_rate,
            diagonal_fov_degrees=self.config.diagonal_fov_degrees,
        )

        # Face presence (strict): zero tolerance for any frame with a
        # confident face, even if Phase 1 let a short flash through.
        check_results["ml_face_presence"] = check_face_presence(
            per_frame_faces,
            confidence_threshold=self.config.face_confidence_threshold,
        )

        # Collect motion results
        stability_result, frozen_result = motion_future.result()
        check_results["motion_camera_stability"] = stability_result
        check_results["motion_frozen_segments"] = frozen_result

        # Determine pass/fail
        failing_checks = [
            name for name, r in check_results.items()
            if r.status == "fail"
        ]
        passed = len(failing_checks) == 0

        for name, r in check_results.items():
            print(f"    {name:<35} {r.status:>7}  "
                  f"metric={r.metric_value:.4f}")

        return SegmentValidationResult(
            segment=segment,
            passed=passed,
            check_results=check_results,
            failing_checks=failing_checks,
        )

    # ── Phase 3 ──────────────────────────────────────────────────────

    def _phase3_yield(
        self,
        original_duration: float,
        segment_results: list[SegmentValidationResult],
    ) -> tuple[float, float, float]:
        """Calculate usable/unusable durations and yield.

        Returns:
            (usable_sec, unusable_sec, yield_ratio)
        """
        usable_sec = sum(
            sr.segment.duration for sr in segment_results if sr.passed
        )
        unusable_sec = original_duration - usable_sec
        yield_ratio = usable_sec / original_duration if original_duration > 0 else 0.0

        return (
            round(usable_sec, 2),
            round(unusable_sec, 2),
            round(yield_ratio, 4),
        )


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "bachman_cortex/sample_data/test_30s.mp4"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "bachman_cortex/results/test"

    config = PipelineConfig(
        sampling_fps=1.0,
    )

    pipeline = ValidationProcessingPipeline(config)
    result = pipeline.process_video(video_path, output_dir)
    print(f"\nYield: {result.yield_ratio:.1%}")
    print(f"Usable: {result.usable_duration_sec:.1f}s / {result.original_duration_sec:.1f}s")
