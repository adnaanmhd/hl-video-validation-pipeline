"""Unified video validation pipeline.

Runs all checks in order:
  1. Video metadata (short-circuits on failure)
  2. Luminance & blur (includes brightness stability)
  3. Motion analysis
  4. ML detection (SCRFD, YOLO11m, YOLO11m-pose, Hands23, Grounding DINO)

Results are keyed by category: meta_*, luminance_blur,
motion_*, ml_*.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from ml_checks.checks.check_results import CheckResult
from ml_checks.utils.video_metadata import get_video_metadata
from ml_checks.checks.video_metadata import run_all_metadata_checks
from ml_checks.checks.luminance_blur import check_luminance_blur, LuminanceBlurConfig
from ml_checks.checks.motion_analysis import check_camera_stability, check_frozen_segments
from ml_checks.checks.face_presence import check_face_presence
from ml_checks.checks.participants import check_participants
from ml_checks.checks.hand_visibility import check_hand_visibility
from ml_checks.checks.hand_object_interaction import check_hand_object_interaction
from ml_checks.checks.privacy_safety import check_privacy_safety
from ml_checks.checks.view_obstruction import check_view_obstruction
from ml_checks.checks.pov_hand_angle import check_pov_hand_angle
from ml_checks.checks.body_part_visibility import check_body_part_visibility
from ml_checks.utils.frame_extractor import extract_frames
from ml_checks.utils.early_stop import (
    EarlyStopMonitor,
    CheckSpec,
    eval_hand_visibility,
    eval_face_presence,
    eval_participants,
    eval_hand_object_interaction,
    eval_pov_hand_angle,
    eval_body_part_visibility,
)

# Batch size for YOLO inference in the ML loop
_BATCH_SIZE = 16

# All non-metadata check keys, used for skipped results on metadata failure
NON_META_CHECK_KEYS = [
    "luminance_blur",
    "motion_camera_stability",
    "motion_frozen_segments",
    "ml_face_presence",
    "ml_participants",
    "ml_hand_visibility",
    "ml_hand_object_interaction",
    "ml_privacy_safety",
    "ml_view_obstruction",
    "ml_pov_hand_angle",
    "ml_body_part_visibility",
]


@dataclass
class PipelineConfig:
    """Configuration for the full validation pipeline."""
    # Frame sampling
    sampling_fps: float = 1.0
    max_frames: int | None = None

    # Model paths
    scrfd_root: str = "ml_checks/models/weights/insightface"
    yolo_model: str = "yolo11s.pt"
    yolo_pose_model: str = "yolo11m-pose.pt"
    hand_detector_repo: str = "ml_checks/models/weights/hands23_detector"
    gdino_cache: str = "ml_checks/models/weights/grounding_dino"

    # ML thresholds
    face_confidence_threshold: float = 0.8
    person_confidence_threshold: float = 0.4
    hand_confidence_threshold: float = 0.7
    hand_pass_rate: float = 0.90
    interaction_pass_rate: float = 0.70
    participant_pass_rate: float = 0.90
    privacy_gdino_threshold: float = 0.3
    obstruction_max_ratio: float = 0.10
    angle_threshold: float = 40.0
    angle_pass_rate: float = 0.80
    diagonal_fov_degrees: float = 90.0
    body_part_pass_rate: float = 0.90
    body_part_keypoint_conf: float = 0.5
    hand_frame_margin: int = 2

    # Grounding DINO
    run_grounding_dino: bool = True
    gdino_text_prompt: str = (
        "laptop screen . computer monitor . smartphone screen . "
        "paper document . credit card . ID card . identification card . bank card"
    )

    # Brightness stability threshold
    max_brightness_std: float = 60.0

    # Camera stability (two-stage LK optical flow)
    shaky_score_threshold: float = 0.30
    deep_score_threshold: float = 0.25
    fast_scale: float = 0.333
    frame_skip: int = 2
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
    frozen_ssim_threshold: float = 0.99
    frozen_target_fps: float = 10.0

    # Luminance & blur
    luminance_blur_min_good_ratio: float = 0.80

    # Hands23 input resolution cap (downscale long edge to this before inference).
    # Set to None to disable and run at native resolution.
    hands23_max_resolution: int | None = 720

    # Fail-fast: skip ML inference when quality checks fail
    fail_fast: bool = False


class ValidationPipeline:
    """Unified pipeline that runs metadata, quality, motion, and ML checks."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._models_loaded = False

    def load_models(self):
        """Load all ML models. Call once at worker startup."""
        print("Loading ML models...")
        t0 = time.perf_counter()

        # 1. SCRFD face detector
        from ml_checks.models.scrfd_detector import SCRFDDetector
        self.face_detector = SCRFDDetector(root=self.config.scrfd_root)
        print("  SCRFD loaded")

        # 2. YOLO11m
        from ml_checks.models.yolo_detector import YOLODetector
        self.yolo_detector = YOLODetector(model_path=self.config.yolo_model)
        print("  YOLO11m loaded")

        # 3. YOLO11m-pose
        from ml_checks.models.yolo_pose_detector import YOLOPoseDetector
        self.pose_detector = YOLOPoseDetector(model_path=self.config.yolo_pose_model)
        print("  YOLO11m-pose loaded")

        # 4. Hands23 hand-object detector
        from ml_checks.models.hand_detector import HandObjectDetectorHands23
        self.hand_detector = HandObjectDetectorHands23(
            repo_dir=self.config.hand_detector_repo,
            max_resolution=self.config.hands23_max_resolution,
        )
        print("  Hands23 loaded")

        # 5. Grounding DINO (optional, for privacy)
        self.gdino_detector = None
        if self.config.run_grounding_dino:
            from ml_checks.models.grounding_dino_detector import GroundingDINODetector
            self.gdino_detector = GroundingDINODetector(
                cache_dir=self.config.gdino_cache,
            )
            print("  Grounding DINO loaded")

        elapsed = time.perf_counter() - t0
        print(f"All models loaded in {elapsed:.1f}s")
        self._models_loaded = True

    def process_video(self, video_path: str | Path) -> dict[str, CheckResult]:
        """Run all validation checks on a video.

        Returns:
            Dict mapping check name to CheckResult.
            Keys are prefixed by category: meta_*, quality_*, luminance_blur,
            motion_*, ml_*.
        """
        video_path = str(video_path)
        results = {}
        timings = {}

        # ── Step 1: Metadata checks (short-circuit gate) ──────────
        t0 = time.perf_counter()
        metadata = get_video_metadata(video_path)
        meta_results = run_all_metadata_checks(metadata)
        timings["metadata"] = time.perf_counter() - t0
        results.update(meta_results)

        print(f"Metadata checks ({timings['metadata']:.2f}s):")
        for name, r in meta_results.items():
            print(f"  {name:<25} {r.status:>4}")

        # If any metadata check fails, skip all other checks
        meta_failed = any(r.status == "fail" for r in meta_results.values())
        if meta_failed:
            print("  Metadata check FAILED -- skipping all other checks")
            for key in NON_META_CHECK_KEYS:
                results[key] = CheckResult(
                    status="skipped",
                    metric_value=0.0,
                    confidence=1.0,
                    details={"reason": "metadata check failed"},
                )
            return results

        # ── Step 2: Extract sampled frames ────────────────────────
        t0 = time.perf_counter()
        frames, frame_meta = extract_frames(
            video_path,
            fps=self.config.sampling_fps,
            max_frames=self.config.max_frames,
        )
        timings["frame_extraction"] = time.perf_counter() - t0
        frame_h, frame_w = frame_meta["height"], frame_meta["width"]

        # Compute per-frame timestamps in seconds (for privacy timestamp reporting)
        frame_timestamps_sec = [i / self.config.sampling_fps for i in range(len(frames))]

        print(f"Extracted {len(frames)} frames ({frame_meta['duration_s']}s video) "
              f"in {timings['frame_extraction']:.2f}s")

        # ── Step 3: Luminance & blur ──────────────────────────────
        t0 = time.perf_counter()
        lb_config = LuminanceBlurConfig(
            min_good_ratio=self.config.luminance_blur_min_good_ratio,
            max_brightness_std=self.config.max_brightness_std,
        )
        results["luminance_blur"] = check_luminance_blur(frames, config=lb_config)
        timings["luminance_blur"] = time.perf_counter() - t0
        print(f"Luminance & blur check ({timings['luminance_blur']:.2f}s)")

        # ── Fail-fast gate on luminance ─────────────────────────
        if self.config.fail_fast and results["luminance_blur"].status == "fail":
            print("  Fail-fast: skipping motion + ML (luminance_blur failed)")
            for key in [k for k in NON_META_CHECK_KEYS if k != "luminance_blur"]:
                results[key] = CheckResult(
                    status="skipped", metric_value=0.0, confidence=1.0,
                    details={"reason": "fail-fast: luminance_blur failed"},
                )
            return results

        # ── Step 4+5: Motion analysis + ML inference ─────────────
        # When fail-fast is on, run motion first so we can skip ML on failure.
        # When fail-fast is off, run motion in a background thread while ML
        # runs on the main thread (parallel execution).

        def _run_motion():
            t_stab = time.perf_counter()
            stability = check_camera_stability(
                video_path,
                shaky_score_threshold=self.config.shaky_score_threshold,
                deep_score_threshold=self.config.deep_score_threshold,
                fast_scale=self.config.fast_scale,
                frame_skip=self.config.frame_skip,
                trans_threshold=self.config.stability_trans_threshold,
                jump_threshold=self.config.stability_jump_threshold,
                rot_threshold=self.config.stability_rot_threshold,
                variance_threshold=self.config.stability_variance_threshold,
                w_trans=self.config.stability_w_trans,
                w_var=self.config.stability_w_var,
                w_rot=self.config.stability_w_rot,
                w_jump=self.config.stability_w_jump,
                max_corners=self.config.stability_max_corners,
                lk_win_size=self.config.stability_lk_win_size,
                lk_max_level=self.config.stability_lk_max_level,
                target_fps=self.config.stability_target_fps,
            )
            t_stab = time.perf_counter() - t_stab
            t_froz = time.perf_counter()
            frozen = check_frozen_segments(
                video_path,
                max_consecutive=self.config.frozen_max_consecutive,
                ssim_threshold=self.config.frozen_ssim_threshold,
                target_fps=self.config.frozen_target_fps,
            )
            t_froz = time.perf_counter() - t_froz
            return stability, frozen, t_stab, t_froz

        if self.config.fail_fast:
            # Sequential: run motion first, gate ML on result
            stability_result, frozen_result, t_stab, t_froz = _run_motion()
            results["motion_camera_stability"] = stability_result
            results["motion_frozen_segments"] = frozen_result
            timings["camera_stability"] = t_stab
            timings["frozen_segments"] = t_froz
            print(f"Motion analysis ({t_stab:.2f}s stability, {t_froz:.2f}s frozen)")

            failed_motion = [
                k for k in ["motion_camera_stability", "motion_frozen_segments"]
                if results[k].status == "fail"
            ]
            if failed_motion:
                print(f"  Fail-fast: skipping ML ({', '.join(failed_motion)} failed)")
                ml_keys = [k for k in NON_META_CHECK_KEYS if k.startswith("ml_")]
                for key in ml_keys:
                    results[key] = CheckResult(
                        status="skipped", metric_value=0.0, confidence=1.0,
                        details={"reason": f"fail-fast: {', '.join(failed_motion)} failed"},
                    )
                return results
            motion_future = None
        else:
            # Parallel: launch motion in background thread
            motion_executor = ThreadPoolExecutor(max_workers=1)
            motion_future = motion_executor.submit(_run_motion)

        # ML inference runs in the main thread
        if not self._models_loaded:
            self.load_models()

        per_frame_faces = []
        per_frame_yolo = []
        per_frame_yolo_persons = []
        per_frame_hands = []
        per_frame_objects = []
        per_frame_poses = []

        t_scrfd = 0.0
        t_yolo = 0.0
        t_pose = 0.0
        t_hands23 = 0.0

        # Early stopping monitor
        monitor = EarlyStopMonitor(len(frames), [
            CheckSpec("hand_visibility", self.config.hand_pass_rate, False),
            CheckSpec("face_presence", 1.0, True),
            CheckSpec("participants", self.config.participant_pass_rate, False),
            CheckSpec("hand_object_interaction", self.config.interaction_pass_rate, False),
            CheckSpec("pov_hand_angle", self.config.angle_pass_rate, False),
            CheckSpec("body_part_visibility", self.config.body_part_pass_rate, False),
        ])

        total_frames = len(frames)
        frames_inferred = 0

        for batch_start in range(0, total_frames, _BATCH_SIZE):
            if monitor.all_determined():
                break

            batch_end = min(batch_start + _BATCH_SIZE, total_frames)
            batch_frames = frames[batch_start:batch_end]

            # Batched YOLO inference
            t0 = time.perf_counter()
            batch_yolo = self.yolo_detector.detect_batch(batch_frames)
            t_yolo += time.perf_counter() - t0

            t0 = time.perf_counter()
            batch_poses = self.pose_detector.detect_batch(batch_frames)
            t_pose += time.perf_counter() - t0

            # Per-frame inference for SCRFD + Hands23 (no batch API)
            for j, frame in enumerate(batch_frames):
                i = batch_start + j  # global frame index

                t0 = time.perf_counter()
                faces = self.face_detector.detect(frame)
                t_scrfd += time.perf_counter() - t0
                per_frame_faces.append(faces)

                yolo_dets = batch_yolo[j]
                per_frame_yolo.append(yolo_dets)
                persons = self.yolo_detector.get_persons(yolo_dets)
                per_frame_yolo_persons.append(persons)

                poses = batch_poses[j]
                per_frame_poses.append(poses)

                t0 = time.perf_counter()
                hands, objects = self.hand_detector.detect(frame)
                t_hands23 += time.perf_counter() - t0
                per_frame_hands.append(hands)
                per_frame_objects.append(objects)

                # Update early stop monitor
                monitor.update("hand_visibility",
                    eval_hand_visibility(hands, self.config.hand_confidence_threshold,
                                         frame_w, frame_h, self.config.hand_frame_margin))
                monitor.update("face_presence",
                    eval_face_presence(faces, self.config.face_confidence_threshold))
                monitor.update("participants",
                    eval_participants(persons, faces, hands, frame_h, frame_w,
                        self.config.person_confidence_threshold, 0.5, 0.15))
                monitor.update("hand_object_interaction",
                    eval_hand_object_interaction(hands))
                monitor.update("pov_hand_angle",
                    eval_pov_hand_angle(hands, frame_h, frame_w,
                        self.config.angle_threshold, self.config.diagonal_fov_degrees))
                monitor.update("body_part_visibility",
                    eval_body_part_visibility(poses, hands, frame_h, frame_w,
                        0.4, self.config.body_part_keypoint_conf))

                monitor.advance_frame()
                frames_inferred = i + 1

                if (frames_inferred) % 10 == 0 or frames_inferred == total_frames:
                    print(f"  Processed frame {frames_inferred}/{total_frames}")

                if monitor.all_determined():
                    break

        if frames_inferred < total_frames:
            print(f"  Early stopped at frame {frames_inferred}/{total_frames} "
                  f"(all checks determined)")

        timings["scrfd"] = t_scrfd
        timings["yolo"] = t_yolo
        timings["yolo_pose"] = t_pose
        timings["hands23"] = t_hands23

        # Grounding DINO on all frames for privacy detection
        per_frame_gdino = [[] for _ in range(frames_inferred)]
        t_gdino = 0.0
        if self.gdino_detector:
            print(f"  Running Grounding DINO on {frames_inferred} frames...")
            for idx in range(frames_inferred):
                t0 = time.perf_counter()
                gdino_dets = self.gdino_detector.detect(
                    frames[idx],
                    text_prompt=self.config.gdino_text_prompt,
                    box_threshold=self.config.privacy_gdino_threshold,
                )
                t_gdino += time.perf_counter() - t0
                per_frame_gdino[idx] = gdino_dets
        timings["grounding_dino"] = t_gdino

        # ── Collect motion analysis results (if ran in parallel) ──
        if motion_future is not None:
            stability_result, frozen_result, t_stab, t_froz = motion_future.result()
            motion_executor.shutdown(wait=False)
            results["motion_camera_stability"] = stability_result
            results["motion_frozen_segments"] = frozen_result
            timings["camera_stability"] = t_stab
            timings["frozen_segments"] = t_froz
            print(f"Motion analysis ({t_stab:.2f}s stability, {t_froz:.2f}s frozen)")

        # ── Step 6: ML checks ────────────────────────────────────
        # For determined checks, build results directly from monitor.
        # For undetermined checks (all frames processed), use original functions.
        print("Running ML checks...")

        _es_details = {"early_stopped": True, "frames_inferred": frames_inferred,
                       "total_frames": total_frames}

        # -- face_presence
        status, passes, processed = monitor.get_result("face_presence")
        if status is not None and frames_inferred < total_frames:
            results["ml_face_presence"] = CheckResult(
                status=status,
                metric_value=round(passes / processed, 4) if processed else 0.0,
                confidence=1.0,
                details={**_es_details, "clean_frames": passes, "frames_processed": processed},
            )
        else:
            results["ml_face_presence"] = check_face_presence(
                per_frame_faces,
                confidence_threshold=self.config.face_confidence_threshold,
            )

        # -- participants
        status, passes, processed = monitor.get_result("participants")
        if status is not None and frames_inferred < total_frames:
            results["ml_participants"] = CheckResult(
                status=status,
                metric_value=round(passes / processed, 4) if processed else 0.0,
                confidence=1.0,
                details={**_es_details, "clean_frames": passes, "frames_processed": processed},
            )
        else:
            results["ml_participants"] = check_participants(
                per_frame_yolo_persons, per_frame_faces, per_frame_hands,
                frame_dims=(frame_h, frame_w),
                pass_rate_threshold=self.config.participant_pass_rate,
                person_conf_threshold=self.config.person_confidence_threshold,
            )

        # -- hand_visibility
        status, passes, processed = monitor.get_result("hand_visibility")
        if status is not None and frames_inferred < total_frames:
            results["ml_hand_visibility"] = CheckResult(
                status=status,
                metric_value=round(passes / processed, 4) if processed else 0.0,
                confidence=1.0,
                details={**_es_details, "both_hands_frames": passes, "frames_processed": processed},
            )
        else:
            results["ml_hand_visibility"] = check_hand_visibility(
                per_frame_hands,
                frame_dims=(frame_h, frame_w),
                confidence_threshold=self.config.hand_confidence_threshold,
                pass_rate_threshold=self.config.hand_pass_rate,
                frame_margin=self.config.hand_frame_margin,
            )

        # -- hand_object_interaction
        status, passes, processed = monitor.get_result("hand_object_interaction")
        if status is not None and frames_inferred < total_frames:
            results["ml_hand_object_interaction"] = CheckResult(
                status=status,
                metric_value=round(passes / processed, 4) if processed else 0.0,
                confidence=1.0,
                details={**_es_details, "interaction_frames": passes, "frames_processed": processed},
            )
        else:
            results["ml_hand_object_interaction"] = check_hand_object_interaction(
                per_frame_hands,
                pass_rate_threshold=self.config.interaction_pass_rate,
            )

        # -- privacy_safety (GDINO on all frames, collects timestamps)
        results["ml_privacy_safety"] = check_privacy_safety(
            per_frame_gdino,
            gdino_conf_threshold=self.config.privacy_gdino_threshold,
            frame_timestamps_sec=frame_timestamps_sec,
        )

        # -- view_obstruction (CPU-only, no ML models, no early stopping)
        results["ml_view_obstruction"] = check_view_obstruction(
            frames,
            max_obstructed_ratio=self.config.obstruction_max_ratio,
        )

        # -- pov_hand_angle
        status, passes, processed = monitor.get_result("pov_hand_angle")
        if status is not None and frames_inferred < total_frames:
            results["ml_pov_hand_angle"] = CheckResult(
                status=status,
                metric_value=round(passes / processed, 4) if processed else 0.0,
                confidence=1.0,
                details={**_es_details, "passing_frames": passes, "frames_processed": processed},
            )
        else:
            results["ml_pov_hand_angle"] = check_pov_hand_angle(
                per_frame_hands,
                frame_dims=(frame_h, frame_w),
                angle_threshold=self.config.angle_threshold,
                pass_rate_threshold=self.config.angle_pass_rate,
                diagonal_fov_degrees=self.config.diagonal_fov_degrees,
            )

        # -- body_part_visibility
        status, passes, processed = monitor.get_result("body_part_visibility")
        if status is not None and frames_inferred < total_frames:
            results["ml_body_part_visibility"] = CheckResult(
                status=status,
                metric_value=round(passes / processed, 4) if processed else 0.0,
                confidence=1.0,
                details={**_es_details, "clean_frames": passes, "frames_processed": processed},
            )
        else:
            results["ml_body_part_visibility"] = check_body_part_visibility(
                per_frame_poses, per_frame_hands,
                frame_dims=(frame_h, frame_w),
                pass_rate_threshold=self.config.body_part_pass_rate,
                keypoint_conf_threshold=self.config.body_part_keypoint_conf,
            )

        # ── Summary ──────────────────────────────────────────────
        total_time = sum(timings.values())
        timings["total"] = total_time

        print(f"\nTiming breakdown:")
        print(f"  Metadata:         {timings['metadata']:.2f}s")
        print(f"  Frame extraction: {timings['frame_extraction']:.2f}s")
        print(f"  Luminance & blur: {timings['luminance_blur']:.2f}s")
        print(f"  Camera stability: {timings['camera_stability']:.2f}s")
        print(f"  Frozen segments:  {timings['frozen_segments']:.2f}s")
        print(f"  SCRFD:            {timings['scrfd']:.2f}s")
        print(f"  YOLO:             {timings['yolo']:.2f}s")
        print(f"  YOLO-pose:        {timings['yolo_pose']:.2f}s")
        print(f"  Hands23:          {timings['hands23']:.2f}s")
        print(f"  Grounding DINO:   {timings['grounding_dino']:.2f}s")
        print(f"  TOTAL:            {total_time:.2f}s")
        if frames_inferred < total_frames:
            print(f"  (early stopped at {frames_inferred}/{total_frames} frames)")

        print(f"\nCheck results:")
        for name, result in results.items():
            flag = ""
            if result.status == "pass" and result.confidence < 0.7:
                flag = " (low confidence)"
            print(f"  {name:<35} {result.status:>7}  "
                  f"metric={result.metric_value:.4f}  conf={result.confidence:.4f}{flag}")

        return results


# Backward compatibility alias
MLCheckPipeline = ValidationPipeline


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "ml_checks/sample_data/test_30s.mp4"

    config = PipelineConfig(
        sampling_fps=1.0,
        max_frames=10,
        run_grounding_dino=True,
    )

    pipeline = ValidationPipeline(config)
    results = pipeline.process_video(video_path)
