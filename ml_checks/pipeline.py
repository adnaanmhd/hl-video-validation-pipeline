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
    yolo_model: str = "yolo11m.pt"
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
    privacy_yolo_threshold: float = 0.6
    privacy_gdino_threshold: float = 0.3
    obstruction_max_ratio: float = 0.10
    angle_threshold: float = 40.0
    angle_pass_rate: float = 0.80
    diagonal_fov_degrees: float = 90.0
    body_part_pass_rate: float = 0.90
    body_part_keypoint_conf: float = 0.5

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

    # Frozen segments
    frozen_max_consecutive: int = 30
    frozen_ssim_threshold: float = 0.99

    # Luminance & blur
    luminance_blur_min_good_ratio: float = 0.80


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

        # ── Step 4: Motion analysis ───────────────────────────────
        t0 = time.perf_counter()
        results["motion_camera_stability"] = check_camera_stability(
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
        )
        timings["camera_stability"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        results["motion_frozen_segments"] = check_frozen_segments(
            video_path,
            max_consecutive=self.config.frozen_max_consecutive,
            ssim_threshold=self.config.frozen_ssim_threshold,
        )
        timings["frozen_segments"] = time.perf_counter() - t0
        print(f"Motion analysis ({timings['camera_stability']:.2f}s stability, "
              f"{timings['frozen_segments']:.2f}s frozen)")

        # ── Step 5: ML model inference ────────────────────────────
        if not self._models_loaded:
            self.load_models()

        per_frame_faces = []
        per_frame_yolo = []
        per_frame_yolo_persons = []
        per_frame_yolo_sensitive = []
        per_frame_hands = []
        per_frame_objects = []
        per_frame_poses = []
        flagged_for_gdino = []

        from ml_checks.models.yolo_detector import PERSON_CLASS, SENSITIVE_CLASSES

        t_scrfd = 0
        t_yolo = 0
        t_pose = 0
        t_hands23 = 0

        for i, frame in enumerate(frames):
            t0 = time.perf_counter()
            faces = self.face_detector.detect(frame)
            t_scrfd += time.perf_counter() - t0
            per_frame_faces.append(faces)

            t0 = time.perf_counter()
            yolo_dets = self.yolo_detector.detect(frame)
            t_yolo += time.perf_counter() - t0
            per_frame_yolo.append(yolo_dets)
            per_frame_yolo_persons.append(self.yolo_detector.get_persons(yolo_dets))
            sensitive = self.yolo_detector.get_sensitive_objects(yolo_dets)
            per_frame_yolo_sensitive.append(sensitive)
            if sensitive:
                flagged_for_gdino.append(i)

            t0 = time.perf_counter()
            poses = self.pose_detector.detect(frame)
            t_pose += time.perf_counter() - t0
            per_frame_poses.append(poses)

            t0 = time.perf_counter()
            hands, objects = self.hand_detector.detect(frame)
            t_hands23 += time.perf_counter() - t0
            per_frame_hands.append(hands)
            per_frame_objects.append(objects)

            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                print(f"  Processed frame {i + 1}/{len(frames)}")

        timings["scrfd"] = t_scrfd
        timings["yolo"] = t_yolo
        timings["yolo_pose"] = t_pose
        timings["hands23"] = t_hands23

        # Grounding DINO on flagged frames
        per_frame_gdino = [[] for _ in frames]
        t_gdino = 0
        if self.gdino_detector and flagged_for_gdino:
            print(f"  Running Grounding DINO on {len(flagged_for_gdino)} flagged frames...")
            for idx in flagged_for_gdino:
                t0 = time.perf_counter()
                gdino_dets = self.gdino_detector.detect(
                    frames[idx],
                    text_prompt=self.config.gdino_text_prompt,
                    box_threshold=self.config.privacy_gdino_threshold,
                )
                t_gdino += time.perf_counter() - t0
                per_frame_gdino[idx] = gdino_dets
        timings["grounding_dino"] = t_gdino

        # ── Step 6: ML checks ────────────────────────────────────
        print("Running ML checks...")

        results["ml_face_presence"] = check_face_presence(
            per_frame_faces,
            confidence_threshold=self.config.face_confidence_threshold,
        )

        results["ml_participants"] = check_participants(
            per_frame_yolo_persons,
            per_frame_faces,
            per_frame_hands,
            frame_dims=(frame_h, frame_w),
            pass_rate_threshold=self.config.participant_pass_rate,
            person_conf_threshold=self.config.person_confidence_threshold,
        )

        results["ml_hand_visibility"] = check_hand_visibility(
            per_frame_hands,
            confidence_threshold=self.config.hand_confidence_threshold,
            pass_rate_threshold=self.config.hand_pass_rate,
        )

        results["ml_hand_object_interaction"] = check_hand_object_interaction(
            per_frame_hands,
            pass_rate_threshold=self.config.interaction_pass_rate,
        )

        results["ml_privacy_safety"] = check_privacy_safety(
            per_frame_yolo_sensitive,
            per_frame_gdino if self.gdino_detector else None,
            yolo_conf_threshold=self.config.privacy_yolo_threshold,
            gdino_conf_threshold=self.config.privacy_gdino_threshold,
        )

        results["ml_view_obstruction"] = check_view_obstruction(
            frames,
            max_obstructed_ratio=self.config.obstruction_max_ratio,
        )

        results["ml_pov_hand_angle"] = check_pov_hand_angle(
            per_frame_hands,
            frame_dims=(frame_h, frame_w),
            angle_threshold=self.config.angle_threshold,
            pass_rate_threshold=self.config.angle_pass_rate,
            diagonal_fov_degrees=self.config.diagonal_fov_degrees,
        )

        results["ml_body_part_visibility"] = check_body_part_visibility(
            per_frame_poses,
            per_frame_hands,
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
        print(f"  YOLO11m:          {timings['yolo']:.2f}s")
        print(f"  YOLO11m-pose:     {timings['yolo_pose']:.2f}s")
        print(f"  Hands23:          {timings['hands23']:.2f}s")
        print(f"  Grounding DINO:   {timings['grounding_dino']:.2f}s")
        print(f"  TOTAL:            {total_time:.2f}s")

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
