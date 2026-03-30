"""Unified ML inference pipeline for all 7 video quality checks.

Runs all models on sampled frames and distributes results to check functions.
Pipeline:
  Per frame: SCRFD -> YOLO11m -> 100DOH -> (Grounding DINO if flagged) -> heuristic
  Distribute results to all 7 checks -> aggregate
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from ml_checks.checks.check_results import CheckResult
from ml_checks.checks.face_presence import check_face_presence
from ml_checks.checks.participants import check_participants
from ml_checks.checks.hand_visibility import check_hand_visibility
from ml_checks.checks.hand_object_interaction import check_hand_object_interaction
from ml_checks.checks.privacy_safety import check_privacy_safety
from ml_checks.checks.view_obstruction import check_view_obstruction
from ml_checks.checks.pov_hand_angle import check_pov_hand_angle
from ml_checks.utils.frame_extractor import extract_frames


@dataclass
class PipelineConfig:
    """Configuration for the ML check pipeline."""
    # Frame sampling
    sampling_fps: float = 1.0
    max_frames: int | None = None

    # Model paths
    scrfd_root: str = "ml_checks/models/weights/insightface"
    yolo_model: str = "yolo11m.pt"
    hand_detector_repo: str = "ml_checks/models/weights/hand_object_detector"
    gdino_cache: str = "ml_checks/models/weights/grounding_dino"

    # Thresholds
    face_confidence_threshold: float = 0.8
    person_confidence_threshold: float = 0.4
    hand_confidence_threshold: float = 0.7
    hand_pass_rate: float = 0.90
    interaction_pass_rate: float = 0.70
    participant_pass_rate: float = 0.95
    privacy_yolo_threshold: float = 0.6
    privacy_gdino_threshold: float = 0.3
    obstruction_max_ratio: float = 0.10
    angle_threshold: float = 40.0
    angle_pass_rate: float = 0.80
    diagonal_fov_degrees: float = 90.0

    # Grounding DINO
    run_grounding_dino: bool = True
    gdino_text_prompt: str = (
        "laptop screen . computer monitor . smartphone screen . "
        "paper document . credit card . ID card . identification card . bank card"
    )


class MLCheckPipeline:
    """Unified pipeline that loads all models once and runs all 7 checks."""

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

        # 3. 100DOH hand-object detector
        from ml_checks.models.hand_detector import HandObjectDetector100DOH
        self.hand_detector = HandObjectDetector100DOH(
            repo_dir=self.config.hand_detector_repo,
        )
        print("  100DOH loaded")

        # 4. Grounding DINO (optional, for privacy)
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
        """Run all 7 ML checks on a video.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict mapping check name to CheckResult.
        """
        if not self._models_loaded:
            self.load_models()

        video_path = str(video_path)
        results = {}
        timings = {}

        # Step 1: Extract frames
        t0 = time.perf_counter()
        frames, meta = extract_frames(
            video_path,
            fps=self.config.sampling_fps,
            max_frames=self.config.max_frames,
        )
        timings["frame_extraction"] = time.perf_counter() - t0
        frame_h, frame_w = meta["height"], meta["width"]

        print(f"Extracted {len(frames)} frames ({meta['duration_s']}s video) in {timings['frame_extraction']:.2f}s")

        # Step 2: Run all models per frame
        per_frame_faces = []
        per_frame_yolo = []
        per_frame_yolo_persons = []
        per_frame_yolo_sensitive = []
        per_frame_hands = []
        per_frame_objects = []
        flagged_for_gdino = []  # Frame indices that need Grounding DINO

        from ml_checks.models.yolo_detector import PERSON_CLASS, SENSITIVE_CLASSES

        t_scrfd = 0
        t_yolo = 0
        t_100doh = 0

        for i, frame in enumerate(frames):
            # SCRFD face detection
            t0 = time.perf_counter()
            faces = self.face_detector.detect(frame)
            t_scrfd += time.perf_counter() - t0
            per_frame_faces.append(faces)

            # YOLO detection
            t0 = time.perf_counter()
            yolo_dets = self.yolo_detector.detect(frame)
            t_yolo += time.perf_counter() - t0
            per_frame_yolo.append(yolo_dets)
            per_frame_yolo_persons.append(self.yolo_detector.get_persons(yolo_dets))
            sensitive = self.yolo_detector.get_sensitive_objects(yolo_dets)
            per_frame_yolo_sensitive.append(sensitive)
            if sensitive:
                flagged_for_gdino.append(i)

            # 100DOH hand-object detection
            t0 = time.perf_counter()
            hands, objects = self.hand_detector.detect(frame)
            t_100doh += time.perf_counter() - t0
            per_frame_hands.append(hands)
            per_frame_objects.append(objects)

            if (i + 1) % 10 == 0 or i == len(frames) - 1:
                print(f"  Processed frame {i + 1}/{len(frames)}")

        timings["scrfd"] = t_scrfd
        timings["yolo"] = t_yolo
        timings["100doh"] = t_100doh

        # Step 3: Grounding DINO on flagged frames (if enabled)
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

        # Step 4: Run all 7 checks
        print("Running checks...")

        # Check 1: Face Presence
        results["face_presence"] = check_face_presence(
            per_frame_faces,
            confidence_threshold=self.config.face_confidence_threshold,
        )

        # Check 2: Participants
        results["participants"] = check_participants(
            per_frame_yolo_persons,
            per_frame_faces,
            per_frame_hands,
            frame_dims=(frame_h, frame_w),
            pass_rate_threshold=self.config.participant_pass_rate,
            person_conf_threshold=self.config.person_confidence_threshold,
        )

        # Check 3: Hand Visibility
        results["hand_visibility"] = check_hand_visibility(
            per_frame_hands,
            confidence_threshold=self.config.hand_confidence_threshold,
            pass_rate_threshold=self.config.hand_pass_rate,
        )

        # Check 4: Hand-Object Interaction
        results["hand_object_interaction"] = check_hand_object_interaction(
            per_frame_hands,
            pass_rate_threshold=self.config.interaction_pass_rate,
        )

        # Check 5: Privacy Safety
        results["privacy_safety"] = check_privacy_safety(
            per_frame_yolo_sensitive,
            per_frame_gdino if self.gdino_detector else None,
            yolo_conf_threshold=self.config.privacy_yolo_threshold,
            gdino_conf_threshold=self.config.privacy_gdino_threshold,
        )

        # Check 6: View Obstruction
        results["view_obstruction"] = check_view_obstruction(
            frames,
            max_obstructed_ratio=self.config.obstruction_max_ratio,
        )

        # Check 7: POV-Hand Angle
        results["pov_hand_angle"] = check_pov_hand_angle(
            per_frame_hands,
            frame_dims=(frame_h, frame_w),
            angle_threshold=self.config.angle_threshold,
            pass_rate_threshold=self.config.angle_pass_rate,
            diagonal_fov_degrees=self.config.diagonal_fov_degrees,
        )

        # Summary
        total_time = sum(timings.values())
        timings["total"] = total_time

        print(f"\nTiming breakdown:")
        print(f"  Frame extraction: {timings['frame_extraction']:.2f}s")
        print(f"  SCRFD:            {timings['scrfd']:.2f}s")
        print(f"  YOLO11m:          {timings['yolo']:.2f}s")
        print(f"  100DOH:           {timings['100doh']:.2f}s")
        print(f"  Grounding DINO:   {timings['grounding_dino']:.2f}s")
        print(f"  TOTAL:            {total_time:.2f}s")

        print(f"\nCheck results:")
        for name, result in results.items():
            flag = " ⚠️" if result.confidence < 0.7 and result.status == "pass" else ""
            print(f"  {name:<30} {result.status:>4}  metric={result.metric_value:.4f}  conf={result.confidence:.4f}{flag}")

        return results


if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "ml_checks/sample_data/test_30s.mp4"

    config = PipelineConfig(
        sampling_fps=1.0,
        max_frames=10,  # Limit for quick testing
        run_grounding_dino=True,
    )

    pipeline = MLCheckPipeline(config)
    results = pipeline.process_video(video_path)
