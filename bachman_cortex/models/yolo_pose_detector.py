"""YOLO11m-pose keypoint detection wrapper using Ultralytics."""

import numpy as np
from dataclasses import dataclass
from typing import Optional
import time


# COCO keypoint indices
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# Keypoints that are acceptable for the camera wearer in egocentric video
# (hands up to elbows)
ALLOWED_WEARER_KEYPOINTS = {7, 8, 9, 10}  # left/right elbow, left/right wrist


@dataclass
class PoseDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    keypoints: np.ndarray  # (17, 3) — x, y, confidence per keypoint


class YOLOPoseDetector:
    """YOLO11m-pose for body keypoint detection.

    Used to detect whether body parts beyond hands/forearms are visible
    in egocentric video frames.
    """

    def __init__(
        self,
        model_path: str = "yolo11m-pose.pt",
        conf_threshold: float = 0.25,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> list[PoseDetection]:
        """Run pose estimation on a BGR frame.

        Returns:
            List of PoseDetection with bounding boxes and 17 keypoints each.
        """
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
            boxes = r.boxes
            kpts = r.keypoints
            for i in range(len(boxes)):
                detections.append(
                    PoseDetection(
                        bbox=boxes[i].xyxy[0].cpu().numpy(),
                        confidence=float(boxes[i].conf[0]),
                        keypoints=kpts[i].data[0].cpu().numpy(),  # (17, 3)
                    )
                )
        return detections

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[PoseDetection]]:
        """Run pose estimation on a batch of BGR frames.

        Args:
            frames_bgr: List of OpenCV BGR images.

        Returns:
            List of pose detection lists, one per input frame.
        """
        if not frames_bgr:
            return []
        results = self.model.predict(
            frames_bgr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        batch_detections = []
        for r in results:
            frame_dets = []
            if r.boxes is not None and r.keypoints is not None:
                boxes = r.boxes
                kpts = r.keypoints
                for i in range(len(boxes)):
                    frame_dets.append(
                        PoseDetection(
                            bbox=boxes[i].xyxy[0].cpu().numpy(),
                            confidence=float(boxes[i].conf[0]),
                            keypoints=kpts[i].data[0].cpu().numpy(),
                        )
                    )
            batch_detections.append(frame_dets)
        return batch_detections
