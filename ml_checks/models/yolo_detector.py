"""YOLO11m object detection wrapper using Ultralytics."""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import time


# COCO class IDs relevant to our checks
PERSON_CLASS = 0


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """YOLO11m object detector for person counting.

    Single forward pass serves:
    - Check 2 (Participants): filter class_id == 0 (person)
    """

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        conf_threshold: float = 0.25,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run object detection on a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image.

        Returns:
            List of Detection objects.
        """
        results = self.model.predict(
            frame_bgr,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append(
                    Detection(
                        bbox=box.xyxy[0].cpu().numpy(),
                        confidence=float(box.conf[0]),
                        class_id=cls_id,
                        class_name=r.names[cls_id],
                    )
                )
        return detections

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """Run object detection on a batch of BGR frames.

        Args:
            frames_bgr: List of OpenCV BGR images.

        Returns:
            List of detection lists, one per input frame.
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
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    frame_dets.append(
                        Detection(
                            bbox=box.xyxy[0].cpu().numpy(),
                            confidence=float(box.conf[0]),
                            class_id=cls_id,
                            class_name=r.names[cls_id],
                        )
                    )
            batch_detections.append(frame_dets)
        return batch_detections

    def get_persons(self, detections: list[Detection], min_conf: float = 0.4) -> list[Detection]:
        """Filter person detections."""
        return [d for d in detections if d.class_id == PERSON_CLASS and d.confidence >= min_conf]

    def benchmark(self, frames: list[np.ndarray]) -> dict:
        """Benchmark inference speed."""
        times = []
        for frame in frames:
            t0 = time.perf_counter()
            self.detect(frame)
            times.append(time.perf_counter() - t0)
        times_ms = [t * 1000 for t in times]
        return {
            "model": "YOLO11m",
            "frames": len(frames),
            "p50_ms": round(np.percentile(times_ms, 50), 2),
            "p95_ms": round(np.percentile(times_ms, 95), 2),
            "p99_ms": round(np.percentile(times_ms, 99), 2),
            "mean_ms": round(np.mean(times_ms), 2),
            "total_s": round(sum(times), 3),
        }
