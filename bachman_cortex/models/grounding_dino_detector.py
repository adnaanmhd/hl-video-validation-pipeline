"""Grounding DINO zero-shot object detection for privacy safety checks."""

import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class ZeroShotDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    label: str


# Default prompt for sensitive objects
DEFAULT_PRIVACY_PROMPT = (
    "laptop screen . computer monitor . smartphone screen . "
    "paper document . credit card . ID card . identification card . bank card"
)


class GroundingDINODetector:
    """Grounding DINO for zero-shot detection of sensitive objects.

    Used as a second-stage detector on frames flagged by YOLO for
    fine-grained privacy detection (credit cards, ID cards, documents).
    """

    def __init__(
        self,
        model_id: str = "IDEA-Research/grounding-dino-base",
        cache_dir: str | Path = "bachman_cortex/models/weights/grounding_dino",
        device: str | None = None,
    ):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=str(cache_dir))
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=str(cache_dir)
        ).to(self.device)
        self.model.eval()

    def detect(
        self,
        frame_bgr: np.ndarray,
        text_prompt: str = DEFAULT_PRIVACY_PROMPT,
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> list[ZeroShotDetection]:
        """Detect objects matching text prompt in a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image.
            text_prompt: Period-separated text labels to detect.
            box_threshold: Minimum confidence for box detection.
            text_threshold: Minimum confidence for text matching.

        Returns:
            List of ZeroShotDetection objects.
        """
        from PIL import Image
        import cv2

        # Convert BGR to RGB PIL Image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]

        detections = []
        h, w = frame_bgr.shape[:2]
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
            bbox = box.cpu().numpy()
            detections.append(
                ZeroShotDetection(
                    bbox=bbox,
                    confidence=float(score),
                    label=label,
                )
            )
        return detections

    def benchmark(self, frames: list[np.ndarray], text_prompt: str = DEFAULT_PRIVACY_PROMPT) -> dict:
        """Benchmark inference speed."""
        times = []
        for frame in frames:
            t0 = time.perf_counter()
            self.detect(frame, text_prompt=text_prompt)
            times.append(time.perf_counter() - t0)
        times_ms = [t * 1000 for t in times]
        return {
            "model": "Grounding DINO (base)",
            "frames": len(frames),
            "p50_ms": round(np.percentile(times_ms, 50), 2),
            "p95_ms": round(np.percentile(times_ms, 95), 2),
            "p99_ms": round(np.percentile(times_ms, 99), 2),
            "mean_ms": round(np.mean(times_ms), 2),
            "total_s": round(sum(times), 3),
        }
