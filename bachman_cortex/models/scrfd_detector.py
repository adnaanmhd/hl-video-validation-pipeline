"""SCRFD face detection wrapper using InsightFace."""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class FaceDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    landmarks: np.ndarray | None  # 5 landmarks (eyes, nose, mouth corners)


class SCRFDDetector:
    """SCRFD-2.5GF face detector via InsightFace.

    Returns face detections with confidence scores for privacy checking.
    """

    def __init__(
        self,
        model_name: str = "buffalo_sc",
        root: str | Path = "bachman_cortex/models/weights/insightface",
        det_size: tuple[int, int] = (640, 640),
    ):
        from insightface.app import FaceAnalysis
        import onnxruntime

        available = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self.app = FaceAnalysis(
            name=model_name,
            root=str(root),
            providers=providers,
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        """Detect faces in a BGR frame.

        Args:
            frame_bgr: OpenCV BGR image (numpy array).

        Returns:
            List of FaceDetection objects with bbox, confidence, landmarks.
        """
        faces = self.app.get(frame_bgr)
        results = []
        for face in faces:
            results.append(
                FaceDetection(
                    bbox=face.bbox,
                    confidence=float(face.det_score),
                    landmarks=face.kps if hasattr(face, "kps") else None,
                )
            )
        return results

    def benchmark(self, frames: list[np.ndarray]) -> dict:
        """Benchmark inference speed on a list of frames."""
        times = []
        for frame in frames:
            t0 = time.perf_counter()
            self.detect(frame)
            times.append(time.perf_counter() - t0)
        times_ms = [t * 1000 for t in times]
        return {
            "model": "SCRFD (buffalo_sc)",
            "frames": len(frames),
            "p50_ms": round(np.percentile(times_ms, 50), 2),
            "p95_ms": round(np.percentile(times_ms, 95), 2),
            "p99_ms": round(np.percentile(times_ms, 99), 2),
            "mean_ms": round(np.mean(times_ms), 2),
            "total_s": round(sum(times), 3),
        }
