"""Hand-Object detection for egocentric video quality checks.

Active backend: Hands23 (NeurIPS 2023) — successor to 100DOH by the same research group.
Outputs: hand bboxes, left/right, contact state, grasp type, object bboxes + masks.

Previous backend (commented out): 100DOH (CVPR 2020)
To revert to 100DOH, see hand_detector_100doh.py and swap the import where
HandObjectDetectorHands23 is instantiated (bachman_cortex/scoring_engine.py).

Setup:
    python bachman_cortex/models/download_models.py --all
"""

import os
import sys
import numpy as np
import torch
import cv2
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ============================================================
# Shared types — used by all check functions.
# These stay the same regardless of which backend is active.
# ============================================================

class ContactState(Enum):
    NO_CONTACT = 0
    SELF_CONTACT = 1
    OTHER_PERSON = 2
    PORTABLE_OBJ = 3
    STATIONARY_OBJ = 4


class HandSide(Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


@dataclass
class HandDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    side: HandSide
    contact_state: ContactState
    contact_state_confidence: float = 0.0  # Hands23: dz[9] interaction score
    grasp_type: str | None = None  # Hands23 only: e.g. "NP-Palm", "Pow-Pris", etc.
    offset_vector: np.ndarray | None = None  # 100DOH only


@dataclass
class ObjectDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    touch_type: str | None = None  # Hands23: "tool-object" relationship


# ============================================================
# Hands23 contact/grasp/side label mappings
# ============================================================

HANDS23_CONTACT_MAP = {
    0: ContactState.NO_CONTACT,
    1: ContactState.SELF_CONTACT,
    2: ContactState.OTHER_PERSON,
    3: ContactState.PORTABLE_OBJ,   # "object_contact" in Hands23
    4: ContactState.STATIONARY_OBJ,  # mapped from tool-object if needed
}

HANDS23_GRASP_NAMES = {
    0: "NP-Palm",
    1: "NP-Fin",
    2: "Pow-Pris",
    3: "Pre-Pris",
    4: "Pow-Circ",
    5: "Pre-Circ",
    6: "Later",
    7: "Other",
}

HANDS23_SIDE_MAP = {
    0: HandSide.LEFT,
    1: HandSide.RIGHT,
}


# ============================================================
# Hands23 Detector (active backend)
# ============================================================

class HandObjectDetectorHands23:
    """Hands23 hand-object detector (NeurIPS 2023).

    Faster R-CNN X-101-FPN trained on 250K images with rich annotations:
    hand bbox, left/right, contact state, grasp type, object bbox, masks.

    Uses Detectron2's DefaultPredictor for inference.
    """

    def __init__(
        self,
        repo_dir: str | Path = "bachman_cortex/models/weights/hands23_detector",
        weight_file: str = "model_weights/model_hands23.pth",
        hand_thresh: float = 0.5,
        obj_thresh: float = 0.3,
        hand_rela_thresh: float = 0.3,
        max_resolution: int | None = 720,
    ):
        self.repo_dir = Path(repo_dir).resolve()
        self.hand_thresh = hand_thresh
        self.obj_thresh = obj_thresh
        self.hand_rela_thresh = hand_rela_thresh
        self.max_resolution = max_resolution

        # Validate setup
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"Hands23 repo not found at {self.repo_dir}. "
                f"Run: python bachman_cortex/models/download_models.py --all"
            )
        weights_path = self.repo_dir / weight_file
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Hands23 weights not found at {weights_path}. "
                f"Run: python bachman_cortex/models/download_models.py --all"
            )

        # Add Hands23 repo to sys.path so hodetector package is importable
        repo_str = str(self.repo_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        # Import Hands23 custom Detectron2 components (registers custom ROI heads)
        import hodetector.modeling  # noqa: F401 — registers custom components with Detectron2

        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        # Build config
        cfg = get_cfg()

        # Add Hands23 custom config keys (extends Detectron2 cfg with custom heads)
        from hodetector.config import get_meshrcnn_cfg_defaults
        get_meshrcnn_cfg_defaults(cfg)

        cfg.merge_from_file(str(self.repo_dir / "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml"))
        cfg.MODEL.WEIGHTS = str(weights_path)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = hand_thresh
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Hands23 custom thresholds (used by roi_heads.py inference logic)
        cfg.HAND = hand_thresh
        cfg.FIRSTOBJ = obj_thresh
        cfg.SECONDOBJ = 0.1
        cfg.HAND_RELA = hand_rela_thresh
        cfg.OBJ_RELA = 0.7

        self.predictor = DefaultPredictor(cfg)
        self.cfg = cfg

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[HandDetection], list[ObjectDetection]]:
        """Detect hands and objects in a BGR frame.

        Hands23 pred_dz layout per instance:
            [0:4]   — associated object bbox (x1, y1, x2, y2)
            [4]     — interaction index (-1 = no interaction)
            [5]     — hand_side (0=left, 1=right)
            [6]     — grasp type index
            [7]     — touch type index
            [8]     — contact_state index
            [9]     — interaction score
            [10:18] — grasp scores (8 classes)
            [18:25] — touch scores (7 classes)

        Class mapping: 0=hand, 1=first_object, 2=second_object

        Returns:
            Tuple of (hand_detections, object_detections).
        """
        # Downscale to max_resolution if configured (scale bboxes back afterwards)
        h_orig, w_orig = frame_bgr.shape[:2]
        scale = 1.0
        if self.max_resolution and max(h_orig, w_orig) > self.max_resolution:
            scale = self.max_resolution / max(h_orig, w_orig)
            new_w = int(w_orig * scale)
            new_h = int(h_orig * scale)
            frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        outputs = self.predictor(frame_bgr)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            return [], []

        boxes = instances.pred_boxes.tensor.numpy()
        if scale != 1.0:
            boxes = boxes / scale
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        pred_dz = instances.pred_dz.numpy()

        hand_detections = []
        object_detections = []

        for i in range(len(instances)):
            cls = int(classes[i])
            bbox = boxes[i]
            conf = float(scores[i])
            dz = pred_dz[i]

            if cls == 0:
                # Hand detection
                side_idx = int(dz[5])
                side = HANDS23_SIDE_MAP.get(side_idx, HandSide.UNKNOWN)

                contact_idx = int(dz[8])
                contact_state = HANDS23_CONTACT_MAP.get(contact_idx, ContactState.NO_CONTACT)

                contact_conf = float(dz[9])

                grasp_idx = int(dz[6])
                grasp_type = HANDS23_GRASP_NAMES.get(grasp_idx)

                hand_detections.append(HandDetection(
                    bbox=bbox,
                    confidence=conf,
                    side=side,
                    contact_state=contact_state,
                    contact_state_confidence=contact_conf,
                    grasp_type=grasp_type,
                ))
            elif cls in (1, 2):
                # Object detection (first or second object)
                touch_idx = int(dz[7]) if dz[7] != 100 else None
                object_detections.append(ObjectDetection(
                    bbox=bbox,
                    confidence=conf,
                    touch_type=f"type_{touch_idx}" if touch_idx is not None else None,
                ))

        return hand_detections, object_detections

    def benchmark(self, frames: list[np.ndarray]) -> dict:
        """Benchmark inference speed."""
        times = []
        for frame in frames:
            t0 = time.perf_counter()
            self.detect(frame)
            times.append(time.perf_counter() - t0)
        times_ms = [t * 1000 for t in times]
        return {
            "model": f"Hands23 (Faster R-CNN X-101-FPN, max_res={self.max_resolution})",
            "frames": len(frames),
            "p50_ms": round(np.percentile(times_ms, 50), 2),
            "p95_ms": round(np.percentile(times_ms, 95), 2),
            "p99_ms": round(np.percentile(times_ms, 99), 2),
            "mean_ms": round(np.mean(times_ms), 2),
            "total_s": round(sum(times), 3),
        }


# ============================================================
# 100DOH Detector (previous backend — kept for reference)
# To use: from bachman_cortex.models.hand_detector_100doh import HandObjectDetector100DOH
# ============================================================
