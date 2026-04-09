"""Hand-Object detection using 100DOH (Understanding Human Hands in Contact at Internet Scale).

The 100DOH model is a Faster R-CNN (ResNet-101) trained on 100K+ egocentric images.
It detects hands and objects, classifying:
- Hand side (left/right)
- Contact state: N (no contact), S (self), O (other person), P (portable object), F (stationary)
- Offset vectors from hand to interacted object

Setup:
    python bachman_cortex/models/download_models.py --all

Runtime (macOS):
    The C extension links against libtorch. On macOS, set DYLD_LIBRARY_PATH *before* launching Python:
        export DYLD_LIBRARY_PATH=$(python -c "import torch; print(torch.__file__.replace('__init__.py','lib'))")
    On Linux this is typically not needed.
"""

import os
import sys
import platform
import numpy as np
import torch
import cv2
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


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
    offset_vector: np.ndarray | None = None  # unit vector + magnitude to interacted object


@dataclass
class ObjectDetection:
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float


def _resolve_repo_dir(repo_dir: str | Path) -> Path:
    """Resolve the 100DOH repo directory, checking it exists and is set up."""
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(
            f"100DOH repo not found at {repo_dir}. "
            f"Run: python bachman_cortex/models/download_models.py --all"
        )
    weight_file = repo_dir / "models" / "res101_handobj_100K" / "pascal_voc" / "faster_rcnn_1_8_132028.pth"
    if not weight_file.exists():
        raise FileNotFoundError(
            f"100DOH weights not found at {weight_file}. "
            f"Run: python bachman_cortex/models/download_models.py --all"
        )
    so_files = list((repo_dir / "lib" / "model").glob("_C*.so")) + \
               list((repo_dir / "lib" / "model").glob("_C*.pyd"))
    if not so_files:
        raise FileNotFoundError(
            f"100DOH C extension not compiled. "
            f"Run: cd {repo_dir / 'lib'} && python setup.py build develop"
        )
    return repo_dir


def _check_torch_lib_path():
    """Verify torch shared libraries are findable at runtime (macOS issue)."""
    if platform.system() != "Darwin":
        return  # Linux/Windows typically don't need this

    torch_lib = str(Path(torch.__file__).parent / "lib")
    dyld = os.environ.get("DYLD_LIBRARY_PATH", "")

    if torch_lib not in dyld:
        # Try to load the extension anyway — it may work if rpath is set
        try:
            import ctypes
            ctypes.CDLL(str(Path(torch_lib) / "libc10.dylib"))
        except OSError:
            raise RuntimeError(
                f"torch libraries not found at runtime. On macOS, set DYLD_LIBRARY_PATH "
                f"*before* launching Python:\n\n"
                f"  export DYLD_LIBRARY_PATH=$(python -c \"import torch; "
                f"print(torch.__file__.replace('__init__.py','lib'))\")\n\n"
                f"Then re-run your command."
            )


class HandObjectDetector100DOH:
    """100DOH hand-object detector wrapper.

    Loads the ResNet-101 Faster R-CNN model trained on 100K+ego dataset.
    Returns hand detections with contact state and object detections.
    """

    def __init__(
        self,
        repo_dir: str | Path = "bachman_cortex/models/weights/hand_object_detector",
        weight_file: str = "models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth",
        thresh_hand: float = 0.5,
        thresh_obj: float = 0.5,
    ):
        self.repo_dir = _resolve_repo_dir(repo_dir)
        self.thresh_hand = thresh_hand
        self.thresh_obj = thresh_obj

        # Verify torch libs are loadable (fails fast with clear message on macOS)
        _check_torch_lib_path()

        # Add 100DOH lib to path so `model._C`, `model.utils.config`, etc. are importable
        lib_dir = str(self.repo_dir / "lib")
        repo_str = str(self.repo_dir)
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        import _init_paths  # noqa: F401 — sets up additional paths within the 100DOH repo
        from model.utils.config import cfg, cfg_from_file, cfg_from_list
        from model.faster_rcnn.resnet import resnet
        from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
        from model.roi_layers import nms
        from model.utils.blob import im_list_to_blob

        self._cfg = cfg
        self._bbox_transform_inv = bbox_transform_inv
        self._clip_boxes = clip_boxes
        self._nms = nms
        self._im_list_to_blob = im_list_to_blob

        # Configure
        cfg_from_file(str(self.repo_dir / "cfgs" / "res101.yml"))
        cfg_from_list(["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"])
        cfg.USE_GPU_NMS = False

        # Build and load model
        pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
        self.classes = pascal_classes

        self.model = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
        self.model.create_architecture()

        load_path = str(self.repo_dir / weight_file)
        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if "pooling_mode" in checkpoint:
            cfg.POOLING_MODE = checkpoint["pooling_mode"]

        self.model.eval()

        # Pre-allocate tensors
        self._im_data = torch.FloatTensor(1)
        self._im_info = torch.FloatTensor(1)
        self._num_boxes = torch.LongTensor(1)
        self._gt_boxes = torch.FloatTensor(1)
        self._box_info = torch.FloatTensor(1)

    def _get_image_blob(self, im: np.ndarray):
        """Convert image to network input blob."""
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self._cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self._cfg.TEST.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self._cfg.TEST.MAX_SIZE:
                im_scale = float(self._cfg.TEST.MAX_SIZE) / float(im_size_max)
            resized = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                                 interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(resized)

        blob = self._im_list_to_blob(processed_ims)
        return blob, np.array(im_scale_factors)

    def detect(self, frame_bgr: np.ndarray) -> tuple[list[HandDetection], list[ObjectDetection]]:
        """Detect hands and objects in a BGR frame.

        Returns:
            Tuple of (hand_detections, object_detections).
        """
        blobs, im_scales = self._get_image_blob(frame_bgr)
        im_blob = blobs
        im_info_np = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32,
        )

        im_data_pt = torch.from_numpy(im_blob).permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)

        with torch.no_grad():
            self._im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            self._im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            self._gt_boxes.resize_(1, 1, 5).zero_()
            self._num_boxes.resize_(1).zero_()
            self._box_info.resize_(1, 1, 5).zero_()

            rois, cls_prob, bbox_pred, _, _, _, _, _, loss_list = self.model(
                self._im_data, self._im_info, self._gt_boxes,
                self._num_boxes, self._box_info,
            )

        scores = cls_prob.data.squeeze()
        boxes = rois.data[:, :, 1:5]

        # Extract predicted params
        contact_vector = loss_list[0][0]
        offset_vector = loss_list[1][0].detach()
        lr_vector = loss_list[2][0].detach()

        # Contact state (argmax of 5-class softmax)
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # Hand side (sigmoid > 0.5 = right)
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        # Apply bbox regression
        box_deltas = bbox_pred.data
        if self._cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                self._cfg.TRAIN.BBOX_NORMALIZE_STDS
            ) + torch.FloatTensor(self._cfg.TRAIN.BBOX_NORMALIZE_MEANS)
            box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))

        pred_boxes = self._bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = self._clip_boxes(pred_boxes, self._im_info.data, 1)
        pred_boxes /= im_scales[0]

        pred_boxes = pred_boxes.squeeze()

        # Parse detections
        hand_detections = []
        object_detections = []

        for j in range(1, len(self.classes)):
            cls_name = self.classes[j]
            thresh = self.thresh_hand if cls_name == "hand" else self.thresh_obj

            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() == 0:
                continue

            cls_scores = scores[:, j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

            cls_dets = torch.cat(
                (cls_boxes, cls_scores.unsqueeze(1),
                 contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]),
                1,
            )
            cls_dets = cls_dets[order]
            keep = self._nms(cls_boxes[order, :], cls_scores[order], self._cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()].cpu().numpy()

            if cls_name == "hand":
                for det in cls_dets:
                    bbox = det[:4]
                    conf = float(det[4])
                    contact_idx = int(det[5])
                    offset = det[6:9]  # unit vector (3 values)
                    is_right = bool(det[-1] > 0.5) if len(det) > 10 else True

                    hand_detections.append(HandDetection(
                        bbox=bbox,
                        confidence=conf,
                        side=HandSide.RIGHT if is_right else HandSide.LEFT,
                        contact_state=ContactState(min(contact_idx, 4)),
                        offset_vector=offset,
                    ))
            elif cls_name == "targetobject":
                for det in cls_dets:
                    object_detections.append(ObjectDetection(
                        bbox=det[:4],
                        confidence=float(det[4]),
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
            "model": "100DOH hand_object_detector (ResNet-101)",
            "frames": len(frames),
            "p50_ms": round(np.percentile(times_ms, 50), 2),
            "p95_ms": round(np.percentile(times_ms, 95), 2),
            "p99_ms": round(np.percentile(times_ms, 99), 2),
            "mean_ms": round(np.mean(times_ms), 2),
            "total_s": round(sum(times), 3),
        }
