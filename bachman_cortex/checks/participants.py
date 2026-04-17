"""Participants detection + wearer filter.

Three independent signals combined into a single per-frame pass/fail:
  1. YOLO person detection (conf ≥ yolo_conf), wearer-filtered.
  2. SCRFD face detection (conf ≥ scrfd_conf), wearer-filtered.
  3. Hands23 extra hands — >2 total hands, 3rd+ hand conf ≥ extra_hand_conf.

Wearer filter (applied to YOLO persons AND SCRFD faces):
  - exclude detections overlapping a Hands23 hand (wearer's body reflected),
  - exclude detections anchored in the bottom-centre region (arm/torso
    coming in from below),
  - exclude bboxes whose height < min_bbox_height_frac of frame height.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from bachman_cortex.models.hand_detector import HandDetection
from bachman_cortex.models.scrfd_detector import FaceDetection
from bachman_cortex.models.yolo_detector import Detection as YOLODetection


@dataclass(frozen=True)
class ParticipantsThresholds:
    yolo_conf: float = 0.6
    scrfd_conf: float = 0.6
    extra_hand_conf: float = 0.7
    min_bbox_height_frac: float = 0.15
    # Wearer-filter knobs
    hand_overlap_fraction: float = 0.3
    bottom_anchor_y_frac: float = 0.85
    bottom_anchor_center_lo_frac: float = 0.2
    bottom_anchor_center_hi_frac: float = 0.8


@dataclass
class ParticipantsFinalizeResult:
    participant_pass: list[bool]
    participant_conf: list[float]
    participant_source: list[str | None]    # "yolo" | "scrfd" | "extra_hands"
    extra_hands_count: list[int]
    sample_indices: list[int]


def _is_wearer(
    bbox: np.ndarray,
    hands: list[HandDetection],
    frame_w: int,
    frame_h: int,
    th: ParticipantsThresholds,
) -> bool:
    """Heuristics to drop bboxes that likely belong to the camera wearer."""
    x1, y1, x2, y2 = bbox
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)

    # Too small → treat as noise.
    if bh < frame_h * th.min_bbox_height_frac:
        return True

    # Bottom-centre anchor rule: bottom edge well into the lower fifth,
    # centre within the middle 60%.
    centre_x = (x1 + x2) / 2
    if (y2 > frame_h * th.bottom_anchor_y_frac
            and frame_w * th.bottom_anchor_center_lo_frac
            < centre_x
            < frame_w * th.bottom_anchor_center_hi_frac):
        return True

    # Hand-overlap rule: >30% of the bbox area overlaps a detected hand.
    bbox_area = bw * bh
    for hand in hands:
        hx1, hy1, hx2, hy2 = hand.bbox
        ox1 = max(x1, hx1)
        oy1 = max(y1, hy1)
        ox2 = min(x2, hx2)
        oy2 = min(y2, hy2)
        if ox2 > ox1 and oy2 > oy1:
            overlap = (ox2 - ox1) * (oy2 - oy1)
            if overlap / bbox_area > th.hand_overlap_fraction:
                return True

    return False


@dataclass
class ParticipantsAccumulator:
    thresholds: ParticipantsThresholds = field(default_factory=ParticipantsThresholds)

    _pass: list[bool] = field(init=False, default_factory=list)
    _conf: list[float] = field(init=False, default_factory=list)
    _source: list[str | None] = field(init=False, default_factory=list)
    _extra: list[int] = field(init=False, default_factory=list)
    _sample_indices: list[int] = field(init=False, default_factory=list)

    def process_frame(
        self,
        yolo_persons: list[YOLODetection],
        scrfd_faces: list[FaceDetection],
        hands: list[HandDetection],
        frame_idx: int,
        frame_wh: tuple[int, int],
    ) -> None:
        th = self.thresholds
        frame_w, frame_h = frame_wh

        best_yolo_conf = 0.0
        for det in yolo_persons:
            if det.confidence < th.yolo_conf:
                continue
            if _is_wearer(det.bbox, hands, frame_w, frame_h, th):
                continue
            best_yolo_conf = max(best_yolo_conf, float(det.confidence))

        best_scrfd_conf = 0.0
        for face in scrfd_faces:
            if face.confidence < th.scrfd_conf:
                continue
            if _is_wearer(face.bbox, hands, frame_w, frame_h, th):
                continue
            best_scrfd_conf = max(best_scrfd_conf, float(face.confidence))

        # Extra-hands signal: Hands23 sees >2 hands, the 3rd+ hand is above
        # extra_hand_conf. Sort by confidence descending, check index 2+.
        hands_sorted = sorted(
            hands, key=lambda h: float(h.confidence), reverse=True
        )
        extra_count = 0
        best_extra_conf = 0.0
        if len(hands_sorted) > 2:
            for h in hands_sorted[2:]:
                if h.confidence >= th.extra_hand_conf:
                    extra_count += 1
                    best_extra_conf = max(best_extra_conf, float(h.confidence))

        signals = {
            "yolo": best_yolo_conf,
            "scrfd": best_scrfd_conf,
            "extra_hands": best_extra_conf,
        }
        source = max(signals, key=lambda k: signals[k])
        best_conf = signals[source]
        passes = best_conf > 0.0
        self._pass.append(passes)
        self._conf.append(best_conf if passes else 0.0)
        self._source.append(source if passes else None)
        self._extra.append(extra_count)
        self._sample_indices.append(frame_idx)

    def finalize(self) -> ParticipantsFinalizeResult:
        return ParticipantsFinalizeResult(
            participant_pass=list(self._pass),
            participant_conf=list(self._conf),
            participant_source=list(self._source),
            extra_hands_count=list(self._extra),
            sample_indices=list(self._sample_indices),
        )
