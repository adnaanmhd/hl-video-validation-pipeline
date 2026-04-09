"""Early stopping monitor for per-frame ML checks.

Tracks per-frame pass/fail outcomes during the inference loop and determines
when a check's final result is mathematically locked in, allowing the pipeline
to skip remaining frames.
"""

import math
from dataclasses import dataclass, field

from bachman_cortex.models.hand_detector import HandDetection, HandSide, ContactState
from bachman_cortex.models.scrfd_detector import FaceDetection
from bachman_cortex.models.yolo_detector import Detection
from bachman_cortex.models.yolo_pose_detector import PoseDetection, ALLOWED_WEARER_KEYPOINTS
from bachman_cortex.checks.pov_hand_angle import compute_hand_angle
from bachman_cortex.checks.participants import _is_wearer_body_part
from bachman_cortex.checks.body_part_visibility import _is_wearer_pose, _get_disallowed_keypoints


# ── Check specification ──────────────────────────────────────────────────────

@dataclass
class CheckSpec:
    """Defines a check for early stopping tracking."""
    name: str
    threshold: float        # pass rate needed (e.g. 0.90)
    zero_tolerance: bool    # True = fails on first failure (face, privacy)


# ── Early stop monitor ───────────────────────────────────────────────────────

class EarlyStopMonitor:
    """Tracks per-frame outcomes and determines when checks are decided."""

    def __init__(self, total_frames: int, checks: list[CheckSpec]):
        self.total_frames = total_frames
        self._checks = {c.name: c for c in checks}
        self._passes: dict[str, int] = {c.name: 0 for c in checks}
        self._failures: dict[str, int] = {c.name: 0 for c in checks}
        self._determined: dict[str, str | None] = {c.name: None for c in checks}
        self._frames_processed = 0

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    def advance_frame(self) -> None:
        """Call once per frame after all update() calls for that frame."""
        self._frames_processed += 1
        self._evaluate_all()

    def update(self, check_name: str, frame_passed: bool) -> None:
        """Record one frame's outcome for a check."""
        if self._determined[check_name] is not None:
            return  # already decided
        if frame_passed:
            self._passes[check_name] += 1
        else:
            self._failures[check_name] += 1

    def _evaluate_all(self) -> None:
        """Re-evaluate determination for all undecided checks."""
        k = self._frames_processed
        remaining = self.total_frames - k

        for name, spec in self._checks.items():
            if self._determined[name] is not None:
                continue

            p = self._passes[name]
            f = self._failures[name]

            if spec.zero_tolerance:
                # Fails on first failure, passes only after all frames clean
                if f > 0:
                    self._determined[name] = "fail"
                elif remaining == 0:
                    self._determined[name] = "pass"
            else:
                needed = math.ceil(spec.threshold * self.total_frames)
                # Guaranteed pass: already have enough passes
                if p >= needed:
                    self._determined[name] = "pass"
                # Guaranteed fail: even if all remaining pass, not enough
                elif p + remaining < needed:
                    self._determined[name] = "fail"

    def is_determined(self, check_name: str) -> bool:
        return self._determined[check_name] is not None

    def all_determined(self) -> bool:
        return all(v is not None for v in self._determined.values())

    def get_result(self, check_name: str) -> tuple[str | None, int, int]:
        """Return (status_or_None, passes, frames_processed)."""
        return (
            self._determined[check_name],
            self._passes[check_name],
            self._frames_processed,
        )


# ── Per-frame evaluation helpers ─────────────────────────────────────────────
#
# Each function replicates the per-frame logic from the corresponding check
# module, returning True if the frame "passes" for that check.


_INTERACTION_STATES = {ContactState.PORTABLE_OBJ, ContactState.STATIONARY_OBJ}


def _hand_in_frame(bbox, frame_w: int, frame_h: int, margin: int) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 > margin and y1 > margin
            and x2 < frame_w - margin and y2 < frame_h - margin)


def eval_hand_visibility(
    hands: list[HandDetection],
    confidence_threshold: float,
    frame_w: int,
    frame_h: int,
    frame_margin: int = 2,
) -> bool:
    """True if >= 2 confident hand detections fully within the frame."""
    return sum(
        1 for h in hands
        if h.confidence >= confidence_threshold
        and _hand_in_frame(h.bbox, frame_w, frame_h, frame_margin)
    ) >= 2


def eval_face_presence(
    faces: list[FaceDetection],
    confidence_threshold: float,
) -> bool:
    """True if NO face exceeds the confidence threshold (clean frame)."""
    return all(f.confidence < confidence_threshold for f in faces)


def eval_participants(
    persons: list[Detection],
    faces: list[FaceDetection],
    hands: list[HandDetection],
    frame_h: int,
    frame_w: int,
    person_conf_threshold: float,
    face_conf_threshold: float,
    min_person_height_ratio: float,
) -> bool:
    """True if no other person detected in this frame."""
    min_person_height = frame_h * min_person_height_ratio

    for p in persons:
        if p.confidence < person_conf_threshold:
            continue
        bbox_height = p.bbox[3] - p.bbox[1]
        if bbox_height < min_person_height:
            continue
        if _is_wearer_body_part(p.bbox, hands, frame_h, frame_w):
            continue
        return False  # other person detected

    for f in faces:
        if f.confidence >= face_conf_threshold:
            return False  # face = another person

    return True


def eval_hand_object_interaction(hands: list[HandDetection]) -> bool:
    """True if any hand is in contact with an object."""
    return any(h.contact_state in _INTERACTION_STATES for h in hands)


def eval_pov_hand_angle(
    hands: list[HandDetection],
    frame_h: int,
    frame_w: int,
    angle_threshold: float,
    diagonal_fov_degrees: float,
) -> bool:
    """True if all hands are within angle threshold. False if no hands."""
    if not hands:
        return False
    for hand in hands:
        angle = compute_hand_angle(hand.bbox, frame_w, frame_h, diagonal_fov_degrees)
        if angle >= angle_threshold:
            return False
    return True


def eval_body_part_visibility(
    poses: list[PoseDetection],
    hands: list[HandDetection],
    frame_h: int,
    frame_w: int,
    pose_conf_threshold: float,
    keypoint_conf_threshold: float,
) -> bool:
    """True if no disallowed body parts are visible on the wearer."""
    for pose in poses:
        if pose.confidence < pose_conf_threshold:
            continue
        if not _is_wearer_pose(pose, hands, frame_h, frame_w):
            continue
        disallowed = _get_disallowed_keypoints(pose, keypoint_conf_threshold)
        if disallowed:
            return False
    return True


