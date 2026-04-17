"""Tests for ParticipantsAccumulator — YOLO + SCRFD + extra-hands signal."""

import numpy as np

from bachman_cortex.checks.participants import (
    ParticipantsAccumulator,
    ParticipantsThresholds,
)
from bachman_cortex.models.hand_detector import (
    ContactState,
    HandDetection,
    HandSide,
)
from bachman_cortex.models.scrfd_detector import FaceDetection
from bachman_cortex.models.yolo_detector import Detection as YOLODetection


FRAME_WH = (1280, 720)


def _yolo(bbox, conf=0.9) -> YOLODetection:
    return YOLODetection(
        bbox=np.array(bbox, dtype=np.float32),
        confidence=conf,
        class_id=0,
        class_name="person",
    )


def _face(bbox, conf=0.9) -> FaceDetection:
    return FaceDetection(
        bbox=np.array(bbox, dtype=np.float32),
        confidence=conf,
        landmarks=None,
    )


def _hand(side=HandSide.LEFT, conf=0.9, bbox=(100, 100, 200, 200)) -> HandDetection:
    return HandDetection(
        bbox=np.array(bbox, dtype=np.float32),
        confidence=conf,
        side=side,
        contact_state=ContactState.NO_CONTACT,
        contact_state_confidence=0.0,
    )


def test_high_conf_person_not_wearer_passes():
    acc = ParticipantsAccumulator()
    acc.process_frame(
        yolo_persons=[_yolo((500, 100, 700, 600), conf=0.9)],
        scrfd_faces=[],
        hands=[],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_pass == [True]
    assert r.participant_source == ["yolo"]
    assert r.participant_conf == [0.9]


def test_bottom_centre_person_filtered_as_wearer():
    # Bottom of frame, centre → treated as wearer's own torso.
    acc = ParticipantsAccumulator()
    acc.process_frame(
        yolo_persons=[_yolo((500, 620, 800, 719), conf=0.95)],
        scrfd_faces=[], hands=[],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_pass == [False]


def test_person_overlapping_hand_filtered_as_wearer():
    acc = ParticipantsAccumulator()
    hand = _hand(bbox=(400, 200, 550, 350))
    acc.process_frame(
        yolo_persons=[_yolo((390, 190, 560, 360), conf=0.9)],
        scrfd_faces=[], hands=[hand],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_pass == [False]


def test_small_bbox_filtered_by_min_height():
    acc = ParticipantsAccumulator(
        ParticipantsThresholds(min_bbox_height_frac=0.15)
    )
    # Height 50 vs frame 720 → 6.9% → too small.
    acc.process_frame(
        yolo_persons=[_yolo((500, 100, 600, 150), conf=0.9)],
        scrfd_faces=[], hands=[],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_pass == [False]


def test_scrfd_face_wearer_filter_applies():
    """Plan §1: wearer filter applies to SCRFD faces too."""
    acc = ParticipantsAccumulator()
    # Face bbox dead centre-bottom — would be filtered by bottom-anchor rule.
    acc.process_frame(
        yolo_persons=[],
        scrfd_faces=[_face((500, 630, 700, 715), conf=0.95)],
        hands=[],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_pass == [False]


def test_extra_hands_signal_triggers_on_3rd_hand():
    acc = ParticipantsAccumulator(
        ParticipantsThresholds(extra_hand_conf=0.7)
    )
    hands = [
        _hand(conf=0.95, bbox=(100, 100, 200, 200)),
        _hand(conf=0.9, bbox=(300, 100, 400, 200)),
        _hand(conf=0.85, bbox=(500, 100, 600, 200)),   # the "extra" hand
    ]
    acc.process_frame(
        yolo_persons=[], scrfd_faces=[], hands=hands,
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_pass == [True]
    assert r.participant_source == ["extra_hands"]
    assert r.extra_hands_count == [1]


def test_highest_conf_source_wins():
    """Multiple signals → best-conf source reported."""
    acc = ParticipantsAccumulator()
    acc.process_frame(
        yolo_persons=[_yolo((500, 100, 700, 600), conf=0.65)],
        scrfd_faces=[_face((520, 150, 620, 350), conf=0.95)],
        hands=[],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_source == ["scrfd"]
    assert r.participant_conf == [0.95]


def test_no_signals_fails():
    acc = ParticipantsAccumulator()
    acc.process_frame(
        yolo_persons=[], scrfd_faces=[], hands=[],
        frame_idx=0, frame_wh=FRAME_WH,
    )
    r = acc.finalize()
    assert r.participant_pass == [False]
    assert r.participant_source == [None]
    assert r.participant_conf == [0.0]
    assert r.extra_hands_count == [0]
