"""Tests for bachman_cortex.per_frame_store."""

import math

import pyarrow.parquet as pq
import pytest

from bachman_cortex import per_frame_store as pfs


def test_schema_columns_match_plan_section_4():
    """All §4-listed parquet columns are present in the canonical schema."""
    expected = {
        "frame_idx", "timestamp_s", "motion_jitter", "frozen_state",
        "luminance_class", "luminance_flicker", "pixelation_ratio",
        "both_hands_pass", "both_hands_conf",
        "single_hand_pass", "single_hand_conf",
        "hand_obj_pass", "hand_obj_contact",
        "hand_angle_pass", "hand_angle_mean_deg",
        "participant_pass", "participant_conf", "participant_source",
        "extra_hands_count", "obstructed",
    }
    assert set(pfs.schema().names) == expected


def test_append_row_fills_missing_columns_with_null():
    store = pfs.PerFrameStore()
    store.append_row(0, 0.0, motion_jitter=0.05, frozen_state=False)
    row = store.rows()[0]
    assert row["frame_idx"] == 0
    assert row["motion_jitter"] == 0.05
    assert row["frozen_state"] is False
    # Unset columns remain None
    assert row["luminance_class"] is None
    assert row["both_hands_pass"] is None


def test_unknown_column_raises():
    store = pfs.PerFrameStore()
    with pytest.raises(ValueError, match="bogus_col"):
        store.append_row(0, 0.0, bogus_col=True)


def test_nan_in_float_column_serialises_as_null(tmp_path):
    store = pfs.PerFrameStore()
    store.append_row(0, 0.0, hand_angle_mean_deg=float("nan"))
    store.append_row(1, 0.033, hand_angle_mean_deg=12.5)

    out = store.flush(tmp_path / "p.parquet")
    tbl = pq.read_table(out)
    angles = tbl["hand_angle_mean_deg"].to_pylist()
    assert angles[0] is None     # NaN → null
    assert angles[1] == pytest.approx(12.5)


def test_flush_roundtrip_preserves_rows(tmp_path):
    store = pfs.PerFrameStore()
    for i in range(5):
        store.append_row(
            i,
            i / 30.0,
            motion_jitter=0.01 * i if i % 2 == 0 else None,
            luminance_class=(i % 4) if i % 3 == 0 else None,
            both_hands_pass=(i == 2),
            both_hands_conf=0.9 if i == 2 else 0.0,
            participant_source="yolo" if i == 4 else None,
            obstructed=(i == 3),
        )

    out = store.flush(tmp_path / "roundtrip.parquet")
    tbl = pq.read_table(out)
    assert tbl.num_rows == 5
    assert tbl["frame_idx"].to_pylist() == [0, 1, 2, 3, 4]
    assert tbl["timestamp_s"].to_pylist()[0] == 0.0
    assert tbl["participant_source"].to_pylist()[4] == "yolo"
    assert tbl["obstructed"].to_pylist()[3] is True
    # Null-fill semantics hold across the roundtrip
    assert tbl["motion_jitter"].to_pylist()[1] is None
    assert tbl["luminance_class"].to_pylist()[1] is None


def test_flush_creates_parent_dirs(tmp_path):
    store = pfs.PerFrameStore()
    store.append_row(0, 0.0)
    nested = tmp_path / "results" / "run_001" / "video" / "v.parquet"
    store.flush(nested)
    assert nested.exists()


def test_column_accessor_returns_raw_values_including_nulls():
    store = pfs.PerFrameStore()
    store.append_row(0, 0.0, motion_jitter=0.1)
    store.append_row(1, 0.033)
    assert store.column("motion_jitter") == [0.1, None]
    assert store.column("frame_idx") == [0, 1]
