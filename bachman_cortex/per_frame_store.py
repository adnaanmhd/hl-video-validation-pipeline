"""Per-frame row buffer and single-shot parquet writer.

Buffers one row per native frame in memory during decode and writes the
whole table in a single `flush()` call at the end. See plan §3 invariants
and §4 parquet schema.

Why not streaming writes: the row count is bounded
(~`native_fps * duration`), typical worst case ~10 MB, and a single-shot
write eliminates any chunk-boundary reasoning for rolling windows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


# Ordered column schema for the parquet artefact. Order also determines
# the column order in the emitted file.
_SCHEMA: pa.Schema = pa.schema([
    pa.field("frame_idx", pa.int32()),
    pa.field("timestamp_s", pa.float64()),
    pa.field("motion_jitter", pa.float32()),
    pa.field("frozen_state", pa.bool_()),
    pa.field("luminance_class", pa.int8()),
    pa.field("luminance_flicker", pa.bool_()),
    pa.field("pixelation_ratio", pa.float32()),
    pa.field("both_hands_pass", pa.bool_()),
    pa.field("both_hands_conf", pa.float32()),
    pa.field("single_hand_pass", pa.bool_()),
    pa.field("single_hand_conf", pa.float32()),
    pa.field("hand_obj_pass", pa.bool_()),
    pa.field("hand_obj_contact", pa.string()),
    pa.field("hand_angle_pass", pa.bool_()),
    pa.field("hand_angle_mean_deg", pa.float32()),
    pa.field("participant_pass", pa.bool_()),
    pa.field("participant_conf", pa.float32()),
    pa.field("participant_source", pa.string()),
    pa.field("extra_hands_count", pa.int8()),
    pa.field("obstructed", pa.bool_()),
])

# Columns that must be present on every row (never null).
_REQUIRED_COLS: tuple[str, ...] = ("frame_idx", "timestamp_s")

_NULLABLE_COLS: tuple[str, ...] = tuple(
    f.name for f in _SCHEMA if f.name not in _REQUIRED_COLS
)


@dataclass
class PerFrameStore:
    """In-memory buffer of per-native-frame rows, flushed once at end.

    Use `append_row(frame_idx, timestamp_s, **cadence_values)` for each
    native frame. Missing cadence values are left null. Call
    `flush(path)` exactly once to write the parquet.
    """

    _rows: list[dict[str, Any]] = field(default_factory=list, repr=False)

    def append_row(
        self,
        frame_idx: int,
        timestamp_s: float,
        **values: Any,
    ) -> None:
        """Append one row. Unknown keys raise immediately to surface typos."""
        unknown = set(values.keys()) - set(_NULLABLE_COLS)
        if unknown:
            raise ValueError(
                f"PerFrameStore: unknown column(s) {sorted(unknown)!r}; "
                f"valid optional columns are {list(_NULLABLE_COLS)}"
            )
        row: dict[str, Any] = {
            "frame_idx": int(frame_idx),
            "timestamp_s": float(timestamp_s),
        }
        for col in _NULLABLE_COLS:
            row[col] = values.get(col)
        self._rows.append(row)

    def __len__(self) -> int:
        return len(self._rows)

    def rows(self) -> list[dict[str, Any]]:
        """Read-only access — for segmentation at report time."""
        return self._rows

    def column(self, name: str) -> list[Any]:
        """Return the raw column as a list (preserves nulls)."""
        if name not in _SCHEMA.names:
            raise KeyError(name)
        return [r.get(name) for r in self._rows]

    def to_table(self) -> pa.Table:
        """Materialise the row buffer as an Arrow Table."""
        columns: dict[str, list[Any]] = {name: [] for name in _SCHEMA.names}
        for row in self._rows:
            for name in _SCHEMA.names:
                columns[name].append(row.get(name))
        # float NaN → null for float columns so Arrow statistics stay clean.
        for name in ("motion_jitter", "pixelation_ratio",
                     "both_hands_conf", "single_hand_conf",
                     "hand_angle_mean_deg", "participant_conf"):
            columns[name] = [None if (v is not None and isinstance(v, float) and math.isnan(v))
                             else v for v in columns[name]]
        arrays: list[pa.Array] = []
        for f in _SCHEMA:
            arrays.append(pa.array(columns[f.name], type=f.type))
        return pa.Table.from_arrays(arrays, schema=_SCHEMA)

    def flush(self, path: str | Path) -> Path:
        """Write the buffered rows to `path` as parquet. Returns the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        table = self.to_table()
        pq.write_table(table, out, compression="zstd")
        return out


def schema() -> pa.Schema:
    """Expose the canonical schema for tests / downstream readers."""
    return _SCHEMA
