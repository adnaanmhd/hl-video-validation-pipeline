"""TOML-configurable thresholds and cadences for the scoring engine.

Schema mirrors SCORING_ENGINE_PLAN.md §7. `dump_default_toml()` is the
canonical template and must round-trip through `loads()` to an
equal `Config()` — enforced by `tests/test_config.py`.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


# ── Schema ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Cadences:
    quality_fps: float = 2.0
    motion_fps: float = 30.0       # cap
    frozen_fps: float = 10.0
    luminance_fps: float = 10.0
    pixelation_fps: float = 10.0


@dataclass(frozen=True)
class Segmentation:
    merge_threshold_s: float = 1.0


@dataclass(frozen=True)
class Decode:
    chunk_size: int = 256


@dataclass(frozen=True)
class Metadata:
    min_duration_s: float = 59.0
    min_fps: float = 28.0
    min_width: int = 1920
    min_height: int = 1080


@dataclass(frozen=True)
class Stability:
    shaky_score_threshold: float = 0.181
    trans_threshold: float = 8.0
    jump_threshold: float = 30.0
    rot_threshold: float = 0.3
    variance_threshold: float = 6.0
    w_trans: float = 0.35
    w_var: float = 0.25
    w_rot: float = 0.20
    w_jump: float = 0.20
    highpass_window_sec: float = 0.5


@dataclass(frozen=True)
class Frozen:
    max_consecutive: int = 30
    trans_threshold: float = 0.1
    rot_threshold: float = 0.001


@dataclass(frozen=True)
class Luminance:
    good_frame_ratio: float = 0.80
    dead_black_max: float = 15.0
    too_dark_max: float = 45.0
    blown_out_min: float = 230.0
    flicker_window: int = 10
    flicker_stddev_threshold: float = 30.0


@dataclass(frozen=True)
class Pixelation:
    good_frame_ratio: float = 0.80
    max_blockiness_ratio: float = 1.5
    block_size: int = 8


@dataclass(frozen=True)
class Technical:
    stability: Stability = field(default_factory=Stability)
    frozen: Frozen = field(default_factory=Frozen)
    luminance: Luminance = field(default_factory=Luminance)
    pixelation: Pixelation = field(default_factory=Pixelation)


@dataclass(frozen=True)
class Hands:
    hands23_conf: float = 0.7


@dataclass(frozen=True)
class Angle:
    max_degrees: float = 40.0


@dataclass(frozen=True)
class Participants:
    yolo_conf: float = 0.6
    scrfd_conf: float = 0.6
    extra_hand_conf: float = 0.7
    min_bbox_height_frac: float = 0.15


@dataclass(frozen=True)
class Quality:
    hands: Hands = field(default_factory=Hands)
    angle: Angle = field(default_factory=Angle)
    participants: Participants = field(default_factory=Participants)


@dataclass(frozen=True)
class Config:
    cadences: Cadences = field(default_factory=Cadences)
    segmentation: Segmentation = field(default_factory=Segmentation)
    decode: Decode = field(default_factory=Decode)
    metadata: Metadata = field(default_factory=Metadata)
    technical: Technical = field(default_factory=Technical)
    quality: Quality = field(default_factory=Quality)


# ── Loading ────────────────────────────────────────────────────────────────

def load(path: str | Path) -> Config:
    """Load a Config from a TOML file. Missing fields fall back to defaults."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return _merge_into(Config(), data)


def loads(text: str) -> Config:
    """Load a Config from a TOML string. Missing fields fall back to defaults."""
    return _merge_into(Config(), tomllib.loads(text))


def _merge_into(base: Any, overrides: dict[str, Any]) -> Any:
    """Return a new dataclass with `overrides` merged into `base`.

    Recurses into dataclass-valued fields. Unknown keys raise ValueError so
    typos surface loudly instead of being silently ignored.
    """
    known = {f.name for f in fields(base)}
    kwargs: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in known:
            raise ValueError(
                f"Unknown config key {key!r} in [{type(base).__name__.lower()}]"
            )
        current = getattr(base, key)
        if is_dataclass(current) and isinstance(value, dict):
            kwargs[key] = _merge_into(current, value)
        else:
            kwargs[key] = value
    return dataclasses.replace(base, **kwargs)


# ── Emission ───────────────────────────────────────────────────────────────

_DEFAULT_TEMPLATE = """\
# hl-score configuration — defaults.
#
# Emitted by `hl-score --dump-default-config`. Schema: SCORING_ENGINE_PLAN.md §7.
# Parsing this file with `bachman_cortex.config.load()` yields an object equal
# to `Config()` — any edit shifts only the fields you change.

[cadences]
quality_fps = 2.0
motion_fps = 30.0          # cap
frozen_fps = 10.0
luminance_fps = 10.0
pixelation_fps = 10.0

[segmentation]
merge_threshold_s = 1.0

[decode]
chunk_size = 256

[metadata]
min_duration_s = 59.0
min_fps = 28.0
min_width = 1920
min_height = 1080

[technical.stability]
shaky_score_threshold = 0.181
trans_threshold = 8.0
jump_threshold = 30.0
rot_threshold = 0.3
variance_threshold = 6.0
w_trans = 0.35
w_var = 0.25
w_rot = 0.20
w_jump = 0.20
highpass_window_sec = 0.5

[technical.frozen]
max_consecutive = 30
trans_threshold = 0.1
rot_threshold = 0.001

[technical.luminance]
good_frame_ratio = 0.80
dead_black_max = 15.0
too_dark_max = 45.0
blown_out_min = 230.0
flicker_window = 10
flicker_stddev_threshold = 30.0

[technical.pixelation]
good_frame_ratio = 0.80
max_blockiness_ratio = 1.5
block_size = 8

[quality.hands]
hands23_conf = 0.7

[quality.angle]
max_degrees = 40.0

[quality.participants]
yolo_conf = 0.6
scrfd_conf = 0.6
extra_hand_conf = 0.7
min_bbox_height_frac = 0.15
"""


def dump_default_toml() -> str:
    """Return the canonical default-config TOML template.

    Backs `hl-score --dump-default-config`. Round-trips through `loads()` to
    `Config()` — tested in `tests/test_config.py::test_default_roundtrip`.
    """
    return _DEFAULT_TEMPLATE
