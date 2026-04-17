"""Tests for bachman_cortex.config."""

import tomllib

import pytest

from bachman_cortex import config as cfg


def test_default_roundtrip():
    """Dumped default template must parse back to an equal Config()."""
    text = cfg.dump_default_toml()
    parsed = cfg.loads(text)
    assert parsed == cfg.Config()


def test_default_template_is_valid_toml():
    """The emitted template parses cleanly with stdlib tomllib."""
    tomllib.loads(cfg.dump_default_toml())


def test_partial_override_preserves_defaults():
    """Overriding one field leaves siblings and peer sections untouched."""
    text = """
    [cadences]
    quality_fps = 4.0

    [technical.luminance]
    good_frame_ratio = 0.9
    """
    parsed = cfg.loads(text)

    assert parsed.cadences.quality_fps == 4.0
    assert parsed.cadences.motion_fps == 30.0
    assert parsed.technical.luminance.good_frame_ratio == 0.9
    assert parsed.technical.luminance.dead_black_max == 15.0
    assert parsed.technical.stability.shaky_score_threshold == 0.181


def test_file_load_roundtrip(tmp_path):
    """load(path) yields the same object as loads(text) on matching content."""
    text = cfg.dump_default_toml()
    path = tmp_path / "hl-score.toml"
    path.write_text(text)

    assert cfg.load(path) == cfg.loads(text) == cfg.Config()


def test_unknown_top_level_key_raises():
    with pytest.raises(ValueError, match="bogus_section"):
        cfg.loads("[bogus_section]\nx = 1\n")


def test_unknown_nested_key_raises():
    text = """
    [technical.stability]
    shaky_score_threshold = 0.2
    typo_threshold = 1.0
    """
    with pytest.raises(ValueError, match="typo_threshold"):
        cfg.loads(text)
