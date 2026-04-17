"""`hl-score` command-line entrypoint.

Usage:
    hl-score path/to/video.mp4
    hl-score path/to/dir/
    hl-score path/to/dir/ --config ./hl-score.toml
    hl-score --dump-default-config > hl-score.toml

Only top-level knobs are exposed on the command line; fine-grained
thresholds live in the TOML config (see SCORING_ENGINE_PLAN.md §7).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bachman_cortex import config as cfg_mod
from bachman_cortex.batch import BatchOptions, auto_worker_count, score_batch


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hl-score",
        description="Score egocentric videos against acceptance criteria.",
    )
    p.add_argument(
        "input",
        nargs="?",
        help="Single MP4 file or directory to walk recursively.",
    )
    p.add_argument(
        "--out-dir",
        default="results",
        help="Output root. run_NNN is created under this (default: results/).",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config with cadences + thresholds.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Reserved. Auto-detected if omitted.",
    )
    p.add_argument(
        "--hand-detector-repo",
        default=None,
        help="Override Hands23 repo/weights path.",
    )
    p.add_argument(
        "--scrfd-root",
        default=None,
        help="Override SCRFD weights root.",
    )
    p.add_argument(
        "--yolo-model",
        default=None,
        help="Override YOLO weights path.",
    )
    p.add_argument(
        "--dump-default-config",
        action="store_true",
        help="Write the default-config TOML template to stdout and exit.",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Log per-video progress at INFO level.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.dump_default_config:
        sys.stdout.write(cfg_mod.dump_default_toml())
        return 0

    if not args.input:
        parser.error("INPUT is required unless --dump-default-config is set.")
        return 2

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.config:
        cfg = cfg_mod.load(args.config)
    else:
        cfg = cfg_mod.Config()

    if args.workers is None:
        args.workers = auto_worker_count()

    options = BatchOptions(
        out_root=args.out_dir,
        config=cfg,
        workers=args.workers,
        hand_detector_repo=args.hand_detector_repo,
        scrfd_root=args.scrfd_root,
        yolo_model=args.yolo_model,
    )
    batch, run_dir = score_batch([Path(args.input)], options=options)

    print(f"run_dir: {run_dir}")
    print(f"video_count: {batch.video_count}")
    print(f"errors: {len(batch.errors)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
