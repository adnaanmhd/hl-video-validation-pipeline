#!/usr/bin/env python
"""Run the validation pipeline on a batch of videos.

Usage:
    # Single video
    python -m bachman_cortex.run_batch /path/to/video.mp4

    # Multiple videos
    python -m bachman_cortex.run_batch /path/to/video1.mp4 /path/to/video2.mp4

    # Directory of videos
    python -m bachman_cortex.run_batch /path/to/videos/

    # With options
    python -m bachman_cortex.run_batch /path/to/videos/ --fps 2 --max-frames 50 --no-gdino --output results/
"""

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from bachman_cortex.pipeline import ValidationPipeline, PipelineConfig
from bachman_cortex.utils.video_metadata import get_video_metadata


# Check display names grouped by category
CHECK_CATEGORIES = [
    ("Video Metadata", [
        ("meta_format", "Format"),
        ("meta_encoding", "Encoding"),
        ("meta_resolution", "Resolution"),
        ("meta_frame_rate", "Frame Rate"),
        ("meta_duration", "Duration"),
        ("meta_orientation", "Orientation"),
    ]),
    ("Luminance & Blur", [
        ("luminance_blur", "Luminance & Blur"),
    ]),
    ("Motion Analysis", [
        ("motion_camera_stability", "Camera Stability"),
        ("motion_frozen_segments", "Frozen Segments"),
    ]),
    ("ML Detection", [
        ("ml_face_presence", "Face Presence"),
        ("ml_participants", "Participants"),
        ("ml_hand_visibility", "Hand Visibility"),
        ("ml_hand_object_interaction", "Hand-Object Interaction"),
        ("ml_privacy_safety", "Privacy Safety"),
        ("ml_view_obstruction", "View Obstruction"),
        ("ml_pov_hand_angle", "POV-Hand Angle"),
        ("ml_body_part_visibility", "Body Part Visibility"),
    ]),
]


def collect_videos(paths: list[str]) -> list[Path]:
    """Collect .mp4 video files from paths (files or directories)."""
    videos = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() == ".mp4":
            videos.append(path)
        elif path.is_dir():
            videos.extend(sorted(path.glob("*.mp4")))
            videos.extend(sorted(path.glob("*.MP4")))
    # Deduplicate preserving order
    seen = set()
    unique = []
    for v in videos:
        resolved = v.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(v)
    return unique


def _status_icon(status: str) -> str:
    """Map status string to display label."""
    return {
        "pass": "PASS",
        "fail": "FAIL",
        "review": "REVIEW",
        "skipped": "SKIPPED",
    }.get(status, status.upper())


# Acceptance conditions and actual-value extractors per check key
ACCEPTANCE_CONDITIONS = {
    "meta_format": "MP4 container (MPEG-4)",
    "meta_encoding": "H.264 video codec",
    "meta_resolution": ">= 1920 x 1080 pixels",
    "meta_frame_rate": ">= 28 FPS",
    "meta_duration": ">= 180 seconds",
    "meta_orientation": "Rotation = 0 or 180 degrees and width > height",
    "luminance_blur": "(Accept + review) frames >= 80% and brightness std dev <= 60",
    "motion_camera_stability": "Two-stage LK shakiness score <= 0.30",
    "motion_frozen_segments": "No > 30 consecutive frames with SSIM > 0.99",
    "ml_face_presence": "Face detection confidence < 0.8 in all frames",
    "ml_participants": "Persons detected <= 1 in >= 90% frames",
    "ml_hand_visibility": ">= 90% frames with both hands confidence >= 0.7",
    "ml_hand_object_interaction": "Interaction detected in >= 70% frames",
    "ml_privacy_safety": "Sensitive object detections = 0 in all frames",
    "ml_view_obstruction": "<= 10% frames obstructed",
    "ml_pov_hand_angle": "Hands within 40 deg of center in >= 80% frames",
    "ml_body_part_visibility": "Only hands/forearms (up to elbows) in >= 90% frames",
}


def _format_actual(key: str, details: dict) -> str:
    """Extract a human-readable actual result from a check's details dict."""
    if not details:
        return "-"
    d = details

    formatters = {
        "meta_format": lambda: d.get("container_format", "-"),
        "meta_encoding": lambda: d.get("video_codec", "-"),
        "meta_resolution": lambda: f"{d.get('width', '?')} x {d.get('height', '?')}",
        "meta_frame_rate": lambda: f"{d.get('fps', '?')} FPS",
        "meta_duration": lambda: f"{d.get('duration_s', '?')}s",
        "meta_orientation": lambda: f"rotation={d.get('rotation', '?')}, {d.get('width', '?')} x {d.get('height', '?')}",
        "luminance_blur": lambda: f"{d.get('good_ratio', 0):.0%} good ({d.get('accept_frames', 0)} accept + {d.get('review_frames', 0)} review + {d.get('reject_frames', 0)} reject), brightness std={d.get('brightness_std', '?')}",
        "motion_camera_stability": lambda: f"score={d.get('overall_score', '?')} ({d.get('shaky_seconds_count', 0)} shaky / {d.get('total_seconds', '?')}s, stage2={d.get('stage2_seconds_flagged', 0)}s deep-analysed)",
        "motion_frozen_segments": lambda: f"Longest frozen run = {d.get('longest_frozen_run', '?')} frames ({d.get('frozen_duration_s', '?')}s)",
        "ml_face_presence": lambda: f"{d.get('frames_with_prominent_face', '?')} frames with face >= 0.8 (max conf = {d.get('max_face_confidence_seen', '?')})",
        "ml_participants": lambda: f"{d.get('clean_frames', '?')}/{d.get('total_frames', '?')} clean frames ({d.get('clean_frames', 0) / max(d.get('total_frames', 1), 1):.1%})",
        "ml_hand_visibility": lambda: f"{d.get('both_hands_frames', '?')}/{d.get('total_frames', '?')} frames with both hands ({d.get('both_hands_frames', 0) / max(d.get('total_frames', 1), 1):.1%})",
        "ml_hand_object_interaction": lambda: f"{d.get('interaction_frames', '?')}/{d.get('total_frames', '?')} frames with interaction ({d.get('interaction_frames', 0) / max(d.get('total_frames', 1), 1):.1%})",
        "ml_privacy_safety": lambda: f"{d.get('frames_with_sensitive_objects', '?')} frames with sensitive objects",
        "ml_view_obstruction": lambda: f"{d.get('obstructed_frames', '?')}/{d.get('total_frames', '?')} obstructed ({d.get('obstructed_ratio', 0):.1%})",
        "ml_pov_hand_angle": lambda: f"{d.get('passing_frames', '?')}/{d.get('total_frames', '?')} frames within angle (mean = {d.get('mean_angle', '?')} deg, max = {d.get('max_angle', '?')} deg)",
        "ml_body_part_visibility": lambda: f"{d.get('clean_frames', '?')}/{d.get('total_frames', '?')} clean frames ({d.get('clean_frames', 0) / max(d.get('total_frames', 1), 1):.1%})" + (f" — flagged: {', '.join(f'{k}({v})' for k, v in d['flagged_body_parts'].items())}" if d.get('flagged_body_parts') else ""),
    }

    formatter = formatters.get(key)
    if formatter:
        try:
            return formatter()
        except Exception:
            return str(d)
    return str(d)


def write_report(
    all_results: list[dict],
    output_dir: Path,
    config: PipelineConfig,
):
    """Write a markdown batch report with detailed per-check results."""
    report_path = output_dir / "batch_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Batch Pipeline Report",
        "",
        f"**Generated:** {timestamp}  ",
        f"**Videos processed:** {len(all_results)}  ",
        f"**Sampling FPS:** {config.sampling_fps}  ",
        f"**Grounding DINO:** {'enabled' if config.run_grounding_dino else 'disabled'}",
        "",
        "---",
        "",
    ]

    # ── Summary table ─────────────────────────────────────────
    lines.append("## Summary")
    lines.append("")
    lines.append("| # | Video | Passed | Review | Failed | Skipped | Status | Time |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for idx, entry in enumerate(all_results, 1):
        name = entry["filename"]
        if "error" in entry:
            lines.append(f"| {idx} | {name} | - | - | - | - | ERROR | - |")
            continue

        checks = entry["results"]
        passed = sum(1 for r in checks.values() if r["status"] == "pass")
        review = sum(1 for r in checks.values() if r["status"] == "review")
        failed = sum(1 for r in checks.values() if r["status"] == "fail")
        skipped = sum(1 for r in checks.values() if r["status"] == "skipped")

        if failed > 0:
            status = "FAIL"
        elif review > 0:
            status = "REVIEW"
        elif skipped > 0:
            status = "METADATA FAIL"
        else:
            status = "ALL PASS"

        proc_time = f"{entry['processing_time_s']:.1f}s"
        lines.append(f"| {idx} | {name} | {passed} | {review} | {failed} | {skipped} | {status} | {proc_time} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Per-video detailed results ────────────────────────────
    for entry in all_results:
        name = entry["filename"]
        lines.append(f"## {name}")
        lines.append("")

        if "error" in entry:
            lines.append(f"**Error:** {entry['error']}")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        # Video info header
        meta = entry["video_meta"]
        lines.append(
            f"**File:** {meta['file_size_mb']} MB | "
            f"**Resolution:** {meta['width']}x{meta['height']} | "
            f"**FPS:** {meta['fps']} | "
            f"**Duration:** {meta['duration_s']}s | "
            f"**Codec:** {meta['video_codec']} | "
            f"**Rotation:** {meta['rotation']}"
        )
        lines.append(
            f"**Frames sampled:** {entry.get('frames_sampled', 'N/A')} | "
            f"**Processing time:** {entry['processing_time_s']:.1f}s"
        )
        lines.append("")

        # Detailed check table grouped by category
        for category_name, check_list in CHECK_CATEGORIES:
            category_checks = [
                (key, display) for key, display in check_list
                if key in entry["results"]
            ]
            if not category_checks:
                continue

            lines.append(f"### {category_name}")
            lines.append("")
            lines.append("| Check | Status | Acceptance Condition | Actual Result |")
            lines.append("|---|---|---|---|")

            for key, display in category_checks:
                r = entry["results"][key]
                status = _status_icon(r["status"])
                acceptance = ACCEPTANCE_CONDITIONS.get(key, "-")

                if r["status"] == "skipped":
                    actual = "Skipped (metadata failed)"
                else:
                    actual = _format_actual(key, r.get("details", {}))

                lines.append(f"| {display} | **{status}** | {acceptance} | {actual} |")

            lines.append("")

        lines.append("---")
        lines.append("")

    report_content = "\n".join(lines)
    report_path.write_text(report_content)
    return report_path


def _get_display_name(key: str) -> str:
    """Look up display name for a check key."""
    for _, checks in CHECK_CATEGORIES:
        for k, display in checks:
            if k == key:
                return display
    return key


def _auto_detect_workers() -> int:
    """Determine worker count based on available resources."""
    cpu_count = os.cpu_count() or 1
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            # ~3 GB VRAM per worker for all models
            return max(1, min(int(vram_gb // 3), cpu_count // 2, 4))
    except ImportError:
        pass
    # CPU-only: conservative (each worker loads all models, ~4-6 GB RAM)
    return max(1, min(cpu_count // 4, 4))


def _process_video_worker(args_tuple: tuple) -> dict:
    """Process a single video in a worker process.

    Each worker creates its own pipeline instance and loads models independently.
    """
    video_path_str, config_dict, output_dir_str = args_tuple
    video_path = Path(video_path_str)
    output_dir = Path(output_dir_str)

    # Reconstruct config (tuples become lists via asdict, fix lk_win_size)
    if "stability_lk_win_size" in config_dict and isinstance(config_dict["stability_lk_win_size"], list):
        config_dict["stability_lk_win_size"] = tuple(config_dict["stability_lk_win_size"])
    config = PipelineConfig(**config_dict)
    pipeline = ValidationPipeline(config)

    entry = {
        "filename": video_path.name,
        "path": str(video_path),
    }

    try:
        metadata = get_video_metadata(str(video_path))
        entry["video_meta"] = metadata

        t0 = time.perf_counter()
        results = pipeline.process_video(str(video_path))
        elapsed = time.perf_counter() - t0

        entry["processing_time_s"] = round(elapsed, 2)

        lb_result = results.get("luminance_blur")
        if lb_result and lb_result.details:
            entry["frames_sampled"] = lb_result.details.get("total_frames", "N/A")
        else:
            entry["frames_sampled"] = "N/A (metadata failed)"

        entry["results"] = {}
        for name, r in results.items():
            entry["results"][name] = {
                "status": r.status,
                "metric_value": r.metric_value,
                "confidence": r.confidence,
                "details": r.details,
            }

    except Exception as e:
        entry["error"] = str(e)
        entry["processing_time_s"] = 0.0
        traceback.print_exc()

    # Save per-video JSON
    video_json = output_dir / f"{video_path.stem}.json"
    with open(video_json, "w") as f:
        json.dump(entry, f, indent=2, default=str)

    return entry


def main():
    parser = argparse.ArgumentParser(
        description="Run validation pipeline on a batch of videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bachman_cortex.run_batch video.mp4
  python -m bachman_cortex.run_batch videos_dir/
  python -m bachman_cortex.run_batch *.mp4 --output results/
  python -m bachman_cortex.run_batch videos/ --fps 2 --max-frames 100 --no-gdino
        """,
    )
    parser.add_argument("paths", nargs="+", help="Video files or directories to process (.mp4 only)")
    parser.add_argument("--output", "-o", default="bachman_cortex/results", help="Output directory for reports (default: bachman_cortex/results)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate in FPS (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to sample per video (default: no limit)")
    parser.add_argument("--no-gdino", action="store_true", help="Disable Grounding DINO (faster, but no fine-grained privacy detection)")
    parser.add_argument("--hand-detector-repo", default="bachman_cortex/models/weights/hands23_detector", help="Path to Hands23 repo")
    parser.add_argument("--scrfd-root", default="bachman_cortex/models/weights/insightface", help="Path to InsightFace models")
    parser.add_argument("--gdino-cache", default="bachman_cortex/models/weights/grounding_dino", help="Path to Grounding DINO cache")
    parser.add_argument("--fail-fast", action="store_true", default=False, help="Skip ML inference when quality checks fail")
    parser.add_argument("--workers", type=int, default=0, help="Parallel video workers (0=auto-detect, 1=sequential)")
    parser.add_argument("--yolo-model", default="yolo11s.pt", help="YOLO model for object detection (default: yolo11s.pt)")

    args = parser.parse_args()

    # Collect videos
    videos = collect_videos(args.paths)
    if not videos:
        print("No .mp4 video files found.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to process.")
    for v in videos:
        print(f"  - {v.name} ({os.path.getsize(v) / 1024 / 1024:.1f} MB)")

    # Setup output directory with sequential run numbering
    base_output_dir = Path(args.output)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Determine next run number
    existing_runs = [
        d for d in base_output_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    existing_numbers = []
    for d in existing_runs:
        try:
            existing_numbers.append(int(d.name.split("_", 1)[1]))
        except (ValueError, IndexError):
            pass
    next_run = max(existing_numbers, default=0) + 1
    output_dir = base_output_dir / f"run_{next_run:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline
    config = PipelineConfig(
        sampling_fps=args.fps,
        max_frames=args.max_frames,
        run_grounding_dino=not args.no_gdino,
        scrfd_root=args.scrfd_root,
        hand_detector_repo=args.hand_detector_repo,
        gdino_cache=args.gdino_cache,
        fail_fast=args.fail_fast,
        yolo_model=args.yolo_model,
    )

    # Determine worker count
    workers = args.workers if args.workers > 0 else _auto_detect_workers()
    print(f"Using {workers} worker(s)")

    # Process videos
    all_results = []

    if workers <= 1:
        # Sequential: single pipeline instance, models loaded once
        pipeline = ValidationPipeline(config)

        for i, video_path in enumerate(videos):
            print(f"\n{'='*70}")
            print(f"[{i+1}/{len(videos)}] {video_path.name}")
            print(f"{'='*70}")

            entry = {
                "filename": video_path.name,
                "path": str(video_path),
            }

            try:
                metadata = get_video_metadata(str(video_path))
                entry["video_meta"] = metadata

                t0 = time.perf_counter()
                results = pipeline.process_video(str(video_path))
                elapsed = time.perf_counter() - t0

                entry["processing_time_s"] = round(elapsed, 2)

                lb_result = results.get("luminance_blur")
                if lb_result and lb_result.details:
                    entry["frames_sampled"] = lb_result.details.get("total_frames", "N/A")
                else:
                    entry["frames_sampled"] = "N/A (metadata failed)"

                entry["results"] = {}
                for name, r in results.items():
                    entry["results"][name] = {
                        "status": r.status,
                        "metric_value": r.metric_value,
                        "confidence": r.confidence,
                        "details": r.details,
                    }

            except Exception as e:
                entry["error"] = str(e)
                entry["processing_time_s"] = 0.0
                traceback.print_exc()

            all_results.append(entry)

            video_json = output_dir / f"{video_path.stem}.json"
            with open(video_json, "w") as f:
                json.dump(entry, f, indent=2, default=str)
    else:
        # Parallel: each worker loads its own pipeline and models
        config_dict = dataclasses.asdict(config)
        work_items = [
            (str(v), config_dict, str(output_dir)) for v in videos
        ]

        print(f"Processing {len(videos)} videos with {workers} parallel workers...")
        with Pool(processes=workers) as pool:
            for i, entry in enumerate(pool.imap_unordered(_process_video_worker, work_items)):
                all_results.append(entry)
                status = "error" if "error" in entry else "done"
                t = entry.get("processing_time_s", 0)
                print(f"  [{i+1}/{len(videos)}] {entry['filename']} ({status}, {t:.1f}s)")

    # Write batch report
    report_path = write_report(all_results, output_dir, config)
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(all_results)} videos processed")
    print(f"Run directory: {output_dir}/")
    print(f"Report: {report_path}")
    print(f"{'='*70}")

    # Save full batch JSON
    batch_json = output_dir / "batch_results.json"
    with open(batch_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Update index.md in the base output directory
    _update_index(base_output_dir, output_dir, all_results, config)


def _update_index(
    base_output_dir: Path,
    run_dir: Path,
    all_results: list[dict],
    config: PipelineConfig,
):
    """Append this run to the index.md in the base output directory."""
    index_path = base_output_dir / "index.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = run_dir.name
    num_videos = len(all_results)

    # Count overall status
    total_passed = 0
    total_failed = 0
    total_errors = 0
    for entry in all_results:
        if "error" in entry:
            total_errors += 1
        elif any(r["status"] == "fail" for r in entry.get("results", {}).values()):
            total_failed += 1
        else:
            total_passed += 1

    status_summary = f"{total_passed} passed, {total_failed} failed"
    if total_errors:
        status_summary += f", {total_errors} errors"

    entry_line = f"| [{run_name}]({run_name}/batch_report.md) | {timestamp} | {num_videos} | {status_summary} |"

    if index_path.exists():
        content = index_path.read_text()
        content = content.rstrip("\n") + "\n" + entry_line + "\n"
    else:
        content = (
            "# Pipeline Run Index\n"
            "\n"
            "| Run | Timestamp | Videos | Result |\n"
            "|-----|-----------|--------|--------|\n"
            f"{entry_line}\n"
        )

    index_path.write_text(content)


if __name__ == "__main__":
    main()
