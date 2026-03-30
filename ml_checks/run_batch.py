#!/usr/bin/env python
"""Run the ML check pipeline on a batch of videos.

Usage:
    # Single video
    python -m ml_checks.run_batch /path/to/video.mp4

    # Multiple videos
    python -m ml_checks.run_batch /path/to/video1.mp4 /path/to/video2.mp4

    # Directory of videos
    python -m ml_checks.run_batch /path/to/videos/

    # With options
    python -m ml_checks.run_batch /path/to/videos/ --fps 2 --max-frames 50 --no-gdino --output results/

    # Auto-rotate portrait videos to landscape before processing
    python -m ml_checks.run_batch /path/to/videos/ --auto-rotate
"""

import argparse
import json
import os
import sys
import time
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

# Ensure project root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from ml_checks.pipeline import MLCheckPipeline, PipelineConfig
from ml_checks.utils.frame_extractor import extract_frames


def get_video_meta(path: str) -> dict:
    """Get video metadata without loading the full file."""
    cap = cv2.VideoCapture(path)
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": round(cap.get(cv2.CAP_PROP_FPS), 2),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_s": round(
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(cap.get(cv2.CAP_PROP_FPS), 1), 2
        ),
        "file_size_mb": round(os.path.getsize(path) / 1024 / 1024, 1),
    }
    cap.release()
    return meta


def check_rotation(path: str) -> int:
    """Check if video has a rotation metadata tag. Returns degrees (0, 90, 180, 270)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True,
        )
        data = json.loads(result.stdout)
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                # Check tags
                rotation = stream.get("tags", {}).get("rotate", "0")
                if rotation != "0":
                    return abs(int(rotation))
                # Check side_data_list
                for sd in stream.get("side_data_list", []):
                    if "rotation" in sd:
                        return abs(int(sd["rotation"]))
    except Exception:
        pass
    return 0


def rotate_video(input_path: str, output_path: str, rotation: int) -> str:
    """Rotate video to correct orientation using ffmpeg."""
    # Map rotation to ffmpeg transpose value
    # rotation = degrees the video is rotated FROM upright
    if rotation == 90:
        vf = "transpose=1"  # 90° clockwise
    elif rotation == 270:
        vf = "transpose=2"  # 90° counter-clockwise
    elif rotation == 180:
        vf = "rotate=PI"
    else:
        return input_path  # No rotation needed

    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-vf", vf,
         "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-an", output_path],
        capture_output=True,
    )
    return output_path


def is_portrait(path: str) -> bool:
    """Check if video displays as portrait (accounting for rotation metadata)."""
    meta = get_video_meta(path)
    rotation = check_rotation(path)
    w, h = meta["width"], meta["height"]
    # If rotated 90 or 270, the display dimensions are swapped
    if rotation in (90, 270):
        w, h = h, w
    return h > w


def collect_videos(paths: list[str], extensions: set[str] = {".mp4", ".mov", ".avi", ".mkv"}) -> list[Path]:
    """Collect video files from paths (files or directories)."""
    videos = []
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() in extensions:
            videos.append(path)
        elif path.is_dir():
            for ext in extensions:
                videos.extend(sorted(path.glob(f"*{ext}")))
                videos.extend(sorted(path.glob(f"*{ext.upper()}")))
    # Deduplicate preserving order
    seen = set()
    unique = []
    for v in videos:
        resolved = v.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(v)
    return unique


def format_result_row(name: str, result) -> str:
    """Format a single check result as a markdown table row."""
    status_icon = "PASS" if result.status == "pass" else "FAIL"
    flag = " (flagged)" if result.confidence < 0.7 and result.status == "pass" else ""
    return f"| {name} | **{status_icon}** | {result.metric_value:.4f} | {result.confidence:.4f}{flag} |"


def write_report(
    all_results: list[dict],
    output_dir: Path,
    config: PipelineConfig,
):
    """Write a markdown summary report for all processed videos."""
    report_path = output_dir / "batch_report.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Batch Pipeline Report",
        f"",
        f"**Generated:** {timestamp}",
        f"**Videos processed:** {len(all_results)}",
        f"**Sampling FPS:** {config.sampling_fps}",
        f"**Grounding DINO:** {'enabled' if config.run_grounding_dino else 'disabled'}",
        f"",
        f"---",
        f"",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Video | Checks Passed | Status | Processing Time |")
    lines.append("|---|---|---|---|")

    for entry in all_results:
        name = entry["filename"]
        if "error" in entry:
            lines.append(f"| {name} | - | ERROR: {entry['error']} | - |")
            continue
        checks = entry["results"]
        passed = sum(1 for r in checks.values() if r["status"] == "pass")
        total = len(checks)
        status = "ALL PASS" if passed == total else f"{passed}/{total}"
        proc_time = f"{entry['processing_time_s']:.1f}s"
        lines.append(f"| {name} | {passed}/{total} | {status} | {proc_time} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-video detail
    lines.append("## Per-Video Results")
    lines.append("")

    for entry in all_results:
        name = entry["filename"]
        lines.append(f"### {name}")
        lines.append("")

        if "error" in entry:
            lines.append(f"**Error:** {entry['error']}")
            lines.append("")
            continue

        meta = entry["video_meta"]
        lines.append(f"- Resolution: {meta['width']}x{meta['height']}, {meta['fps']} FPS, {meta['duration_s']}s, {meta['file_size_mb']} MB")
        if entry.get("rotated"):
            lines.append(f"- Auto-rotated from portrait to landscape")
        lines.append(f"- Frames sampled: {entry['frames_sampled']}")
        lines.append(f"- Processing time: {entry['processing_time_s']:.1f}s")
        lines.append("")

        lines.append("| Check | Status | Metric | Confidence |")
        lines.append("|---|---|---|---|")

        check_names = [
            ("face_presence", "Face Presence"),
            ("participants", "Participants"),
            ("hand_visibility", "Hand Visibility"),
            ("hand_object_interaction", "Hand-Object Interaction"),
            ("privacy_safety", "Privacy Safety"),
            ("view_obstruction", "View Obstruction"),
            ("pov_hand_angle", "POV-Hand Angle"),
        ]

        for key, display in check_names:
            if key in entry["results"]:
                r = entry["results"][key]
                status_icon = "PASS" if r["status"] == "pass" else "FAIL"
                flag = " (flagged)" if r["confidence"] < 0.7 and r["status"] == "pass" else ""
                lines.append(f"| {display} | **{status_icon}** | {r['metric_value']:.4f} | {r['confidence']:.4f}{flag} |")

        lines.append("")

        # Details for failing checks
        failing = {k: v for k, v in entry["results"].items() if v["status"] == "fail"}
        if failing:
            lines.append("**Failing checks:**")
            lines.append("")
            for key, r in failing.items():
                display = dict(check_names).get(key, key)
                lines.append(f"- **{display}**: {json.dumps(r['details'], default=str)}")
            lines.append("")

        lines.append("---")
        lines.append("")

    report_content = "\n".join(lines)
    report_path.write_text(report_content)
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Run ML check pipeline on a batch of videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ml_checks.run_batch video.mp4
  python -m ml_checks.run_batch videos_dir/
  python -m ml_checks.run_batch *.mp4 --auto-rotate --output results/
  python -m ml_checks.run_batch videos/ --fps 2 --max-frames 100 --no-gdino
        """,
    )
    parser.add_argument("paths", nargs="+", help="Video files or directories to process")
    parser.add_argument("--output", "-o", default="ml_checks/results", help="Output directory for reports (default: ml_checks/results)")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate in FPS (default: 1.0)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to sample per video (default: no limit)")
    parser.add_argument("--no-gdino", action="store_true", help="Disable Grounding DINO (faster, but no fine-grained privacy detection)")
    parser.add_argument("--auto-rotate", action="store_true", help="Auto-rotate portrait videos to landscape before processing")
    parser.add_argument("--hand-detector-repo", default="ml_checks/models/weights/hand_object_detector", help="Path to 100DOH repo")
    parser.add_argument("--scrfd-root", default="ml_checks/models/weights/insightface", help="Path to InsightFace models")
    parser.add_argument("--gdino-cache", default="ml_checks/models/weights/grounding_dino", help="Path to Grounding DINO cache")

    args = parser.parse_args()

    # Collect videos
    videos = collect_videos(args.paths)
    if not videos:
        print("No video files found.")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to process.")
    for v in videos:
        print(f"  - {v.name} ({os.path.getsize(v) / 1024 / 1024:.1f} MB)")

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure pipeline
    config = PipelineConfig(
        sampling_fps=args.fps,
        max_frames=args.max_frames,
        run_grounding_dino=not args.no_gdino,
        scrfd_root=args.scrfd_root,
        hand_detector_repo=args.hand_detector_repo,
        gdino_cache=args.gdino_cache,
    )

    # Load models once
    pipeline = MLCheckPipeline(config)
    pipeline.load_models()

    # Process each video
    all_results = []
    temp_files = []

    for i, video_path in enumerate(videos):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(videos)}] {video_path.name}")
        print(f"{'='*70}")

        entry = {
            "filename": video_path.name,
            "path": str(video_path),
        }

        try:
            # Get metadata
            meta = get_video_meta(str(video_path))
            entry["video_meta"] = meta

            # Auto-rotate if needed
            process_path = str(video_path)
            entry["rotated"] = False

            if args.auto_rotate and is_portrait(str(video_path)):
                rotation = check_rotation(str(video_path))
                if rotation == 0:
                    rotation = 90  # Default: portrait without tag = needs 90° rotation
                print(f"  Portrait detected (rotation={rotation}°). Auto-rotating...")
                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                tmp.close()
                process_path = rotate_video(str(video_path), tmp.name, rotation)
                temp_files.append(tmp.name)
                entry["rotated"] = True
                # Update meta for rotated video
                meta = get_video_meta(process_path)
                entry["video_meta_rotated"] = meta

            # Run pipeline
            t0 = time.perf_counter()
            results = pipeline.process_video(process_path)
            elapsed = time.perf_counter() - t0

            entry["processing_time_s"] = round(elapsed, 2)
            entry["frames_sampled"] = len(
                extract_frames(process_path, fps=config.sampling_fps, max_frames=config.max_frames)[0]
            )

            # Serialize results
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
            import traceback
            traceback.print_exc()

        all_results.append(entry)

        # Save per-video JSON
        video_json = output_dir / f"{video_path.stem}.json"
        with open(video_json, "w") as f:
            json.dump(entry, f, indent=2, default=str)

    # Cleanup temp files
    for tmp in temp_files:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    # Write batch report
    report_path = write_report(all_results, output_dir, config)
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE: {len(all_results)} videos processed")
    print(f"Report: {report_path}")
    print(f"Per-video JSON: {output_dir}/")
    print(f"{'='*70}")

    # Save full batch JSON
    batch_json = output_dir / "batch_results.json"
    with open(batch_json, "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
