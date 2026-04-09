"""Frame extraction utility for video quality checks.

Extracts frames at a configurable rate from video files using OpenCV.
Designed for egocentric video processing pipeline.
"""

import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional


def extract_frames(
    video_path: str | Path,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
) -> tuple[list[np.ndarray], dict]:
    """Extract frames from a video at the specified sampling rate.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to sample. Default 1.0 = 1 frame/second.
        max_frames: Maximum number of frames to extract. None = no limit.

    Returns:
        Tuple of (frames, metadata) where:
        - frames: List of BGR numpy arrays (OpenCV format)
        - metadata: Dict with video_fps, total_frames, duration_s,
                     width, height, frames_extracted, extraction_time_s
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / video_fps if video_fps > 0 else 0

    # Calculate frame interval in milliseconds
    interval_ms = 1000.0 / fps

    frames = []
    t_start = time.perf_counter()
    timestamp_ms = 0.0

    while timestamp_ms < (duration_s * 1000):
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
        timestamp_ms += interval_ms

    cap.release()
    extraction_time = time.perf_counter() - t_start

    metadata = {
        "video_fps": video_fps,
        "total_frames": total_frames,
        "duration_s": round(duration_s, 2),
        "width": width,
        "height": height,
        "frames_extracted": len(frames),
        "extraction_time_s": round(extraction_time, 3),
        "sampling_fps": fps,
    }

    return frames, metadata


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path> [fps]")
        sys.exit(1)

    video_path = sys.argv[1]
    fps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    print(f"Extracting frames from {video_path} at {fps} FPS...")
    frames, meta = extract_frames(video_path, fps=fps)
    print(f"Extracted {meta['frames_extracted']} frames in {meta['extraction_time_s']}s")
    print(f"Video: {meta['width']}x{meta['height']}, {meta['duration_s']}s, {meta['video_fps']} FPS")
