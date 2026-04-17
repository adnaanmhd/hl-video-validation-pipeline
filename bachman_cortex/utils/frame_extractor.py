"""Native-FPS frame generator for the scoring engine.

Yields `(frame_idx, frame_bgr_720p)` for every native frame in the video.
Downstream accumulators own cadence gating — this module does not
subsample.

Backend selection is automatic: NVDEC via `cv2.cudacodec.VideoReader`
when the cv2 build has `cudacodec` and a CUDA device is present,
`cv2.VideoCapture` (CPU) otherwise.

720p long-edge downscale happens here, once, so every downstream check
can treat the input resolution as fixed.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


_DEFAULT_LONG_EDGE = 1280   # 720p long-edge


@dataclass(frozen=True)
class VideoStreamInfo:
    """Container + backend metadata reported up-front by `iter_native_frames`."""
    native_fps: float
    total_frames: int
    width: int
    height: int
    duration_s: float
    backend: str            # "nvdec" | "cpu"


def _nvdec_available() -> bool:
    """True if this cv2 build has cudacodec + at least one CUDA device."""
    if not hasattr(cv2, "cudacodec"):
        return False
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


_NVDEC_OK = _nvdec_available()


def _resize_long_edge(frame: np.ndarray, long_edge: int) -> np.ndarray:
    """Resize in-place so the longest edge equals `long_edge` (or smaller)."""
    fh, fw = frame.shape[:2]
    longest = max(fh, fw)
    if longest <= long_edge:
        return frame
    s = long_edge / longest
    return cv2.resize(
        frame,
        (int(round(fw * s)), int(round(fh * s))),
        interpolation=cv2.INTER_AREA,
    )


def probe_video(video_path: str | Path) -> VideoStreamInfo:
    """Read container headers without decoding — cheap, used by the engine."""
    path = str(video_path)
    probe = cv2.VideoCapture(path)
    if not probe.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = probe.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()
    duration = (total / fps) if fps > 0 else 0.0
    backend = "nvdec" if _NVDEC_OK else "cpu"
    return VideoStreamInfo(
        native_fps=round(fps, 6),
        total_frames=total,
        width=width,
        height=height,
        duration_s=round(duration, 3),
        backend=backend,
    )


def iter_native_frames(
    video_path: str | Path,
    long_edge: int = _DEFAULT_LONG_EDGE,
) -> tuple[VideoStreamInfo, Iterator[tuple[int, np.ndarray]]]:
    """Open the video and return `(info, generator)`.

    The generator yields `(frame_idx, frame_bgr_720p)` for every native
    frame. Consumers must pump the generator to completion — it
    releases the underlying reader on exhaustion.

    This is a two-tuple (not a plain generator) so the caller has
    up-front access to `native_fps`, `total_frames`, and the backend
    label before they start iterating. That lets the engine stamp
    those values into the report without a pre-probe pass.
    """
    info = probe_video(video_path)
    path = str(video_path)

    def _gen_nvdec() -> Iterator[tuple[int, np.ndarray]]:
        reader = cv2.cudacodec.createVideoReader(path)
        frame_idx = 0
        try:
            while True:
                ok, gpu_frame = reader.nextFrame()
                if not ok:
                    break
                gpu_bgr = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGRA2BGR)
                frame = gpu_bgr.download()
                frame = _resize_long_edge(frame, long_edge)
                yield frame_idx, frame
                frame_idx += 1
        finally:
            # cv2.cudacodec.VideoReader has no explicit close; drop ref.
            del reader

    def _gen_cpu() -> Iterator[tuple[int, np.ndarray]]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = _resize_long_edge(frame, long_edge)
                yield frame_idx, frame
                frame_idx += 1
        finally:
            cap.release()

    if info.backend == "nvdec":
        try:
            return info, _gen_nvdec()
        except cv2.error:
            # Fall through to CPU on NVDEC init failure.
            pass

    return VideoStreamInfo(
        native_fps=info.native_fps,
        total_frames=info.total_frames,
        width=info.width,
        height=info.height,
        duration_s=info.duration_s,
        backend="cpu",
    ), _gen_cpu()


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python frame_extractor.py <video_path>")
        sys.exit(1)

    info, frames = iter_native_frames(sys.argv[1])
    t0 = time.perf_counter()
    count = 0
    for _idx, _frame in frames:
        count += 1
    dt = time.perf_counter() - t0
    print(
        f"{info.backend}: {count} frames in {dt:.2f}s "
        f"({count / dt:.1f} FPS), native_fps={info.native_fps}, "
        f"dims={info.width}x{info.height}"
    )
