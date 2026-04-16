"""Motion analysis checks.

Camera stability via single-pass sparse Lucas-Kanade optical flow at 0.5x
resolution, per-frame at target FPS (default 30).  CUDA GPU acceleration
used when available (cv2.cuda.SparsePyrLKOpticalFlow).
Frozen segment detection derived from LK signal (near-zero translation + rotation).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from bachman_cortex.checks.check_results import CheckResult


def _compute_ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two grayscale images using OpenCV.

    Lightweight implementation -- no scikit-image dependency.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


# ── Lucas-Kanade helpers ─────────────────────────────────────────────────────

def _feature_params(max_corners: int = 300) -> dict:
    return dict(maxCorners=max_corners, qualityLevel=0.01,
                minDistance=10, blockSize=3)


def _lk_params(win_size: tuple[int, int] = (21, 21),
               max_level: int = 3) -> dict:
    return dict(winSize=win_size, maxLevel=max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          30, 0.01))


# ── CUDA LK acceleration ───────────────────────────────────────────────────

_CUDA_LK = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cv2.cuda.SparsePyrLKOpticalFlow.create()  # smoke-test
        _CUDA_LK = True
except Exception:
    pass

_cuda_lk_solvers: dict = {}


def _get_cuda_lk(win_size: tuple[int, int], max_level: int):
    key = (win_size, max_level)
    if key not in _cuda_lk_solvers:
        _cuda_lk_solvers[key] = cv2.cuda.SparsePyrLKOpticalFlow.create(
            winSize=win_size, maxLevel=max_level)
    return _cuda_lk_solvers[key]


def _lk_track(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    prev_pts: np.ndarray,
    lk_cpu_params: dict,
    win_size: tuple[int, int],
    max_level: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sparse LK optical flow — CUDA when available, CPU fallback.

    Returns (next_pts, status) where status is a 1D uint8 array.
    """
    if _CUDA_LK:
        try:
            solver = _get_cuda_lk(win_size, max_level)
            g_prev = cv2.cuda_GpuMat(prev_gray)
            g_curr = cv2.cuda_GpuMat(curr_gray)
            g_pts = cv2.cuda_GpuMat(
                prev_pts.reshape(1, -1, 2).astype(np.float32))
            g_next, g_status, _ = solver.calc(g_prev, g_curr, g_pts, None)
            return (g_next.download().reshape(-1, 1, 2),
                    g_status.download().reshape(-1))
        except cv2.error:
            pass  # fall through to CPU
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_cpu_params)
    return curr_pts, status.reshape(-1)


def _highpass_signal(values: np.ndarray, window: int) -> np.ndarray:
    """Remove low-frequency component, isolating high-frequency jitter.

    Uses a variable-width rolling mean so edge frames are not distorted by
    zero-padding.  Returns ``|values - rolling_mean|``.
    """
    n = len(values)
    if n <= window or window < 2:
        return values
    cs = np.cumsum(np.concatenate(([0.0], values.astype(np.float64))))
    half = window // 2
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    low_freq = (cs[hi] - cs[lo]) / (hi - lo)
    return np.abs(values - low_freq)


def _transforms_to_score(
    translations: list[float],
    rotations: list[float],
    scale: float,
    trans_threshold: float,
    jump_threshold: float,
    rot_threshold: float,
    variance_threshold: float,
    w_trans: float,
    w_var: float,
    w_rot: float,
    w_jump: float,
) -> tuple[float, dict]:
    """Convert per-frame transform lists to a shakiness score in [0, 1].

    `scale` converts downsampled-pixel values back to native-pixel equivalents.
    """
    if not translations:
        return 0.0, {}

    t = np.array(translations) * scale
    r = np.array(rotations)

    avg_t = float(np.mean(t))
    std_t = float(np.std(t))
    avg_r = float(np.mean(r))
    jumps = int(np.sum(t > jump_threshold))
    n = len(t)

    t_s = min(avg_t / (trans_threshold * 3), 1.0)
    v_s = min(std_t / (variance_threshold * 2), 1.0)
    r_s = min(avg_r / (rot_threshold * 3), 1.0)
    j_s = min((jumps / n) * 10, 1.0)

    score = w_trans * t_s + w_var * v_s + w_rot * r_s + w_jump * j_s
    stats = dict(avg_t=round(avg_t, 2), std_t=round(std_t, 2),
                 avg_r=round(avg_r, 4), jumps=jumps)
    return round(score, 3), stats


def _analyse_segment(
    cap: cv2.VideoCapture,
    start_frame: int,
    end_frame: int,
    scale: float,
    frame_skip: int,
    max_corners: int,
    lk_win_size: tuple[int, int],
    lk_max_level: int,
    trans_threshold: float,
    jump_threshold: float,
    rot_threshold: float,
    variance_threshold: float,
    w_trans: float,
    w_var: float,
    w_rot: float,
    w_jump: float,
) -> tuple[float, dict]:
    """Run LK optical-flow analysis on [start_frame, end_frame).

    Returns (score, stats).
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    ret, frame = cap.read()
    if not ret:
        return 0.0, {}

    if scale != 1.0:
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, **_feature_params(max_corners))

    lk = _lk_params(lk_win_size, lk_max_level)
    translations: list[float] = []
    rotations: list[float] = []
    local_idx = 0

    for fno in range(start_frame + 1, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        local_idx += 1

        if local_idx % frame_skip != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_pts is None or len(prev_pts) < 4:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, **_feature_params(max_corners))
            prev_gray = curr_gray
            continue

        curr_pts, status = _lk_track(
            prev_gray, curr_gray, prev_pts, lk,
            lk_win_size, lk_max_level)

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) >= 4:
            T, _ = cv2.estimateAffinePartial2D(
                good_prev, good_curr,
                method=cv2.RANSAC, ransacReprojThreshold=3)
            if T is not None:
                dx, dy = T[0, 2], T[1, 2]
                translations.append(np.sqrt(dx**2 + dy**2))
                rotations.append(abs(np.degrees(np.arctan2(T[1, 0], T[0, 0]))))

        # Refresh corners every 60 real frames (~1s) or on degradation
        if fno % 60 == 0 or len(good_curr) < 20:
            prev_pts = cv2.goodFeaturesToTrack(curr_gray, **_feature_params(max_corners))
        else:
            prev_pts = good_curr.reshape(-1, 1, 2)
        prev_gray = curr_gray

    native_scale = 1.0 / scale if scale != 1.0 else 1.0
    return _transforms_to_score(
        translations, rotations, scale=native_scale,
        trans_threshold=trans_threshold, jump_threshold=jump_threshold,
        rot_threshold=rot_threshold, variance_threshold=variance_threshold,
        w_trans=w_trans, w_var=w_var, w_rot=w_rot, w_jump=w_jump,
    )


# ── Camera stability (two-stage) ─────────────────────────────────────────────

def check_camera_stability(
    video_path: str | Path,
    *,
    # Thresholds
    shaky_score_threshold: float = 0.181,
    deep_score_threshold: float = 0.25,
    # Stage 1 tunables
    fast_scale: float = 0.333,
    frame_skip: int = 2,
    # Per-frame thresholds (native-res px/frame)
    trans_threshold: float = 8.0,
    jump_threshold: float = 30.0,
    rot_threshold: float = 0.3,
    variance_threshold: float = 6.0,
    # Scoring weights
    w_trans: float = 0.35,
    w_var: float = 0.25,
    w_rot: float = 0.20,
    w_jump: float = 0.20,
    # LK params
    max_corners: int = 300,
    lk_win_size: tuple[int, int] = (21, 21),
    lk_max_level: int = 3,
    # FPS cap
    target_fps: float = 30.0,
    # Optional sub-range (for clip-level analysis)
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> CheckResult:
    """Two-stage camera shakiness detection using sparse Lucas-Kanade optical flow.

    Stage 1: Fast pre-screen at downscaled resolution, subsampled to target_fps.
             Produces per-second shakiness scores.
    Stage 2: Full-resolution analysis on seconds flagged by Stage 1,
             also capped at target_fps.

    Pass if overall_score <= shaky_score_threshold.

    Args:
        video_path: Path to video file.
        shaky_score_threshold: Score above this marks a second as shaky and
            the video as shaky if overall score exceeds it.
        deep_score_threshold: Stage-1 score above this queues the second
            for Stage-2 deep analysis.
        fast_scale: Downscale factor for Stage 1 (e.g. 0.333 = ~360p).
        frame_skip: Base frame skip for Stage 1 (multiplied by fps_skip).
        trans_threshold: Translation px/frame threshold (native res).
        jump_threshold: Sudden jolt px/frame threshold.
        rot_threshold: Rotation degrees/frame threshold.
        variance_threshold: Translation std-dev threshold.
        w_trans: Scoring weight for mean translation.
        w_var: Scoring weight for translation variance.
        w_rot: Scoring weight for mean rotation.
        w_jump: Scoring weight for sudden jumps.
        max_corners: Max corners for goodFeaturesToTrack.
        lk_win_size: Lucas-Kanade window size.
        lk_max_level: Lucas-Kanade pyramid levels.
        target_fps: Max analysis FPS — videos above this are subsampled.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_sec = int(fps)

    # Sub-range support: limit analysis to [start_sec, end_sec)
    range_start_frame = int(start_sec * fps) if start_sec is not None else 0
    range_end_frame = int(end_sec * fps) if end_sec is not None else total_frames_video
    range_end_frame = min(range_end_frame, total_frames_video)
    total_frames = range_end_frame - range_start_frame
    total_seconds = total_frames / fps

    range_start_sec_int = int(start_sec) if start_sec is not None else 0

    # FPS-based subsampling: if native > target, increase frame_skip
    fps_skip = max(1, round(fps / target_fps))
    effective_stage1_skip = frame_skip * fps_skip
    effective_stage2_skip = fps_skip  # Stage 2 was frame_skip=1, now fps_skip

    # Shared kwargs for scoring
    score_kwargs = dict(
        trans_threshold=trans_threshold, jump_threshold=jump_threshold,
        rot_threshold=rot_threshold, variance_threshold=variance_threshold,
        w_trans=w_trans, w_var=w_var, w_rot=w_rot, w_jump=w_jump,
        max_corners=max_corners, lk_win_size=lk_win_size,
        lk_max_level=lk_max_level,
    )

    # ── Stage 1: fast per-second scan ─────────────────────────────────────
    stage1_scores: dict[int, float] = {}
    stage1_frame_count = 0
    deep_queue: list[int] = []

    for sec in range(int(total_seconds) + 1):
        sf = range_start_frame + sec * frames_per_sec
        ef = min(sf + frames_per_sec, range_end_frame)
        if sf >= range_end_frame:
            break

        score, _ = _analyse_segment(
            cap, sf, ef, scale=fast_scale, frame_skip=effective_stage1_skip,
            **score_kwargs,
        )
        stage1_scores[sec] = score
        stage1_frame_count += max(1, (ef - sf) // effective_stage1_skip)

        if score >= deep_score_threshold:
            deep_queue.append(sec)

    # ── Stage 2: deep re-analysis on flagged seconds ──────────────────────
    stage2_scores: dict[int, float] = {}
    stage2_frame_count = 0

    if deep_queue:
        for sec in deep_queue:
            sf = range_start_frame + sec * frames_per_sec
            ef = min(sf + frames_per_sec, range_end_frame)

            score, _ = _analyse_segment(
                cap, sf, ef, scale=1.0, frame_skip=effective_stage2_skip,
                **score_kwargs,
            )
            stage2_scores[sec] = score
            stage2_frame_count += max(1, (ef - sf) // effective_stage2_skip)

    cap.release()

    # ── Merge: prefer deep results where available ────────────────────────
    final_scores: dict[int, float] = {}
    for sec in sorted(stage1_scores):
        final_scores[sec] = stage2_scores.get(sec, stage1_scores[sec])

    shaky_seconds = [s for s, sc in final_scores.items() if sc >= shaky_score_threshold]
    overall_score = float(np.mean(list(final_scores.values()))) if final_scores else 0.0
    passes = overall_score <= shaky_score_threshold

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(overall_score, 4),
        confidence=1.0,
        details={
            "overall_score": round(overall_score, 4),
            "shaky_score_threshold": shaky_score_threshold,
            "shaky_seconds": shaky_seconds,
            "shaky_seconds_count": len(shaky_seconds),
            "total_seconds": int(total_seconds),
            "stage1_frames_analysed": stage1_frame_count,
            "stage2_frames_analysed": stage2_frame_count,
            "stage2_seconds_flagged": len(deep_queue),
            "fast_scale": fast_scale,
            "frame_skip": frame_skip,
            "native_fps": round(fps, 2),
            "target_fps": target_fps,
            "fps_skip": fps_skip,
            "effective_stage1_skip": effective_stage1_skip,
            "effective_stage2_skip": effective_stage2_skip,
        },
    )


# ── Frozen segments ──────────────────────────────────────────────────────────

def check_frozen_segments(
    video_path: str | Path,
    max_consecutive: int = 30,
    ssim_threshold: float = 0.99,
    downscale_height: int = 480,
    target_fps: float = 10.0,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> CheckResult:
    """Check for frozen segments by subsampling at target_fps.

    Reads every Nth frame (N = native_fps / target_fps) and computes SSIM
    between consecutive sampled pairs.  Fails if any run of consecutive
    sampled frames exceeds the scaled threshold.

    Optimizations:
    - Subsample at target_fps instead of reading every native frame.
    - Downscale to 480p grayscale before SSIM.
    - Fast pre-filter: skip SSIM if mean absolute difference > 5.
    - Streams 2 frames at a time (constant memory).

    Args:
        video_path: Path to video file.
        max_consecutive: Max allowed consecutive frozen frames (at native FPS).
        ssim_threshold: SSIM above this = frozen.
        downscale_height: Height to downscale to before comparison.
        target_fps: Target sampling rate for frozen detection (default 10).
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Sub-range support
    range_start_frame = int(start_sec * native_fps) if start_sec is not None else 0
    range_end_frame = int(end_sec * native_fps) if end_sec is not None else total_frames_video
    range_end_frame = min(range_end_frame, total_frames_video)
    total_frames = range_end_frame - range_start_frame

    # Subsampling: read every frame_step-th frame
    frame_step = max(1, round(native_fps / target_fps))
    effective_fps = native_fps / frame_step
    # Scale the threshold proportionally to the sampling rate
    effective_max = max(1, round(max_consecutive * effective_fps / native_fps))

    # Seek to range start and read first frame
    if range_start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, range_start_frame)
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return CheckResult(status="pass", metric_value=0.0, confidence=1.0,
                           details={"error": "could not read first frame"})

    prev_gray = _downscale_gray(prev_frame, downscale_height)

    current_run = 0
    longest_run = 0
    frozen_segments = []
    run_start = 0
    frames_sampled = 1
    frame_idx = range_start_frame  # native frame index of last read

    while True:
        # Skip ahead by frame_step
        frame_idx += frame_step
        if frame_idx >= range_end_frame:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, curr_frame = cap.read()
        if not ret:
            break
        frames_sampled += 1

        curr_gray = _downscale_gray(curr_frame, downscale_height)

        # Fast pre-filter: if frames are clearly different, skip SSIM
        mean_abs_diff = float(np.mean(np.abs(
            curr_gray.astype(np.float32) - prev_gray.astype(np.float32)
        )))

        if mean_abs_diff <= 5.0:
            ssim = _compute_ssim_gray(prev_gray, curr_gray)
            is_frozen = ssim > ssim_threshold
        else:
            is_frozen = False

        if is_frozen:
            if current_run == 0:
                run_start = frame_idx - frame_step
            current_run += 1
        else:
            if current_run > 0:
                if current_run > longest_run:
                    longest_run = current_run
                if current_run > effective_max:
                    frozen_segments.append({"start_frame": run_start, "length_sampled": current_run})
                current_run = 0

        prev_gray = curr_gray

    # Handle run at end of video
    if current_run > 0:
        if current_run > longest_run:
            longest_run = current_run
        if current_run > effective_max:
            frozen_segments.append({"start_frame": run_start, "length_sampled": current_run})

    cap.release()

    passes = longest_run <= effective_max
    metric = longest_run / effective_max if effective_max > 0 else 0.0

    # Convert sampled run length back to estimated native-fps duration
    longest_run_native_est = longest_run * frame_step
    frozen_duration_s = round(longest_run_native_est / native_fps, 2) if native_fps > 0 else 0

    return CheckResult(
        status="pass" if passes else "fail",
        metric_value=round(min(metric, 10.0), 4),
        confidence=1.0,
        details={
            "longest_frozen_run_sampled": longest_run,
            "longest_frozen_run_native_est": longest_run_native_est,
            "effective_max_consecutive": effective_max,
            "max_consecutive_native": max_consecutive,
            "ssim_threshold": ssim_threshold,
            "frames_sampled": frames_sampled,
            "total_native_frames": total_frames,
            "native_fps": round(native_fps, 2),
            "target_fps": target_fps,
            "effective_fps": round(effective_fps, 2),
            "frame_step": frame_step,
            "frozen_duration_s": frozen_duration_s,
            "frozen_segments": frozen_segments[:10],
        },
    )


def _downscale_gray(frame: np.ndarray, target_height: int) -> np.ndarray:
    """Convert to grayscale and downscale to target height."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h <= target_height:
        return gray
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(gray, (new_w, target_height), interpolation=cv2.INTER_AREA)


# ── Combined single-pass motion analysis ────────────────────────────────────

def check_motion_combined(
    video_path: str | Path,
    *,
    # Stability params
    shaky_score_threshold: float = 0.181,
    fast_scale: float = 0.5,
    trans_threshold: float = 8.0,
    jump_threshold: float = 30.0,
    rot_threshold: float = 0.3,
    variance_threshold: float = 6.0,
    w_trans: float = 0.35,
    w_var: float = 0.25,
    w_rot: float = 0.20,
    w_jump: float = 0.20,
    max_corners: int = 300,
    lk_win_size: tuple[int, int] = (21, 21),
    lk_max_level: int = 3,
    stability_target_fps: float = 30.0,
    # High-pass filter: isolate jitter from intentional camera motion
    highpass_window_sec: float = 0.5,
    # Frozen params (derived from LK signal — no separate SSIM pass)
    frozen_max_consecutive: int = 30,
    frozen_trans_threshold: float = 0.1,
    frozen_rot_threshold: float = 0.001,
    # Range
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[CheckResult, CheckResult]:
    """Single-pass motion analysis: camera stability + frozen segment detection.

    Single pass at 0.5x resolution with per-frame LK optical flow (every
    frame at target_fps).  A high-pass filter separates intentional camera
    movement from jitter before scoring.  Frozen detection derived from the
    same LK signal (near-zero translation + rotation = frozen).  CUDA GPU
    acceleration used when available.

    Returns:
        (stability_result, frozen_result)
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_sec = int(native_fps)

    # Sub-range
    range_start = int(start_sec * native_fps) if start_sec is not None else 0
    range_end = int(end_sec * native_fps) if end_sec is not None else total_frames_video
    range_end = min(range_end, total_frames_video)
    total_frames = range_end - range_start
    total_seconds = total_frames / native_fps

    # --- Stability sampling config ---
    stab_fps_skip = max(1, round(native_fps / stability_target_fps))
    eff_skip = stab_fps_skip  # frame_skip=1, only fps-based subsampling

    # --- Frozen config (derived from LK signal, same sampling as stability) ---
    frozen_eff_fps = native_fps / eff_skip
    frozen_eff_max = max(1, round(frozen_max_consecutive / eff_skip))

    # --- LK helpers ---
    feat_p = _feature_params(max_corners)
    lk_p = _lk_params(lk_win_size, lk_max_level)
    score_kw = dict(
        trans_threshold=trans_threshold, jump_threshold=jump_threshold,
        rot_threshold=rot_threshold, variance_threshold=variance_threshold,
        w_trans=w_trans, w_var=w_var, w_rot=w_rot, w_jump=w_jump,
    )
    inv_scale = 1.0 / fast_scale if fast_scale != 1.0 else 1.0

    # --- Stability state (raw per-second data; filtered + scored after pass) ---
    per_sec_raw_trans: dict[int, list[float]] = {}
    per_sec_raw_rots: dict[int, list[float]] = {}
    lk_prev_gray = None
    lk_prev_pts = None
    lk_trans: list[float] = []
    lk_rots: list[float] = []
    cur_sec = -1
    frame_count = 0

    # --- Frozen state (tracked from LK signal) ---
    frozen_run = 0
    frozen_longest = 0
    frozen_segments_found: list[dict] = []
    frozen_run_start = 0
    frozen_sampled = 0

    # Seek to start
    if range_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, range_start)

    # === Single pass: stability + frozen detection ===
    for offset in range(total_frames):
        sec = offset // frames_per_sec
        frame_in_sec = offset % frames_per_sec

        # First frame of each second + every eff_skip-th after
        need_stab = (frame_in_sec == 0) or (
            frame_in_sec > 0 and frame_in_sec % eff_skip == 0
        )
        if not cap.grab():
            break

        if not need_stab:
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        # --- Stability processing ---
        if need_stab:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if fast_scale != 1.0:
                gray = cv2.resize(gray, (0, 0), fx=fast_scale, fy=fast_scale)

            if sec != cur_sec:
                # Store previous second's raw data
                if cur_sec >= 0:
                    per_sec_raw_trans[cur_sec] = lk_trans
                    per_sec_raw_rots[cur_sec] = lk_rots
                    frame_count += len(lk_trans) + 1

                cur_sec = sec
                lk_trans = []
                lk_rots = []
                lk_prev_gray = gray
                lk_prev_pts = cv2.goodFeaturesToTrack(gray, **feat_p)
            else:
                # LK against previous stability frame
                if lk_prev_pts is not None and len(lk_prev_pts) >= 4:
                    curr_pts, status = _lk_track(
                        lk_prev_gray, gray, lk_prev_pts, lk_p,
                        lk_win_size, lk_max_level)
                    good_prev = lk_prev_pts[status == 1]
                    good_curr = curr_pts[status == 1]

                    if len(good_prev) >= 4:
                        T, _ = cv2.estimateAffinePartial2D(
                            good_prev, good_curr,
                            method=cv2.RANSAC, ransacReprojThreshold=3)
                        if T is not None:
                            dx, dy = T[0, 2], T[1, 2]
                            trans_mag = np.sqrt(dx**2 + dy**2)
                            rot_mag = abs(np.degrees(
                                np.arctan2(T[1, 0], T[0, 0])))
                            lk_trans.append(trans_mag)
                            lk_rots.append(rot_mag)

                            # Frozen: near-zero motion = frozen frame
                            frozen_sampled += 1
                            trans_native = trans_mag * inv_scale
                            is_frozen = (trans_native < frozen_trans_threshold
                                         and rot_mag < frozen_rot_threshold)
                            if is_frozen:
                                if frozen_run == 0:
                                    frozen_run_start = range_start + max(0, offset - eff_skip)
                                frozen_run += 1
                            else:
                                if frozen_run > 0:
                                    if frozen_run > frozen_longest:
                                        frozen_longest = frozen_run
                                    if frozen_run > frozen_eff_max:
                                        frozen_segments_found.append({
                                            "start_frame": frozen_run_start,
                                            "length_sampled": frozen_run,
                                        })
                                    frozen_run = 0

                    abs_frame = range_start + offset
                    if abs_frame % 60 == 0 or len(good_curr) < 20:
                        lk_prev_pts = cv2.goodFeaturesToTrack(gray, **feat_p)
                    else:
                        lk_prev_pts = good_curr.reshape(-1, 1, 2)
                else:
                    lk_prev_pts = cv2.goodFeaturesToTrack(gray, **feat_p)

                lk_prev_gray = gray

    # Store last second's raw data
    if cur_sec >= 0:
        per_sec_raw_trans[cur_sec] = lk_trans
        per_sec_raw_rots[cur_sec] = lk_rots
        frame_count += len(lk_trans) + 1

    # --- High-pass filter full timeseries, then score per-second ---
    hp_window = max(1, int(round(highpass_window_sec * stability_target_fps)))
    all_trans: list[float] = []
    all_rots: list[float] = []
    sec_order: list[int] = []
    sec_counts: list[int] = []
    for sec in sorted(per_sec_raw_trans):
        t = per_sec_raw_trans[sec]
        all_trans.extend(t)
        all_rots.extend(per_sec_raw_rots[sec])
        sec_order.append(sec)
        sec_counts.append(len(t))

    per_sec_scores: dict[int, float] = {}
    if all_trans:
        filtered_trans = _highpass_signal(np.array(all_trans), hp_window)
        filtered_rots = _highpass_signal(np.array(all_rots), hp_window)
        idx = 0
        for sec, count in zip(sec_order, sec_counts):
            sec_t = filtered_trans[idx:idx + count].tolist()
            sec_r = filtered_rots[idx:idx + count].tolist()
            score, _ = _transforms_to_score(
                sec_t, sec_r, scale=inv_scale, **score_kw)
            per_sec_scores[sec] = score
            idx += count

    # Finalize frozen run at end
    if frozen_run > 0:
        if frozen_run > frozen_longest:
            frozen_longest = frozen_run
        if frozen_run > frozen_eff_max:
            frozen_segments_found.append({
                "start_frame": frozen_run_start,
                "length_sampled": frozen_run,
            })

    cap.release()

    # --- Stability result ---
    shaky_secs = [s for s, sc in per_sec_scores.items() if sc >= shaky_score_threshold]
    overall = float(np.mean(list(per_sec_scores.values()))) if per_sec_scores else 0.0
    stab_pass = overall <= shaky_score_threshold

    stability_result = CheckResult(
        status="pass" if stab_pass else "fail",
        metric_value=round(overall, 4),
        confidence=1.0,
        details={
            "overall_score": round(overall, 4),
            "shaky_score_threshold": shaky_score_threshold,
            "shaky_seconds": shaky_secs,
            "shaky_seconds_count": len(shaky_secs),
            "total_seconds": int(total_seconds),
            "frames_analysed": frame_count,
            "fast_scale": fast_scale,
            "highpass_window_sec": highpass_window_sec,
            "native_fps": round(native_fps, 2),
            "target_fps": stability_target_fps,
            "fps_skip": stab_fps_skip,
            "effective_skip": eff_skip,
        },
    )

    # --- Frozen result (derived from LK signal) ---
    frozen_pass = frozen_longest <= frozen_eff_max
    frozen_metric = frozen_longest / frozen_eff_max if frozen_eff_max > 0 else 0.0
    frozen_longest_native = frozen_longest * eff_skip
    frozen_dur_s = round(frozen_longest_native / native_fps, 2) if native_fps > 0 else 0

    frozen_result = CheckResult(
        status="pass" if frozen_pass else "fail",
        metric_value=round(min(frozen_metric, 10.0), 4),
        confidence=1.0,
        details={
            "longest_frozen_run_sampled": frozen_longest,
            "longest_frozen_run_native_est": frozen_longest_native,
            "effective_max_consecutive": frozen_eff_max,
            "max_consecutive_native": frozen_max_consecutive,
            "frozen_trans_threshold": frozen_trans_threshold,
            "frozen_rot_threshold": frozen_rot_threshold,
            "lk_pairs_checked": frozen_sampled,
            "total_native_frames": total_frames,
            "native_fps": round(native_fps, 2),
            "effective_fps": round(frozen_eff_fps, 2),
            "effective_frame_step": eff_skip,
            "frozen_duration_s": frozen_dur_s,
            "frozen_segments": frozen_segments_found[:10],
        },
    )

    return stability_result, frozen_result


# ── Streaming single-pass motion analyzer ───────────────────────────────────
#
# The analyzer consumes the video's native-rate frame stream once (typically
# alongside frame extraction in frame_extractor.extract_frames) and stores
# per-second translation/rotation magnitudes plus per-sample frozen flags.
# Clip-level stability/frozen results are then derived cheaply by
# `check_motion_combined_from_analyzer` without re-opening the video.


@dataclass
class MotionAnalyzer:
    """Stateful LK analyzer populated during single-pass video decode.

    Call `process_frame(frame_bgr, native_frame_idx)` for every native frame
    whose index is a multiple of `frame_skip` (computed from
    `native_fps / target_fps`). Call `finalize()` once decode completes.
    """

    native_fps: float
    total_frames: int
    fast_scale: float = 0.5
    target_fps: float = 30.0
    max_corners: int = 300
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3

    # populated during processing
    frame_skip: int = field(init=False, default=1)
    per_second_trans: dict[int, list[float]] = field(init=False, default_factory=dict)
    per_second_rot: dict[int, list[float]] = field(init=False, default_factory=dict)
    # sampled frames as (native_frame_idx, trans_mag_scaled, rot_mag); one row
    # per processed frame pair (skips the first frame of each second).
    sampled: list[tuple[int, float, float]] = field(init=False, default_factory=list)
    frames_processed: int = field(init=False, default=0)

    # internal LK state
    _prev_gray: np.ndarray | None = field(init=False, default=None, repr=False)
    _prev_pts: np.ndarray | None = field(init=False, default=None, repr=False)
    _cur_sec: int = field(init=False, default=-1, repr=False)
    _feat_params: dict = field(init=False, default_factory=dict, repr=False)
    _lk_params: dict = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.frame_skip = max(1, round(self.native_fps / self.target_fps))
        self._feat_params = _feature_params(self.max_corners)
        self._lk_params = _lk_params(self.lk_win_size, self.lk_max_level)

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> None:
        """Consume a single native-rate frame at LK sampling cadence.

        Caller is responsible for only invoking this on frames with
        `frame_idx % frame_skip == 0`.
        """
        self.frames_processed += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.fast_scale != 1.0:
            gray = cv2.resize(gray, (0, 0),
                              fx=self.fast_scale, fy=self.fast_scale)

        # Map frame_idx → second using native_fps.
        sec = int(frame_idx // self.native_fps) if self.native_fps > 0 else 0

        if sec != self._cur_sec:
            self._cur_sec = sec
            self.per_second_trans.setdefault(sec, [])
            self.per_second_rot.setdefault(sec, [])
            self._prev_gray = gray
            self._prev_pts = cv2.goodFeaturesToTrack(gray, **self._feat_params)
            return

        if self._prev_pts is None or len(self._prev_pts) < 4:
            self._prev_pts = cv2.goodFeaturesToTrack(gray, **self._feat_params)
            self._prev_gray = gray
            return

        curr_pts, status = _lk_track(
            self._prev_gray, gray, self._prev_pts, self._lk_params,
            self.lk_win_size, self.lk_max_level,
        )
        good_prev = self._prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_prev) >= 4:
            T, _ = cv2.estimateAffinePartial2D(
                good_prev, good_curr,
                method=cv2.RANSAC, ransacReprojThreshold=3,
            )
            if T is not None:
                dx, dy = T[0, 2], T[1, 2]
                trans_mag = float(np.sqrt(dx ** 2 + dy ** 2))
                rot_mag = float(abs(np.degrees(np.arctan2(T[1, 0], T[0, 0]))))
                self.per_second_trans[sec].append(trans_mag)
                self.per_second_rot[sec].append(rot_mag)
                self.sampled.append((frame_idx, trans_mag, rot_mag))

        if frame_idx % 60 == 0 or len(good_curr) < 20:
            self._prev_pts = cv2.goodFeaturesToTrack(gray, **self._feat_params)
        else:
            self._prev_pts = good_curr.reshape(-1, 1, 2)
        self._prev_gray = gray


def check_motion_combined_from_analyzer(
    analyzer: MotionAnalyzer,
    *,
    start_sec: float,
    end_sec: float,
    shaky_score_threshold: float = 0.181,
    trans_threshold: float = 8.0,
    jump_threshold: float = 30.0,
    rot_threshold: float = 0.3,
    variance_threshold: float = 6.0,
    w_trans: float = 0.35,
    w_var: float = 0.25,
    w_rot: float = 0.20,
    w_jump: float = 0.20,
    frozen_max_consecutive: int = 30,
    frozen_trans_threshold: float = 0.1,
    frozen_rot_threshold: float = 0.001,
    highpass_window_sec: float = 0.5,
) -> tuple[CheckResult, CheckResult]:
    """Derive segment stability + frozen results from a pre-populated analyzer.

    Replaces `check_motion_combined`'s per-segment video re-open. Uses the
    per-second trans/rot arrays and the sampled-frame sequence already
    collected during single-pass decode.  A high-pass filter separates
    intentional camera movement from jitter before scoring.
    """
    native_fps = analyzer.native_fps
    inv_scale = 1.0 / analyzer.fast_scale if analyzer.fast_scale != 1.0 else 1.0
    eff_skip = analyzer.frame_skip

    sec_start = int(start_sec)
    sec_end = int(np.ceil(end_sec))
    total_seconds = end_sec - start_sec

    score_kw = dict(
        trans_threshold=trans_threshold, jump_threshold=jump_threshold,
        rot_threshold=rot_threshold, variance_threshold=variance_threshold,
        w_trans=w_trans, w_var=w_var, w_rot=w_rot, w_jump=w_jump,
    )

    # Concatenate translations/rotations across all seconds in range,
    # apply high-pass filter to isolate jitter, then re-split and score.
    hp_window = max(1, int(round(highpass_window_sec * (native_fps / eff_skip))))
    all_trans: list[float] = []
    all_rots: list[float] = []
    sec_order: list[int] = []
    sec_counts: list[int] = []
    frame_count = 0
    for sec in range(sec_start, sec_end):
        trans = analyzer.per_second_trans.get(sec, [])
        rots = analyzer.per_second_rot.get(sec, [])
        if not trans:
            continue
        all_trans.extend(trans)
        all_rots.extend(rots)
        sec_order.append(sec)
        sec_counts.append(len(trans))
        frame_count += len(trans) + 1

    per_sec_scores: dict[int, float] = {}
    if all_trans:
        filtered_trans = _highpass_signal(np.array(all_trans), hp_window)
        filtered_rots = _highpass_signal(np.array(all_rots), hp_window)
        idx = 0
        for sec, count in zip(sec_order, sec_counts):
            sec_t = filtered_trans[idx:idx + count].tolist()
            sec_r = filtered_rots[idx:idx + count].tolist()
            score, _ = _transforms_to_score(
                sec_t, sec_r, scale=inv_scale, **score_kw,
            )
            per_sec_scores[sec] = score
            idx += count

    shaky_secs = [s for s, sc in per_sec_scores.items()
                  if sc >= shaky_score_threshold]
    overall = float(np.mean(list(per_sec_scores.values()))) if per_sec_scores else 0.0
    stab_pass = overall <= shaky_score_threshold

    stability_result = CheckResult(
        status="pass" if stab_pass else "fail",
        metric_value=round(overall, 4),
        confidence=1.0,
        details={
            "overall_score": round(overall, 4),
            "shaky_score_threshold": shaky_score_threshold,
            "shaky_seconds": shaky_secs,
            "shaky_seconds_count": len(shaky_secs),
            "total_seconds": int(total_seconds),
            "frames_analysed": frame_count,
            "fast_scale": analyzer.fast_scale,
            "highpass_window_sec": highpass_window_sec,
            "native_fps": round(native_fps, 2),
            "target_fps": analyzer.target_fps,
            "effective_skip": eff_skip,
            "source": "single_pass_analyzer",
        },
    )

    # Frozen detection: walk sampled-frame sequence restricted to clip range.
    range_start = int(start_sec * native_fps)
    range_end = int(end_sec * native_fps)
    frozen_run = 0
    frozen_longest = 0
    frozen_segments_found: list[dict] = []
    frozen_run_start = 0
    lk_pairs_checked = 0

    for frame_idx, trans_mag, rot_mag in analyzer.sampled:
        if frame_idx < range_start:
            continue
        if frame_idx >= range_end:
            break
        lk_pairs_checked += 1
        trans_native = trans_mag * inv_scale
        is_frozen = (trans_native < frozen_trans_threshold
                     and rot_mag < frozen_rot_threshold)
        if is_frozen:
            if frozen_run == 0:
                frozen_run_start = max(0, frame_idx - eff_skip)
            frozen_run += 1
        else:
            if frozen_run > 0:
                if frozen_run > frozen_longest:
                    frozen_longest = frozen_run
                if frozen_run > max(1, round(frozen_max_consecutive / eff_skip)):
                    frozen_segments_found.append({
                        "start_frame": frozen_run_start,
                        "length_sampled": frozen_run,
                    })
                frozen_run = 0

    if frozen_run > 0:
        if frozen_run > frozen_longest:
            frozen_longest = frozen_run
        if frozen_run > max(1, round(frozen_max_consecutive / eff_skip)):
            frozen_segments_found.append({
                "start_frame": frozen_run_start,
                "length_sampled": frozen_run,
            })

    frozen_eff_max = max(1, round(frozen_max_consecutive / eff_skip))
    frozen_pass = frozen_longest <= frozen_eff_max
    frozen_metric = frozen_longest / frozen_eff_max if frozen_eff_max > 0 else 0.0
    frozen_longest_native = frozen_longest * eff_skip
    frozen_dur_s = round(frozen_longest_native / native_fps, 2) if native_fps > 0 else 0

    frozen_result = CheckResult(
        status="pass" if frozen_pass else "fail",
        metric_value=round(min(frozen_metric, 10.0), 4),
        confidence=1.0,
        details={
            "longest_frozen_run_sampled": frozen_longest,
            "longest_frozen_run_native_est": frozen_longest_native,
            "effective_max_consecutive": frozen_eff_max,
            "max_consecutive_native": frozen_max_consecutive,
            "frozen_trans_threshold": frozen_trans_threshold,
            "frozen_rot_threshold": frozen_rot_threshold,
            "lk_pairs_checked": lk_pairs_checked,
            "native_fps": round(native_fps, 2),
            "effective_frame_step": eff_skip,
            "frozen_duration_s": frozen_dur_s,
            "frozen_segments": frozen_segments_found[:10],
            "source": "single_pass_analyzer",
        },
    )

    return stability_result, frozen_result
