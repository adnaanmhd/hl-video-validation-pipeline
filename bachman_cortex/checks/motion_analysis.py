"""Motion analysis checks.

Camera stability via two-stage sparse Lucas-Kanade optical flow:
  Stage 1 — fast pre-screen: downscaled + every Nth frame → per-second scores
  Stage 2 — deep analysis: full-res on flagged seconds only
Frozen segment detection via SSIM at native FPS (streaming, memory-efficient).
"""

import cv2
import numpy as np
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

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **lk)

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
    shaky_score_threshold: float = 0.30,
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_sec = int(fps)
    total_seconds = total_frames / fps

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
        sf = sec * frames_per_sec
        ef = min(sf + frames_per_sec, total_frames)
        if sf >= total_frames:
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
            sf = sec * frames_per_sec
            ef = min(sf + frames_per_sec, total_frames)

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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Subsampling: read every frame_step-th frame
    frame_step = max(1, round(native_fps / target_fps))
    effective_fps = native_fps / frame_step
    # Scale the threshold proportionally to the sampling rate
    effective_max = max(1, round(max_consecutive * effective_fps / native_fps))

    # Read first frame
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
    frame_idx = 0  # native frame index of last read

    while True:
        # Skip ahead by frame_step
        frame_idx += frame_step
        if frame_idx >= total_frames:
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
