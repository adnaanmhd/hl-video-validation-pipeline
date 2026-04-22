"""Motion analysis — stability (camera jitter) and frozen-frame detection.

Single streaming pass over the native-FPS frame stream. The analyzer is
fed only at its cadence (`native_fps / target_fps_motion`) via
`process_frame`. At end of video, `finalize_whole_video()` returns the
pass/fail verdicts plus per-native-frame arrays (motion jitter broadcast
to every frame within its second, frozen flag per-frame) for the
parquet schema.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field


# ── Lucas-Kanade helpers ─────────────────────────────────────────────────────

def _feature_params(max_corners: int = 300) -> dict:
    return dict(maxCorners=max_corners, qualityLevel=0.01,
                minDistance=10, blockSize=3)


def _lk_params(win_size: tuple[int, int] = (21, 21),
               max_level: int = 3) -> dict:
    return dict(winSize=win_size, maxLevel=max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          30, 0.01))


# CUDA LK acceleration

_CUDA_LK = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cv2.cuda.SparsePyrLKOpticalFlow.create()   # smoke test
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
    """Sparse LK optical flow — CUDA when available, CPU fallback."""
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
            pass
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_pts, None, **lk_cpu_params)
    return curr_pts, status.reshape(-1)


def _highpass_signal(values: np.ndarray, window: int) -> np.ndarray:
    """Remove low-frequency component, isolating high-frequency jitter.

    Uses a variable-width rolling mean so edge frames aren't distorted by
    zero-padding. Returns `|values - rolling_mean|`.
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


def _score_second(
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
) -> float:
    """Collapse per-frame LK transforms into a per-second shakiness score."""
    if not translations:
        return 0.0

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

    return round(w_trans * t_s + w_var * v_s + w_rot * r_s + w_jump * j_s, 6)


# ── Stability thresholds ────────────────────────────────────────────────────

@dataclass(frozen=True)
class StabilityThresholds:
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
class FrozenThresholds:
    max_consecutive: int = 900
    trans_threshold: float = 0.1
    rot_threshold: float = 0.001


# ── Finalize result ─────────────────────────────────────────────────────────

@dataclass
class MotionFinalizeResult:
    """Return type of `MotionAnalyzer.finalize_whole_video`."""
    stability_pass: bool
    stability_detected: str
    frozen_pass: bool
    frozen_detected: str

    # Per-native-frame arrays. Aligned with frame_idx 0..total_frames-1.
    # Null in the parquet is represented by None in these lists.
    per_frame_jitter: list[float | None]
    per_frame_frozen: list[bool]

    # Raw diagnostics
    overall_jitter_score: float
    shaky_seconds: list[int]
    longest_frozen_run_sampled: int
    longest_frozen_run_native_est: int


# ── Streaming analyzer ──────────────────────────────────────────────────────

@dataclass
class MotionAnalyzer:
    """Stateful LK analyzer for a single-pass decode.

    The caller invokes `process_frame(frame_bgr, frame_idx)` only on
    frames whose index is a multiple of `frame_skip`. Results are
    finalised once the decode generator is exhausted.
    """

    native_fps: float
    total_frames: int
    fast_scale: float = 0.5
    target_fps: float = 30.0
    max_corners: int = 300
    lk_win_size: tuple[int, int] = (21, 21)
    lk_max_level: int = 3

    frame_skip: int = field(init=False, default=1)
    per_second_trans: dict[int, list[float]] = field(init=False, default_factory=dict)
    per_second_rot: dict[int, list[float]] = field(init=False, default_factory=dict)
    # (native_frame_idx, trans_mag_scaled, rot_mag) for every processed
    # frame pair (skips the first frame of each new second).
    sampled: list[tuple[int, float, float]] = field(init=False, default_factory=list)
    frames_processed: int = field(init=False, default=0)

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
        """Consume a single native-rate frame at LK cadence."""
        self.frames_processed += 1

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.fast_scale != 1.0:
            gray = cv2.resize(gray, (0, 0),
                              fx=self.fast_scale, fy=self.fast_scale)

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

    def finalize_whole_video(
        self,
        stability: StabilityThresholds | None = None,
        frozen: FrozenThresholds | None = None,
    ) -> MotionFinalizeResult:
        """Derive stability + frozen verdicts and per-native-frame arrays."""
        stab = stability or StabilityThresholds()
        fro = frozen or FrozenThresholds()

        inv_scale = (1.0 / self.fast_scale) if self.fast_scale != 1.0 else 1.0
        eff_skip = self.frame_skip

        # ── Stability: high-pass filter then score each second ────────────
        hp_window = max(1, int(round(stab.highpass_window_sec
                                     * (self.native_fps / eff_skip))))
        secs = sorted(self.per_second_trans.keys())
        all_trans: list[float] = []
        all_rots: list[float] = []
        sec_counts: list[int] = []
        for sec in secs:
            trans = self.per_second_trans.get(sec, [])
            rots = self.per_second_rot.get(sec, [])
            if not trans:
                sec_counts.append(0)
                continue
            all_trans.extend(trans)
            all_rots.extend(rots)
            sec_counts.append(len(trans))

        per_sec_scores: dict[int, float] = {}
        if all_trans:
            filt_t = _highpass_signal(np.array(all_trans), hp_window)
            filt_r = _highpass_signal(np.array(all_rots), hp_window)
            idx = 0
            for sec, count in zip(secs, sec_counts):
                if count == 0:
                    continue
                score = _score_second(
                    filt_t[idx:idx + count].tolist(),
                    filt_r[idx:idx + count].tolist(),
                    scale=inv_scale,
                    trans_threshold=stab.trans_threshold,
                    jump_threshold=stab.jump_threshold,
                    rot_threshold=stab.rot_threshold,
                    variance_threshold=stab.variance_threshold,
                    w_trans=stab.w_trans, w_var=stab.w_var,
                    w_rot=stab.w_rot, w_jump=stab.w_jump,
                )
                per_sec_scores[sec] = score
                idx += count

        overall = (float(np.mean(list(per_sec_scores.values())))
                   if per_sec_scores else 0.0)
        shaky_secs = sorted(
            s for s, sc in per_sec_scores.items()
            if sc >= stab.shaky_score_threshold
        )
        stability_pass = overall <= stab.shaky_score_threshold
        stability_detected = f"mean_jitter={overall:.4f}"

        # ── Per-native-frame jitter: broadcast per-second score ───────────
        per_frame_jitter: list[float | None] = [None] * self.total_frames
        for sec, score in per_sec_scores.items():
            lo = int(round(sec * self.native_fps))
            hi = int(round((sec + 1) * self.native_fps))
            hi = min(hi, self.total_frames)
            for i in range(lo, hi):
                per_frame_jitter[i] = score

        # ── Frozen: walk sampled sequence, derive native-frame mask ───────
        eff_max = max(1, round(fro.max_consecutive / eff_skip))
        frozen_run = 0
        frozen_longest = 0
        frozen_runs_native: list[tuple[int, int]] = []  # (start_native, end_native_exclusive)
        run_start_sample_idx = 0
        for sample_idx, (frame_idx, trans_mag, rot_mag) in enumerate(self.sampled):
            trans_native = trans_mag * inv_scale
            is_frozen = (trans_native < fro.trans_threshold
                         and rot_mag < fro.rot_threshold)
            if is_frozen:
                if frozen_run == 0:
                    run_start_sample_idx = sample_idx
                frozen_run += 1
            else:
                if frozen_run > fro.max_consecutive / eff_skip:
                    start_frame = self.sampled[run_start_sample_idx][0]
                    end_frame = self.sampled[sample_idx - 1][0] + eff_skip
                    frozen_runs_native.append((start_frame,
                                               min(end_frame, self.total_frames)))
                if frozen_run > frozen_longest:
                    frozen_longest = frozen_run
                frozen_run = 0

        if frozen_run > 0:
            if frozen_run > frozen_longest:
                frozen_longest = frozen_run
            if frozen_run > fro.max_consecutive / eff_skip:
                start_frame = self.sampled[run_start_sample_idx][0]
                end_frame = self.sampled[-1][0] + eff_skip
                frozen_runs_native.append((start_frame,
                                           min(end_frame, self.total_frames)))

        per_frame_frozen = [False] * self.total_frames
        for lo, hi in frozen_runs_native:
            for i in range(lo, hi):
                if 0 <= i < self.total_frames:
                    per_frame_frozen[i] = True

        frozen_pass = frozen_longest <= eff_max
        longest_native_est = frozen_longest * eff_skip
        frozen_detected = (
            f"longest_run_sampled={frozen_longest} "
            f"(~{longest_native_est}f at native FPS)"
        )

        return MotionFinalizeResult(
            stability_pass=stability_pass,
            stability_detected=stability_detected,
            frozen_pass=frozen_pass,
            frozen_detected=frozen_detected,
            per_frame_jitter=per_frame_jitter,
            per_frame_frozen=per_frame_frozen,
            overall_jitter_score=round(overall, 4),
            shaky_seconds=shaky_secs,
            longest_frozen_run_sampled=frozen_longest,
            longest_frozen_run_native_est=longest_native_est,
        )
