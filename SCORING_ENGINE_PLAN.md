# Scoring Engine — Implementation Plan

**Status:** locked, ready to implement
**Branch:** `scoring-engine-rewrite` (to be created from `main` after a cleanup commit)
**Source of truth for the spec:** [`idea-brief.md`](./idea-brief.md) — this plan supersedes it where they disagree.

Every decision below has been explicitly agreed with Adnaan. Do not re-open closed questions — implement as specified.

---

## 0. What this is

A ground-up replacement for the existing 4-phase validation pipeline (`bachman_cortex/pipeline.py`). The old pipeline classified time ranges as USABLE / UNUSABLE / REJECTED, computed yield, and had a HITL construct. All of that is **gone**.

The new engine:
- Runs three **gated** stages: metadata checks → technical checks → quality metrics.
- Uses a **single decode pass**. Quality metrics compute even when technical ultimately fails — justified by `P(technical pass) > 0.7` in the target corpus (wasted inference on the failing minority is cheaper than a second decode pass on the passing majority).
- On technical fail: report renders quality as `skipped: true`, but the raw per-frame quality columns remain in the parquet (debuggability).
- No segmentation, no yield, no HITL, no pass/fail on quality metrics — quality metrics are readings only.
- Outputs per-video reports (`.md` + `.json` + `.parquet`) and batch reports (`.md` + `.json` + `.csv`).

Rename scope: class + CLI + docs only. Package stays `bachman_cortex`.

---

## 1. Locked decisions

### Execution model

| Area | Decision |
|---|---|
| **Flow** | metadata → technical → quality. Stage failure propagates downstream as `skipped: true` at report level. Single decode pass always runs all checks. |
| **Decode strategy** | Single pass per video. NVDEC via `cv2.cudacodec` when available, `cv2.VideoCapture` fallback. Both downscale to 720p long-edge at download time. |
| **Per-frame storage** | **In-memory row buffer during decode, single-shot parquet write at end-of-video.** Worst case ~10 MB (1h × 30 FPS × 80 B). Simpler than streaming flush; eliminates chunk-boundary state concerns for rolling windows. |
| **Input** | Single MP4 file OR directory of MP4s (recursive). `.mp4` case-insensitive. Follow symlinks with cycle detection (visited-inode set). Skip hidden files (dot-prefix). Audio-only MP4s → mark as error + skip. |
| **Package name** | `bachman_cortex` (unchanged). |
| **CLI name** | `hl-score` (replaces `hl-validate`). |
| **Branch** | `scoring-engine-rewrite` from `main` (post-cleanup commit). |
| **Commit author** | `m.adnaan161@gmail.com`. |
| **Parallelism** | Auto-detect: `workers = max(1, min(os.cpu_count() // 2, gpu_count × 2))`. Override via `--workers N`. |
| **Error policy (batch)** | One video's failure does not abort the batch; captured in `BatchScoreReport.errors`. |

### Cadences (all configurable via config file)

| Check | Target FPS | Resolution |
|---|---|---|
| Motion (stability source) | 30 (cap) | 360p (in-place 0.5× from 720p) |
| Frozen (derived from motion) | 10 | Same signal as motion |
| Luminance | 10 | 360p (in-place downscale from 720p) |
| Pixelation | 10 | 720p |
| Quality metrics (hands, HOI, angle, participants, obstruction) | 2 | 720p |

All cadences rooted at `frame_idx = 0`. Frame `i` ticks cadence `C` iff `i % round(native_fps / target_fps_C) == 0`.

### Per-metric condition and value semantics

**face_presence as a standalone metric is REMOVED.** SCRFD face detection is folded into `participants`.

| Metric | Per-frame PASS condition | Value (raw per-frame) | Value on fail |
|---|---|---|---|
| Both hands visibility | Hands23 detects L **and** R, each conf ≥ 0.7 (if L or R missing → fail) | `max(L_conf, R_conf)` | 0.0 |
| Single hand visibility | ≥1 hand with Hands23 conf ≥ 0.7 | max Hands23 conf | 0.0 |
| Hand-object interaction | ≥1 detected hand with contact state ∈ {P, F} | contact char of qualifying hand; on fail, most-confident hand's char ("N"/"S"/"O"); null if zero hands | same column |
| Hand angle | all detected hands have angle ≤ 40° (zero hands → fail) | mean angle (°) of detected hands | mean angle (°); NaN if no hands |
| Participants | ≥1 "other person" signal from {YOLO person conf ≥ 0.6 wearer-filtered, SCRFD face conf ≥ 0.6 wearer-filtered, Hands23 detects >2 hands with 3rd+ hand conf ≥ 0.7} | `max(YOLO_conf, SCRFD_conf, extra_hand_conf)` | 0.0 |
| Obstructed (renamed from view_obstruction) | heuristic triggers ≥2 of 4 signals | `true` | `false` |

Both-hands and single-hand metrics may pass for the same frame (not mutually exclusive).

### Wearer filter (applies to YOLO persons AND SCRFD faces)

- Exclude detections overlapping Hands23 hands (wearer's own body/face via reflection).
- Exclude bottom-center region detections.
- Min bbox height > 15% of frame height.

### Technical check thresholds (unchanged from current)

| Check | Rule |
|---|---|
| Luminance | ≥ 80% good frames (no dead black <15, too dark 15–45, blown out >230, no flicker) |
| Stability | whole-video mean jitter score ≤ 0.181 (high-pass filtered LK) |
| Frozen | no run > 60 consecutive native-frame-equivalents with near-zero motion (trans < 0.1 px, rot < 0.001°) |
| Pixelation | ≥ 80% frames with blockiness ratio ≤ 1.5 |

### Metadata thresholds (unchanged from current code; duration bump already committed)

| Check | Rule |
|---|---|
| format | container is MP4 |
| encoding | H.264 or H.265 (HEVC) |
| resolution | ≥ 1920×1080 displayed (after rotation) |
| frame_rate | ≥ 28 FPS |
| duration | ≥ 59 seconds |
| orientation | rotation ∈ {0, 90, 270} **AND** displayed landscape |

All 6 must pass for technical to run. Any single failure marks the video metadata-failed; technical and quality are reported SKIPPED (no decode performed).

### Segment merging rules

| Rule | Decision |
|---|---|
| Merge threshold | 1.0s default, configurable. |
| Absorption | Runs shorter than threshold absorb into preceding neighbor (flip state). First-run edge case: absorb into following run. |
| Iteration | Single left-to-right pass. No cascade re-merging. |
| Whole-video single short run | Left as-is (no neighbor). |
| Trailing short run after merge | Left as-is (boundary artifact). |
| Merged segment value | Inherits absorbing-neighbor's value. |
| Merged pass-segment confidence | Max over qualifying frames only; does NOT include absorbed-fail frames. |
| Hand-angle value per segment | Segment-wide mean angle (over all frames in the segment, merged or not). |
| Hand-object contact char per segment | Most common contact char in the segment. |
| Raw parquet | Un-merged (raw per-frame). Merging applies only to JSON/MD report output. |
| Percentage computation | From un-merged per-frame array. |

### Parquet row model

- **Dense rows:** one row per native frame. Null-fill where a cadence didn't tick on that frame.
- **`motion_jitter`:** per-second score broadcast to every native frame within that second.
- **`frozen_state`:** per-native-frame flag, true iff the frame is part of a detected frozen run.
- Quality columns are present in the parquet regardless of whether technical failed (JSON/MD are the curated view; parquet is raw truth).

---

## 2. Architecture

### Runtime flow for one video

```
ffprobe metadata → run_all_metadata_checks
   │
   ├── any fail? ──▶ VideoScoreReport with all technical/quality = SKIPPED.
   │                 Write report.md + {video}.json. NO parquet. Exit.
   ▼
[Stage 2 + 3] Single streaming decode pass @ native FPS
   │   chunked (default 256 frames) for frame-memory discipline
   │   each native frame dispatched to accumulators by cadence:
   │     • MotionAnalyzer          (30 FPS cap; 0.5× → 360p)
   │     • LuminanceAccumulator    (10 FPS; 360p)
   │     • PixelationAccumulator   (10 FPS; 720p)
   │     • ObstructionAccumulator  (10 FPS; 720p)
   │     • Quality model bank      (1 FPS;  720p) — Hands23 + SCRFD + YOLO
   │         → HandsAccumulator (both + single + angle + HOI)
   │         → ParticipantsAccumulator (YOLO + SCRFD + Hands23 extra-hand signal)
   │   per-frame results appended to in-memory row buffer (one row per native frame)
   ▼
[Stage 2 finalize] whole-video aggregation of technical checks
   │   • stability: highpass-filter per-second scores → mean → threshold
   │   • frozen: walk MotionAnalyzer.sampled → detect runs > max
   │   • luminance / pixelation / obstruction: % good-frames
   │
   ├── any technical fail? ──▶ quality_skipped = True; report renders quality = SKIPPED.
   ▼
[Stage 3 finalize] whole-video aggregation of quality metrics
   │   group per-frame condition states → runs → merge <1s → segments
   │   percent_frames computed from un-merged per-frame array
   ▼
Write {video}.parquet (raw per-frame, all columns populated per schema)
Write {video}.json (merged segments; skipped flags respected)
Write report.md
```

### Invariants

- **One decode per video.**
- **Peak frame memory** ≈ `chunk_size × 2.7 MB` (720p BGR). Does not grow with video length.
- **Per-frame row buffer** ≈ `native_fps × duration × ~80 B`. 1h × 30 FPS ≈ 8.6 MB.
- Segmentation and merging run **only at report time**, from the completed row buffer — not during decode.
- Parquet is the ground truth; report files are the curated view.

---

## 3. File-level changes

### Delete
- `bachman_cortex/pipeline.py`
- `bachman_cortex/reporting.py` (replaced)
- `bachman_cortex/run_batch.py`
- `bachman_cortex/data_types.py` (replaced)
- `bachman_cortex/utils/segment_ops.py`
- `bachman_cortex/utils/early_stop.py`
- `bachman_cortex/utils/transcode.py` (no callers remain after `pipeline.py` deletion — verified)
- `bachman_cortex/utils/review_runs.py` (only used by `reporting.py`, which is also being deleted — verified)
- `bachman_cortex/checks/face_presence.py` (metric removed; SCRFD integrated into participants)
- `run_lb_check.py` at project root (untracked scratch)

### Keep unchanged
- `scripts/install_opencv_cuda.sh`
- `bachman_cortex/_cv2_dnn_shim.py`
- `bachman_cortex/__init__.py` (cv2.dnn shim install)
- `bachman_cortex/models/**` (all detectors + weights; SCRFD still loaded for participants)
- `bachman_cortex/utils/video_metadata.py`
- `bachman_cortex/checks/check_results.py` (CheckResult dataclass — still useful)
- `bachman_cortex/checks/video_metadata.py` (duration 10→59s already committed)

### Refactor

| File | Change |
|---|---|
| `utils/frame_extractor.py` | Convert from list-returning function to a **generator** yielding `(frame_idx, native_fps, frame_720p_bgr)`. Yields **every native frame**; consumers subsample by cadence. Keep NVDEC + cv2 fallback. Remove `motion_analyzer` kwarg. |
| `checks/motion_analysis.py` | Keep streaming `MotionAnalyzer`. Add `finalize_whole_video() -> (stability_result, frozen_result, per_frame_jitter_array, per_frame_frozen_array)`. Remove `check_motion_combined_from_analyzer`. Per-second score broadcast to all native frames in that second. |
| `checks/luminance.py` | New `LuminanceAccumulator`. `process_frame(frame_360p)` + `finalize() -> (class_array, flicker_array, pass/fail)`. 10-frame rolling stddev for flicker; state persists across chunks (buffered-in-memory → no boundary issue). |
| `checks/pixelation.py` | Same pattern. `PixelationAccumulator`. |
| `checks/view_obstruction.py` | `ObstructionAccumulator`. Per-frame bool via existing heuristic. Metric name in data model: `obstructed`. |
| `checks/hand_visibility.py` | Per-frame `{both_hands_pass, both_conf, single_hand_pass, single_conf}`. Both-hands requires explicit L AND R detection. |
| `checks/hand_object_interaction.py` | Per-frame `{pass, contact_char}`. Pass iff ≥1 detected hand has contact ∈ {P, F}. |
| `checks/pov_hand_angle.py` | Per-frame `{pass, mean_angle}`. Zero hands → pass=False, mean_angle=NaN. |
| `checks/participants.py` | Per-frame `{pass, conf, source, extra_hands_count}`. Sources: `yolo` (≥0.6, wearer-filtered), `scrfd` (≥0.6, wearer-filtered), `extra_hands` (Hands23 detections beyond L+R, conf ≥ 0.7). Wearer filter applies to YOLO and SCRFD. |

### New

| File | Purpose |
|---|---|
| `bachman_cortex/scoring_engine.py` | `ScoringEngine` class. `score_video(path) -> VideoScoreReport`, `score_batch(paths) -> BatchScoreReport`. Owns model loading, warmup, decode orchestration, stage gating. |
| `bachman_cortex/data_types.py` | New dataclasses (see §4). |
| `bachman_cortex/reporting.py` | `write_video_report(result, out_dir)`, `write_batch_report(batch, out_dir)`. |
| `bachman_cortex/per_frame_store.py` | In-memory row buffer + single-shot parquet writer. One public method `flush(path)`. |
| `bachman_cortex/segmentation.py` | Pure functions: `group_runs(per_frame_array, condition_fn) -> list[Run]`, `merge_short_runs(runs, min_duration_s) -> list[Run]`. |
| `bachman_cortex/config.py` | TOML config loader. Schema mirrors all thresholds + cadences. `--dump-default-config` emits a fully-populated template. |
| `bachman_cortex/cli.py` | `hl-score` entrypoint. |

### `pyproject.toml`
- Rename console script: `hl-validate` → `hl-score` pointing at `bachman_cortex.cli:main`.
- Add `pyarrow` dependency.
- `tomllib` is stdlib on Python 3.11+; add `tomli` as a fallback only if supporting <3.11.

---

## 4. Data model (`data_types.py`)

```python
@dataclass
class MetadataCheckResult:
    check: str          # "format" | "encoding" | "resolution" | "frame_rate" | "duration" | "orientation"
    status: str         # "pass" | "fail"
    accepted: str       # human-readable threshold
    detected: str       # human-readable observed

@dataclass
class TechnicalCheckResult:
    check: str          # "stability" | "frozen" | "luminance" | "pixelation"
    status: str         # "pass" | "fail" | "skipped"
    accepted: str
    detected: str
    skipped: bool = False

@dataclass
class QualitySegment:
    start_s: float      # inclusive
    end_s: float        # exclusive (half-open [start_s, end_s))
    duration_s: float
    value: float | str  # type per metric; see §1 table
    value_label: str    # "confidence" | "angle" | "contact_state" | "obstructed"

@dataclass
class QualityMetricResult:
    metric: str         # "both_hands_visibility" | "single_hand_visibility" | "hand_obj_interaction" | "hand_angle" | "participants" | "obstructed"
    percent_frames: float       # 0.0 if skipped; computed from un-merged per-frame array
    segments: list[QualitySegment]
    skipped: bool = False

@dataclass
class CheckStats:
    pass_count: int
    pass_duration_s: float
    fail_count: int
    fail_duration_s: float
    skipped_count: int = 0

@dataclass
class QualityStats:
    mean_percent: float
    median_percent: float
    min_percent: float
    max_percent: float

@dataclass
class ProcessingErrorReport:
    video_path: str
    video_name: str
    error_reason: str   # "audio_only" | "decode_failed" | "corrupt" | "metadata_probe_failed" | ...

@dataclass
class VideoScoreReport:
    video_path: str
    video_name: str
    generated_at: str                # ISO 8601 UTC
    processing_wall_time_s: float    # wall-clock measured inside the worker task
    duration_s: float

    metadata_checks: list[MetadataCheckResult]
    technical_checks: list[TechnicalCheckResult]
    quality_metrics: list[QualityMetricResult]

    technical_skipped: bool          # True iff metadata failed
    quality_skipped: bool            # True iff metadata OR technical failed

@dataclass
class BatchScoreReport:
    generated_at: str                # ISO 8601 UTC
    video_count: int
    total_duration_s: float
    total_wall_time_s: float         # real elapsed time of the batch run

    metadata_check_stats: dict[str, CheckStats]
    technical_check_stats: dict[str, CheckStats]
    quality_metric_stats: dict[str, QualityStats]

    videos: list[VideoScoreReport]          # in input order
    errors: list[ProcessingErrorReport]     # audio-only, decode-failed, etc.
```

### Parquet schema (per video)

One row per native frame, dense. Null where cadence didn't tick. Quality columns always present, regardless of technical outcome.

| Column | Type | Notes |
|---|---|---|
| `frame_idx` | int32 | native frame index |
| `timestamp_s` | float64 | `frame_idx / native_fps` |
| `motion_jitter` | float32 | per-second score broadcast to all native frames in that second; null if motion didn't process that second |
| `frozen_state` | bool | true if the native frame is part of a detected frozen run |
| `luminance_class` | int8 | 0=dead black, 1=too dark, 2=usable, 3=blown out; null outside luminance cadence |
| `luminance_flicker` | bool | from rolling stddev; null outside luminance cadence |
| `pixelation_ratio` | float32 | null outside pixelation cadence |
| `both_hands_pass` | bool | null outside quality cadence |
| `both_hands_conf` | float32 | 0.0 on fail |
| `single_hand_pass` | bool | null outside quality cadence |
| `single_hand_conf` | float32 | 0.0 on fail |
| `hand_obj_pass` | bool | null outside quality cadence |
| `hand_obj_contact` | str | "N"/"S"/"O"/"P"/"F" or null when no hands |
| `hand_angle_pass` | bool | null outside quality cadence |
| `hand_angle_mean_deg` | float32 | NaN when no hands |
| `participant_pass` | bool | null outside quality cadence |
| `participant_conf` | float32 | 0.0 on none |
| `participant_source` | str | "yolo" / "scrfd" / "extra_hands" / null |
| `extra_hands_count` | int8 | Hands23 detections beyond L+R (debug aid) |
| `obstructed` | bool | true = obstructed |

---

## 5. Output layout

```
results/run_NNN/
├── {video_name}/
│   ├── report.md
│   ├── {video_name}.json
│   └── {video_name}.parquet      # omitted if metadata failed (nothing decoded)
├── batch_report.md
├── batch_results.json
└── batch_results.csv
```

- `NNN` zero-padded 3-digit minimum; auto-extends to 4+ digits after `run_999`.
- Run dir allocated via atomic `os.makedirs(run_dir, exist_ok=False)` with bump-on-collision retry (cap 100) — safe for concurrent invocations.
- Single-video runs still write batch files (video_count = 1).

---

## 6. Report formatting rules

1. **Skipped stages:** MD table renders `Status = SKIPPED`, `-` in accepted/detected. JSON has `skipped: true`, `percent_frames: 0.0` (for quality), `segments: []`.
2. **Metadata fail:** all 4 technical rows rendered as SKIPPED; all 6 quality rows rendered as SKIPPED. Report + JSON produced; **no parquet** (nothing was decoded).
3. **Technical fail (any one check):** quality rendered as SKIPPED in MD/JSON. Parquet **is** written with all columns populated.
4. **Technical pass:** quality section fully populated.
5. **Quality segment column header:** per-metric — `confidence` / `angle` / `contact_state` / `obstructed`.
6. **Segment value on condition-fail frames:** `0.00` for conf-based metrics; actual mean angle (or `NaN`) for hand-angle; actual contact char for hand-object; `false` (0.0) for obstructed fail.
7. **Segment merging:** runs shorter than 1.0s absorbed into preceding neighbor (or following for first run); single left-to-right pass, no cascade. Parquet remains un-merged.
8. **Percent in quality section:** computed from the un-merged per-frame array.
9. **Mixed-type value formatting:** floats → 2 decimals (`0.92`); strings → as-is (`P`). CSV string-casts everything.
10. **Timestamps:** half-open `[start_s, end_s)`.
11. **`generated_at`:** ISO 8601 UTC (e.g., `2026-04-17T14:32:05Z`).
12. **CSV columns for checks:** two columns per check — `{check}_status` (PASS/FAIL/SKIPPED) and `{check}_value` (string-cast numeric metric). Quality metric cells store `% frames` or `SKIPPED`. Video rows in input order.

---

## 7. CLI

```bash
hl-score path/to/video.mp4
hl-score path/to/dir/
hl-score path/to/dir/ --config ./hl-score.toml
hl-score --dump-default-config > hl-score.toml
```

### Flags (top-level knobs only)

| Flag | Default | Meaning |
|---|---|---|
| `--out-dir` | `results/` | Output root. `run_NNN` created under this. |
| `--config` | (none) | Path to TOML config with cadences + thresholds. Overrides built-in defaults. |
| `--workers` | auto | Auto-detected: `max(1, min(os.cpu_count() // 2, gpu_count × 2))`. |
| `--hand-detector-repo` | (built-in) | Override Hands23 weights path. |
| `--scrfd-root` | (built-in) | Override SCRFD weights path. |
| `--yolo-model` | (built-in) | Override YOLO weights path. |
| `--dump-default-config` | — | Write full default TOML to stdout and exit. |

### Config file (TOML) — schema summary

```toml
[cadences]
quality_fps = 1.0
motion_fps = 30.0      # cap
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
max_consecutive = 60
trans_threshold = 0.1
rot_threshold = 0.001

[technical.luminance]
good_frame_ratio = 0.80
# ... all sub-thresholds exposed

[technical.pixelation]
good_frame_ratio = 0.80
max_blockiness_ratio = 1.5

[quality.hands]
hands23_conf = 0.7

[quality.angle]
max_degrees = 40.0

[quality.participants]
yolo_conf = 0.6
scrfd_conf = 0.6
extra_hand_conf = 0.7
min_bbox_height_frac = 0.15
```

`hl-score --dump-default-config` emits the full populated template for discoverability.

---

## 8. Implementation order

Each step is one commit. Stops at any step leave `main` unaffected.

0. **Cleanup commit on main:** commit existing modified `bachman_cortex/checks/video_metadata.py` (duration 10→59s) and staged `.planning/**` deletions. Untracked scratch files remain untracked.
1. **Branch:** `git checkout -b scoring-engine-rewrite` from post-cleanup main. Commit `SCORING_ENGINE_PLAN.md` as the first commit on the branch.
2. **Housekeeping:** delete files listed in §3. Empty `data_types.py` and `reporting.py` to stubs. Acknowledge any now-broken tests.
3. **Config loader:** `config.py` + TOML schema + `--dump-default-config`. Unit test: roundtrip + default-emission.
4. **Data types:** build new `data_types.py`.
5. **Per-frame store:** in-memory buffer + single-shot parquet writer. Unit test on synthetic arrays.
6. **Frame streamer refactor:** `utils/frame_extractor.py` → generator yielding every native frame. One-file smoke test on NVDEC and cv2 fallback.
7. **Check accumulators** (one commit each, each with per-accumulator unit tests):
   - motion (stability + frozen finalize → per-frame jitter + frozen arrays)
   - luminance
   - pixelation
   - view_obstruction (rename metric to `obstructed`)
   - hand_visibility (both + single)
   - hand_object_interaction
   - pov_hand_angle
   - participants (YOLO + SCRFD + Hands23 extra-hand signal; wearer filter applied to both YOLO and SCRFD)
8. **Segmentation utility:** `group_runs`, `merge_short_runs`. Unit tests covering first-run, trailing-run, whole-video-short-run, and pass-segment-confidence-excludes-absorbed-fail-frames cases.
9. **ScoringEngine:** orchestrator, stage gating, model loading, warmup. Single-video end-to-end.
10. **Reporting:** per-video MD + JSON + parquet. Golden-file comparison for MD formatting.
11. **Batch driver:** multi-video, batch MD + JSON + CSV, worker auto-detect, error collection (audio-only, decode-failed, corrupt).
12. **CLI:** `cli.py`, update `pyproject.toml` console script.
13. **Validation runs:** `GH011093.MP4`, `20251223-172252-6ba8971.mp4`, `20251223-202254-01a0213.mp4`. Sanity cases: low-res clip (metadata fail), all-black clip (luminance fail), frozen clip.
14. **Docs rewrite:** `checks.md`, `bachman_cortex/CONTEXT.md`, `README.md`.
15. **PR:** `scoring-engine-rewrite` → `main` with validation output attached.

---

## 9. Known risks / things to watch

- **Hands23 warmup cost** (~2–3s per run). Pre-warm once in `ScoringEngine.__init__`. Amortized across batch.
- **Parquet size at scale:** ~8.6 MB/h at 30 FPS. 5000 × 1h videos ≈ 43 GB. Manageable.
- **MotionAnalyzer across generator refactor:** analyzer already stores only the prior frame. Dispatch layer owns the cadence gate — must call `process_frame` only on frames matching motion skip, not every native frame.
- **cv2.cudacodec with generator:** `nextFrame()` is already lazy. Verify no prefetch when the extractor is converted to `yield`.
- **SCRFD wearer filter (new):** current code applies the filter only to YOLO persons. Extending to SCRFD means wearer-face detections (rare POV reflections) get filtered. Verify on reference clips.
- **Participant source priority:** a frame may qualify via multiple signals. Report the highest-conf source in `participant_source`.
- **Worker count × GPU contention:** `gpu_count × 2` is conservative. May need tuning per machine.
- **Symlink cycles:** visited-inode set prevents loops. Bails with an explicit error if a cycle is detected.
- **Concurrent `run_NNN` allocation:** atomic `mkdir(exist_ok=False)` + retry. Capped at 100 attempts.
- **Processing time in GPU-heavy pipeline:** `processing_wall_time_s` is wall-clock inside the worker (meaningful under parallelism because each worker sees only its own video).

---

## 10. Reference to existing context

- `idea-brief.md` — original spec. This plan supersedes it where they disagree (face_presence removed, segment value semantics, obstruction renaming, single-pass with report-level skip).
- `checks.md` — **to be rewritten** after implementation.
- `bachman_cortex/CONTEXT.md` — **to be rewritten** after implementation.
- `bachman_cortex/pipeline.py` — to delete, but useful reference for the existing model-loading sequence.
- `bachman_cortex/checks/*` — refactor targets; existing per-frame logic can be extracted.
- Git history — `a9adc1e` onwards contains the most recent check implementations (luminance/pixelation split, stability high-pass filter, HEVC acceptance).
