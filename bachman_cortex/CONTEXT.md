# bachman_cortex — Implementation Context

This document captures the architecture and key decisions of the scoring
engine as it exists today. For the locked design plan that preceded the
rewrite, see `SCORING_ENGINE_PLAN.md` at the repo root.

---

## 1. What the package does

`bachman_cortex` scores egocentric videos against a fixed set of
acceptance criteria and emits a per-video report plus a batch rollup.
It is **not** a clip extractor and **not** a yield calculator — the
engine reports observations; decisions about what to do with videos
that fail the technical stage happen downstream.

---

## 2. Runtime flow (single video)

```
ffprobe metadata → run_all_metadata_checks
   │              → build_observations (non-gating; 7 fields always populated)
   │
   ├── any fail? ──▶ VideoScoreReport with all technical/quality = SKIPPED.
   │                 Observations still populated. Write report.md +
   │                 {video}.json. NO parquet.
   ▼
[Stage 2 + 3] Single streaming decode pass @ native FPS
   │   each native frame dispatched to accumulators by cadence:
   │     • MotionAnalyzer          (30 FPS cap; 0.5× → 360p)
   │     • LuminanceAccumulator    (10 FPS; 360p)
   │     • PixelationAccumulator   (10 FPS; 720p)
   │     • ObstructionAccumulator  (10 FPS; 720p)
   │     • Quality model bank      (1 FPS;  720p) — Hands23 + SCRFD + YOLO
   │         → HandsAccumulator (both + single + angle + HOI)
   │         → ParticipantsAccumulator (YOLO + SCRFD + Hands23 extra-hand signal)
   │   per-frame results appended to an in-memory row buffer
   ▼
[Stage 2 finalize] whole-video aggregation of technical checks
   │
   ├── any technical fail? ──▶ quality rendered as SKIPPED; parquet still
   │                            gets the raw columns for debuggability.
   ▼
[Stage 3 finalize] whole-video aggregation of quality metrics
   │   group per-frame → runs → merge short runs → segments
   │
Write {video}.parquet, {video}.json, report.md.
```

Batch mode calls `ScoringEngine.score_video` per input and aggregates
the reports; see `batch.score_batch`.

---

## 3. Module map

| File                                   | Purpose                                                       |
| -------------------------------------- | ------------------------------------------------------------- |
| `scoring_engine.py`                    | `ScoringEngine` — orchestrator + stage gating.                |
| `batch.py`                             | Directory walk + batch driver + error collection.             |
| `cli.py`                               | `hl-score` entrypoint (argparse + config loading).            |
| `config.py`                            | TOML config schema + loader + default-template emitter.       |
| `data_types.py`                        | Report dataclasses + canonical name tuples.                   |
| `segmentation.py`                      | `group_runs` + `merge_short_runs` + per-segment value helpers. |
| `per_frame_store.py`                   | In-memory row buffer + single-shot parquet writer.            |
| `reporting.py`                         | Per-video + batch report writers (MD / JSON / CSV).            |
| `utils/frame_extractor.py`             | `iter_native_frames` — NVDEC/cv2 generator + 720p resize.     |
| `utils/video_metadata.py`              | ffprobe wrapper + avg-GOP packet scan + tag-surface flattener. |
| `utils/metadata_observations.py`       | Seven non-gating metadata observations + vendor registry for stabilization / FOV. |
| `utils/gpmd.py`                        | GoPro GPMD timed-metadata stub (real KLV parser lands with IMU extraction). |
| `checks/video_metadata.py`             | Six metadata checks against ffprobe output.                    |
| `checks/motion_analysis.py`            | `MotionAnalyzer` (stability + frozen finalise).                |
| `checks/luminance.py`                  | `LuminanceAccumulator`.                                        |
| `checks/pixelation.py`                 | `PixelationAccumulator`.                                       |
| `checks/view_obstruction.py`           | `ObstructionAccumulator` ("obstructed" metric).                |
| `checks/hand_visibility.py`            | `HandsAccumulator` — both/single/angle/HOI in one pass.        |
| `checks/participants.py`               | `ParticipantsAccumulator` + wearer filter.                     |
| `checks/check_results.py`              | Shared `CheckResult` dataclass.                                |
| `models/hand_detector.py`              | Hands23 (Faster R-CNN X-101-FPN via Detectron2).               |
| `models/scrfd_detector.py`             | SCRFD-2.5GF face detector via InsightFace.                     |
| `models/yolo_detector.py`              | YOLO11 person detection via Ultralytics.                       |
| `_cv2_dnn_shim.py`                     | Numpy implementation of `cv2.dnn.blobFromImage` so the custom  |
|                                        | OpenCV-CUDA build works with InsightFace.                      |

---

## 4. Invariants

- **One decode per video.** The accumulator-per-cadence pattern routes
  each native frame to the right subset of checks during a single
  streaming pass.
- **Peak frame memory** ≈ a few tens of MB (720p BGR frames in
  NVDEC/CPU pipeline). Does not grow with video length.
- **Per-frame row buffer** ≈ `native_fps × duration × ~80 B`. ~8.6 MB
  for a 1h 30 FPS clip.
- Segmentation and merging run **only at report time** from the
  completed row buffer — never during decode.
- Parquet is the ground truth (always emitted when the engine
  decoded); MD/JSON are the curated view and can hide quality when
  technical failed.
- **Metadata observations are one-per-video**, not per-frame. They
  live on `VideoScoreReport.metadata_observations` and in the batch
  CSV; they are **not** in parquet. They are populated unconditionally
  — even on metadata-gate failure — because they come from the same
  ffprobe call plus one cheap packet-level GOP scan. No decode, no
  model inference.

---

## 5. Configuration

All thresholds and cadences live in a single TOML schema
(`config.py`). `hl-score --dump-default-config` writes the full
populated template to stdout. The schema uses frozen dataclasses, and
unknown keys raise loud `ValueError`s at load time so typos surface
instead of being silently ignored.

Only top-level knobs are on the CLI (`--out-dir`, `--config`,
`--workers`, model-weight overrides). Everything fine-grained is in
the TOML.

---

## 6. Testing

`tests/` contains per-accumulator unit tests using fabricated inputs
(so no model weights required for the unit pass). The accumulators
under `bachman_cortex/checks/` each have a narrow test of pass/fail
semantics and per-frame array shape. `test_segmentation.py` covers
the absorption rules from plan §1 — especially that absorbed
fail-frames' confidences do NOT contribute to the merged pass-run's
`segment_confidence_value`.

End-to-end validation (decode + model inference + report emission)
is exercised manually via `hl-score` on real clips; the automated
tests are enough to catch regressions in the deterministic logic
without needing GPU fixtures.

---

## 7. Model setup

All three models expect weights under
`bachman_cortex/models/weights/` (gitignored). Run
`python bachman_cortex/models/download_models.py --all` to fetch
Hands23 + SCRFD + YOLO at setup time. The top-level CLI accepts
`--hand-detector-repo`, `--scrfd-root`, `--yolo-model` to override
those paths.

GPU acceleration is used throughout when available (CUDA OpenCV for
LK optical flow + NVDEC for decode, ONNXRuntime CUDAExecutionProvider
for SCRFD, Detectron2 CUDA for Hands23, Ultralytics CUDA for YOLO).
CPU fallbacks exist for every path so the engine still runs on a
machine without a GPU — just slowly.

---

## 8. Running

```bash
hl-score path/to/video.mp4
hl-score path/to/dir/
hl-score path/to/dir/ --config ./hl-score.toml
hl-score --dump-default-config > hl-score.toml
```

Outputs land under `results/run_NNN/`. `NNN` auto-allocates via atomic
mkdir-with-bump, safe under concurrent invocation.
