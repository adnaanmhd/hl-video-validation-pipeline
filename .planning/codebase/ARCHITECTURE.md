# Architecture

**Analysis Date:** 2026-04-15

## Pattern Overview

**Overall:** 4-phase sequential validation pipeline with ML-driven segment filtering and acceptance criteria checking.

**Key Characteristics:**
- Phase-gated architecture: metadata → segment filtering → segment validation → yield calculation
- Check-based composition: result aggregation across multiple deterministic and ML-based checks
- Per-frame to segment pipeline: frame-level classifications aggregate into time segments
- Caching for efficiency: Phase 1 frames and ML detections reused in Phase 2 without redundant inference
- Parallel video processing: multi-worker batch mode with single-process model isolation per worker

## Layers

**Entry Point / CLI:**
- Purpose: Batch video processing orchestration and configuration
- Location: `bachman_cortex/run_batch.py`
- Contains: CLI argument parsing, batch video collection, worker pool setup, report aggregation
- Depends on: `pipeline.py`, `reporting.py`, `data_types.py`
- Used by: Shell scripts and direct CLI invocation

**Pipeline Core:**
- Purpose: Main 4-phase validation logic and orchestration
- Location: `bachman_cortex/pipeline.py`
- Contains: `ValidationProcessingPipeline` class implementing phases 0–3, model loading, frame extraction
- Depends on: All check modules, all model modules, utils (frame extraction, segment ops, motion analysis)
- Used by: `run_batch.py`, direct imports

**Check Implementations:**
- Purpose: Deterministic and ML-based acceptance criteria
- Location: `bachman_cortex/checks/` directory
- Contains: Individual check functions (e.g., `check_hand_visibility()`, `check_motion_camera_stability()`)
- Depends on: Model outputs, `CheckResult`, shared utilities
- Used by: `pipeline.py` in phases 0 and 2

**Model Wrappers:**
- Purpose: Interface to external ML models (SCRFD face, YOLO persons, Hands23 hands)
- Location: `bachman_cortex/models/` directory
- Contains: `SCRFDDetector`, `YOLODetector`, `HandObjectDetectorHands23`, model weight downloader
- Depends on: InsightFace, ultralytics, onnxruntime, detectron2
- Used by: Phase 1 inference in `pipeline.py`

**Utilities:**
- Purpose: Shared algorithms and data processing
- Location: `bachman_cortex/utils/` directory
- Contains: Frame extraction, segment operations, motion analysis, video transcoding, metadata extraction
- Depends on: OpenCV, FFmpeg, numpy
- Used by: `pipeline.py`, check functions

**Data Types:**
- Purpose: Shared data structures across pipeline
- Location: `bachman_cortex/data_types.py`
- Contains: `TimeSegment`, `FrameLabel`, `CheckableSegment`, `SegmentValidationResult`, `VideoProcessingResult`
- Depends on: `check_results.py`
- Used by: All pipeline, check, and reporting modules

**Reporting:**
- Purpose: Report generation and result serialization
- Location: `bachman_cortex/reporting.py`
- Contains: Timeline construction, per-video HTML/Markdown reports, batch summaries
- Depends on: Data types, check results
- Used by: `run_batch.py` after processing completes

## Data Flow

**Phase 0: Metadata Gate**

1. Load video and extract metadata (codec, resolution, framerate, duration, rotation)
2. Run deterministic checks: format (MP4), encoding (H.264), resolution (≥1920x1080), framerate (≥28 FPS), duration (≥120s), orientation (landscape)
3. Short-circuit: if any check fails, reject entire video immediately
4. Output: `metadata_passed` flag, `metadata_results` per-check

**Phase 1: Segment Filtering**

1. Extract frames at configurable sampling FPS (default 1.0 FPS)
2. Run single-pass video decode, tee into motion analyzer (no re-open)
3. Run batch YOLO inference for person detection
4. Run threaded SCRFD inference for face detection (InsightFace)
5. Run Hands23 inference for hand detection
6. Per-frame labels: `face_presence` (bool), `participants` (bool)
7. Convert per-frame bools → bad segments (contiguous ranges of failed frames)
8. Merge bad segments across all checks
9. Compute good segments (inverse of bad)
10. Filter segments by minimum duration (default 60s)
11. Output: `prefiltered_segments` list, reusable cache (frames, detections)

**Phase 2: Segment Validation**

1. For each prefiltered segment:
   - Extract segment-scoped frames and cached detections from Phase 1
   - Run motion analysis (camera stability, frozen segments) via cached motion analyzer
   - Run luminance/blur check on segment frames
   - Run hand visibility check (both hands ≥80% OR single hand ≥90%)
   - Run hand-object interaction check
   - Run view obstruction check
   - Run POV hand angle check
   - Run strict face presence check (zero-tolerance)
2. Aggregate check results: fail if any check fails
3. Output: `usable_segments` (passed), `rejected_segments` (failed), per-check details

**Phase 3: Yield Calculation**

1. Sum usable segment durations
2. Calculate unusable duration = original - usable
3. Calculate yield ratio = usable / original
4. Output: `usable_duration_sec`, `unusable_duration_sec`, `yield_ratio`

**State Management:**

- Phase 1 cache: frames, detections, frame dimensions passed to Phase 2 to avoid redundant inference
- Motion analyzer: single-pass lightweight motion extraction during decode, cached metrics queried per-segment in Phase 2
- Model instances: loaded once per worker in batch mode, reused across videos
- Per-video results: aggregated into JSON and batch reports after all phases complete

## Key Abstractions

**Check Result:**
- Purpose: Standardized pass/fail output from all checks
- Examples: `bachman_cortex/checks/check_results.py`, all check functions
- Pattern: Every check returns `CheckResult(status, metric_value, confidence, details)` where status ∈ {"pass", "fail", "review", "skipped"}

**Time Segment:**
- Purpose: Represent contiguous ranges in seconds
- Examples: `bachman_cortex/data_types.py`
- Pattern: `TimeSegment(start_sec, end_sec)` with computed `duration` property

**Frame Label:**
- Purpose: Per-frame pass/fail result for a single check
- Examples: `bachman_cortex/data_types.py`
- Pattern: `FrameLabel(frame_idx, timestamp_sec, passed, confidence, labels)` for aggregation into bad segments

**Detection Objects:**
- Purpose: Typed results from ML models
- Examples: `FaceDetection(bbox, confidence, landmarks)`, `HandDetection`, `Detection`
- Pattern: Simple dataclasses with bounding box, confidence, optional extra fields

**Segment Validation Result:**
- Purpose: Phase 2 decision for a single segment
- Examples: `bachman_cortex/data_types.py`
- Pattern: `SegmentValidationResult(segment, passed, check_results, failing_checks)`

## Entry Points

**CLI Entry Point (batch):**
- Location: `bachman_cortex/run_batch.py:main()`
- Triggers: `python -m bachman_cortex.run_batch /path/to/videos/ [options]`
- Responsibilities: Parse CLI args, collect video files, auto-detect worker count, spawn worker pool, collect results, write reports

**Shell Script Entry Point:**
- Location: `validate.sh` (project root)
- Triggers: `./validate.sh /path/to/videos/ [options]`
- Responsibilities: Check Python version, install deps, download models, activate venv, invoke `run_batch.py`

**Single-Video Entry Point:**
- Location: `bachman_cortex/pipeline.py:ValidationProcessingPipeline.process_video()`
- Triggers: Instantiate `ValidationProcessingPipeline(config)`, call `.process_video(video_path, output_dir)`
- Responsibilities: Run all 4 phases, return structured `VideoProcessingResult`

**Worker Process Entry Point:**
- Location: `bachman_cortex/run_batch.py:_init_worker()`
- Triggers: Multiprocessing pool initialization (spawn mode)
- Responsibilities: Load models once per worker, prepare shared `_worker_pipeline` instance

## Error Handling

**Strategy:** Fail-fast on metadata, graceful degradation on ML checks; preserve partial results.

**Patterns:**

- Metadata failures: Return early with `metadata_passed=False`, empty segment lists
- Phase 1 segment filtering: If no checkable segments remain after filtering, return early with empty usable segments
- Per-segment Phase 2: If segment validation throws, catch and record as failing check
- Worker failures: Catch exceptions in `_process_video_worker()`, return error dict with traceback
- Model loading failures: Log warning, continue if fail during warmup; raise if fail during actual inference
- Missing frames: Return empty segment result with `failing_checks=["no_frames_extracted"]`

## Cross-Cutting Concerns

**Logging:** Stdout prints progress via `print()` statements at phase boundaries and per-segment validation; designed for batch runner capture.

**Validation:** Every check returns `CheckResult`; status aggregated via `all(r.status != "fail")` pattern throughout.

**Authentication:** None; pipeline is local-only with no remote API calls.

**Model Loading:** Lazy on first video (Phase 1), eager initialization in batch workers via `_init_worker()` and warmup forward passes.

---

*Architecture analysis: 2026-04-15*
