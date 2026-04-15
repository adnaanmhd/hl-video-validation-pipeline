# Codebase Structure

**Analysis Date:** 2026-04-15

## Directory Layout

```
hl-bachman/
├── bachman_cortex/              # Main package
│   ├── __init__.py              # cv2.dnn shim installer
│   ├── _cv2_dnn_shim.py         # Custom cv2.dnn blobFromImage replacement
│   ├── data_types.py            # Shared dataclasses (segments, labels, results)
│   ├── pipeline.py              # 4-phase validation pipeline orchestrator
│   ├── run_batch.py             # Batch processing CLI and worker pool
│   ├── reporting.py             # Report generation (HTML, Markdown, JSON)
│   │
│   ├── checks/                  # Individual acceptance criteria checks
│   │   ├── __init__.py
│   │   ├── check_results.py     # CheckResult dataclass
│   │   ├── video_metadata.py    # Phase 0 checks (format, codec, resolution, etc.)
│   │   ├── face_presence.py     # Phase 2 check: strict face detection
│   │   ├── hand_visibility.py   # Phase 2 check: both-hands OR single-hand visibility
│   │   ├── hand_object_interaction.py  # Phase 2 check: hand in grasp pose
│   │   ├── luminance_blur.py    # Phase 2 check: brightness stability and blur
│   │   ├── motion_analysis.py   # Phase 2 checks: camera stability, frozen segments
│   │   ├── participants.py      # Phase 1 helper: other person detection
│   │   ├── pov_hand_angle.py    # Phase 2 check: hand-camera angle
│   │   └── view_obstruction.py  # Phase 2 check: lens obstruction
│   │
│   ├── models/                  # ML model wrappers and loaders
│   │   ├── __init__.py
│   │   ├── download_models.py   # Download SCRFD, YOLO, Hands23 weights
│   │   ├── scrfd_detector.py    # InsightFace SCRFD face detection wrapper
│   │   ├── yolo_detector.py     # Ultralytics YOLO person/object detection
│   │   ├── hand_detector.py     # Hands23 hand detection wrapper
│   │   ├── hand_detector_100doh.py  # Hands23 alternative implementation
│   │   └── weights/             # Model checkpoint directory
│   │       ├── insightface/     # SCRFD weights (buffalo_sc)
│   │       └── hands23_detector/    # Hands23 fine-tuned weights + training code
│   │
│   ├── utils/                   # Shared utilities and algorithms
│   │   ├── __init__.py
│   │   ├── video_metadata.py    # FFprobe-based metadata extraction
│   │   ├── frame_extractor.py   # OpenCV video decode with NVDEC fallback
│   │   ├── segment_ops.py       # Per-frame→segment conversions, merging, filtering
│   │   ├── early_stop.py        # Phase 1 per-frame decision helpers
│   │   └── transcode.py         # HEVC→H.264 lossless transcoding (opt-in)
│   │
│   └── tests/                   # Benchmarks and test utilities
│       ├── __init__.py
│       ├── benchmark_models.py  # Model inference timing
│       ├── benchmark_phase_correlation.py  # Motion analysis benchmarking
│       └── generate_test_video.py  # Synthetic test video generation
│
├── validate.sh                  # One-command setup and run script
├── setup.py                     # Package metadata and dependencies
├── pyproject.toml               # Build system config
├── checks.md                    # Acceptance criteria documentation
├── README.md                    # Project overview and architecture diagram
├── .planning/
│   └── codebase/               # Architecture documentation (generated)
│       ├── ARCHITECTURE.md     # Pipeline pattern, layers, data flow
│       └── STRUCTURE.md        # This file
│
└── bachman_cortex/results/     # Output directory (created at runtime)
    ├── run_001/
    │   ├── index.md            # Run index across all videos
    │   ├── batch_report.md     # Batch-level summary
    │   ├── batch_results.json  # Raw results JSON
    │   └── video_name/         # Per-video output
    │       ├── video_name.json      # Per-video structured result
    │       └── video_name.html      # Per-video timeline report
    └── run_002/
        └── ...
```

## Directory Purposes

**`bachman_cortex/`:**
- Purpose: Main Python package root
- Contains: Core pipeline logic, checks, models, utilities
- Key files: `pipeline.py` (orchestrator), `run_batch.py` (CLI), `data_types.py` (schemas)

**`bachman_cortex/checks/`:**
- Purpose: Acceptance criteria checks (deterministic and ML-based)
- Contains: Individual check functions, each returning `CheckResult`
- Key files: 
  - Phase 0: `video_metadata.py` (6 deterministic checks)
  - Phase 1: `face_presence.py`, `participants.py` (early-stopping gates)
  - Phase 2: All other checks (hand visibility, motion, luminance, etc.)

**`bachman_cortex/models/`:**
- Purpose: ML model wrappers and weight management
- Contains: Detector classes for SCRFD (face), YOLO (persons), Hands23 (hands)
- Key files: `download_models.py` (invoked by `validate.sh`), detector wrappers

**`bachman_cortex/utils/`:**
- Purpose: Shared algorithms and video processing utilities
- Contains: Frame extraction, segment operations, motion analysis, transcoding
- Key files:
  - `frame_extractor.py`: Video decode with NVDEC GPU acceleration fallback
  - `segment_ops.py`: Per-frame→segment conversions, merging, filtering logic
  - `motion_analysis.py`: Lucas-Kanade optical flow camera stability analysis

**`bachman_cortex/tests/`:**
- Purpose: Benchmarking and test utilities (not unit tests)
- Contains: Model inference timing, motion analysis benchmarks
- Generated outputs: Not committed; for development analysis only

**`bachman_cortex/results/`:**
- Purpose: Batch run outputs (created at runtime)
- Contains: Per-run directories with indexed reports and per-video results
- Structure: `run_NNN/` → `batch_report.md`, `batch_results.json`, `video_name/` subdirs
- Committed: No; .gitignore'd

## Key File Locations

**Entry Points:**
- `validate.sh`: Shell wrapper for one-command setup + run
- `bachman_cortex/run_batch.py`: CLI batch processor (`python -m bachman_cortex.run_batch`)
- `bachman_cortex/pipeline.py:ValidationProcessingPipeline.process_video()`: Single-video entry point

**Configuration:**
- `bachman_cortex/pipeline.py:PipelineConfig`: Dataclass with all tunable parameters (thresholds, model paths, FPS, etc.)
- `validate.sh`: Env var overrides (e.g., `FORCE_CPU=1`)
- Model weight paths: `bachman_cortex/models/weights/insightface`, `hands23_detector`

**Core Logic:**
- `bachman_cortex/pipeline.py`: 4-phase orchestration
- `bachman_cortex/checks/`: Check functions invoked from pipeline
- `bachman_cortex/utils/segment_ops.py`: Per-frame→segment aggregation logic

**Testing:**
- `bachman_cortex/tests/`: Benchmarks (not unit tests)
- No pytest/unittest suite; integration testing via `validate.sh` on sample videos

**Reporting:**
- `bachman_cortex/reporting.py`: HTML/Markdown/JSON generation
- Output: `bachman_cortex/results/run_NNN/batch_report.md`, per-video HTML, structured JSON

## Naming Conventions

**Files:**
- `check_*.py`: Individual check implementations (Phase-specific)
- `*_detector.py`: ML model wrapper classes in `models/`
- `.py` extension: Python modules
- `*.md`: Markdown documentation
- `*.sh`: Bash scripts (e.g., `validate.sh`)

**Directories:**
- `checks/`: All check implementations, Phase 0–2
- `models/`: ML detector wrappers and weight management
- `utils/`: Shared algorithms (frame extraction, segment ops, etc.)
- `tests/`: Benchmarks and test utilities
- `weights/`: Committed model weight directories

**Python Modules:**
- Package name: `bachman_cortex` (renamed from legacy `vald_pipeline`)
- Private modules: `_cv2_dnn_shim.py`, `_worker_pipeline` (global in run_batch)
- Dataclass names: PascalCase (`TimeSegment`, `FrameLabel`, `CheckResult`, etc.)
- Function names: snake_case (`check_hand_visibility()`, `extract_frames()`, etc.)
- Class names: PascalCase (`ValidationProcessingPipeline`, `MotionAnalyzer`, etc.)

## Where to Add New Code

**New Check (Phase 0, 1, or 2):**
- Implementation: `bachman_cortex/checks/new_check_name.py`
- Pattern: Define `check_new_check_name(args) -> CheckResult` function
- Integration: Import and call from appropriate phase in `pipeline.py` (Phase 0: `_phase0_metadata()`, Phase 2: `_phase2_validate_single_segment()`)
- Naming: Use `ml_` prefix for learned checks, no prefix for deterministic

**New Model Wrapper:**
- Implementation: `bachman_cortex/models/new_detector.py`
- Pattern: Class wrapping external model with `.detect(frame)` or `.detect_batch(frames)` method
- Integration: Instantiate in `pipeline.load_models()`, cache instance for reuse
- Weight management: Update `download_models.py` if new weights need downloading

**New Utility Function:**
- Shared algorithm: `bachman_cortex/utils/new_algorithm.py` or append to existing `*.py` in utils/
- Frame-level processing: Consider adding to `frame_extractor.py` or segment processing
- Pattern: Pure functions preferred, stateless or explicitly managed state

**New CLI Option:**
- Implementation: Add argument to `run_batch.py:main()` argparse
- Integration: Pass through `PipelineConfig` dataclass to `ValidationProcessingPipeline`
- Validation: Pre-validate in `main()` before spawning workers

## Special Directories

**`bachman_cortex/models/weights/`:**
- Purpose: Committed model weight checkpoints
- Generated: No (weights are committed to repo)
- Committed: Yes, large files (1.5+ GB)
- Note: Downloaded incrementally by `download_models.py` if missing

**`bachman_cortex/results/`:**
- Purpose: Batch run output artifacts
- Generated: Yes (created at runtime by `run_batch.py`)
- Committed: No (.gitignore'd)
- Structure: Sequential `run_NNN/` directories with reports and JSON results

**`bachman_cortex/tests/`:**
- Purpose: Development benchmarks and test utilities
- Generated: Outputs created at runtime
- Committed: Yes, source files only (benchmark code); generated outputs .gitignore'd

**`.venv/`:**
- Purpose: Python virtual environment (created by `validate.sh`)
- Generated: Yes (created by `python -m venv`)
- Committed: No (.gitignore'd)

---

*Structure analysis: 2026-04-15*
