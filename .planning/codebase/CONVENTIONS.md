# Coding Conventions

**Analysis Date:** 2026-04-15

## Naming Patterns

**Files:**
- Lowercase with underscores (snake_case): `hand_visibility.py`, `frame_extractor.py`, `motion_analysis.py`
- Model files follow detector type names: `scrfd_detector.py`, `yolo_detector.py`, `hand_detector.py`
- Utility modules in `bachman_cortex/utils/`: `frame_extractor.py`, `early_stop.py`, `segment_ops.py`, `transcode.py`, `video_metadata.py`
- Check modules in `bachman_cortex/checks/`: `hand_visibility.py`, `luminance_blur.py`, `motion_analysis.py`, `face_presence.py`, `hand_object_interaction.py`, `view_obstruction.py`, `pov_hand_angle.py`, `participants.py`, `video_metadata.py`
- Test files in `bachman_cortex/tests/`: `benchmark_models.py`, `benchmark_phase_correlation.py`, `generate_test_video.py`

**Functions:**
- Snake_case: `check_hand_visibility()`, `extract_frames()`, `_hand_in_frame()` (private prefix with single underscore)
- Helper functions prefixed with underscore: `_compute_frame_metrics()`, `_transforms_to_score()`, `_lk_track()`, `_feature_params()`, `_phase_corr_score()`
- Check functions follow pattern: `check_<criterion_name>()` — examples: `check_hand_visibility()`, `check_face_presence()`, `check_luminance_blur()`, `check_motion_combined_from_analyzer()`, `check_hand_object_interaction()`

**Variables:**
- Snake_case: `per_frame_hands`, `total_frames`, `frame_h`, `frame_w`, `confidence_threshold`, `both_hands_pass_rate`
- Short abbreviations where appropriate: `h` for height, `w` for width, `p` for person, `f` for face
- Loop variables: `for frame in frames`, `for hands in per_frame_hands`, `for h in hands`

**Types/Classes:**
- PascalCase for classes: `CheckResult`, `HandDetection`, `FaceDetection`, `ObjectDetection`, `FrameMetrics`, `FrameLabel`, `TimeSegment`, `CheckableSegment`, `SegmentValidationResult`, `VideoProcessingResult`, `PipelineConfig`, `ValidationProcessingPipeline`, `LuminanceBlurConfig`, `ContactState`, `HandSide`, `MotionAnalyzer`
- Enum members: UPPERCASE (e.g., `ContactState.NO_CONTACT`, `HandSide.LEFT`, `HandSide.RIGHT`)
- Dataclass fields: Snake_case with type hints
- Config parameter names: Snake_case with descriptive suffixes (e.g., `confidence_threshold`, `pass_rate`, `frame_margin`)

## Code Style

**Formatting:**
- Python 3.11+ features used throughout (type hints, modern syntax)
- 4-space indentation
- Line length not strictly enforced but generally kept reasonable
- Multiple return values in tuples: `return frames, metadata` (from `extract_frames()`)
- Dictionary-based configuration and results

**Linting:**
- No `.eslintrc` or `.prettierrc` files found
- Code follows PEP 8 conventions implicitly
- Imports organized but no strict enforcement detected

**Type Hints:**
- Comprehensive use of type hints throughout
- Examples from code:
  - `def check_hand_visibility(per_frame_hands: list[list[HandDetection]], frame_dims: tuple[int, int], ...) -> CheckResult:`
  - `def extract_frames(video_path: str | Path, fps: float = 1.0, max_frames: Optional[int] = None, ...) -> tuple[list[np.ndarray], dict]:`
  - `def _hand_in_frame(bbox: np.ndarray, frame_w: int, frame_h: int, margin: int) -> bool:`
  - `frames: list[np.ndarray] = []` (explicit variable type hints)
- Union types: `str | Path`, `dict | None`, `list[str] | None`
- Optional type: `Optional[int]` and `str | None` (both patterns used)
- Dataclass fields have type hints: `confidence: float`, `bbox: np.ndarray`, `side: HandSide`

## Import Organization

**Order:**
1. Standard library (sys, os, time, json, dataclasses, pathlib, enum, etc.)
2. Third-party imports (numpy, cv2, torch, etc.)
3. Local imports (from bachman_cortex.*)

**Examples from codebase:**
```python
# motion_analysis.py
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from bachman_cortex.checks.check_results import CheckResult

# hand_visibility.py
import numpy as np

from bachman_cortex.checks.check_results import CheckResult
from bachman_cortex.models.hand_detector import HandDetection, HandSide

# run_batch.py
import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

from bachman_cortex.pipeline import ValidationProcessingPipeline, PipelineConfig
```

**Path Aliases:**
- Relative imports used: `from bachman_cortex.models.hand_detector import ...`
- Absolute imports preferred (no alias shortcuts observed)
- Entry point script uses: `ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))`

## Docstring Style

**Module-level docstrings:**
- Always present at file top with triple quotes
- Describe purpose and high-level behavior
- Example from `motion_analysis.py`:
  ```python
  """Motion analysis checks.
  
  Camera stability via single-pass sparse Lucas-Kanade optical flow at 0.5x
  resolution, per-frame at target FPS (default 30).  CUDA GPU acceleration
  used when available (cv2.cuda.SparsePyrLKOpticalFlow).
  Frozen segment detection derived from LK signal (near-zero translation + rotation).
  """
  ```

**Function docstrings:**
- Use triple-quoted strings (not Sphinx/Google style strictly)
- Include Args, Returns, and notes where relevant
- Example from `check_hand_visibility()`:
  ```python
  """Check hand visibility using an OR of two sub-conditions.

  Passes if either:
    - ``both_hands_ratio >= both_hands_pass_rate`` (primary), or
    - ``single_hand_ratio >= single_hand_pass_rate`` (fallback).

  Args:
      per_frame_hands: Hand detections per frame.
      frame_dims: (height, width) of each frame.
      confidence_threshold: Min confidence for a hand to count as detected.
      both_hands_pass_rate: Min fraction of frames with both hands visible.
      single_hand_pass_rate: Min fraction of frames with at least one hand
          visible (fallback when the both-hands condition fails).
      frame_margin: Min pixel clearance from frame edges.

  Returns:
      CheckResult. ``metric_value`` is the both-hands ratio; the single-hand
      ratio and both thresholds are reported in ``details``.
  """
  ```

**Class docstrings:**
- Concise one-liner describing purpose
- Example: `class CheckResult: "Result of a single acceptance criteria check."`

**Inline comments:**
- Used for complex logic, heuristics, and workarounds
- Examples from code:
  - `# The Hands23 model's left/right classifier can mislabel...`
  - `# Confidence: use whichever sub-condition has the best margin...`
  - `# HEURISTIC avoids the multi-second EXHAUSTIVE autotune...`
  - `# Phase 1 frame downscale (long-edge). None disables downscaling.`

## Error Handling

**Patterns:**
- Explicit None checks: `if total_frames == 0: return CheckResult(status="fail", ...)`
- Exception catching for GPU fallback: `except cv2.error:` (in Lucas-Kanade optical flow)
- Try-except for optional features: `try: ... except Exception:` (in frame_extractor NVDEC check)
- Validation through explicit checks rather than exceptions
- Default values for optional components (e.g., `motion_analyzer: Optional[_FrameProcessor] = None`)

**Return types for errors:**
- Checks return `CheckResult` with status="fail" on error
- Example: `return CheckResult(status="fail", metric_value=0.0, confidence=0.0, details={"error": "no frames"})`
- GPU decode fallback to CPU in `frame_extractor.py`: graceful degradation

## Logging

**Framework:** `print()` statements only — no logging library used

**Patterns:**
- Status messages prefixed with category: "=" signs for section headers in benchmarks
- Example from `benchmark_models.py`:
  ```python
  print("\n" + "=" * 60)
  print("SCRFD-2.5GF Face Detector")
  print("=" * 60)
  ```
- Progress messages: `print("Warmup (3 frames)...")`, `print(f"Benchmarking on {len(frames)} frames...")`
- Result messages: `print(f"  p50: {result['p50_ms']:.1f}ms | p95: ...")`
- Setup/execution messages in `validate.sh` using `info()` and `error()` helper functions

**Message format:**
- Descriptive but concise
- Metrics formatted with 1-2 decimal places (`.1f`, `.2f`, `.3f`, `.4f`)
- Example: `print(f"Extracted {meta['frames_extracted']} frames in {meta['extraction_time_s']}s")`

## Dataclass Usage

**Heavy use of dataclasses:**
- `@dataclass` decorator for all data structures
- Examples: `CheckResult`, `HandDetection`, `FaceDetection`, `FrameMetrics`, `TimeSegment`, `CheckableSegment`, `FrameLabel`, `CheckFrameResults`, `SegmentValidationResult`, `VideoProcessingResult`, `PipelineConfig`, `LuminanceBlurConfig`
- Fields have type hints and optional defaults
- Example:
  ```python
  @dataclass
  class HandDetection:
      bbox: np.ndarray  # [x1, y1, x2, y2]
      confidence: float
      side: HandSide
      contact_state: ContactState
      grasp_type: str | None = None
      offset_vector: np.ndarray | None = None
  ```

## Module Organization

**Checks module structure (`bachman_cortex/checks/`):**
- Each check is a standalone module with one or more check functions
- Shared types in `check_results.py` and `participants.py`
- Each check imports only what it needs (tight dependencies)
- Consistent signature: `check_<name>(per_frame_<type>, ...) -> CheckResult`

**Models module structure (`bachman_cortex/models/`):**
- Model wrapper classes (detector classes) encapsulate inference
- Detection result dataclasses (HandDetection, FaceDetection, etc.)
- Enum classes for classification (HandSide, ContactState)
- Example: `SCRFDDetector`, `YOLODetector`, `HandObjectDetectorHands23`

**Utils module structure (`bachman_cortex/utils/`):**
- Reusable functions for frame extraction, video metadata, segment operations
- Protocol class `_FrameProcessor` for callback interfaces
- No class-based state management (mostly functional)

## Dictionary Keys and Constants

**Configuration dictionaries:**
- Snake_case keys: `sampling_fps`, `face_confidence_threshold`, `hand_pass_rate`
- Serializable via `dataclasses.asdict()`

**Enum-based mappings:**
- Contact states and grasp types mapped via dictionaries: `HANDS23_CONTACT_MAP`, `HANDS23_GRASP_NAMES`, `HANDS23_SIDE_MAP`
- Located in `hand_detector.py` for easy reference

---

*Convention analysis: 2026-04-15*
