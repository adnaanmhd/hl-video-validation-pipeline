# Testing Patterns

**Analysis Date:** 2026-04-15

## Test Framework

**Runner:**
- No dedicated test framework configured (pytest, unittest not in dependencies)
- Benchmarking via standalone scripts instead of test framework
- Manual test videos generated for verification

**Test Files:**
- Location: `bachman_cortex/tests/`
- Files: `benchmark_models.py`, `benchmark_phase_correlation.py`, `generate_test_video.py`
- Executed directly as Python modules: `python -m bachman_cortex.tests.benchmark_models`

**Run Commands:**
```bash
# Benchmark all ML models
python -m bachman_cortex.tests.benchmark_models

# Benchmark phase-correlation motion analysis
python -m bachman_cortex.tests.benchmark_phase_correlation

# Generate test video
python -m bachman_cortex.tests.generate_test_video
```

## Benchmark Structure

**Entry Point Pattern:**
- Each benchmark is a standalone module with a `__main__` guard
- Example from `benchmark_models.py`:
  ```python
  if len(sys.argv) < 2:
      print("Usage: python -m ...")
      sys.exit(1)
  ```

**Test Data Setup:**
- `generate_test_video.py` creates synthetic test videos
- Benchmarks load real videos from command-line arguments
- Frame extraction via `extract_frames()` utility with configurable FPS

## Benchmark Functions

**Model Benchmarking Pattern:**
- Separate function per detector: `benchmark_scrfd()`, `benchmark_yolo()`, `benchmark_100doh()`
- Shared structure:
  1. Warmup phase (2-3 frames)
  2. Measurement phase on full frame set
  3. Detection counting
  4. Result aggregation

**Example from `benchmark_models.py`:**
```python
def benchmark_scrfd(frames: list[np.ndarray]) -> dict:
    """Benchmark SCRFD face detector."""
    print("\n" + "=" * 60)
    print("SCRFD-2.5GF Face Detector")
    print("=" * 60)

    from bachman_cortex.models.scrfd_detector import SCRFDDetector

    detector = SCRFDDetector(
        root=os.path.join(ROOT, "bachman_cortex/models/weights/insightface"),
    )

    # Warmup
    print("Warmup (3 frames)...")
    for f in frames[:3]:
        detector.detect(f)

    # Benchmark
    print(f"Benchmarking on {len(frames)} frames...")
    result = detector.benchmark(frames)

    # Count detections
    total_faces = 0
    for f in frames:
        faces = detector.detect(f)
        total_faces += len(faces)

    result["total_detections"] = total_faces
    print(f"  p50: {result['p50_ms']:.1f}ms | p95: {result['p95_ms']:.1f}ms | mean: {result['mean_ms']:.1f}ms")
    print(f"  Total faces detected: {total_faces}")
    return result
```

**Metrics Collected:**
- Per-frame latency percentiles: p50, p95, p99 (in milliseconds)
- Mean latency and total time
- Detection counts (total faces, persons, hands, etc.)
- Returned as dictionary for aggregation

## Model Benchmark Methods

**Detector.benchmark() method signature:**
Each detector implements a `benchmark()` method:
```python
def benchmark(self, frames: list[np.ndarray]) -> dict:
    """Benchmark inference speed on a list of frames.
    
    Returns dict with:
    - model: str (model name)
    - frames: int (frame count)
    - p50_ms, p95_ms, p99_ms, mean_ms: float (latency percentiles)
    - total_s: float (total time in seconds)
    """
```

**Example from `scrfd_detector.py`:**
```python
def benchmark(self, frames: list[np.ndarray]) -> dict:
    """Benchmark inference speed on a list of frames."""
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        self.detect(frame)
        times.append(time.perf_counter() - t0)
    times_ms = [t * 1000 for t in times]
    return {
        "model": "SCRFD (buffalo_sc)",
        "frames": len(frames),
        "p50_ms": round(np.percentile(times_ms, 50), 2),
        "p95_ms": round(np.percentile(times_ms, 95), 2),
        "p99_ms": round(np.percentile(times_ms, 99), 2),
        "mean_ms": round(np.mean(times_ms), 2),
        "total_s": round(sum(times), 3),
    }
```

## Phase-Correlation Benchmark

**Purpose:**
- Validates new motion detection algorithm against existing Lucas-Kanade + SSIM baseline
- File: `bachman_cortex/tests/benchmark_phase_correlation.py`

**Structure:**
```python
def check_camera_stability_phase_corr(
    video_path: str,
    *,
    shaky_score_threshold: float = 0.30,
    target_fps: float = 30.0,
    scale: float = 0.25,
    # ... additional parameters
) -> CheckResult:
    """Phase-correlation stability check implementation."""
```

**Comparison Logic:**
- Implements same signature as production `check_camera_stability()` from motion_analysis.py
- Allows side-by-side timing and results comparison
- Scoring function `_phase_corr_score()` mirrors production logic but without rotation component

## Testing Without Framework

**Validation approach:**
- Real video data testing via benchmarks
- Precision metrics (percentiles, mean latency)
- Visual/numerical output inspection
- Reproducibility via fixed frame sets

**Test data:**
- Generated synthetic videos (from `generate_test_video.py`)
- Real egocentric video samples (in `vid_samples/` directory if present)
- Frame extraction utilities support various FPS rates

## Implicit Testing

**Error Paths:**
- GPU fallback tested in frame_extractor: CUDA path crashes gracefully to CPU
- Example from `frame_extractor.py`:
  ```python
  if _NVDEC_OK:
      try:
          reader = cv2.cudacodec.createVideoReader(video_path)
          backend = "nvdec"
      except cv2.error:
          reader = None
  ```

**Edge Cases:**
- Empty frame list handling in all check functions
- Example from `check_hand_visibility.py`:
  ```python
  if total_frames == 0:
      return CheckResult(status="fail", metric_value=0.0, confidence=0.0,
                         details={"error": "no frames"})
  ```

**Type Checking:**
- Comprehensive type hints enable static type checking (no explicit mypy runs observed)
- Dataclass validation through type system

## Benchmark Output

**Console output format:**
- Descriptive headers with section separators (60-char lines)
- Warmup status messages
- Per-model result summaries with formatted metrics

**Example output pattern:**
```
============================================================
SCRFD-2.5GF Face Detector
============================================================
Warmup (3 frames)...
Benchmarking on 100 frames...
  p50: 45.2ms | p95: 52.1ms | mean: 46.3ms
  Total faces detected: 150
```

**Return values:**
- Benchmarks return dict suitable for JSON serialization
- Pipeline results use dataclass-to-dict conversion: `dataclasses.asdict(result)`

## Testing Integration

**Pipeline validation:**
- `run_batch.py` loads models once per worker process (initialization pattern)
- Early-stopping gates tested implicitly via pipeline execution
- Per-video error handling: `try/except/traceback` around pipeline processing

**Batch processing:**
- Multiprocessing pattern: `ProcessPoolExecutor` with model preloading
- `_init_worker()` function ensures models load once per worker
- Errors captured and stored in result dict: `error: str | None` field in `VideoProcessingResult`

## Setup Validation

**`validate.sh` testing:**
- Script includes idempotent checks for dependencies
- Python version verification: `check_python_version()` function
- GPU detection and conditional PyTorch installation
- Model weight download verification before pipeline run

**Dependency validation:**
```bash
# Check Python version
python3.12 --version

# Check system tools
ffprobe --version  # FFmpeg metadata extraction
git --version      # For detectron2 installation

# Check Python packages
python -c "import torch"
python -c "import onnxruntime"
python -c "import detectron2"
```

---

*Testing analysis: 2026-04-15*
