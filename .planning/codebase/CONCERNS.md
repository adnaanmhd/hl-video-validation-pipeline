# Codebase Concerns

**Analysis Date:** 2026-04-15

## Tech Debt

### Dead Code: Deleted ML Checks (Active Refactoring)

**What's happening:**
Git status shows recent deletions of five check files:
- `bachman_cortex/checks/body_part_visibility.py` (deleted)
- `bachman_cortex/checks/frame_quality.py` (deleted)
- `bachman_cortex/checks/privacy_safety.py` (deleted)
- `bachman_cortex/models/grounding_dino_detector.py` (deleted)
- `bachman_cortex/models/yolo_pose_detector.py` (deleted)

**Impact:**
These deletions represent an intentional architecture redesign documented in OPTIMIZATION_PLAN_V3.md. Three checks (privacy, egocentric perspective, body-part visibility) moved behind opt-in CLI flags (`--privacy-check`, `--egocentric-check`, `--body-part-visibility`). They are no longer on the default hot path but remain conditional. YOLO-pose is only loaded when those checks are enabled, saving ~4-5s of model load on default runs.

**Status:** This is intentional optimization work, not legacy code. However, references to deleted modules may still exist in codebase.

**Action:** Verify no import statements reference the deleted modules in unconditional code paths.

---

### Custom OpenCV Build with Missing DNN Module

**Files:** `bachman_cortex/_cv2_dnn_shim.py` (new), `bachman_cortex/__init__.py` (modified)

**Problem:**
Custom CUDA-enabled OpenCV build was compiled with `BUILD_opencv_dnn=OFF` to work around a CMake/OpenCV 4.10 bug (`pyopencv_dnn.hpp` fails on gcc-12). This removes `cv2.dnn` entirely. Since `insightface` depends on `cv2.dnn.blobFromImage` for SCRFD preprocessing, a shim was created (`_cv2_dnn_shim.py`) that patches pure-numpy implementations into the `cv2.dnn` namespace.

**Risk:**
- **Fragility**: The shim must be installed before any `insightface` import (happens in `__init__.py`). If load order changes, `insightface` fails.
- **Maintainability**: The shim is architecture-specific to this one OpenCV build. Upgrading OpenCV or switching to a standard PyPI build will break it silently if not reimported.
- **Numpy path mismatch**: The shim uses `np.transpose()` for NCHW layout; any future insightface version that expects stricter tensor properties may fail.

**Mitigation in place:** `__init__.py` imports `_cv2_dnn_shim` and calls `.install()` before importing insightface. Clear comments explain why.

**Future risk:** When OpenCV upstream fixes the CMake bug, remove the shim and standard `cv2.dnn` will be available.

---

### Hands23 Hardcoded `.cuda()` Calls Patched at Runtime

**Files:** `bachman_cortex/models/download_models.py` (lines 103-160, CPU compatibility patching)

**Problem:**
Hands23 upstream code has hardcoded `.cuda()` calls. The download script automatically patches these to device-aware `.to(device)` during setup (`patch_hands23_cuda_calls()`). This is applied at download time, not import time.

**Risk:**
- If a user re-clones the hands23 repo from upstream, the patch is lost and CPU runs fail.
- The patch is brittle — applied via string replacement on Python source files. If upstream code structure changes, the regex may not match.
- **Not documented in CONTEXT.md or README** — a user who manually updates Hands23 will not know to re-run the patcher.

**Mitigation:** The patch is applied in `download_models.py` and only runs once. But there's no automated check to verify the patch is in place.

---

## Known Bugs

### TODO Marker in Hands23 Weight Preprocessor (Not Critical)

**File:** `bachman_cortex/models/weights/hands23_detector/data_prep/data_util.py:208`

**Comment:**
```python
scalar = 1000 # TODO: the scalar needs testing
```

**Impact:** Low. This is in data preprocessing for training, not the inference path. The scalar is used during dataset preparation, not at validation time.

---

## Security Considerations

### Subprocess Calls to FFmpeg without Shell Escaping

**Files:**
- `bachman_cortex/utils/video_metadata.py:23-29` (`ffprobe` subprocess call)
- `bachman_cortex/utils/transcode.py:106` (`ffmpeg` subprocess call)

**Current approach:**
Both calls use `subprocess.run()` with a list of arguments (not shell=True), which is safe from injection. File paths are passed through string conversion and included in the command array directly.

**Risk assessment:**
- ✅ **Safe**: List-form subprocess.run() prevents shell injection.
- ⚠️ **Precondition**: Video file paths must be validated before reaching these functions. If paths come from untrusted sources (e.g., URL parameters in a web service), symlink attacks or malicious filenames could cause issues.

**Current safeguards:**
- `run_batch.py` validates files via `collect_videos()` — only .mp4 files in user-provided directories
- No remote URL downloads (only local file paths)
- No dynamic path construction from user input

**Recommendation:** If this code is ever exposed to web input, add explicit path validation:
```python
from pathlib import Path
video_path = Path(video_path).resolve()  # Canonicalize; raises if symlink loops
if not video_path.exists():
    raise ValueError(f"File does not exist: {video_path}")
```

---

### Model Download URL Security

**File:** `bachman_cortex/models/download_models.py:72` (Hands23 weights download)

**Issue:**
The Hands23 model weights are downloaded from `https://fouheylab.eecs.umich.edu/...` via `wget --no-check-certificate` or `curl -k`. The `--no-check-certificate` / `-k` flags disable SSL verification.

**Justification:** Server SSL certificate was failing with Python's strict verification in the past.

**Risk:**
- **MITM vulnerability**: On untrusted networks, disabling SSL verification allows attacker-in-the-middle attacks to inject malicious model weights.
- **Silent failure**: If the server certificate is actually bad, no user-facing error is raised — the insecure download happens silently.

**Mitigation:**
- This is a one-time download during setup (`download_models.py`), not part of the validation pipeline.
- The weights file is large (~400MB) and would be noticeable if corrupted.
- Recommend: Update the URL or request the upstream maintainer fix their SSL certificate.

**Recommendation:**
```python
# Instead of --no-check-certificate:
try:
    subprocess.run(["wget", "-O", str(weight_file), url], check=True)
except subprocess.CalledProcessError:
    # Fall back to Python urllib with proper certificate handling
    import urllib.request
    urllib.request.urlretrieve(url, str(weight_file))
```

---

## Performance Bottlenecks

### Hands23 is the Residual Bottleneck on Default Config

**Files:** `bachman_cortex/pipeline.py:759-765` (Phase 2 pose loop, now conditional), `bachman_cortex/models/hand_detector.py`

**Current cost:**
Per `OPTIMIZATION_PLAN_V3.md`, after P0 + P1 optimizations (gating pose on flags, single-pass decode, phase1 720p downscale):
- Hands23 inference: ~30-40s per 180s clip
- Motion analysis (GPU LK): ~15s
- Other: ~10-15s
- **Total:** ~60s for a default 180s 1080p clip

**Why:**
- Hands23 runs per-frame (not batched) via Detectron2's `DefaultPredictor`
- ~180ms/frame on CPU, ~100-200ms/frame on GPU (untested on this host)
- Cannot be easily batched through the public API

**Plan documented:**
OPTIMIZATION_PLAN_V3 §6 explicitly states "Does not modify Hands23." Hands23 remains out of scope per user constraint.

**Impact:** At scale (5,000 videos/day), Hands23 dominates. A single optimized pass would cut end-to-end time by ~30-40%.

**Feasibility for future:**
P2.2 (Hands23 backbone swap) is listed as speculative after all P0 + P1 work. Would require retraining or finding a compatible faster backbone.

---

### Motion Analysis Reopens Video per Clip (FIXED in P0.2)

**Status:** ✅ **RESOLVED** per OPTIMIZATION_PLAN_V3.md

Previously, `check_motion_combined()` reopened the video file per clip, requiring N+1 decodes for N clips. Now resolved via `MotionAnalyzer` which collects LK data during frame extraction in a single pass.

**Files affected (now optimized):**
- `bachman_cortex/utils/frame_extractor.py` (accepts `motion_analyzer` kwarg)
- `bachman_cortex/checks/motion_analysis.py:597` (old re-open, replaced with analyzer-based slicing)

---

### OpenCV CUDA LK Disabled on This Host

**Files:** `bachman_cortex/checks/motion_analysis.py:62-78` (CUDA LK path)

**Situation:**
The code has a CUDA-accelerated LK path:
```python
_CUDA_LK = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cv2.cuda.SparsePyrLKOpticalFlow.create()  # smoke-test
        _CUDA_LK = True
except Exception:
    pass
```

Per OPTIMIZATION_PLAN_V3.md §2: `cv2.cuda.getCudaEnabledDeviceCount()` returns 0 because the pip `opencv-python` doesn't include CUDA support. **CUDA OpenCV was built from source and installed locally, but this code path is still dead** because the installed cv2 was not actually detected as CUDA-capable during the test.

**Impact:** Motion analysis runs on CPU. Should be 5-10x slower than CUDA path.

**Fix approach:** The OPTIMIZATION_PLAN_V3.md §5 P2.1 notes that CUDA OpenCV was "done" via `scripts/install_opencv_cuda.sh`, but the `_CUDA_LK = False` flag in `motion_analysis.py` remains unchecked. The path exists and will activate IF cv2 is rebuilt correctly.

**Test needed:**
```python
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())  # Should be > 0 if CUDA build is active
```

---

## Fragile Areas

### GDINO Parallel Thread Ordering Issue (Now Opt-In)

**Files:** `bachman_cortex/pipeline.py:407-413, 555-557` (GDINO thread + _gdino_skip_ready event)

**Situation:**
GDINO runs in a separate thread but waits on `_gdino_skip_ready`, which is only set AFTER the Phase 1 inner loop completes. This makes GDINO effectively sequential, not parallel.

**Status per OPTIMIZATION_PLAN_V3.md §3.5:**
GDINO is now opt-in (behind `--privacy-check` flag). Its sequential bottleneck no longer affects the default hot path. However, when privacy checks are enabled, this serialization still exists.

**Fix documented but not implemented:**
P0.priv.B (un-block GDINO from skip gate) is listed as future work for privacy-enabled runs. Would require structural change to launch GDINO speculatively and handle its results after Phase 1.

**Risk:** Privacy-enabled runs pay full GDINO cost (112.5s per original profile) without parallelism gains.

---

### Frame Cache Memory Spike at Native Resolution (FIXED in P1.2)

**Status:** ✅ **RESOLVED** per OPTIMIZATION_PLAN_V3.md

Previously, Phase 1 cache held full-resolution (1080p) frames. Now (`pipeline.py:654`) it holds 720p downscaled frames after P1.1, cutting RAM from ~1.1GB to ~475MB for 180-frame clips.

**Current state:** Downscaling applied at `PipelineConfig.phase1_long_edge = 720` (default).

**Fallback:** `phase1_long_edge=None` disables downscaling if face recall issues arise.

---

### Model Weight Checksum Validation Not Implemented

**Files:** `bachman_cortex/models/download_models.py` (downloads SCRFD, YOLO, Hands23, legacy 100DOH)

**Issue:**
Downloaded model weights are never checksummed or validated. If a download is interrupted or corrupted, the pipeline will fail only at inference time, not during setup.

**Risk:**
- User runs `download_models.py`, which partially downloads a 400MB Hands23 model, then disconnects.
- Setup completes without error.
- Later, during batch processing, inference fails obscurely when model weights are corrupted.

**Example code locations:**
- `download_models.py:75-85` (wget/curl/urllib download, no checksum)
- `download_models.py:86` (size printed, but no SHA256 check)

**Mitigation in place:** None. Manual verification only.

**Recommendation:**
```python
import hashlib
# After download:
expected_sha256 = "abc123..."  # Hardcode or fetch from source
actual_sha256 = hashlib.sha256(open(weight_file, 'rb').read()).hexdigest()
assert actual_sha256 == expected_sha256, f"Corrupted download: {actual_sha256}"
```

---

## Scaling Limits

### GPU Memory Constrained to Single Worker on RTX 3060

**Files:** `bachman_cortex/run_batch.py:70-81` (worker count auto-detection)

**Current hardware:** RTX 3060 12GB, 16GB system RAM

**Problem:**
The `_auto_detect_workers()` function estimates workers as `min(vram_gb // 8, cpu_count // 2, 4)`. On a 3060 (12GB total), this gives:
```python
workers = min(12 // 8, ...) = min(1, ...) = 1
```

So only 1 concurrent worker can run, meaning only 1 video processes at a time.

**Why:**
Each worker loads all models into its own CUDA context (~6-8GB per the code comment). With only 12GB total and OS/PyTorch overhead, 1 worker is safe.

**Scaling path:**
- Use multi-GPU setups (e.g., 2x RTX 3060 = 2 workers, 24GB total)
- Use larger GPUs (A10G 24GB = ~2-3 workers)
- Use CPU-only fallback with parallel workers (much slower, see CONTEXT.md economics)

**Current limitation:** ~5,000 videos/day on single 3060 = ~5,000 / (180 sec/video * 60 / 1-worker) = ~0.46 videos/second. With 86,400 seconds/day and 180s per video, need ~480 videos/day throughput, which is feasible. But at scale, multi-GPU or larger GPU needed.

---

### RAM Swap Thrashing During Frame Cache Peak

**Files:** `bachman_cortex/pipeline.py:650-659` (phase1_cache assembly)

Per OPTIMIZATION_PLAN_V3.md §2:
- System RAM: 16GB total, ~9.4GB free
- Swap: 4GB total, **92% used (3.7GB)**
- Peak usage during Phase 1: ~1.1GB native frames (now 475MB at 720p) + ~2-3GB model weights + PyTorch workspace

**Impact:**
Swap thrashing causes irregular latency spikes (p99 wall-clock varies by 10-20s on same video). This affects batch throughput predictability.

**Mitigation:**
- P1.1 (720p downscale) + P1.2 (drop raw cache) already deployed, halving frame cache size.
- Further wins require model quantization (INT8 ONNX) or model splitting (not realistic for a single worker).

**Recommendation:** For production at scale, use machines with ≥24GB RAM to avoid swap.

---

## Test Coverage Gaps

### No Privacy Validation Set (Blocks GDINO Optimization)

**Files:**
- No `bachman_cortex/tests/privacy_validation/` directory exists
- `OPTIMIZATION_PLAN_V2.md §4 P0.0` describes how to bootstrap one

**Issue:**
Every privacy detection optimization (GDINO-tiny swap, confidence thresholding, tracker propagation) trades accuracy for speed. Without a labelled validation set, there's no way to confirm privacy recall doesn't regress.

**Blocking:**
- P0.5 (swap to GDINO-tiny) cannot ship without validation set
- P2.x privacy-path optimizations cannot be measured

**Current state:**
Privacy checks are opt-in (behind `--privacy-check` flag), so default runs don't use GDINO. But the flag exists as a precondition for future work.

**Recommendation:**
OPTIMIZATION_PLAN_V3.md notes this as "not started" but "prerequisite for P0.5, P2.1, P2.2". Timeline: ~3-4 hours to bootstrap (~200-400 GDINO-harvested frames + manual review).

---

### No Regression Tests for GPU-Accelerated LK vs CPU

**Files:** `bachman_cortex/checks/motion_analysis.py` (both CPU and CUDA paths exist)

**Issue:**
The code has two LK implementations (CPU fallback, CUDA GPU path), but no test compares their outputs. If CUDA LK is activated, motion scores could diverge from CPU baseline without detection.

**OPTIMIZATION_PLAN_V3.md notes:**
"motion scores within 0.001 of CPU baseline" was observed during a manual bench run, but no automated test verifies this.

**Risk:** If CUDA OpenCV build changes or numerical precision differs, motion checks could silently change thresholds.

**Recommendation:**
```python
# unittest in benchmarks
def test_cuda_lk_matches_cpu():
    test_video = "path/to/test.mp4"
    # Run both paths on same video
    cpu_scores = motion_analysis_cpu(test_video)
    gpu_scores = motion_analysis_cuda(test_video)
    np.testing.assert_allclose(cpu_scores, gpu_scores, atol=0.001)
```

---

## Complexity Hotspots

### Motion Analysis Module is 1,059 Lines (Largest Single File)

**File:** `bachman_cortex/checks/motion_analysis.py`

**Functions/classes:**
- `MotionAnalyzer` — stateful LK accumulator (large)
- `check_motion_combined_from_analyzer()` — slice analyzer results per clip
- `check_motion_combined()` — legacy per-clip motion (now unused in Phase 2)
- `_lk_track()` — CPU/CUDA branch for optical flow
- Multiple helper functions for LK, SSIM, frozen detection

**Complexity:**
- CPU and CUDA code paths interleaved
- State machine for LK accumulation (per-second Trans/Rot tracking, sampled frame list for frozen detection)
- Multiple thresholds (stability_trans, stability_rot, frozen_trans, frozen_rot, etc.)

**Maintainability:**
- 1,059 lines is manageable but at the threshold where extraction into separate modules would help
- CUDA path is #ifdef-like code (try/except around cv2.cuda) which makes logic flow harder to follow

**No immediate action needed** — the code is well-documented in OPTIMIZATION_PLAN_V3.md and works. But if motion checks grow more complex (e.g., adding IMU-based fallback), consider extracting CPU and CUDA paths into separate modules.

---

### Pipeline.py has Multiple Phases with Nested Loops (757 Lines)

**File:** `bachman_cortex/pipeline.py`

**Structure:**
- 4 phases (metadata gate, segment filtering, validation, yield)
- Phase 2 has nested loops: clips → per-clip checks → per-frame aggregation
- Many conditional branches based on flags (privacy, egocentric, body-part visibility)

**Complexity sources:**
- `_phase1_run_inference()` — loop over batches of frames, runs YOLO (batch), SCRFD (threaded), Hands23 (per-frame)
- `_phase2_validate_single_clip()` — series of checks, some conditional on flags
- Flag-based loading (`pose_needed`, `privacy_needed`, etc.)

**Maintainability:**
- ~757 lines is large but reasonable for a 4-phase pipeline
- Clear comments mark phase boundaries
- Flag-based conditionals are localized (lines 89-91, 107, 410-411, etc.)

**Future risk:** If more optional checks are added, the conditional logic could become hard to follow. Consider a plugin architecture or strategy pattern.

---

## Dependencies at Risk

### Hands23 GitHub Dependency (Maintenance Risk)

**Files:** `bachman_cortex/models/hand_detector.py`, `download_models.py:53-64` (clones from `github.com/EvaCheng-cty/hands23_detector`)

**Risk factors:**
- **Author maintenance:** Single-author repo (EvaCheng-cty). If author stops maintaining, security patches or PyTorch compatibility fixes may not arrive.
- **No official PyPI package:** Must clone from GitHub at setup time. Network outages or GitHub API changes could break setup.
- **Detectron2 dependency:** Hands23 requires `detectron2` built from source with `--no-build-isolation`, which is fragile across Python/CUDA versions.

**Mitigation:**
- Hands23 is a published NeurIPS 2023 paper with multiple citations — unlikely to disappear immediately.
- Code is read-only (inference only); minimal risk of upstream breakage on version mismatch.
- Recommended: Pin to a specific commit hash instead of cloning `main`.

**Current approach (from download_models.py:59):**
```python
subprocess.run([
    "git", "clone",
    "https://github.com/EvaCheng-cty/hands23_detector.git",
    str(repo_dir)
], check=True)
```

**Recommendation:**
```python
# Add commit pinning:
"https://github.com/EvaCheng-cty/hands23_detector.git@a1b2c3d"  # Known working commit
```

---

### GDINO Model Still Bundled (Memory Overhead When Privacy Off)

**Files:** `bachman_cortex/models/download_models.py`, `download_models.py:128-143` (GDINO weights download)

**Issue:**
GDINO weights (~700MB) are downloaded during setup even though privacy checks are opt-in and disabled by default. This wastes storage and initial download time.

**Per OPTIMIZATION_PLAN_V3.md:**
GDINO is now gated in `pipeline.py:89` (conditional flag), so `load_models()` skips it when privacy is off. But the weights are still downloaded.

**Optimization:** Lazy-load GDINO weights only when `--privacy-check` is first requested. Currently, setup downloads all models.

**Impact:** ~700MB disk space and ~2-3 min download time saved per setup if privacy is never used (common for default runs).

---

## Missing Critical Features

### IMU-Based Motion Detection Not Implemented (Blocked on Precondition)

**Files:** OPTIMIZATION_PLAN_V3.md §4 P1.3, OPTIMIZATION_PLAN_V2.md §7 (IMU details)

**Status:** Not started. Blocked on precondition: LATAM samples do not have `gpmd` (GoPro Metadata Format) tracks.

**Why it matters:**
- GoPro and other action cameras embed gyroscope/accelerometer data in metadata
- Motion analysis could be: gyro RMS per second instead of optical flow
- Expected speedup: 40s → 1s on videos with GPMF (if future videos have it)

**Current workaround:** Optical flow (LK) on CPU or CUDA.

**Action:** If future production ingestion includes GoPro videos with GPMF:
1. Verify presence: `ffprobe -v error -select_streams d -show_entries stream=codec_tag_string <video>`
2. Implement `pygpmf` integration in `motion_analysis.py`
3. Fall back to LK for non-GPMF videos

---

### No Automated Model Versioning (Weights Mismatch Risk)

**Files:** `bachman_cortex/models/download_models.py`, `pipeline.py:146-180` (model loading)

**Issue:**
Model weights are downloaded to hardcoded paths (`weights/insightface`, `weights/hands23_detector`, etc.) with no version tracking. If a new version is released, there's no way to distinguish old vs. new weights without manual inspection.

**Risk scenario:**
1. User A downloads SCRFD v2.5GF
2. 6 months later, upstream releases SCRFD v3.0 with different accuracy
3. User B runs `download_models.py`, gets v3.0
4. Batch results are incomparable (different model versions mixed)

**Current mitigation:** None. Implicit assumption that all workers use the same setup snapshot.

**Recommendation:**
Track model versions in a metadata file:
```python
# models/weights/manifest.json
{
  "scrfd": {"url": "...", "sha256": "...", "date": "2026-04-15"},
  "yolo11s": {"url": "...", "sha256": "...", "date": "2026-04-15"},
  ...
}
```

---

## Security & Infrastructure Gaps

### No Runtime Limits on Model Inference (GPU OOM Risk)

**Files:** `bachman_cortex/pipeline.py` (model inference loops)

**Issue:**
Models run without timeout or memory limits. If a malformed video causes a model to hang or exhaust VRAM:
- Worker thread deadlocks
- GPU memory leaked
- Batch processing stalls

**Example scenario:**
1. Very high-resolution video (e.g., 4K) reaches Phase 1
2. YOLO or Hands23 batches exceed VRAM
3. CUDA OOM error kills the worker
4. No retry, no cleanup

**Current safeguards:**
- Phase 0 metadata gate rejects videos outside 1080p (checks `width <= 1920, height <= 1080`)
- Hands23 is per-frame, so memory bounded
- YOLO batches are fixed size (_BATCH_SIZE = 16)

**Remaining risk:** Edge cases (corrupted metadata, unexpected container formats) could still cause issues.

**Recommendation:**
```python
# In _phase1_run_inference:
import signal
def timeout_handler(signum, frame):
    raise TimeoutError("Inference timeout")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5-minute timeout per clip
try:
    # inference loops
finally:
    signal.alarm(0)  # Cancel alarm
```

(Note: `signal` is Unix-only; use `timeout` process wrapper for cross-platform safety.)

---

### Multiprocessing Pool Worker State Not Isolated (Crash Propagation)

**Files:** `bachman_cortex/run_batch.py:90-100` (pool initialization)

**Issue:**
Workers are initialized via `_init_worker()`, which loads all models into the worker's CUDA context. If a worker crashes (e.g., CUDA OOM), the models are left in an inconsistent state. The pool may reuse the crashed worker for subsequent videos, leading to cascading failures.

**Current mitigation:** `multiprocessing.Pool` has a `maxtasksperchild` parameter (not used in current code). Setting this to a small number (e.g., 10) would recycle workers periodically.

**Recommendation:**
```python
with Pool(
    processes=num_workers,
    initializer=_init_worker,
    initargs=(config_dict,),
    maxtasksperchild=10  # Recycle worker after 10 videos
) as pool:
    ...
```

This limits crash propagation — a worker that fails will be replaced within 10 videos.

---

## Known Unresolved Issues from Optimization Plans

### YOLO Batching Not Extended to SCRFD/Hands23 (P0.3 Partial)

**Status:** ✅ **DONE** but may have room for improvement

Per OPTIMIZATION_PLAN_V3.md §3.4:
- YOLO uses batched inference ✅
- SCRFD batched via `ThreadPoolExecutor(scrfd_threads=4)` ✅
- Hands23 remains per-frame (no batching possible via public API)

**Current state:** P0.4 uses threaded SCRFD, which is ~1.2x faster than serial but not true batching. Expected impact: 5-10% Phase 1 speedup.

**Future improvement:** If Hands23 is ever optimized (P2.2), would require either:
- Direct Detectron2 batching (requires source code knowledge)
- Model export to ONNX + batched inference
- Alternative hand detector (breaks compatibility)

---

### Privacy GDINO Parallelization Not Implemented (P0.priv.B)

**Status:** Not started, opt-in path only

Per OPTIMIZATION_PLAN_V3.md, when `--privacy-check` is enabled:
- GDINO still waits on `_gdino_skip_ready` (set after Phase 1 finishes)
- This makes GDINO sequential, despite having its own thread
- Could be un-blocked to run speculatively, accept the overhead of processing skip-marked frames

**Impact:** When privacy is enabled (not default), GDINO adds 50-100s to wall-clock. Un-blocking could save 60-80s by overlapping with Hands23-dominated Phase 1.

**Why not done:** User constraint: focus on default-run performance first.

---

## Summary of Risk Priorities

| Issue | Severity | Effort | Notes |
|-------|----------|--------|-------|
| Custom OpenCV dnn shim fragility | **Medium** | Low | Works now, breaks if OpenCV rebuilt. Clear mitigation path. |
| Hands23 GitHub dependency | Medium | Low | Pin to commit hash recommended. |
| Model weight validation (checksums) | **Medium** | Low | Setup could fail silently if download corrupts. |
| GPU OOM/timeout handling | **Medium** | Medium | Workers can deadlock. Add signal-based timeouts. |
| Worker crash propagation | Low | Low | Add `maxtasksperchild` to recycle workers. |
| Privacy validation set missing | Low | High | Blocks future optimizations, not default path. |
| CUDA LK on CPU fallback | Low | Low | Works, not critical since P0.2/P1.1 compensate. |
| GDINO parallelization (privacy path) | Low | Medium | Opt-in path, documented as future work. |

---

*Concerns audit: 2026-04-15*
