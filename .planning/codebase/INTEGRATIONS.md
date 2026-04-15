# External Integrations

**Analysis Date:** 2026-04-15

## APIs & External Services

**Model Repositories (GitHub):**
- Hands23 hand-object detector: https://github.com/EvaCheng-cty/hands23_detector
  - SDK/Client: Git clone in `download_models.py`
  - Authentication: None (public repo)
  - Weight source: HTTP download from University of Michigan server (fouheylab.eecs.umich.edu)

- 100DOH hand-object detector (legacy, optional): https://github.com/ddshan/hand_object_detector
  - SDK/Client: Git clone via setuptools
  - Authentication: None (public repo)
  - Weight source: Google Drive via gdown package

- Detectron2: https://github.com/facebookresearch/detectron2
  - SDK/Client: pip install from Git source
  - Authentication: None (public repo)
  - Used by: Hands23 detector as backbone (region-based CNN)

**Model Hub Services:**
- Ultralytics (YOLO11s): Automatic download via API in `ultralytics` package
  - Cached in: `~/.cache/ultralytics/models/` (OS default) or specified model path

- HuggingFace Transformers: Model hub integration for Grounding DINO variant
  - SDK: `transformers>=4.36.0` package
  - Auth: None (public models)
  - Used in: Fallback vision-language grounding (optional, not currently active)

- Google Drive (legacy 100DOH weights): https://drive.google.com/uc?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE
  - SDK/Client: `gdown>=5.0` package
  - Authentication: None (public link)
  - Setup: Called from `bachman_cortex/models/download_models.py` with `--100doh` or `--all` flag

**Inference Optimization:**
- ONNX Model Zoo: Pre-trained models for custom optimizations (not currently integrated)
- NVIDIA TensorRT: Available via ONNX Runtime GPU variant for inference optimization (not forced)

## Data Storage

**Databases:**
- None - This is a stateless batch processing pipeline

**File Storage:**
- Local filesystem only
  - Input: Video files (`.mp4`) from user-provided paths
  - Output: JSON reports and text timelines in `--output` directory (default: `bachman_cortex/results/`)
  - Models: Cached in `bachman_cortex/models/weights/` directory tree
  - Preprocessed videos: Optional HEVC→H.264 transcoded files in temporary directories

**Caching:**
- Model weights: Persistent local cache in `bachman_cortex/models/weights/`
  - InsightFace SCRFD: `insightface/` subdirectory
  - Hands23: `hands23_detector/model_weights/`
  - 100DOH: `hand_object_detector/models/` (legacy, optional)
  - YOLO11s: Default Ultralytics cache or specified model path
- Frame extraction cache: None (streaming from video file via OpenCV)
- Optical flow: Computed per-batch, not cached

## Authentication & Identity

**Auth Provider:**
- None - No user authentication required
- No API keys or credentials needed
- Public GitHub repos and model URLs only

**Setup Requirements:**
- Environment variable `FORCE_CPU=1` (optional): Force CPU mode instead of GPU auto-detection
- No env var secrets or credentials files required

## Monitoring & Observability

**Error Tracking:**
- None - Error handling is in-process with structured exception output to stderr

**Logs:**
- Standard output: Pipeline progress messages (frame counts, model info, timing)
- Standard error: Error traces for video processing failures
- No external logging service (no Sentry, CloudWatch, Datadog integration)

**Video Metadata & Reporting:**
- ffprobe integration: Single JSON output for container format, codec, resolution, FPS, duration, rotation tag
- Per-video JSON reports: `{video_name}/{video_name}.json` with structured results
- Batch summary: `batch_results.json` with aggregated yield metrics
- Timeline HTML reports: Human-readable validation results per video (optional)

## CI/CD & Deployment

**Hosting:**
- Single-machine batch processing (development: Linux/macOS, production: AWS GPU instances)
- No containerization (Docker/Kubernetes) currently used
- Direct filesystem input/output model

**CI Pipeline:**
- None - No external CI service integration (GitHub Actions, Jenkins, etc.)
- Local validation via `validate.sh` setup script and `hl-validate` CLI

**Model Loading Strategy:**
- On-demand loading in main process or per-worker pool processes (`ValidationProcessingPipeline.load_models()`)
- Cached in `_worker_pipeline` global for multiprocessing pool workers
- Pool initialization via `_init_worker()` in `run_batch.py` with config serialization

## Environment Configuration

**Required env vars:**
- None mandated; optional:
  - `FORCE_CPU=1` - Use CPU-only PyTorch (default: auto-detect via nvidia-smi)

**System Dependencies (checked in validate.sh):**
- Python 3.11+ (detected from PATH)
- ffprobe (FFmpeg tools)
- git (for detectron2 and model repos)
- C++ compiler (gcc/g++, or clang on macOS)
- NVIDIA driver + CUDA 12.1 (optional, if GPU desired)

**Secrets location:**
- No secrets required or stored
- `.env` or credentials files: Not applicable

## Video I/O & Codec Integration

**FFmpeg/FFprobe:**
- `ffprobe`: JSON-format metadata extraction (container, codec, dimensions, FPS, duration, rotation)
  - Called from: `bachman_cortex/utils/video_metadata.py`
  - Invocation: `ffprobe -v quiet -show_format -show_streams -print_json <video_path>`

- `ffmpeg`: Optional lossless transcoding (HEVC → H.264)
  - Called from: `bachman_cortex/utils/transcode.py` if `--hevc-to-h264` flag set
  - Encoder selection:
    - GPU: `h264_nvenc -preset p7 -tune lossless` (NVIDIA Video Codec SDK 12.2.72)
    - CPU fallback: `libx264 -crf 0 -preset veryfast`
  - Audio stream handling: Copy as-is (`-c:a copy`)
  - Rotation tag handling: Strip via `-display_rotation:v:0 0`

**OpenCV Video Codecs:**
- `cv2.VideoCapture`: CPU-based frame extraction fallback
- `cv2.cudacodec.createVideoReader()`: GPU-accelerated decoding (NVDEC) if available
  - Fallback to CPU if custom OpenCV-CUDA build lacks `cudacodec` module
  - Input: H.264 or HEVC bitstreams
  - Output: Planar YUV or RGB frames to NumPy arrays

**NVIDIA Video Codec SDK:**
- Location: `Video_Codec_SDK_12.2.72/` at project root
- Purpose: Provides NVENC (encoding) and NVDEC (decoding) libraries
- Integration: Used by FFmpeg's `h264_nvenc` encoder and OpenCV's CUDA codec module
- Libraries: Precompiled in `Video_Codec_SDK_12.2.72/Lib/` for link-time dependency

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Download & Model Management

**Model Download Strategy:**
- Centralized via `bachman_cortex/models/download_models.py`
- Triggered during setup: `validate.sh` → `python download_models.py`
- Resumable: Checks existence; only downloads missing weights

**Download Methods:**
- InsightFace API: Automatic SDK download (buffalo_sc model bundle)
- Ultralytics API: Automatic SDK download (YOLO11s.pt)
- HTTP direct: wget/curl with SSL override for Hands23 (certificate issues on some servers)
- Google Drive: gdown package for 100DOH (legacy, optional)
- Git clone: Hands23 and 100DOH repo checkouts to `models/weights/`

**Post-Download Patching:**
- Hands23: CPU compatibility patches for hardcoded `.cuda()` calls in detector source
  - Files patched: `hodetector/modeling/roi_heads/roi_heads.py`, `hodetector/utils/positional_encoding.py`
  - Pattern: Replace `.cuda()` with `.to(_get_device())` helper

- 100DOH: PyTorch 2.x C++ API compatibility patches (optional)
  - Files patched: `lib/model/csrc/` C++ source files
  - Pattern: `.type().is_cuda()` → `.is_cuda()`, `.type()` → `.scalar_type()`, `.data<T>()` → `.data_ptr<T>()`

## Execution Metadata

**Per-Video Processing:**
- Input: Video file path (`.mp4`)
- Output: Structured JSON with:
  - Per-frame labels (USABLE/UNUSABLE/REJECTED with reasons)
  - Segment boundaries (start_sec, end_sec)
  - Usable/unusable/rejected duration (seconds)
  - Yield ratio (usable_duration / total_duration)
  - Metadata test results
  - Phase 1 gate results (face presence, participant count)
  - Phase 2 segment test results (luminance, motion, hand/face visibility, interaction, obstruction, angle)

**Batch-Level Aggregation:**
- Aggregate stats across all videos (mean yield, pass/fail counts by check)
- Batch JSON summary: `batch_results.json`

---

*Integration audit: 2026-04-15*
