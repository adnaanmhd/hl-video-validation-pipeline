# Technology Stack

**Analysis Date:** 2026-04-15

## Languages

**Primary:**
- Python 3.11+ - Core application, ML pipeline, video processing, CLI entry point
- Bash - Setup and deployment (`validate.sh`), environment configuration

**Secondary:**
- C++ - Optional legacy hand detector extensions (100DOH) via PyTorch extension modules

## Runtime

**Environment:**
- Python 3.11, 3.12, or 3.13 (minimum 3.11, maximum <3.14 per `pyproject.toml`)
- Virtual environment managed by `venv` module
- Requires system-level dependencies: `ffmpeg`, `ffprobe`, `git`, build tools (gcc/g++)

**Package Manager:**
- pip (via venv)
- Lockfile: `pyproject.toml` (setuptools-based) and `bachman_cortex/requirements.txt` (reference only)

## Frameworks

**Core Application:**
- ultralytics (YOLO11s) - Object and person detection
- insightface>=0.7.3 - Face detection (SCRFD-2.5GF model)
- detectron2 - Hand-object detection backbone (Hands23 detector from FaceBook Research)
- transformers>=4.36.0 - Model hub integration for Grounding DINO

**Inference Engines:**
- PyTorch - Deep learning runtime (CPU or CUDA 12.1, installed separately)
- ONNX Runtime (onnxruntime or onnxruntime-gpu) - Model inference optimization
- OpenCV (opencv-python>=4.9.0) - Video I/O, frame processing, optical flow

**Data & Processing:**
- NumPy>=1.24.0 - Array operations
- Pillow>=10.0 - Image processing fallback
- SciPy - Scientific computing (legacy hand detector dependencies)
- supervision>=0.18.0 - Detection result handling and visualization

**Utilities:**
- tqdm>=4.65 - Progress bars
- gdown>=5.0 - Google Drive model weight downloads
- pyyaml - Configuration parsing (legacy hand detector)
- easydict - Dict-like attribute access (legacy hand detector)
- cython - C extensions compilation (legacy hand detector)

**Build & Installation:**
- setuptools>=68.0 - Package management
- pip with `--index-url` for GPU-specific PyTorch wheels

## Key Dependencies

**Critical:**
- `torch`/`torchvision` - Deep learning (GPU-aware, installed via platform-specific index URL in `validate.sh`)
- `detectron2` - Installed from GitHub source, requires no-build-isolation flag
- `onnxruntime` or `onnxruntime-gpu` - Model inference (GPU variant on compute instances)
- `opencv-python` with CUDA support (custom OpenCV-CUDA build for frame extraction acceleration via `cv2.cudacodec`)
- `insightface` - Pre-trained face detection models

**Infrastructure:**
- ffmpeg/ffprobe - Video metadata extraction and lossless transcoding (HEVC to H.264 via NVENC or libx264)
- NVIDIA CUDA Toolkit 12.1 - GPU acceleration (detected via `nvidia-smi`)
- Video Codec SDK 12.2.72 - NVENC/NVDEC hardware video codec libraries (present in `Video_Codec_SDK_12.2.72/` directory)

## Configuration

**Environment:**
- `FORCE_CPU=1` - Override automatic GPU detection to use CPU-only PyTorch
- GPU detection: automatic via `nvidia-smi` check; falls back to CPU if unavailable

**Build & Runtime:**
- PyTorch index URL selection (automatic in `validate.sh`):
  - GPU: `https://download.pytorch.org/whl/cu121`
  - CPU: `https://download.pytorch.org/whl/cpu`
- Model weight caching: `bachman_cortex/models/weights/` directory (created during setup)
- Model sources:
  - SCRFD (InsightFace): Automatic download via `insightface` API
  - YOLO11s: Automatic download via Ultralytics API
  - Hands23: GitHub clone + manual model weight download from University of Michigan server
  - 100DOH (legacy, optional): GitHub clone + Google Drive weight download via gdown

**Preprocessing:**
- Optional HEVC to H.264 lossless transcoding flag: `--hevc-to-h264` (default: disabled)
- Encoder selection: NVENC (`h264_nvenc -preset p7 -tune lossless`) if GPU present, else libx264 (`-crf 0`)

## Platform Requirements

**Development:**
- Python 3.11+ (macOS via `brew install python@3.12`, Ubuntu via `sudo apt install python3.12 python3.12-venv`)
- FFmpeg (macOS via `brew install ffmpeg`, Ubuntu via `sudo apt install ffmpeg`)
- git (required for detectron2 and model repos)
- C++ build tools (macOS: `xcode-select --install`, Ubuntu: `sudo apt install build-essential`)
- Optional: NVIDIA GPU with CUDA 12.1 support and nvidia-smi

**Production (AWS/EC2):**
- GPU instance with NVIDIA driver and CUDA 12.1 support
- onnxruntime-gpu for accelerated inference
- FFmpeg with NVENC support (NVIDIA Video Codec SDK 12.2.72 libraries)
- Parallel video processing: auto-scales to 4 workers max based on available VRAM (8GB per worker minimum)

**Memory Requirements:**
- Base models: ~6-8 GB VRAM per worker process
- Multi-worker scaling: up to 4 workers on high-VRAM GPUs
- CPU-only: fallback with reduced throughput

## Initial Setup Duration

- First run: 5-10 minutes (model weight downloads ~1.5GB total: SCRFD, YOLO11s, Hands23)
- Subsequent runs: sub-second startup (models cached in `bachman_cortex/models/weights/`)

---

*Stack analysis: 2026-04-15*
