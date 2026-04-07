# Egocentric Video Validation Pipeline

Validates egocentric (first-person POV) video quality across 15 checks for datasets used to train autonomous humanoid robots. Runs 4 check categories: video metadata, luminance & blur, motion analysis, and ML-based detection.

## Quick Start

```bash
git clone <repo-url> && cd hl-bachman
./validate.sh /path/to/video.mp4
```

First run takes ~5-10 minutes (installs dependencies, downloads ~1.5GB of model weights). Subsequent runs start in seconds.

Setup only (no pipeline run):

```bash
./validate.sh --setup-only
```

## Requirements

- **Python 3.11+** (3.12, 3.13 also supported)
- **FFmpeg** (ffprobe used for metadata extraction)
- **~2GB disk** (model weights + virtual environment)
- **macOS** (Apple Silicon) or **Linux** (Ubuntu/Debian)
- **Windows:** Not natively supported (detectron2 lacks Windows support). Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) -- the pipeline runs fully inside a WSL2 Ubuntu environment.
- GPU optional -- auto-detected. Force CPU with `FORCE_CPU=1 ./validate.sh ...`

## What It Checks

See [checks.md](checks.md) for detailed acceptance conditions and thresholds.

### Video Metadata (gate -- failure skips all other checks)

| Check | Acceptance Condition |
|---|---|
| Format | MP4 container |
| Encoding | H.264 codec |
| Resolution | >= 1920 x 1080 |
| Frame Rate | >= 28 FPS |
| Duration | >= 60 seconds |
| Orientation | Rotation = 0 or 180, width > height |

### Luminance & Blur

Per-frame classification using Tenengrad sharpness + luminance zones. Video passes if both conditions are met:
1. (accept + review) frames >= 80% of total
2. Brightness stability: std dev of per-frame mean luminance <= 60

### Motion Analysis

| Check | Acceptance Condition |
|---|---|
| Camera Stability | Two-stage LK shakiness score <= 0.30 |
| Frozen Segments | No > 30 consecutive frames with SSIM > 0.99 |

### ML Detection

| Check | Model | Acceptance Condition |
|---|---|---|
| Face Presence | SCRFD-2.5GF | Face confidence < 0.8 in all frames |
| Participants | YOLO11m + SCRFD | Persons <= 1 in >= 90% frames |
| Hand Visibility | Hands23 | Both hands in >= 90% frames |
| Hand-Object Interaction | Hands23 | Interaction in >= 70% frames |
| Privacy Safety | YOLO11m + Grounding DINO | 0 sensitive objects in all frames |
| View Obstruction | OpenCV heuristics | <= 10% frames obstructed |
| POV-Hand Angle | Geometric | Hands within 40 deg of center in >= 80% frames |
| Body Part Visibility | YOLO11m-pose | Only hands/forearms (up to elbows) in >= 90% frames |

## Usage

```bash
# Single video
./validate.sh /path/to/video.mp4

# Directory of videos
./validate.sh /path/to/videos/

# Multiple files
./validate.sh video1.mp4 video2.mp4

# Custom output directory and sampling rate
./validate.sh /path/to/videos/ --output results/ --fps 2

# Skip Grounding DINO (faster, less precise privacy detection)
./validate.sh /path/to/videos/ --no-gdino

# Limit frames per video
./validate.sh /path/to/videos/ --max-frames 50
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output`, `-o` | `ml_checks/results` | Output directory for reports |
| `--fps` | `1.0` | Frame sampling rate (FPS) |
| `--max-frames` | unlimited | Max frames to sample per video |
| `--no-gdino` | disabled | Skip Grounding DINO (faster) |

### After Initial Setup

Once `validate.sh` has run at least once, you can also use the CLI directly:

```bash
source .venv/bin/activate
hl-validate /path/to/videos/
```

## Output

Each run produces:

- `batch_report.md` -- Markdown report with per-video results grouped by check category
- `batch_results.json` -- Full batch results as JSON
- `<video_name>.json` -- Per-video JSON with detailed check results

### Report Format

Each check shows: status (PASS/FAIL/REVIEW/SKIPPED), acceptance condition, and actual measured result.

## Models

| Model | Purpose | Size |
|---|---|---|
| SCRFD-2.5GF | Face detection | ~14 MB |
| YOLO11m | Person + object detection | ~40 MB |
| YOLO11m-pose | Body part keypoint detection | ~40 MB |
| Hands23 | Hand detection + contact state | ~446 MB |
| Grounding DINO | Zero-shot privacy detection | ~700 MB |

Total: ~1.2 GB downloaded on first run.

## Performance

Estimated per-video processing time (120s video, 1 FPS sampling):

| Stage | CPU (M-series Mac) | GPU (A10G) |
|---|---|---|
| Metadata + luminance/blur + motion | ~65s | ~65s |
| SCRFD | ~1.5s | <1s |
| YOLO11m | ~12s | ~2s |
| YOLO11m-pose | ~12s | ~2s |
| Hands23 | ~220s | ~15-25s |
| Grounding DINO (flagged only) | ~3s/frame | <1s/frame |
| **Total** | **~5 min** | **~1.5 min** |

Hands23 dominates CPU time. GPU gives ~3-4x end-to-end speedup.

## Advanced

### Manual Setup (without validate.sh)

```bash
# 1. Create venv
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Or for CUDA:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install ONNX Runtime
pip install onnxruntime
# Or for GPU: pip install onnxruntime-gpu

# 4. Install detectron2
pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# 5. Install this package
pip install -e .

# 6. Download model weights
python ml_checks/models/download_models.py

# 7. Run
hl-validate /path/to/videos/
```

### GPU Acceleration

`validate.sh` auto-detects NVIDIA GPUs via `nvidia-smi`. To force CPU-only:

```bash
FORCE_CPU=1 ./validate.sh /path/to/videos/
```

### Pipeline Behavior

- **Metadata gate:** If any metadata check fails, all other checks are skipped.
- **Independent categories:** Luminance & blur, motion, and ML checks run independently of each other.
- **Statuses:** `pass`, `fail`, `review` (borderline), `skipped` (metadata gate).

## Project Structure

```
hl-bachman/
├── validate.sh                  # One-command entry point
├── pyproject.toml               # Package configuration
├── checks.md                    # Check specifications and thresholds
├── README.md
└── ml_checks/
    ├── pipeline.py              # ValidationPipeline orchestrator
    ├── run_batch.py             # CLI entry point (hl-validate)
    ├── CONTEXT.md               # Model selection rationale
    ├── checks/
    │   ├── check_results.py     # CheckResult dataclass
    │   ├── video_metadata.py    # 6 metadata checks
    │   ├── luminance_blur.py    # Tenengrad + luminance + brightness stability
    │   ├── motion_analysis.py   # Optical flow + frozen segment detection
    │   ├── face_presence.py     # Face detection check
    │   ├── participants.py      # Person count check
    │   ├── hand_visibility.py   # Hand detection check
    │   ├── hand_object_interaction.py  # Contact state check
    │   ├── privacy_safety.py    # Sensitive object detection
    │   ├── view_obstruction.py  # Lens obstruction heuristic
    │   ├── pov_hand_angle.py    # Hand angle check
    │   └── body_part_visibility.py  # Keypoint-based body part check
    ├── models/
    │   ├── download_models.py   # Model weight downloader
    │   ├── scrfd_detector.py    # SCRFD face detector
    │   ├── yolo_detector.py     # YOLO11m object detector
    │   ├── yolo_pose_detector.py # YOLO11m-pose keypoint detector
    │   ├── hand_detector.py     # Hands23 hand-object detector
    │   └── grounding_dino_detector.py  # Zero-shot privacy detector
    └── utils/
        ├── frame_extractor.py   # Video frame sampling
        └── video_metadata.py    # FFprobe metadata extraction
```
