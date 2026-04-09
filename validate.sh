#!/usr/bin/env bash
# validate.sh -- One-command setup and run for the video validation pipeline.
#
# Usage:
#   ./validate.sh /path/to/videos/          # setup (if needed) + run
#   ./validate.sh --setup-only              # setup only, no pipeline run
#   FORCE_CPU=1 ./validate.sh /path/to/vid  # force CPU torch even if GPU present
#
# First run: installs deps, downloads ~1.5GB of model weights (~5-10 min).
# Subsequent runs: starts in seconds (all checks are idempotent).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MIN_PYTHON_MAJOR=3
MIN_PYTHON_MINOR=11

# ─────────────────────────────────────────────────────────────
# Parse arguments
# ─────────────────────────────────────────────────────────────

SETUP_ONLY=0
PIPELINE_ARGS=()

for arg in "$@"; do
    if [ "$arg" = "--setup-only" ]; then
        SETUP_ONLY=1
    else
        PIPELINE_ARGS+=("$arg")
    fi
done

if [ "$SETUP_ONLY" -eq 0 ] && [ "${#PIPELINE_ARGS[@]}" -eq 0 ]; then
    echo "Usage:"
    echo "  ./validate.sh /path/to/video.mp4        # validate a video"
    echo "  ./validate.sh /path/to/videos/           # validate a directory of videos"
    echo "  ./validate.sh --setup-only               # install deps and download models only"
    echo ""
    echo "Options (passed through to hl-validate):"
    echo "  --output, -o DIR    Output directory (default: bachman_cortex/results)"
    echo "  --fps N             Sampling FPS (default: 1.0)"
    echo "  --max-frames N      Max frames to sample per video"
    echo "  --no-gdino          Disable Grounding DINO (faster)"
    echo "  --fail-fast         Skip ML inference when quality checks fail"
    echo "  --workers N         Parallel video workers (0=auto, 1=sequential)"
    echo "  --yolo-model FILE   YOLO model for object detection (default: yolo11s.pt)"
    echo ""
    echo "Environment variables:"
    echo "  FORCE_CPU=1         Force CPU-only PyTorch even if GPU is available"
    exit 0
fi

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

info()  { echo "==> $*"; }
error() { echo "ERROR: $*" >&2; }

check_python_version() {
    local bin="$1"
    if ! command -v "$bin" &>/dev/null; then
        return 1
    fi
    local version
    version=$("$bin" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -z "$version" ]; then
        return 1
    fi
    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    if [ "$major" -gt "$MIN_PYTHON_MAJOR" ]; then
        return 0
    elif [ "$major" -eq "$MIN_PYTHON_MAJOR" ] && [ "$minor" -ge "$MIN_PYTHON_MINOR" ]; then
        return 0
    fi
    return 1
}

# ─────────────────────────────────────────────────────────────
# Step 1: Check system dependencies
# ─────────────────────────────────────────────────────────────

info "Checking system dependencies..."

# Find suitable Python
PYTHON=""
for candidate in python3.12 python3.11 python3.13 python3 python; do
    if check_python_version "$candidate"; then
        PYTHON="$candidate"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python >= $MIN_PYTHON_MAJOR.$MIN_PYTHON_MINOR is required."
    echo ""
    echo "Install Python:"
    if [ "$(uname)" = "Darwin" ]; then
        echo "  brew install python@3.12"
    else
        echo "  sudo apt install python3.12 python3.12-venv"
    fi
    exit 1
fi
info "Python: $($PYTHON --version)"

# Check ffmpeg / ffprobe
if ! command -v ffprobe &>/dev/null; then
    error "ffprobe not found. FFmpeg is required for video metadata extraction."
    echo ""
    echo "Install FFmpeg:"
    if [ "$(uname)" = "Darwin" ]; then
        echo "  brew install ffmpeg"
    else
        echo "  sudo apt install ffmpeg"
    fi
    exit 1
fi

# Check git
if ! command -v git &>/dev/null; then
    error "git not found. Required for installing detectron2 and cloning model repos."
    echo ""
    echo "Install git:"
    if [ "$(uname)" = "Darwin" ]; then
        echo "  xcode-select --install"
    else
        echo "  sudo apt install git"
    fi
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# Step 2: Create virtual environment
# ─────────────────────────────────────────────────────────────

if [ ! -f "$VENV_DIR/bin/activate" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip --quiet 2>/dev/null

# ─────────────────────────────────────────────────────────────
# Step 3: Install PyTorch (GPU-aware)
# ─────────────────────────────────────────────────────────────

USE_GPU=0

if python -c "import torch" 2>/dev/null; then
    info "PyTorch: already installed ($(python -c 'import torch; print(torch.__version__)'))"
else
    # Detect GPU
    if [ "${FORCE_CPU:-0}" = "1" ]; then
        info "FORCE_CPU=1 set, installing CPU-only PyTorch"
    elif command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        USE_GPU=1
        info "NVIDIA GPU detected, installing CUDA PyTorch"
    else
        info "No GPU detected, installing CPU PyTorch"
    fi

    if [ "$USE_GPU" -eq 1 ]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    info "PyTorch installed: $(python -c 'import torch; print(torch.__version__)')"
fi

# ─────────────────────────────────────────────────────────────
# Step 4: Install ONNX Runtime
# ─────────────────────────────────────────────────────────────

if python -c "import onnxruntime" 2>/dev/null; then
    info "ONNX Runtime: already installed"
else
    if [ "$USE_GPU" -eq 1 ]; then
        info "Installing ONNX Runtime (GPU)..."
        pip install onnxruntime-gpu
    else
        info "Installing ONNX Runtime (CPU)..."
        pip install onnxruntime
    fi
fi

# ─────────────────────────────────────────────────────────────
# Step 5: Install detectron2
# ─────────────────────────────────────────────────────────────

if python -c "import detectron2" 2>/dev/null; then
    info "Detectron2: already installed"
else
    info "Installing detectron2 (this takes 1-2 minutes)..."
    pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
fi

# ─────────────────────────────────────────────────────────────
# Step 6: Install hl-video-validation package
# ─────────────────────────────────────────────────────────────

info "Installing hl-video-validation package..."
pip install -e "$SCRIPT_DIR" --quiet 2>/dev/null

# ─────────────────────────────────────────────────────────────
# Step 7: Download model weights
# ─────────────────────────────────────────────────────────────

info "Checking model weights..."
python "$SCRIPT_DIR/bachman_cortex/models/download_models.py"

# ─────────────────────────────────────────────────────────────
# Step 8: Run pipeline (or exit if --setup-only)
# ─────────────────────────────────────────────────────────────

echo ""
info "Setup complete."

if [ "$SETUP_ONLY" -eq 1 ]; then
    echo ""
    echo "Run the pipeline with:"
    echo "  ./validate.sh /path/to/videos/"
    echo "  # or after activating the venv:"
    echo "  source .venv/bin/activate"
    echo "  hl-validate /path/to/videos/"
    exit 0
fi

echo ""
info "Running pipeline..."
echo ""
exec hl-validate "${PIPELINE_ARGS[@]}"
