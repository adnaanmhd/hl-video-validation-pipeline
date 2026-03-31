"""Download all required model weights and set up dependencies for ML checks.

Usage:
    python ml_checks/models/download_models.py          # Download SCRFD, YOLO11m, Grounding DINO, Hands23
    python ml_checks/models/download_models.py --100doh  # Also download + compile 100DOH (legacy)
    python ml_checks/models/download_models.py --all     # Everything including legacy 100DOH
"""

import os
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "weights"
MODELS_DIR.mkdir(exist_ok=True)


def download_scrfd():
    """Download SCRFD-2.5GF face detection model via insightface."""
    print("=" * 60)
    print("1. SCRFD Face Detector (via InsightFace)")
    print("=" * 60)
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_sc",
        root=str(MODELS_DIR / "insightface"),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print(f"SCRFD loaded. Detection model: {app.det_model}")
    return app


def download_yolo11m():
    """Download YOLO11m model."""
    print("\n" + "=" * 60)
    print("2. YOLO11m Object Detector")
    print("=" * 60)
    from ultralytics import YOLO

    model = YOLO("yolo11m.pt")
    print(f"YOLO11m loaded: {model.model_name}")
    return model


def download_grounding_dino():
    """Download Grounding DINO model via HuggingFace transformers."""
    print("\n" + "=" * 60)
    print("3. Grounding DINO (via HuggingFace)")
    print("=" * 60)
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-base"
    cache_dir = str(MODELS_DIR / "grounding_dino")

    print(f"Downloading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, cache_dir=cache_dir)
    print(f"Grounding DINO loaded: {model_id}")
    return model, processor


def download_hands23():
    """Download Hands23 hand-object detector (NeurIPS 2023)."""
    print("\n" + "=" * 60)
    print("4. Hands23 Hand-Object Detector (NeurIPS 2023)")
    print("=" * 60)

    repo_dir = MODELS_DIR / "hands23_detector"

    # Step 1: Clone repo
    if not repo_dir.exists():
        print("Cloning hands23_detector repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/EvaCheng-cty/hands23_detector.git", str(repo_dir)],
            check=True,
        )
    else:
        print(f"Repo already exists at {repo_dir}")

    # Step 2: Download weights
    weights_dir = repo_dir / "model_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_file = weights_dir / "model_hands23.pth"

    if not weight_file.exists():
        print("Downloading Hands23 model weights (~400MB)...")
        url = "https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/model_hands23.pth"
        # Use wget/curl as the server's SSL cert can fail Python's strict verification
        if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
            subprocess.run(["wget", "--no-check-certificate", "-O", str(weight_file), url], check=True)
        elif subprocess.run(["which", "curl"], capture_output=True).returncode == 0:
            subprocess.run(["curl", "-kL", "-o", str(weight_file), url], check=True)
        else:
            import ssl
            import urllib.request
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(url, context=ctx) as resp, open(str(weight_file), "wb") as f:
                f.write(resp.read())
        print(f"Weights saved to {weight_file} ({weight_file.stat().st_size / 1024 / 1024:.0f} MB)")
    else:
        print(f"Weights already exist at {weight_file}")

    # Step 3: Verify Detectron2 is installed
    try:
        import detectron2
        print(f"Detectron2 found: {detectron2.__version__}")
    except ImportError:
        print("Installing Detectron2...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "git+https://github.com/facebookresearch/detectron2.git",
             "--no-build-isolation"],
            check=True,
        )

    # Step 4: Patch hardcoded .cuda() calls for CPU compatibility
    print("Patching hardcoded .cuda() calls for CPU compatibility...")
    _patch_hands23_cuda(repo_dir)

    print("Hands23 setup complete.")
    return str(weight_file)


def _patch_hands23_cuda(repo_dir: Path):
    """Replace hardcoded .cuda() calls with device-aware .to(device) in Hands23 source."""
    import re

    # roi_heads.py — add _get_device helper and replace .cuda()
    roi_heads = repo_dir / "hodetector" / "modeling" / "roi_heads" / "roi_heads.py"
    if roi_heads.exists():
        content = roi_heads.read_text()
        if ".cuda()" in content:
            # Add device helper if not already present
            if "_get_device" not in content:
                content = content.replace(
                    "import cv2",
                    'import cv2\n\ndef _get_device():\n    """Get device (cuda if available, else cpu)."""\n    return torch.device("cuda" if torch.cuda.is_available() else "cpu")',
                )
            content = content.replace(".cuda()", ".to(_get_device())")
            content = content.replace("device='cuda'", "device=_get_device()")
            content = content.replace('device="cuda"', "device=_get_device()")
            roi_heads.write_text(content)
            print(f"  Patched roi_heads.py")

    # positional_encoding.py
    pos_enc = repo_dir / "hodetector" / "utils" / "positional_encoding.py"
    if pos_enc.exists():
        content = pos_enc.read_text()
        if ".cuda()" in content:
            content = content.replace(
                ".cuda()",
                '.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))',
            )
            pos_enc.write_text(content)
            print(f"  Patched positional_encoding.py")


def download_100doh():
    """Download 100DOH hand-object detector (legacy): clone repo, download weights, patch and compile C++ extensions."""
    print("\n" + "=" * 60)
    print("5. 100DOH Hand-Object Detector (legacy)")
    print("=" * 60)

    repo_dir = MODELS_DIR / "hand_object_detector"

    # Step 1: Clone repo
    if not repo_dir.exists():
        print("Cloning hand_object_detector repo...")
        subprocess.run(
            ["git", "clone", "https://github.com/ddshan/hand_object_detector.git", str(repo_dir)],
            check=True,
        )
    else:
        print(f"Repo already exists at {repo_dir}")

    # Step 2: Download weights (100K+ego model)
    weights_dir = repo_dir / "models" / "res101_handobj_100K" / "pascal_voc"
    weights_dir.mkdir(parents=True, exist_ok=True)
    weight_file = weights_dir / "faster_rcnn_1_8_132028.pth"

    if not weight_file.exists():
        print("Downloading 100DOH model weights (100K+ego, ~360MB)...")
        try:
            import gdown
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
            import gdown

        url = "https://drive.google.com/uc?id=1H2tWsZkS7tDF8q1-jdjx6V9XrK25EDbE"
        gdown.download(url, str(weight_file), quiet=False)
        print(f"Weights saved to {weight_file}")
    else:
        print(f"Weights already exist at {weight_file}")

    # Step 3: Install 100DOH Python dependencies
    print("Installing 100DOH Python dependencies...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "easydict", "pyyaml", "scipy", "cython"],
        capture_output=True,
    )

    # Step 4: Patch C++ sources for modern PyTorch (2.x)
    print("Patching C++ extensions for PyTorch 2.x compatibility...")
    _patch_100doh_cpp(repo_dir)

    # Step 5: Compile C++ extensions
    print("Compiling C++ extensions (this may take a minute)...")
    lib_dir = repo_dir / "lib"

    # Clean previous builds to avoid stale objects
    build_dir = lib_dir / "build"
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)

    result = subprocess.run(
        [sys.executable, "setup.py", "build", "develop"],
        cwd=str(lib_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Compilation output:\n{result.stdout}")
        print(f"Compilation errors:\n{result.stderr}")
        raise RuntimeError(
            "Failed to compile 100DOH C++ extensions.\n"
            "Make sure you have a C++ compiler installed:\n"
            "  macOS: xcode-select --install\n"
            "  Ubuntu: sudo apt install build-essential\n"
        )

    # Verify the .so was created
    so_files = list((lib_dir / "model").glob("_C*.so")) + list((lib_dir / "model").glob("_C*.pyd"))
    if not so_files:
        raise RuntimeError("C++ extension compiled but .so file not found in model/")
    print(f"  Extension built: {so_files[0].name}")

    print("100DOH setup complete.")
    return str(weight_file)


def _patch_100doh_cpp(repo_dir: Path):
    """Patch deprecated PyTorch C++ APIs in 100DOH source files.

    Idempotent — safe to run multiple times. Only patches if deprecated APIs are still present.

    Changes:
    - .type().is_cuda() -> .is_cuda()
    - .type() -> .scalar_type()  (in AT_DISPATCH macros)
    - .data<T>() -> .data_ptr<T>()
    """
    import re
    csrc_dir = repo_dir / "lib" / "model" / "csrc"

    def patch_file(rel_path: str, replacements: list[tuple[str, str]]):
        """Apply replacements to a file. Only writes if changes were made."""
        path = csrc_dir / rel_path
        if not path.exists():
            return
        content = path.read_text()
        original = content
        for old, new in replacements:
            content = content.replace(old, new)
        # .data<T>() -> .data_ptr<T>() (regex, but skip if already .data_ptr)
        content = re.sub(r'\.data<([^>]+)>\(\)', r'.data_ptr<\1>()', content)
        if content != original:
            path.write_text(content)
            print(f"  Patched {rel_path}")
        else:
            print(f"  {rel_path} already patched")

    # Header files
    for header in ["ROIAlign.h", "ROIPool.h", "nms.h"]:
        patch_file(header, [(".type().is_cuda()", ".is_cuda()")])

    # CPU source files
    patch_file("cpu/ROIAlign_cpu.cpp", [
        (".type().is_cuda()", ".is_cuda()"),
        ("input.type(),", "input.scalar_type(),"),
    ])
    patch_file("cpu/nms_cpu.cpp", [
        (".type().is_cuda()", ".is_cuda()"),
        ("dets.type(),", "dets.scalar_type(),"),
        ("dets.type() == scores.type()", "dets.scalar_type() == scores.scalar_type()"),
    ])


if __name__ == "__main__":
    print("Downloading all model weights...")
    print(f"Models directory: {MODELS_DIR}\n")

    download_scrfd()
    download_yolo11m()
    download_grounding_dino()
    download_hands23()

    # Legacy 100DOH (only if explicitly requested)
    if "--100doh" in sys.argv or "--all" in sys.argv:
        download_100doh()

    print("\nDone!")
