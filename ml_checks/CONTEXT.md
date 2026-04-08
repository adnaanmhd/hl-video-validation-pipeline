# Video Validation Pipeline: Research, Model Selection & Implementation Context

**Date:** 2026-03-30 (updated 2026-03-31)
**Project:** Bachman — Egocentric Video Validation & Submission Tool
**Author:** Adnaan (PM, Humyn Labs) + Claude Code

---

## 1. Problem & Purpose

Humyn Labs is building an egocentric video dataset for **training autonomous humanoid robots**. Videos are captured by a distributed workforce wearing head-mounted cameras while performing agricultural, commercial, and residential tasks. The validation pipeline checks video quality across 20 criteria before data enters the training pipeline.

**Scale:** ~5,000 videos/day, each ~500MB, 5+ minutes, 1080p, 30FPS.
**Target:** <5 min total processing per video.

The pipeline validates 5 categories of checks across 20 total criteria:
1. **Video Metadata** (6 checks) — format, encoding, resolution, frame rate, duration, orientation. Acts as a gate: failure skips all other checks.
2. **Frame-Level Quality** (3 checks) — average brightness, brightness stability, near-black frames.
3. **Luminance & Blur** (1 check) — per-frame Tenengrad/luminance classification with segment-level aggregation.
4. **Motion Analysis** (2 checks) — camera stability via Farneback optical flow, frozen segments via native-FPS SSIM.
5. **ML Detection** (8 checks) — face presence, participants, hand visibility, hand-object interaction, privacy safety, view obstruction, POV-hand angle, body part visibility.

See [checks.md](../checks.md) for full acceptance conditions and thresholds.

---

## 2. The 8 ML Detection Checks

| # | Check | Criterion | Model |
|---|---|---|---|
| 1 | Face Presence | Face detection confidence < 0.8 in ALL frames | SCRFD-2.5GF |
| 2 | Participants | 0 other persons (face or body parts) in ≥ 90% frames | YOLO11m + SCRFD |
| 3 | Hand Visibility | Both hands fully in frame (bbox > 2 px from every edge, ≥ 0.7 conf) in ≥ 90% frames | Hands23 |
| 4 | Hand-Object Interaction | Interaction detected in ≥ 70% frames | Hands23 (contact state) |
| 5 | Privacy Safety | Sensitive objects = 0 in ALL frames | Grounding DINO zero-shot on all frames |
| 6 | View Obstruction | ≤ 10% frames obstructed | OpenCV heuristic (no ML) |
| 7 | POV-Hand Angle | Angle from center to hands < 40° in ≥ 80% frames | Geometric computation on Hands23 output |
| 8 | Body Part Visibility | Only hands/forearms (up to elbows) visible in ≥ 90% frames | YOLO11m-pose keypoint detection |

---

## 3. Model Selection Rationale

### Constraints (updated from initial brief)
- **Prefer open-source** but do NOT compromise on quality, accuracy, scale, or latency
- **CPU preferred, GPU acceptable** if quality/accuracy/scale/latency demand it
- Must work locally for testing AND on AWS (ECS with GPU, SageMaker)
- No paid per-call APIs (no Google Vision, no Video Intelligence API)
- Models must be testable locally and compatible with AWS managed infrastructure

### Why These Models Were Chosen

#### SCRFD-2.5GF (Face Detection — Check 1)
- **Why over MediaPipe:** 93.8% AP on WIDER Face hard set. MediaPipe is optimized for front-facing webcam use, not egocentric angles where bystander faces appear at unusual angles (profile, top-down, partial occlusion).
- **Why not YOLO:** COCO has no "face" class. SCRFD is purpose-built, tiny (~3MB ONNX), and faster than any general detector for faces.
- **Package:** `insightface` + `onnxruntime`
- **Speed:** ~9.6ms/frame CPU

#### YOLO11m (Person/Object Detection — Checks 2, 5)
- **Why YOLO11m over nano:** 51.5 vs 39.5 mAP on COCO. Person detection is safety-critical (false negatives = missed people in privacy-sensitive data). 22% fewer params than YOLOv8m with higher accuracy.
- **Why not RT-DETR:** Heavier for comparable accuracy, less mature deployment ecosystem.
- **Egocentric note:** Camera wearer is invisible but their arms/hands appear. Pipeline filters out wearer's own body parts by checking overlap with Hands23 hand detections and bottom-center frame position.
- **Package:** `ultralytics`
- **Speed:** ~88ms/frame CPU

#### Hands23 (Hand Detection — Checks 3, 4, 7) [Default]
- **Why Hands23 over 100DOH:** NeurIPS 2023 successor to 100DOH. Trained on 250K images including EPIC-KITCHENS and VISOR egocentric datasets. No custom C++ compilation needed (pure Detectron2). ~3.5x faster than 100DOH on CPU.
- **Why over MediaPipe:** MediaPipe Hand Landmarker has documented AP50 of 29-97% on egocentric data (wildly inconsistent). Hands23 was **trained specifically on egocentric hand images**.
- **Why not HaMeR:** ViT-H backbone (~630M params, ~200-500ms/frame GPU) is overkill for "are hands visible?" detection. HaMeR reconstructs full 3D mesh — we only need bounding boxes + confidence + contact state.
- **Key advantage:** Returns hand contact state (N/S/O/P/F) directly — eliminates need for crude bounding-box overlap heuristic for hand-object interaction detection. Hand bounding boxes are also used for in-frame visibility checks (bbox edge clearance ≥ 2 px).
- **Contact states:** N=no contact, S=self, O=other person, P=portable object, F=stationary object. States P and F = interaction.
- **Architecture:** Faster R-CNN with X-101-FPN backbone, custom heads for contact state, hand side, and grasp type classification.
- **Package:** Custom repo (`github.com/EvaCheng-cty/hands23_detector`) + `detectron2`
- **Speed:** ~1,400ms/frame CPU, **expected ~100-200ms/frame on GPU**

#### 100DOH hand_object_detector (Hand Detection — Legacy)
- **Predecessor to Hands23.** Trained on 131 days of real-world egocentric footage. 90.46% Box AP.
- Requires custom C++ extension compilation (fragile across Python/PyTorch versions).
- **Speed:** ~5,141ms/frame CPU (~3.5x slower than Hands23).
- Available via `./validate.sh --100doh` or `python ml_checks/models/download_models.py --100doh`.

#### Grounding DINO (Privacy — Check 5)
- **Why:** Zero-shot detection of arbitrary text-described objects ("credit card", "ID card", "paper document") without any fine-tuning or training data. 52.5 AP zero-shot on COCO — SOTA.
- **Why not YOLO-World:** Lower zero-shot accuracy (35.4 vs 52.5 AP). For a zero-tolerance privacy check, accuracy > speed.
- **Runs on all inferred frames** (not pre-filtered by YOLO). This ensures every sensitive-object detection is captured with exact timestamps for reporting.
- **Package:** HuggingFace `transformers` (`IDEA-Research/grounding-dino-base`)
- **Speed:** ~2,757ms/frame CPU

#### YOLO11m-pose (Body Part Visibility — Check 8)
- **Why:** Detects 17 COCO body keypoints per person. Used to verify that only the wearer's hands and forearms (up to elbows) are visible — no shoulders, torso, hips, legs, or feet.
- **Why YOLO11m-pose over MediaPipe Pose:** Same rationale as YOLO11m over MediaPipe for other tasks — more robust on egocentric angles.
- **Wearer identification:** Same heuristic as participants check (bottom-center anchored or overlapping with Hands23 hand detections).
- **Allowed keypoints:** Left/right elbows (7, 8) and wrists (9, 10). All others (nose, eyes, ears, shoulders, hips, knees, ankles) are flagged.
- **Package:** `ultralytics` (same as YOLO11m)
- **Speed:** ~114ms/frame CPU

#### View Obstruction Heuristic (Check 6)
- **No ML model.** Combines 4 signals on central 80% of frame:
  1. Low spatial variance (std_dev < 15 → homogeneous/covered)
  2. Low edge density (Laplacian variance < 20)
  3. Color channel uniformity (dominant bin > 80% of pixels)
  4. Brightness anomaly (mean < 15)
- Rule: Frame obstructed if ≥ 2 of 4 signals trigger.
- **Speed:** <1ms/frame

---

## 4. Models That Were Considered and Rejected

| Model | Considered For | Reason Rejected |
|---|---|---|
| MediaPipe Face Detector | Check 1 (Face) | Lower accuracy on WIDER Face hard set vs SCRFD. Optimized for webcam, not egocentric angles. |
| MediaPipe Hand Landmarker | Checks 3, 4, 7 | AP50 of 29-97% on egocentric data — wildly inconsistent. Not reliable for 90% threshold on egocentric video. |
| YOLOv8n / YOLO11n | Checks 2, 5 | 39.5 mAP vs 51.5 for medium. Person detection is safety-critical — accuracy matters more than nano's speed advantage. |
| HaMeR (Hand Mesh Recovery) | Check 3 | ViT-H backbone ~630M params, ~200-500ms/frame GPU. Overkill — reconstructs full 3D mesh when we only need bbox + confidence. |
| EgoHOS | Check 4 | Swin-L backbone ~197M params, ~150-300ms/frame GPU. Pixel-level segmentation overkill for binary "is there interaction?" signal. |
| Ego4D HOI models (SlowFast) | Check 4 | Designed for temporal localization ("when does state change?"), not frame-level binary interaction detection. Can't be directly used. |
| YOLO-World | Check 5 | Lower zero-shot accuracy (35.4 vs 52.5 AP). For zero-tolerance privacy, accuracy > speed. |
| RT-DETR | Check 2 | Heavier than YOLO11 for comparable accuracy, less mature deployment. |
| InsightFace SCRFD (larger variants) | Check 1 | SCRFD-2.5GF sufficient. Larger variants (10GF, 34GF) offer marginal accuracy gain at 4-10x cost. |
| FrankMocap | Check 3 | Research code, not production-ready. Slower than 100DOH for similar accuracy. |
| OWL-ViT / OWLv2 | Check 5 | Lower zero-shot accuracy than Grounding DINO. |
| Florence-2 | Check 5 | General vision foundation model. Slower and less precise than Grounding DINO for detection tasks. |
| SAM 2 | None | Segmentation model — not needed for any of our detection/classification checks. |
| Custom YOLO face model | Check 1 | Requires training. SCRFD already SOTA for faces at this compute budget. |
| Autoencoder anomaly detection | Check 6 | Requires training on data distribution. Overkill for physical lens blockage detection. |

---

## 5. Ego4D & Egocentric Context

This dataset is directly analogous to **Ego4D** (Meta's 3,670-hour egocentric video benchmark). Key references:

- **Ego4D Hands & Objects benchmark** — defines tasks for hand-object interaction in egocentric video. However, the challenge winners use temporal models (SlowFast, CLIP-based) for "when does state change?" — not frame-level "is the hand touching something right now?" which is what we need.
- **100 Days of Hands (100DOH)** — the 100K+ego dataset (131 days of real-world footage) is the most directly relevant training data for our hand detection needs.
- **EgoHOS (ECCV 2022)** — egocentric hand-object segmentation. Good research but too heavy for our throughput requirements.
- **Epic-Kitchens** — egocentric kitchen activity dataset. Useful for sourcing test videos.

**Key insight:** Egocentric-specific models (100DOH, EgoHOS) significantly outperform general-purpose models (MediaPipe, YOLO pose) on egocentric hand detection because hands are viewed from above/behind, often holding tools, with various skin tones and gloves.

---

## 6. GPU vs CPU Economics

| | GPU (g5.xlarge, A10G) | CPU (c6i.2xlarge) |
|---|---|---|
| Per-video time | ~50s | ~27 min |
| Instances for 5K/day | 3 | 10-17 |
| Daily cost (spot) | ~$29 | ~$139 |
| Monthly cost | ~$870 | ~$4,170 |

**GPU is ~4.8x more cost-effective at scale.** Hands23 is the bottleneck — it's ~1.4s/frame on CPU vs ~100-200ms expected on A10G GPU. (Legacy 100DOH was ~5s/frame CPU.)

**Recommended AWS architecture:** ECS on g5.xlarge (A10G) with Celery workers. Single container with all 4 models in GPU memory (~1.1GB model weights, ~3-4GB total GPU memory). A10G has 24GB — plenty of headroom. Spot instances at ~$0.30-0.50/hr.

---

## 7. Benchmark Results (CPU, macOS M-series ARM64)

Tested on 2026-03-30 with synthetic 30s 1080p test video.

| Model | p50 ms/frame | p95 ms/frame | mean ms/frame | 300 frames est. |
|---|---|---|---|---|
| SCRFD-2.5GF | 9.1 | 11.3 | 9.6 | 2.9s |
| YOLO11m | 86.8 | 101.2 | 88.2 | 26.5s |
| 100DOH (ResNet-101) | 5,095 | 5,567 | 5,141 | 25.7 min |
| Grounding DINO (base) | 2,753 | 2,787 | 2,757 | ~41s (15 flagged) |

**Total estimated per 5-min video:**
- CPU: ~27 min (dominated by 100DOH)
- GPU (projected): ~40-65s

---

## 8. Technical Implementation Details

### 100DOH Compilation Fix

The 100DOH hand_object_detector uses custom C++ extensions (ROIAlign, NMS) built against an older PyTorch C++ API. To compile on Python 3.13 + PyTorch 2.11:

**Files patched:**
- `lib/model/csrc/cpu/ROIAlign_cpu.cpp`
- `lib/model/csrc/cpu/nms_cpu.cpp`
- `lib/model/csrc/ROIAlign.h`
- `lib/model/csrc/ROIPool.h`
- `lib/model/csrc/nms.h`

**Changes:**
1. `.type().is_cuda()` → `.is_cuda()`
2. `.type()` → `.scalar_type()` (in `AT_DISPATCH_FLOATING_TYPES`)
3. `.data<T>()` → `.data_ptr<T>()`

After patching, compiled with `python setup.py build develop --no-build-isolation`. Linker warnings about x86_64 architecture (universal binary attempted but torch is arm64-only) are harmless — arm64 version builds and runs correctly.

**Runtime:** Requires `DYLD_LIBRARY_PATH` set to torch lib directory for the C extension to find libtorch at runtime.

### Frame Sampling Strategy

- **1 FPS** (300 frames from 5-min video) — baseline
- Uniform temporal sampling via `cv2.VideoCapture` seeking at 1-second intervals
- Configurable via `PipelineConfig.sampling_fps`
- Sequential processing (bounded memory: ~5MB for current frame + model state)

### Pipeline Architecture

```
Video file
  → Metadata checks (ffprobe, no frames needed)
  → IF metadata fails: return metadata results + SKIPPED for all others
  → Frame extraction (cv2.VideoCapture, 1 FPS)
  → Frame-level quality checks (brightness, stability, near-black)
  → Luminance & blur check (Tenengrad + luminance decision table)
  → Motion analysis:
      - Camera stability (Farneback optical flow on sampled pairs)
      - Frozen segments (SSIM at native FPS, streaming)
  → Per frame (ML):
      1. SCRFD face detection (~10ms)
      2. YOLO11m object detection (~88ms)
      3. YOLO11m-pose keypoint detection (~114ms)
      4. Hands23 hand-object detection (~1.4s CPU / ~100-200ms GPU)
      5. View obstruction heuristic (<1ms)
  → Grounding DINO on all inferred frames for privacy (~2.8s/frame CPU)
  → Distribute results to 8 ML check functions
  → Aggregate all 20 check results
```

### Grounding DINO Privacy Detection

Privacy check runs Grounding DINO on all inferred frames with text prompt: `"laptop screen . computer monitor . smartphone screen . paper document . credit card . ID card . identification card . bank card"`. This ensures complete coverage — every sensitive-object detection is captured with HH:MM:SS timestamps for reporting. Privacy safety is excluded from early stopping so all frames are scanned.

### Check 2 Participants — Wearer Filtering

The camera wearer's arms/hands appear in frame but should NOT be counted as "another person." Filtering logic:
1. Exclude YOLO person detections that overlap significantly with Hands23 hand detections (wearer's hands = wearer's arms)
2. Exclude detections anchored at bottom-center of frame (wearer's torso/arms)
3. Require minimum person bbox height > 15% of frame height (filter tiny partial detections)
4. Combine with SCRFD face detections — even a disembodied face with no body detection counts as "another person present"

---

## 9. Project Structure

```
hl-bachman/
├── validate.sh                         # One-command entry point
├── pyproject.toml                      # Package configuration (hl-video-validation)
├── checks.md                           # Check specifications and thresholds
├── README.md
└── ml_checks/
    ├── __init__.py
    ├── pipeline.py                     # ValidationPipeline orchestrator
    ├── run_batch.py                    # CLI entry point (hl-validate)
    ├── CONTEXT.md                      # This file
    ├── checks/
    │   ├── check_results.py            # CheckResult dataclass
    │   ├── video_metadata.py           # 6 metadata checks
    │   ├── frame_quality.py            # 3 brightness checks
    │   ├── luminance_blur.py           # Tenengrad + luminance decision table
    │   ├── motion_analysis.py          # Optical flow + frozen segment detection
    │   ├── face_presence.py            # Face detection check
    │   ├── participants.py             # Person count check
    │   ├── hand_visibility.py          # Hand detection check
    │   ├── hand_object_interaction.py  # Contact state check
    │   ├── privacy_safety.py           # Sensitive object detection
    │   ├── view_obstruction.py         # Lens obstruction heuristic
    │   ├── pov_hand_angle.py           # Hand angle check
    │   └── body_part_visibility.py    # Body part keypoint check
    ├── models/
    │   ├── download_models.py          # Model weight downloader
    │   ├── scrfd_detector.py           # SCRFD face detector
    │   ├── yolo_detector.py            # YOLO11m object detector
    │   ├── yolo_pose_detector.py       # YOLO11m-pose keypoint detector
    │   ├── hand_detector.py            # Hands23 hand-object detector
    │   ├── grounding_dino_detector.py  # Zero-shot privacy detector
    │   └── weights/                    # ~1.5GB, downloaded on first run
    └── utils/
        ├── frame_extractor.py          # Video frame sampling
        └── video_metadata.py           # FFprobe metadata extraction
```

**YOLO weights** (`yolo11m.pt`, `yolo11m-pose.pt`) are downloaded by ultralytics on first run (~40MB each).

---

## 10. Python Dependencies

```
torch>=2.0
torchvision>=0.15
onnxruntime>=1.17.0
insightface>=0.7.3
ultralytics>=8.1.0
transformers>=4.36.0
supervision>=0.18.0
opencv-python>=4.9.0
numpy>=1.24.0
Pillow>=10.0
tqdm>=4.65
```

**Special:** Hands23 requires `detectron2` (installed from GitHub with `--no-build-isolation`). Legacy 100DOH additionally requires C++ extensions built from source. System dependency: `ffmpeg`/`ffprobe` for video metadata extraction. All handled automatically by `validate.sh`.

---

## 11. Next Steps

1. **Test with real egocentric videos** — synthetic data validates the pipeline but real data is needed to:
   - Tune confidence thresholds for each check
   - Measure false positive/negative rates
   - Validate 100DOH hand detection accuracy on agricultural/commercial/residential tasks
   - Calibrate Grounding DINO text prompts for sensitive objects

2. **GPU testing** — 100DOH at 5s/frame on CPU is the bottleneck. Need to test on A10G GPU to confirm ~100-150ms/frame.

3. **Threshold tuning** — all thresholds are set to idea-brief spec values. May need adjustment based on real data:
   - Face confidence threshold (0.8)
   - Person confidence threshold (0.4)
   - Hand confidence threshold (0.7)
   - Privacy YOLO/GDINO thresholds (0.6/0.3)
   - Obstruction heuristic thresholds

4. **Phase 2 Privacy:** If Grounding DINO has too many false positives on real data, fine-tune YOLO11m on labeled sensitive objects.

5. **Containerization:** Create Docker image with all models + CUDA for AWS ECS deployment.

6. **Integration:** Wire pipeline into Celery workers for the FastAPI backend described in the idea brief.

---

## 12. Key Decisions & Rationale Log

| Decision | Rationale |
|---|---|
| GPU over CPU at scale | 4.8x more cost-effective ($870 vs $4,170/month) |
| Hands23 over 100DOH | NeurIPS 2023 successor, 250K training images, no C++ compilation, ~3.5x faster on CPU |
| Hands23 over MediaPipe for hands | MediaPipe has 29-97% AP on egocentric data; Hands23 trained on egocentric datasets |
| SCRFD over MediaPipe for faces | 93.8% AP on WIDER hard set; MediaPipe optimized for webcam |
| Grounding DINO over fine-tuned YOLO for privacy | Zero-shot capability for arbitrary sensitive objects without training data |
| GDINO on all frames for privacy | Ensures complete timestamp reporting; no frames missed by pre-filter |
| Hands23 input downscaling (720p default) | Reduces inference time on the most expensive model with negligible accuracy impact |
| Heuristic over ML for view obstruction | No standard ML model exists; signal is fundamentally low-level |
| YOLO11m over YOLO11n | 51.5 vs 39.5 mAP; person detection is safety-critical for privacy |
| Metadata gate (short-circuit) | Avoids expensive ML inference on videos that fail basic format/duration requirements |
| Farneback over Lucas-Kanade for stability | Dense optical flow gives more accurate camera motion estimate |
| Native FPS for frozen segments | 30 consecutive frames = 1 second; sampled frames would miss short freezes |
| 1 FPS sampling for ML checks | 120 frames gives sufficient temporal coverage; configurable for GPU |
| YOLO11m-pose for body part visibility | Same ultralytics ecosystem as YOLO11m, no new dependencies. 17-keypoint detection identifies which body parts are visible. Allowed: wrists + elbows only (hands through forearms). |

---

*End of context document*
