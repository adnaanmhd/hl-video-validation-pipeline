# ML Checks: Research, Model Selection & Implementation Context

**Date:** 2026-03-30
**Project:** Bachman — Egocentric Video Validation & Submission Tool
**Author:** Adnaan (PM, Humyn Labs) + Claude Code

---

## 1. Problem & Purpose

Humyn Labs is building an egocentric video dataset for **training autonomous humanoid robots**. Videos are captured by a distributed workforce wearing head-mounted cameras while performing agricultural, commercial, and residential tasks. The ML checks pipeline validates video quality before data enters the training pipeline.

**Scale:** ~5,000 videos/day, each ~500MB, 5+ minutes, 1080p, 30FPS.
**Target:** <5 min total processing per video.

---

## 2. The 7 ML Checks

| # | Check | Criterion | Model |
|---|---|---|---|
| 1 | Face Presence | Face detection confidence < 0.8 in ALL frames | SCRFD-2.5GF |
| 2 | Participants | 0 other persons (face or body parts) in ≥ 95% frames | YOLO11m + SCRFD |
| 3 | Hand Visibility | Both hands detected (≥ 0.7 conf) in ≥ 90% frames | 100DOH |
| 4 | Hand-Object Interaction | Interaction detected in ≥ 70% frames | 100DOH (contact state) |
| 5 | Privacy Safety | Sensitive objects = 0 in ALL frames | YOLO11m pre-filter + Grounding DINO zero-shot |
| 6 | View Obstruction | ≤ 10% frames obstructed | OpenCV heuristic (no ML) |
| 7 | POV-Hand Angle | Angle from center to hands < 40° in ≥ 80% frames | Geometric computation on 100DOH output |

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
- **Egocentric note:** Camera wearer is invisible but their arms/hands appear. Pipeline filters out wearer's own body parts by checking overlap with 100DOH hand detections and bottom-center frame position.
- **Package:** `ultralytics`
- **Speed:** ~88ms/frame CPU

#### 100DOH hand_object_detector (Hand Detection — Checks 3, 4, 7)
- **Why over MediaPipe:** MediaPipe Hand Landmarker has documented AP50 of 29-97% on egocentric data (wildly inconsistent). The 100DOH "100K+ego" model was **trained specifically on egocentric hand images** from 131 days of real-world footage. 90.46% Box AP.
- **Why not HaMeR:** ViT-H backbone (~630M params, ~200-500ms/frame GPU) is overkill for "are hands visible?" detection. HaMeR reconstructs full 3D mesh — we only need bounding boxes + confidence + contact state.
- **Key advantage:** Returns hand contact state (N/S/O/P/F) directly — eliminates need for crude bounding-box overlap heuristic for hand-object interaction detection.
- **Contact states:** N=no contact, S=self, O=other person, P=portable object, F=stationary object. States P and F = interaction.
- **Architecture:** Faster R-CNN with ResNet-101 backbone, custom heads for contact state and hand side classification.
- **Package:** Custom repo (`github.com/ddshan/hand_object_detector`) + `detectron2` (not actually used, compiled C++ extensions from the repo directly)
- **Speed:** ~5,141ms/frame CPU, **expected ~100-150ms/frame on GPU**

#### Grounding DINO (Privacy — Check 5, second stage)
- **Why:** Zero-shot detection of arbitrary text-described objects ("credit card", "ID card", "paper document") without any fine-tuning or training data. 52.5 AP zero-shot on COCO — SOTA.
- **Why not YOLO-World:** Lower zero-shot accuracy (35.4 vs 52.5 AP). For a zero-tolerance privacy check, accuracy > speed.
- **Two-stage approach:** YOLO11m pre-filters frames that contain COCO sensitive classes (tv, laptop, cell_phone). Grounding DINO runs only on those flagged frames (~0-10% of total), keeping it fast.
- **Package:** HuggingFace `transformers` (`IDEA-Research/grounding-dino-base`)
- **Speed:** ~2,757ms/frame CPU

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

**GPU is ~4.8x more cost-effective at scale.** The 100DOH model is the bottleneck — it's ~5s/frame on CPU vs ~100-150ms expected on A10G GPU (~35-50x speedup).

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
  → Frame extraction (cv2.VideoCapture, 1 FPS)
  → Per frame:
      1. SCRFD face detection (~10ms)
      2. YOLO11m object detection (~88ms)
      3. 100DOH hand-object detection (~5s CPU / ~100ms GPU)
      4. View obstruction heuristic (<1ms)
  → If YOLO flagged sensitive objects:
      5. Grounding DINO on flagged frames only (~2.8s/frame CPU)
  → Distribute results to 7 check functions
  → Aggregate results
```

### Grounding DINO Two-Stage Approach

Privacy check uses two stages to balance accuracy and speed:
1. YOLO11m (already running for Check 2) flags frames with COCO classes: tv (62), laptop (63), cell_phone (67)
2. Grounding DINO runs only on flagged frames with text prompt: `"laptop screen . computer monitor . smartphone screen . paper document . credit card . ID card . identification card . bank card"`

This reduces Grounding DINO from 300 frames (~14 min on CPU) to ~0-30 frames (~0-84s).

### Check 2 Participants — Wearer Filtering

The camera wearer's arms/hands appear in frame but should NOT be counted as "another person." Filtering logic:
1. Exclude YOLO person detections that overlap significantly with 100DOH hand detections (wearer's hands = wearer's arms)
2. Exclude detections anchored at bottom-center of frame (wearer's torso/arms)
3. Require minimum person bbox height > 15% of frame height (filter tiny partial detections)
4. Combine with SCRFD face detections — even a disembodied face with no body detection counts as "another person present"

---

## 9. Project Structure

```
ml_checks/
├── __init__.py
├── pipeline.py                     # Unified inference pipeline
├── requirements.txt
├── checks/
│   ├── __init__.py
│   ├── check_results.py            # CheckResult dataclass
│   ├── face_presence.py            # Check 1
│   ├── participants.py             # Check 2
│   ├── hand_visibility.py          # Check 3
│   ├── hand_object_interaction.py  # Check 4
│   ├── privacy_safety.py           # Check 5
│   ├── view_obstruction.py         # Check 6
│   └── pov_hand_angle.py           # Check 7
├── models/
│   ├── __init__.py
│   ├── download_models.py
│   ├── scrfd_detector.py           # SCRFD wrapper
│   ├── yolo_detector.py            # YOLO11m wrapper
│   ├── hand_detector.py            # 100DOH wrapper
│   ├── grounding_dino_detector.py  # Grounding DINO wrapper
│   └── weights/
│       ├── insightface/            # SCRFD model (~14MB)
│       ├── grounding_dino/         # Grounding DINO (~700MB)
│       └── hand_object_detector/   # 100DOH repo + weights (~360MB)
├── utils/
│   ├── __init__.py
│   └── frame_extractor.py
├── tests/
│   ├── __init__.py
│   ├── generate_test_video.py
│   ├── benchmark_models.py
│   └── benchmark_results.json
└── sample_data/
    └── test_30s.mp4                # Synthetic test video
```

**YOLO11m weights** (`yolo11m.pt`) are downloaded by ultralytics to its default cache (~40MB).

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

**Special:** 100DOH requires `faster-rcnn` package built from source (`hand_object_detector/lib/setup.py build develop`). Detectron2 installed but not directly used by 100DOH (it has its own Faster R-CNN implementation).

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
| 100DOH over MediaPipe for hands | MediaPipe has 29-97% AP on egocentric data; 100DOH has 90.4% trained on egocentric |
| SCRFD over MediaPipe for faces | 93.8% AP on WIDER hard set; MediaPipe optimized for webcam |
| Grounding DINO over fine-tuned YOLO for privacy | Zero-shot capability for arbitrary sensitive objects without training data |
| Two-stage privacy (YOLO pre-filter + GDINO) | Reduces GDINO from 300 to ~15 frames, cutting cost by 95% |
| 100DOH over Ego4D HOI models | Ego4D models solve temporal localization, not frame-level interaction detection |
| 100DOH over EgoHOS | EgoHOS provides pixel segmentation — overkill for binary interaction signal |
| Heuristic over ML for view obstruction | No standard ML model exists; signal is fundamentally low-level |
| YOLO11m over YOLO11n | 51.5 vs 39.5 mAP; person detection is safety-critical for privacy |
| 1 FPS sampling | 300 frames gives sufficient temporal coverage; configurable for GPU |

---

*End of context document*
