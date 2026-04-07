# Video Validation Checks

## Video Metadata

| Check       | Acceptance Condition                           | How It Is Estimated              |
| ----------- | ---------------------------------------------- | -------------------------------- |
| Format      | MP4 container (MPEG-4)                         | Container metadata via FFprobe   |
| Encoding    | H.264 video codec                              | Codec metadata via FFprobe       |
| Resolution  | >= 1920 x 1080 pixels                          | Width & height metadata          |
| Frame Rate  | >= 28 FPS                                      | FPS metadata                     |
| Duration    | >= 180 seconds                                 | Duration metadata                |
| Orientation | Rotation = 0 or 180 degrees and width > height | Width, height, rotation metadata |

## Motion Analysis

| Check            | Acceptance Condition                        | How It Is Estimated                            |
| ---------------- | ------------------------------------------- | ---------------------------------------------- |
| Camera Stability | Two-stage LK shakiness score <= 0.30        | Sparse Lucas-Kanade optical flow (two-stage)   |
| Frozen Segments  | No > 30 consecutive frames with SSIM > 0.99 | Native FPS frame similarity                    |

### Camera Stability — Two-Stage Pipeline

**Stage 1 (fast pre-screen):** For each second of video, compute sparse
Lucas-Kanade optical flow on downscaled frames (~360p via `fast_scale=0.333`)
at every Nth frame (`frame_skip=2`). An affine transform is estimated via
RANSAC from tracked corner features, yielding per-frame translation (px) and
rotation (degrees). These are combined into a per-second shakiness score
using weighted components:

| Component              | Weight | Normalisation                       |
| ---------------------- | ------ | ----------------------------------- |
| Mean translation       | 0.35   | avg_t / (trans_threshold * 3)       |
| Translation variance   | 0.25   | std_t / (variance_threshold * 2)    |
| Mean rotation          | 0.20   | avg_r / (rot_threshold * 3)         |
| Sudden jumps (> 30 px) | 0.20   | (jump_count / n) * 10              |

All component scores are clamped to [0, 1]; the weighted sum produces a
per-second score in [0, 1].

**Stage 2 (deep analysis):** Seconds whose Stage-1 score exceeds
`deep_score_threshold` (default 0.25) are re-analysed at full resolution
with every frame (no skip). The deep score replaces the fast score for
those seconds.

**Final verdict:** The overall score is the mean of all per-second scores
(deep where available, fast otherwise). The video passes if overall score
<= `shaky_score_threshold` (default 0.30).

Translation and rotation thresholds are expressed in native-resolution
pixels/frame; Stage-1 values are automatically scaled back to native
equivalents before scoring.

## Luminance & Blur

Per-frame classification using the decision table below, followed by segment-level
aggregation. Two acceptance conditions must both pass:
1. (accept + review) frames >= 80% of total frames
2. Brightness stability: std dev of per-frame mean luminance <= 60

| Condition              | Mean Luminance | Normalized Tenengrad | Raw Tenengrad | Decision |
| ---------------------- | -------------- | -------------------- | ------------- | -------- |
| Dead black             | < 20           | --                   | --            | Reject   |
| Too dark               | 20 - 40        | --                   | --            | Reject   |
| Low light / noise zone | 40 - 70        | Unreliable -- ignore | < 80          | Reject   |
| Low light / noise zone | 40 - 70        | Unreliable -- ignore | 80 - 200      | Review   |
| Low light / noise zone | 40 - 70        | Unreliable -- ignore | > 200         | Accept   |
| Normal range           | 70 - 210       | < 0.04               | --            | Reject   |
| Normal range           | 70 - 210       | 0.04 - 0.10          | --            | Review   |
| Normal range           | 70 - 210       | 0.10 - 0.30          | --            | Accept   |
| Normal range           | 70 - 210       | > 0.30               | --            | Accept   |
| Soft overexposed       | 210 - 235      | < 0.04               | --            | Reject   |
| Soft overexposed       | 210 - 235      | 0.04 - 0.10          | --            | Review   |
| Soft overexposed       | 210 - 235      | > 0.10               | --            | Accept   |
| Blown out              | > 235          | --                   | --            | Reject   |

**Tenengrad computation:** Sobel gradient magnitude. Raw = mean(Gx^2 + Gy^2).
Normalized = raw / (mean_luminance^2 + epsilon). In the low-light noise zone
(luminance 40-70), normalized Tenengrad is unreliable due to noise amplification,
so raw Tenengrad is used instead.

**Segment analysis:** Frames are classified per the table above, then collapsed
into contiguous good/bad segments. "Review" frames count as good for segmentation.
The video passes if the ratio of good frames (accept + review) to total frames
meets the 80% threshold.

## ML Detection

| Check                   | Acceptance Condition                                        | How It Is Estimated              |
| ----------------------- | ----------------------------------------------------------- | -------------------------------- |
| Face Presence           | Face detection confidence < 0.8 in all frames               | Per-frame face detection         |
| Participants            | Persons detected <= 1 in >= 90% frames                      | Person detection                 |
| Hand Visibility         | >= 90% frames with both hands detection confidence >= 0.7   | Per-frame hands detection        |
| Hand-Object Interaction | Interaction detected in >= 70% frames                       | Hand + object proximity analysis |
| View Obstruction        | <= 10% frames obstructed                                    | Occlusion detection              |
| Body Part Visibility    | Only hands/forearms (up to elbows) visible in >= 90% frames | YOLO11m-pose keypoint detection  |
| Privacy Safety          | Sensitive object detections = 0 in all frames               | Detection of documents/screens   |

## Pipeline Behavior

- **Metadata gate:** If any video metadata check fails, all other checks are skipped.
- **Independent categories:** Luminance & blur, motion analysis, and ML detection run
  independently -- a failure in one does not skip others.
- **Statuses:** pass, fail, review (for borderline results), skipped (metadata gate).
