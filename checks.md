# Scoring Engine — Check Specifications

The scoring engine evaluates each video through three gated stages:

1. **Metadata** — cheap ffprobe reads; any single fail skips stages 2 and 3.
2. **Technical** — quality of the capture itself (luminance, stability, frozen, pixelation).
3. **Quality metrics** — readings about what's in frame (hands, HOI, angle, participants, obstruction). Not pass/fail at the video level; reported as `percent_frames` + merged segments.

A single decode pass drives all of stages 2 and 3. If the technical stage fails, quality metrics are rendered as `SKIPPED` in the MD/JSON report — but the raw per-frame quality columns remain in the parquet for debugging.

See `SCORING_ENGINE_PLAN.md` for the locked design decisions.

---

## Stage 1: Metadata checks

Any single failure marks the video metadata-failed; technical and quality are reported SKIPPED, and no decode is performed.

| Check       | Rule                                                                                  | Source  |
| ----------- | ------------------------------------------------------------------------------------- | ------- |
| format      | Container is MP4 (ffprobe: `mov,mp4,m4a,3gp,3g2,mj2`).                                | ffprobe |
| encoding    | Video codec is H.264 or HEVC (H.265).                                                 | ffprobe |
| resolution  | Displayed dims ≥ 1920×1080 (after rotation).                                          | ffprobe |
| frame_rate  | ≥ 28 FPS.                                                                             | ffprobe |
| duration    | ≥ 59 s.                                                                               | ffprobe |
| orientation | Rotation ∈ {0, 90, 270} AND displayed landscape (displayed_width > displayed_height). | ffprobe |

Thresholds are configurable in the TOML config under `[metadata]`.

---

## Metadata observations (non-gating)

Alongside the six gated checks above, each video records seven
observations. They are **recorded, not scored** — no pass/fail, no
effect on whether stages 2 and 3 run. Populated for every video,
including those that fail the metadata gate (the readings come from
the same ffprobe call).

All values come from one existing ffprobe call plus one extra cheap
packet-level scan for GOP (no decode).

| Field              | Value format | Source rule                                                                                                   |
| ------------------ | ------------ | ------------------------------------------------------------------------------------------------------------- |
| bitrate_mbps       | float (Mbps) | Video stream `bit_rate` / 1e6, 2 decimals. Format-level bitrate is **not** used — video only.                  |
| gop                | float        | `total_packets / keyframe_count` on the video stream, packet-level `flags=K` (IDR keyframes), no decode.       |
| color_depth_bits   | 8 / 10 / 12  | `bits_per_raw_sample` when present; else regex on `pix_fmt` (`yuv420p10le` → 10); else 8 for `yuv*`/`nv*`/`rgb*`/`gray*`; else `None`. |
| b_frames           | Y / N        | `has_b_frames > 0` → Y; else N.                                                                                 |
| hdr                | ON / OFF     | ON iff **any** of: `color_transfer ∈ {smpte2084, arib-std-b67}` (PQ / HLG); side_data contains a Dolby Vision record; `codec_tag_string ∈ {dvhe, dvav, dvh1, dva1}`. Else OFF. **BT.2020 primaries alone are not HDR.** |
| stabilization      | Y / N / Unknown | Vendor registry (below). First match wins.                                                                   |
| fov                | string       | Vendor registry (below). First match wins. Raw vendor label is preserved; no cross-vendor normalisation.       |

### HDR rule — rationale

Transfer function is authoritative; primaries alone are not. Some UHD
SDR material is mastered in BT.2020 primaries with `bt709` transfer —
that is not HDR. Dolby Vision is detected separately because DV can
ship with `bt709` transfer while carrying a PQ/HDR layer in side
data.

- `smpte2084` → HDR10 / ST 2084 / PQ
- `arib-std-b67` → HLG (ARIB STD-B67)
- `DOVI configuration record` in `side_data_list` → Dolby Vision
- `dvhe` / `dvav` / `dvh1` / `dva1` codec tags → Dolby Vision fallback

### Stabilization registry

Device-agnostic. Evaluated in priority order; first positive match
wins. Absence of any signal → `Unknown`, never `N` (few containers
carry a "stabilization: off" marker; reporting `N` there would be a
false negative).

| Priority | Vendor / signal          | Detector                                                                                    | Returns   |
| -------- | ------------------------ | ------------------------------------------------------------------------------------------- | --------- |
| 1        | gyroflow / ReelSteady    | video-stream `encoder` tag matches `\b(gyroflow\|reelsteady\|hypersmooth)\b`                 | `Y`       |
| 2a       | GoPro                    | `encoder` contains `gopro`                                                                   | `Y`       |
| 2b       | GoPro (GPMD track)       | any stream has `handler_name` containing `gopro met` or `gpmd`, or `codec_tag_string=gpmd`   | `Y`       |
| 3        | Google CAMM              | any stream `handler_name` contains `camm` (Google Camera Motion Metadata)                    | `Y`       |
| 4        | Samsung                  | `smta` / `svss` atoms or `samsung`-prefixed tags on format/stream                            | `Y`       |
| 5        | DJI                      | `encoder` or `handler_name` contains `dji`                                                   | `Y`       |
| 6        | Apple iPhone             | `com.apple.quicktime.software` tag present                                                   | `Unknown` (container does not reveal EIS/OIS state at capture) |
| —        | no match                 | —                                                                                           | `Unknown` |

### FOV registry

Same ordering principle. Most containers carry no FOV signal, so
`Unknown` is the common case.

| Priority | Vendor       | Detector                                                  | Returns                              |
| -------- | ------------ | --------------------------------------------------------- | ------------------------------------ |
| 1        | GoPro        | GPMD stream present, or `encoder` contains `gopro`        | `GoPro-embedded` (placeholder until the GPMD KLV parser lands with IMU extraction) |
| 2        | DJI          | `encoder`/`handler_name` contains `dji`                   | `DJI-embedded` (placeholder until the `udta` parser lands)                         |
| 3        | Apple iPhone | `com.apple.quicktime.focal.length.35mmEquiv` tag present  | `~{deg}°` computed from 35mm-equivalent focal length via `HFOV = 2·atan(36 / 2f)`  |
| —        | no match     | —                                                         | `Unknown`                            |

GoPro / DJI currently return the `{vendor}-embedded` sentinel rather
than the actual lens preset — the real KLV parser for the GPMD /
`udta` streams is the next piece of work (paired with IMU extraction).
When it lands, this registry promotes the sentinel to the real label
without touching anything else.

### Where these appear

- **Per-video `report.md`:** `## Metadata observations` section with
  a `| Field | Value |` table.
- **Per-video `{video}.json`:** under the top-level
  `metadata_observations` key.
- **Batch `batch_results.csv`:** one column per field, `meta_{field}`,
  value only (no status / accepted columns).
- **Batch `batch_report.md`:** `## Metadata observations (aggregate)`
  section with a numeric sub-table (mean / median / min / max for
  `bitrate_mbps`, `gop`, `color_depth_bits`) and a categorical
  sub-table (distribution histogram for `b_frames`, `hdr`,
  `stabilization`, `fov`). Per-video rows in the batch MD stay
  summary-only.

---

## Stage 2: Technical checks

All four checks run on every video that passes the metadata gate. Any one failure marks technical-failed.

| Check      | Rule                                                                                                   | Cadence / resolution         |
| ---------- | ------------------------------------------------------------------------------------------------------ | ---------------------------- |
| luminance  | ≥ 80% good frames (neither dead black <15, too dark 15–45, blown out >230, nor flicker window).        | 10 FPS, 360p                 |
| stability  | Whole-video mean jitter score ≤ 0.181 (high-pass-filtered LK optical flow).                            | 30 FPS cap, 360p (0.5×)      |
| frozen     | No run > 60 consecutive native-frame-equivalents with near-zero motion (trans < 0.1 px, rot < 0.001°). | Derived from motion samples. |
| pixelation | ≥ 80% frames with blockiness ratio ≤ 1.5.                                                              | 10 FPS, 720p                 |

Thresholds are configurable under `[technical.luminance]`, `[technical.stability]`, `[technical.frozen]`, `[technical.pixelation]`.

---

## Stage 3: Quality metrics

Six per-frame metrics, each reported as `percent_frames` plus merged segments. `face_presence` as a standalone metric is removed — SCRFD face detection now feeds the participants signal.

Cadences: the five hand / participants metrics run at 1 FPS on 720p frames. `obstructed` piggybacks on the pixelation cadence (10 FPS, 720p) — the central-crop heuristic is cheap and benefits from denser sampling.

| Metric                 | Per-frame PASS                                                                                                   | Value (raw per-frame)                       | Value on fail                                          | Cadence |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------ | ------- |
| both_hands_visibility  | Hands23 detects L **and** R, each conf ≥ 0.7.                                                                    | max(L_conf, R_conf)                         | 0.0                                                    | 1 FPS   |
| single_hand_visibility | ≥1 hand with Hands23 conf ≥ 0.7.                                                                                 | max Hands23 conf                            | 0.0                                                    | 1 FPS   |
| hand_obj_interaction   | ≥1 detected hand with contact state ∈ {P, F}.                                                                    | contact char of qualifying hand (P/F)       | most-confident hand's char (N/S/O); null if zero hands | 1 FPS   |
| hand_angle             | all detected hands' angles ≤ 40° (zero hands → fail).                                                            | mean angle of detected hands (degrees)      | mean angle; NaN if no hands                            | 1 FPS   |
| participants           | ≥1 "other person" signal from {YOLO (≥0.6, wearer-filtered), SCRFD (≥0.6, wearer-filtered), extra Hands23 hand}. | max(YOLO_conf, SCRFD_conf, extra_hand_conf) | 0.0                                                    | 1 FPS   |
| obstructed             | Heuristic triggers ≥2 of 4 signals on the central 80% crop.                                                      | `true`                                      | `false`                                                | 10 FPS  |

Both_hands and single_hand may pass for the same frame (not mutually exclusive).

### Wearer filter (applies to YOLO persons AND SCRFD faces)

- Exclude detections that overlap a Hands23 hand (wearer's own body/face via reflection).
- Exclude detections anchored in the bottom-centre region of the frame.
- Exclude bboxes whose height is < 15% of the frame height.

### Segment merging

Runs shorter than `merge_threshold_s` (default 1.0s) absorb into their preceding neighbour (flipping state). First-run edge case: absorb into the following run. Single left-to-right pass — no cascade re-merging. Parquet is un-merged; merging applies only to JSON / MD report output. `percent_frames` is computed from the un-merged per-frame array.

**Segment value semantics (plan §1):**

- Conf-based metrics: max confidence across _qualifying_ (originally pass) frames in the merged window. Absorbed fail-frames never contribute.
- Hand angle: mean angle over all frames in the merged segment.
- Hand-obj contact char: most common contact char over all frames in the merged segment.
- Obstructed: `true` / `false` per the merged segment's state.

---

## Output artefacts

Per video:

- `report.md` — human-readable Markdown tables (`## Metadata`, `## Metadata observations`, `## Technical`, `## Quality`).
- `{video_name}.json` — same content as MD but structured, JSON-safe (NaN serialised as the string `"NaN"`). Observations live under `metadata_observations`.
- `{video_name}.parquet` — one row per native frame, dense schema (see `SCORING_ENGINE_PLAN.md` §4). Omitted when metadata failed (nothing decoded). Observations are **not** in parquet (single value per video, not per frame).

Per batch:

- `batch_report.md`, `batch_results.json`, `batch_results.csv` in the run-dir root. Per-video observations appear as extra columns in the CSV and as an aggregate section at the bottom of the batch MD.

Output layout: `results/run_NNN/{video_name}/report.md`, etc. `NNN` is zero-padded to 3 digits and auto-extends past 999.
