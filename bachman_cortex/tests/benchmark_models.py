"""Benchmark all 4 ML models on sample frames.

Measures per-frame latency (p50/p95/p99) and total throughput for each model.
"""

import sys
import os
import time
import json
import numpy as np

# Setup paths
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

# Set torch lib path for 100DOH
import torch
os.environ["DYLD_LIBRARY_PATH"] = str(os.path.join(os.path.dirname(torch.__file__), "lib"))

from bachman_cortex.utils.frame_extractor import extract_frames


def benchmark_scrfd(frames: list[np.ndarray]) -> dict:
    """Benchmark SCRFD face detector."""
    print("\n" + "=" * 60)
    print("SCRFD-2.5GF Face Detector")
    print("=" * 60)

    from bachman_cortex.models.scrfd_detector import SCRFDDetector

    detector = SCRFDDetector(
        root=os.path.join(ROOT, "bachman_cortex/models/weights/insightface"),
    )

    # Warmup
    print("Warmup (3 frames)...")
    for f in frames[:3]:
        detector.detect(f)

    # Benchmark
    print(f"Benchmarking on {len(frames)} frames...")
    result = detector.benchmark(frames)

    # Count detections
    total_faces = 0
    for f in frames:
        faces = detector.detect(f)
        total_faces += len(faces)

    result["total_detections"] = total_faces
    print(f"  p50: {result['p50_ms']:.1f}ms | p95: {result['p95_ms']:.1f}ms | mean: {result['mean_ms']:.1f}ms")
    print(f"  Total faces detected: {total_faces}")
    return result


def benchmark_yolo(frames: list[np.ndarray]) -> dict:
    """Benchmark YOLO11m object detector."""
    print("\n" + "=" * 60)
    print("YOLO11m Object Detector")
    print("=" * 60)

    from bachman_cortex.models.yolo_detector import YOLODetector

    detector = YOLODetector(model_path="yolo11m.pt")

    # Warmup
    print("Warmup (3 frames)...")
    for f in frames[:3]:
        detector.detect(f)

    # Benchmark
    print(f"Benchmarking on {len(frames)} frames...")
    result = detector.benchmark(frames)

    # Count detections by type
    total_persons = 0
    for f in frames:
        dets = detector.detect(f)
        total_persons += len(detector.get_persons(dets))

    result["total_persons"] = total_persons
    print(f"  p50: {result['p50_ms']:.1f}ms | p95: {result['p95_ms']:.1f}ms | mean: {result['mean_ms']:.1f}ms")
    print(f"  Persons: {total_persons}")
    return result


def benchmark_100doh(frames: list[np.ndarray]) -> dict:
    """Benchmark 100DOH hand-object detector."""
    print("\n" + "=" * 60)
    print("100DOH Hand-Object Detector (ResNet-101)")
    print("=" * 60)

    from bachman_cortex.models.hand_detector import HandObjectDetector100DOH

    detector = HandObjectDetector100DOH(
        repo_dir=os.path.join(ROOT, "bachman_cortex/models/weights/hand_object_detector"),
    )

    # Warmup
    print("Warmup (2 frames)...")
    for f in frames[:2]:
        detector.detect(f)

    # Benchmark (use fewer frames due to CPU cost)
    bench_frames = frames[:10]
    print(f"Benchmarking on {len(bench_frames)} frames (subset for CPU)...")
    result = detector.benchmark(bench_frames)

    # Count detections
    total_hands = 0
    total_objects = 0
    contact_states = {}
    for f in bench_frames:
        hands, objects = detector.detect(f)
        total_hands += len(hands)
        total_objects += len(objects)
        for h in hands:
            state = h.contact_state.name
            contact_states[state] = contact_states.get(state, 0) + 1

    result["total_hands"] = total_hands
    result["total_objects"] = total_objects
    result["contact_states"] = contact_states
    print(f"  p50: {result['p50_ms']:.1f}ms | p95: {result['p95_ms']:.1f}ms | mean: {result['mean_ms']:.1f}ms")
    print(f"  Hands: {total_hands} | Objects: {total_objects}")
    print(f"  Contact states: {contact_states}")
    return result


def benchmark_grounding_dino(frames: list[np.ndarray]) -> dict:
    """Benchmark Grounding DINO zero-shot detector."""
    print("\n" + "=" * 60)
    print("Grounding DINO (zero-shot, base)")
    print("=" * 60)

    from bachman_cortex.models.grounding_dino_detector import GroundingDINODetector

    detector = GroundingDINODetector(
        cache_dir=os.path.join(ROOT, "bachman_cortex/models/weights/grounding_dino"),
    )

    text_prompt = "laptop screen . computer monitor . smartphone screen . paper document . credit card . ID card"

    # Warmup
    print("Warmup (2 frames)...")
    for f in frames[:2]:
        detector.detect(f, text_prompt=text_prompt)

    # Benchmark (use fewer frames due to CPU cost)
    bench_frames = frames[:5]
    print(f"Benchmarking on {len(bench_frames)} frames (subset for CPU)...")
    result = detector.benchmark(bench_frames, text_prompt=text_prompt)

    # Count detections
    total_dets = 0
    labels_seen = {}
    for f in bench_frames:
        dets = detector.detect(f, text_prompt=text_prompt)
        total_dets += len(dets)
        for d in dets:
            labels_seen[d.label] = labels_seen.get(d.label, 0) + 1

    result["total_detections"] = total_dets
    result["labels_seen"] = labels_seen
    print(f"  p50: {result['p50_ms']:.1f}ms | p95: {result['p95_ms']:.1f}ms | mean: {result['mean_ms']:.1f}ms")
    print(f"  Total detections: {total_dets}")
    print(f"  Labels: {labels_seen}")
    return result


def main():
    video_path = os.path.join(ROOT, "bachman_cortex/sample_data/test_30s.mp4")
    print(f"Extracting frames from {video_path}...")
    frames, meta = extract_frames(video_path, fps=1.0, max_frames=30)
    print(f"Extracted {meta['frames_extracted']} frames in {meta['extraction_time_s']}s")
    print(f"Video: {meta['width']}x{meta['height']}, {meta['duration_s']}s")

    results = {}
    results["frame_extraction"] = meta

    # Benchmark each model
    results["scrfd"] = benchmark_scrfd(frames)
    results["yolo11m"] = benchmark_yolo(frames)
    results["100doh"] = benchmark_100doh(frames)
    results["grounding_dino"] = benchmark_grounding_dino(frames)

    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY (CPU, macOS ARM64)")
    print("=" * 60)
    print(f"{'Model':<40} {'p50 ms':>8} {'p95 ms':>8} {'mean ms':>8}")
    print("-" * 64)
    for name in ["scrfd", "yolo11m", "100doh", "grounding_dino"]:
        r = results[name]
        print(f"{r['model']:<40} {r['p50_ms']:>8.1f} {r['p95_ms']:>8.1f} {r['mean_ms']:>8.1f}")

    # Estimate total per-video time (300 frames at 1 FPS)
    scrfd_per_frame = results["scrfd"]["mean_ms"]
    yolo_per_frame = results["yolo11m"]["mean_ms"]
    doh_per_frame = results["100doh"]["mean_ms"]
    gdino_per_frame = results["grounding_dino"]["mean_ms"]

    total_per_frame = scrfd_per_frame + yolo_per_frame + doh_per_frame
    total_300_frames = total_per_frame * 300 / 1000
    # Grounding DINO only runs on ~5% of frames (pre-filtered)
    gdino_estimated = gdino_per_frame * 15 / 1000  # ~15 frames

    print(f"\nEstimated total for 300 frames (5-min video):")
    print(f"  SCRFD:          {scrfd_per_frame * 300 / 1000:.1f}s")
    print(f"  YOLO11m:        {yolo_per_frame * 300 / 1000:.1f}s")
    print(f"  100DOH:         {doh_per_frame * 300 / 1000:.1f}s")
    print(f"  Grounding DINO: {gdino_estimated:.1f}s (est. 15 flagged frames)")
    print(f"  TOTAL:          {total_300_frames + gdino_estimated:.1f}s")

    # Save results
    output_file = os.path.join(ROOT, "bachman_cortex/tests/benchmark_results.json")
    with open(output_file, "w") as f:
        # Convert numpy types for JSON
        def default(o):
            if isinstance(o, (np.integer, np.int64)):
                return int(o)
            if isinstance(o, (np.floating, np.float64)):
                return float(o)
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")
        json.dump(results, f, indent=2, default=default)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
