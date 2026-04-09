"""Generate a synthetic test video for benchmarking ML checks.

Creates a 30-second 1080p 30fps video with:
- Colored gradient backgrounds
- Simulated hand-like shapes in the lower portion
- Some frames with face-like features
"""

import cv2
import numpy as np
from pathlib import Path


def generate_test_video(
    output_path: str = "bachman_cortex/sample_data/test_30s.mp4",
    duration_s: int = 30,
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = duration_s * fps
    print(f"Generating {total_frames} frames ({duration_s}s at {fps}fps)...")

    for i in range(total_frames):
        # Base: gradient background that shifts over time
        t = i / total_frames
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Warm gradient (simulates indoor lighting)
        for y in range(height):
            brightness = int(80 + 80 * (y / height) + 30 * np.sin(t * 2 * np.pi))
            brightness = max(20, min(200, brightness))
            frame[y, :] = [brightness - 20, brightness, brightness + 20]

        # Add some texture (simulates a workspace surface)
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Simulated hand-like shapes in lower-center (skin-colored ellipses)
        if i % 3 != 0:  # ~67% of frames have "hands"
            # Left hand region
            cx_l = int(width * 0.35 + 50 * np.sin(i * 0.05))
            cy_l = int(height * 0.7 + 30 * np.cos(i * 0.03))
            cv2.ellipse(frame, (cx_l, cy_l), (80, 120), -20, 0, 360,
                        (120, 160, 200), -1)  # skin-ish color in BGR

            # Right hand region
            cx_r = int(width * 0.65 + 50 * np.cos(i * 0.05))
            cy_r = int(height * 0.7 + 30 * np.sin(i * 0.03))
            cv2.ellipse(frame, (cx_r, cy_r), (80, 120), 20, 0, 360,
                        (120, 160, 200), -1)

        # Occasional "object" in the center (simulates tool/item)
        if i % 5 == 0:
            ox = int(width * 0.5 + 40 * np.sin(i * 0.02))
            oy = int(height * 0.55)
            cv2.rectangle(frame, (ox - 60, oy - 30), (ox + 60, oy + 30),
                          (80, 80, 80), -1)

        out.write(frame)

        if (i + 1) % (fps * 5) == 0:
            print(f"  {i + 1}/{total_frames} frames written...")

    out.release()
    print(f"Video saved to {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    generate_test_video()
