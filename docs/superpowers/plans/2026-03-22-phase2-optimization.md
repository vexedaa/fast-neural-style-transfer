# Phase 2: TensorRT Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Optimize the Phase 1 pipeline with TensorRT inference, FP16 support, threaded capture, and per-step timing to achieve 60+ FPS (stretch: 144 FPS).

**Architecture:** Two-threaded pipeline — capture thread continuously grabs frames via DXcam, main thread runs TensorRT inference via onnxruntime's TRT provider with engine caching. Shared pure functions extracted into `pipeline.py`. Per-step timing in HUD for profiling.

**Tech Stack:** onnxruntime-gpu (TensorrtExecutionProvider), DXcam, OpenCV, threading

**Spec:** `docs/superpowers/specs/2026-03-22-phase2-optimization-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `pipeline.py` | Shared pure functions: preprocess, postprocess, round_to_mult4, discover_styles, build_hud_text |
| `poc.py` | Phase 1 PoC (modified to import from pipeline.py) |
| `run.py` | Phase 2 optimized pipeline: TRT inference, threaded capture, FP16, per-step timing |
| `build_engines.py` | Pre-build TensorRT engines for all styles |
| `tests/test_poc.py` | Tests for pipeline.py functions (imports updated from poc → pipeline) |
| `tests/test_run.py` | Tests for FrameGrabber and Phase 2 HUD |
| `requirements.txt` | Pin onnxruntime-gpu>=1.24.0 |
| `.gitignore` | Add engines/ |

---

### Task 1: Extract pipeline.py and update imports

Extract the shared pure functions from `poc.py` into `pipeline.py`. Update `poc.py` to import from `pipeline.py`. Update tests to import from `pipeline` instead of `poc`.

**Files:**
- Create: `pipeline.py`
- Modify: `poc.py`
- Modify: `tests/test_poc.py`

- [ ] **Step 1: Create pipeline.py**

```python
"""Shared pure functions for style transfer pipeline."""

import glob
import os

import cv2
import numpy as np


def round_to_mult4(value):
    """Round a dimension to the nearest multiple of 4 (minimum 4)."""
    return max(4, (int(value) + 2) // 4 * 4)


def preprocess(frame, scale):
    """Convert a BGR uint8 frame to model input format.

    Args:
        frame: numpy array (H, W, 3) uint8 BGR
        scale: float, resolution scale factor

    Returns:
        numpy array (1, 3, H', W') float32, RGB, values in [0, 255]
    """
    h, w = frame.shape[:2]
    new_h = round_to_mult4(h * scale)
    new_w = round_to_mult4(w * scale)
    resized = cv2.resize(frame, (new_w, new_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)  # HWC -> CHW
    batch = chw[np.newaxis, :, :, :]  # add batch dim
    return batch.astype(np.float32)


def postprocess(output):
    """Convert model output to a displayable BGR uint8 image.

    Args:
        output: numpy array (1, 3, H, W) float32, RGB, values roughly [0, 255]

    Returns:
        numpy array (H, W, 3) uint8 BGR
    """
    img = output[0]  # remove batch dim -> (3, H, W)
    img = img.transpose(1, 2, 0)  # CHW -> HWC
    img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return bgr


def discover_styles(weights_dir):
    """Find available style names from .onnx files in a directory.

    Returns:
        Sorted list of style names (filename stems).
    """
    onnx_files = glob.glob(os.path.join(weights_dir, "*.onnx"))
    names = [os.path.splitext(os.path.basename(f))[0] for f in onnx_files]
    return sorted(names)


def build_hud_text(style, scale, width, height, fps, num_styles=4):
    """Build the HUD overlay text string."""
    return (
        f"Style: {style} | Scale: {scale:.1f}x ({width}x{height}) | "
        f"FPS: {fps:.0f} | [1-{num_styles}] styles [+/-] scale [Q] quit"
    )
```

- [ ] **Step 2: Update poc.py to import from pipeline.py**

Replace `poc.py` lines 1-78 (the module docstring, imports, and the five pure function definitions) with:

```python
"""Phase 1 PoC: Real-time screen style transfer.

Captures a monitor via DXcam, applies neural style transfer using
a pre-trained ONNX model via onnxruntime (CUDA), and displays the
stylized output in an OpenCV window.

Usage:
    python poc.py --monitor 0 --scale 0.5 --style candy
"""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

from pipeline import (
    build_hud_text,
    discover_styles,
    postprocess,
    preprocess,
    round_to_mult4,
)
```

Everything from `def load_session(...)` onward (line 81+) stays unchanged.

- [ ] **Step 3: Update tests to import from pipeline**

In `tests/test_poc.py`, replace all `from poc import` with `from pipeline import` using find-and-replace. The function signatures are identical.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_poc.py -v`

Expected: All 15 tests PASS (same functions, different import source).

- [ ] **Step 5: Commit**

```bash
git add pipeline.py poc.py tests/test_poc.py
git commit -m "refactor: Extract shared functions to pipeline.py

Move preprocess, postprocess, round_to_mult4, discover_styles,
and build_hud_text into pipeline.py. Both poc.py and the upcoming
run.py import from this shared module.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Update requirements.txt and .gitignore

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Pin onnxruntime-gpu version**

In `requirements.txt`, change `onnxruntime-gpu` to `onnxruntime-gpu>=1.24.0`.

- [ ] **Step 2: Add engines/ to .gitignore**

Append to `.gitignore`:

```
# TensorRT engine cache
engines/
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: Pin onnxruntime-gpu version, gitignore engines/

Pin onnxruntime-gpu>=1.24.0 for TensorRT provider support.
Add engines/ to .gitignore for TensorRT engine cache.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Write tests for Phase 2 components

Tests for `FrameGrabber` (using a mock camera) and the Phase 2 two-line HUD builder.

**Files:**
- Create: `tests/test_run.py`

- [ ] **Step 1: Write test_run.py**

```python
"""Tests for Phase 2 run.py components."""

import threading
import time

import numpy as np
import pytest


class MockCamera:
    """Mock DXcam camera for testing FrameGrabber."""

    def __init__(self, frames):
        self._frames = iter(frames)

    def grab(self):
        try:
            return next(self._frames)
        except StopIteration:
            return None

    def release(self):
        pass


def test_frame_grabber_returns_none_initially():
    from run import FrameGrabber
    camera = MockCamera([])
    grabber = FrameGrabber(camera)
    assert grabber.get_latest() is None


def test_frame_grabber_captures_frame():
    from run import FrameGrabber
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    camera = MockCamera([frame, None, None, None, None])
    grabber = FrameGrabber(camera)
    grabber.start()
    time.sleep(0.1)  # let thread run
    grabber.stop()
    result = grabber.get_latest()
    assert result is not None
    assert result.shape == (100, 200, 3)


def test_frame_grabber_returns_copy():
    from run import FrameGrabber
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    camera = MockCamera([frame, None, None, None, None])
    grabber = FrameGrabber(camera)
    grabber.start()
    time.sleep(0.1)
    grabber.stop()
    result1 = grabber.get_latest()
    result2 = grabber.get_latest()
    assert result1 is not result2  # different objects (copies)
    np.testing.assert_array_equal(result1, result2)  # same content


def test_frame_grabber_gets_latest_frame():
    from run import FrameGrabber
    frame1 = np.full((100, 200, 3), 10, dtype=np.uint8)
    frame2 = np.full((100, 200, 3), 20, dtype=np.uint8)
    frame3 = np.full((100, 200, 3), 30, dtype=np.uint8)
    camera = MockCamera([frame1, frame2, frame3, None, None, None, None])
    grabber = FrameGrabber(camera)
    grabber.start()
    time.sleep(0.1)
    grabber.stop()
    result = grabber.get_latest()
    # Should have the last non-None frame
    assert result[0, 0, 0] == 30


def test_frame_grabber_stop():
    from run import FrameGrabber
    camera = MockCamera([None] * 1000)
    grabber = FrameGrabber(camera)
    grabber.start()
    assert grabber.thread.is_alive()
    grabber.stop()
    grabber.thread.join(timeout=1.0)
    assert not grabber.thread.is_alive()


def test_build_hud_text_phase2():
    from run import build_hud_text_p2
    timings = {"capture": 1.2, "preprocess": 0.8, "inference": 3.1, "postprocess": 0.5, "display": 1.0}
    line1, line2 = build_hud_text_p2("candy", 0.5, 1720, 720, 120.0, 4, True, timings)
    assert "candy" in line1
    assert "FP16" in line1
    assert "120" in line1
    assert "1720x720" in line1
    assert "3.1" in line2
    assert "Infer" in line2


def test_build_hud_text_p2_fp32():
    from run import build_hud_text_p2
    timings = {"capture": 1.0, "preprocess": 1.0, "inference": 5.0, "postprocess": 1.0, "display": 1.0}
    line1, line2 = build_hud_text_p2("mosaic", 1.0, 3440, 1440, 60.0, 4, False, timings)
    assert "FP32" in line1
    assert "mosaic" in line1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run.py -v`

Expected: All tests FAIL with `ModuleNotFoundError` since `run.py` doesn't exist yet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_run.py
git commit -m "test: Add tests for Phase 2 FrameGrabber and HUD

Tests for threaded frame capture with mock camera, frame copying,
latest-frame semantics, stop behavior, and two-line HUD builder.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Implement run.py

The optimized pipeline with TensorRT, threaded capture, FP16 toggle, and per-step timing HUD.

**Files:**
- Create: `run.py`

- [ ] **Step 1: Create run.py**

```python
"""Phase 2: Optimized real-time screen style transfer.

TensorRT inference via onnxruntime, threaded capture, FP16 support,
and per-step timing HUD.

Usage:
    python run.py --monitor 0 --scale 0.5 --style candy --fp16
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort

from pipeline import discover_styles, postprocess, preprocess


class FrameGrabber:
    """Threaded screen capture using DXcam."""

    def __init__(self, camera):
        self.camera = camera
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while self.running:
            grabbed = self.camera.grab()
            if grabbed is not None:
                with self.lock:
                    self.frame = grabbed

    def start(self):
        self.thread.start()

    def get_latest(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


def build_hud_text_p2(style, scale, width, height, fps, num_styles, fp16, timings):
    """Build two-line HUD text with per-step timing.

    Args:
        timings: dict with keys: capture, preprocess, inference, postprocess, display (values in ms)

    Returns:
        Tuple of (line1, line2) strings.
    """
    precision = "FP16" if fp16 else "FP32"
    line1 = (
        f"Style: {style} ({precision}) | Scale: {scale:.1f}x ({width}x{height}) | "
        f"FPS: {fps:.0f} | [1-{num_styles}] styles [+/-] scale [F] fp16 [Q] quit"
    )
    line2 = (
        f"Capture: {timings['capture']:.1f}ms | Pre: {timings['preprocess']:.1f}ms | "
        f"Infer: {timings['inference']:.1f}ms | Post: {timings['postprocess']:.1f}ms | "
        f"Display: {timings['display']:.1f}ms"
    )
    return line1, line2


def draw_hud_p2(frame, line1, line2):
    """Draw two-line HUD with semi-transparent background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    margin = 10

    (w1, h1), b1 = cv2.getTextSize(line1, font, font_scale, thickness)
    (w2, h2), b2 = cv2.getTextSize(line2, font, font_scale, thickness)

    max_w = max(w1, w2)
    total_h = h1 + h2 + 8  # gap between lines

    x = margin
    y1 = margin + h1

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, margin - 5), (x + max_w + 5, y1 + h2 + b2 + 8 + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, line1, (x, y1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.putText(frame, line2, (x, y1 + h2 + 8), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)


def create_session(weights_dir, style_name, engine_dir, fp16):
    """Create an onnxruntime session with TensorRT provider."""
    model_path = os.path.join(weights_dir, f"{style_name}.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    cache_enabled = True
    try:
        os.makedirs(engine_dir, exist_ok=True)
    except OSError as e:
        print(f"WARNING: Cannot create engine cache dir '{engine_dir}': {e}")
        print("Falling back to non-cached inference.")
        cache_enabled = False

    providers = [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": cache_enabled,
            "trt_engine_cache_path": engine_dir if cache_enabled else "",
            "trt_timing_cache_enable": cache_enabled,
            "trt_timing_cache_path": engine_dir if cache_enabled else "",
        }),
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]

    session = ort.InferenceSession(model_path, providers=providers)
    return session


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Optimized real-time screen style transfer")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index (default: 0)")
    parser.add_argument("--scale", type=float, default=0.5, help="Resolution scale 0.1-1.0 (default: 0.5)")
    parser.add_argument("--style", type=str, default="candy", help="Starting style (default: candy)")
    parser.add_argument("--weights-dir", type=str, default="./weights", help="ONNX model directory")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 inference")
    parser.add_argument("--engine-dir", type=str, default="./engines", help="TensorRT engine cache directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate providers
    available_providers = ort.get_available_providers()
    has_trt = "TensorrtExecutionProvider" in available_providers
    has_cuda = "CUDAExecutionProvider" in available_providers
    if not has_cuda:
        print("ERROR: CUDAExecutionProvider not available.")
        print(f"Available providers: {available_providers}")
        return
    if not has_trt:
        print("WARNING: TensorrtExecutionProvider not available, using CUDA only.")

    # Discover styles
    styles = discover_styles(args.weights_dir)
    if not styles:
        print(f"ERROR: No .onnx files found in '{args.weights_dir}'")
        return
    if args.style not in styles:
        print(f"ERROR: Style '{args.style}' not found. Available: {styles}")
        return

    current_style_idx = styles.index(args.style)
    scale = max(0.1, min(1.0, args.scale))
    fp16 = args.fp16

    # Load model
    print(f"Loading style '{styles[current_style_idx]}'...")
    print("Building/loading TensorRT engine (first run may take 30-60s)...", flush=True)
    session = create_session(args.weights_dir, styles[current_style_idx], args.engine_dir, fp16)
    input_name = session.get_inputs()[0].name

    # Warmup
    print("Warming up...", flush=True)
    warmup = np.random.rand(1, 3, 64, 64).astype(np.float32) * 255
    session.run(None, {input_name: warmup})
    print("Ready!")

    # Init DXcam
    import dxcam
    try:
        camera = dxcam.create(device_idx=0, output_idx=args.monitor, output_color="BGR")
    except Exception as e:
        print(f"ERROR: Failed to create capture for monitor {args.monitor}: {e}")
        try:
            n = len(dxcam.device_info()) if hasattr(dxcam, 'device_info') else 0
            print(f"Available monitors: 0-{n - 1}" if n else "Could not enumerate monitors.")
        except Exception:
            pass
        return

    # Start capture thread
    grabber = FrameGrabber(camera)
    grabber.start()

    print(f"Capturing monitor {args.monitor} at scale {scale:.1f}x")
    print(f"Styles: {', '.join(f'[{i+1}] {s}' for i, s in enumerate(styles))}")
    print(f"Precision: {'FP16' if fp16 else 'FP32'}")
    print("Press Q or ESC to quit.")

    window_name = "Style Transfer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps = 0.0
    prev_time = time.perf_counter()
    timings = {"capture": 0.0, "preprocess": 0.0, "inference": 0.0, "postprocess": 0.0, "display": 0.0}

    try:
        while True:
            # Capture
            t0 = time.perf_counter()
            frame = grabber.get_latest()
            t1 = time.perf_counter()

            if frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            timings["capture"] = (t1 - t0) * 1000

            # Preprocess
            t0 = time.perf_counter()
            input_tensor = preprocess(frame, scale)
            _, _, inf_h, inf_w = input_tensor.shape
            t1 = time.perf_counter()
            timings["preprocess"] = (t1 - t0) * 1000

            # Inference
            t0 = time.perf_counter()
            ort_inputs = {input_name: input_tensor}
            ort_outputs = session.run(None, ort_inputs)
            t1 = time.perf_counter()
            timings["inference"] = (t1 - t0) * 1000

            # Postprocess
            t0 = time.perf_counter()
            stylized = postprocess(ort_outputs[0])
            t1 = time.perf_counter()
            timings["postprocess"] = (t1 - t0) * 1000

            # FPS
            current_time = time.perf_counter()
            dt = current_time - prev_time
            prev_time = current_time
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            # HUD
            t0 = time.perf_counter()
            line1, line2 = build_hud_text_p2(
                styles[current_style_idx], scale, inf_w, inf_h, fps,
                len(styles), fp16, timings
            )
            draw_hud_p2(stylized, line1, line2)
            cv2.imshow(window_name, stylized)
            t1 = time.perf_counter()
            timings["display"] = (t1 - t0) * 1000

            # Key handling
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            # Style swap
            if ord("1") <= key <= ord("9"):
                idx = key - ord("1")
                if idx < len(styles) and idx != current_style_idx:
                    current_style_idx = idx
                    print(f"Switching to '{styles[current_style_idx]}'...")
                    session = create_session(
                        args.weights_dir, styles[current_style_idx], args.engine_dir, fp16
                    )
                    input_name = session.get_inputs()[0].name
                    warmup = np.random.rand(1, 3, 64, 64).astype(np.float32) * 255
                    session.run(None, {input_name: warmup})

            # Scale adjust
            if key in (ord("+"), ord("=")):
                scale = min(1.0, round(scale + 0.1, 1))
                print(f"Scale: {scale:.1f}x")
            if key == ord("-"):
                scale = max(0.1, round(scale - 0.1, 1))
                print(f"Scale: {scale:.1f}x")

            # FP16 toggle
            if key in (ord("f"), ord("F")):
                fp16 = not fp16
                print(f"Switching to {'FP16' if fp16 else 'FP32'}...")
                print("Building engine (may take 30-60s if not cached)...", flush=True)
                session = create_session(
                    args.weights_dir, styles[current_style_idx], args.engine_dir, fp16
                )
                input_name = session.get_inputs()[0].name
                warmup = np.random.rand(1, 3, 64, 64).astype(np.float32) * 255
                session.run(None, {input_name: warmup})
                print("Ready!")
    finally:
        grabber.stop()
        cv2.destroyAllWindows()
        camera.release()
        del camera
        print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_run.py tests/test_poc.py -v`

Expected: All tests PASS (both Phase 1 and Phase 2 tests).

- [ ] **Step 3: Commit**

```bash
git add run.py
git commit -m "feat: Add Phase 2 optimized pipeline with TensorRT

TensorRT inference via onnxruntime, threaded capture, FP16 toggle,
per-step timing HUD, and engine caching.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Create build_engines.py

Pre-build TensorRT engines for all styles to avoid cold-start penalty.

**Files:**
- Create: `build_engines.py`

- [ ] **Step 1: Write build_engines.py**

```python
"""Pre-build TensorRT engines for all styles.

Runs a dummy inference per model to trigger onnxruntime's TensorRT
engine compilation and caching. Avoids the 30-60s cold-start penalty
on first run of run.py.

Usage:
    python build_engines.py                  # FP32 engines
    python build_engines.py --fp16           # FP16 engines
    python build_engines.py --both           # Both FP16 and FP32
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort

from pipeline import discover_styles


def build_engine(weights_dir, style_name, engine_dir, fp16):
    """Build a TensorRT engine for one style."""
    model_path = os.path.join(weights_dir, f"{style_name}.onnx")
    os.makedirs(engine_dir, exist_ok=True)

    precision = "FP16" if fp16 else "FP32"
    print(f"  Building {style_name} ({precision})...", end=" ", flush=True)

    providers = [
        ("TensorrtExecutionProvider", {
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": engine_dir,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": engine_dir,
        }),
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]

    start = time.perf_counter()
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name

    # Trigger engine build with dummy inference
    dummy = np.random.rand(1, 3, 64, 64).astype(np.float32) * 255
    session.run(None, {input_name: dummy})

    elapsed = time.perf_counter() - start
    print(f"done ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Pre-build TensorRT engines")
    parser.add_argument("--weights-dir", type=str, default="./weights", help="ONNX model directory")
    parser.add_argument("--engine-dir", type=str, default="./engines", help="Engine cache directory")
    parser.add_argument("--fp16", action="store_true", help="Build FP16 engines")
    parser.add_argument("--both", action="store_true", help="Build both FP16 and FP32 engines")
    args = parser.parse_args()

    styles = discover_styles(args.weights_dir)
    if not styles:
        print(f"No .onnx files found in '{args.weights_dir}'")
        return

    precisions = []
    if args.both:
        precisions = [False, True]
    else:
        precisions = [args.fp16]

    print(f"Found {len(styles)} style(s): {', '.join(styles)}")

    for fp16 in precisions:
        label = "FP16" if fp16 else "FP32"
        print(f"\nBuilding {label} engines:")
        for style in styles:
            build_engine(args.weights_dir, style, args.engine_dir, fp16)

    print(f"\nDone. Engines cached in '{args.engine_dir}'.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add build_engines.py
git commit -m "feat: Add TensorRT engine pre-builder

Pre-builds engines for all styles to avoid cold-start penalty.
Supports FP16, FP32, or both via CLI flags.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Manual integration test

**Files:** None (testing only)

- [ ] **Step 1: Build engines**

Run: `python build_engines.py --both`

Verify: All 4 styles build for both FP16 and FP32. Engine files appear in `./engines/`.

- [ ] **Step 2: Run Phase 2 pipeline**

Run: `python run.py --monitor 0 --scale 0.5 --style candy --fp16`

Verify:
1. Window opens showing stylized desktop
2. Two-line HUD shows style, precision, scale, FPS, and per-step timing
3. Press `2` — style switches
4. Press `+`/`-` — scale adjusts
5. Press `F` — toggles between FP16/FP32 (HUD updates)
6. Press `Q` — exits cleanly, no orphan processes

- [ ] **Step 3: Run Phase 1 for comparison**

Run: `python poc.py --monitor 0 --scale 0.5 --style candy`

Verify: Phase 1 still works (imports from pipeline.py now). Compare FPS with Phase 2.

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/ -v`

Expected: All tests PASS (both test_poc.py and test_run.py).

- [ ] **Step 5: Commit test results as a note (optional)**

If satisfied, no commit needed. Phase 2 is complete.
