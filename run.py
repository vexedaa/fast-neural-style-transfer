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
