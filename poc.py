"""Phase 1 PoC: Real-time screen style transfer.

Captures a monitor via DXcam, applies neural style transfer using
a pre-trained ONNX model via onnxruntime (CUDA), and displays the
stylized output in an OpenCV window.

Usage:
    python poc.py --monitor 0 --scale 0.5 --style candy
"""

import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


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


def load_session(weights_dir, style_name):
    """Load an ONNX inference session for the given style."""
    model_path = os.path.join(weights_dir, f"{style_name}.onnx")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    session = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    return session


def draw_hud(frame, text):
    """Draw HUD text with a semi-transparent background on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    margin = 10

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = margin, margin + text_h

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 PoC: Real-time screen style transfer")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor index to capture (default: 0)")
    parser.add_argument("--scale", type=float, default=0.5, help="Resolution scale factor 0.1-1.0 (default: 0.5)")
    parser.add_argument("--style", type=str, default="candy", help="Starting style name (default: candy)")
    parser.add_argument("--weights-dir", type=str, default="./weights", help="Directory containing .onnx files")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate CUDA
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available_providers:
        print(f"ERROR: CUDAExecutionProvider not available.")
        print(f"Available providers: {available_providers}")
        print("Install onnxruntime-gpu and ensure CUDA is configured.")
        return

    # Discover styles
    styles = discover_styles(args.weights_dir)
    if not styles:
        print(f"ERROR: No .onnx files found in '{args.weights_dir}'")
        return

    # Validate starting style
    if args.style not in styles:
        print(f"ERROR: Style '{args.style}' not found. Available: {styles}")
        return

    current_style_idx = styles.index(args.style)
    scale = max(0.1, min(1.0, args.scale))

    # Init DXcam (imported here to keep pure functions importable cross-platform)
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
        print("Try a different --monitor index.")
        return

    # Load initial model
    print(f"Loading style '{styles[current_style_idx]}'...")
    session = load_session(args.weights_dir, styles[current_style_idx])
    input_name = session.get_inputs()[0].name

    print(f"Capturing monitor {args.monitor} at scale {scale:.1f}x")
    print(f"Styles: {', '.join(f'[{i+1}] {s}' for i, s in enumerate(styles))}")
    print("Press Q or ESC to quit.")

    window_name = "Style Transfer PoC"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    fps = 0.0
    prev_time = time.perf_counter()

    while True:
        frame = camera.grab()
        if frame is None:
            # Screen unchanged, keep displaying last frame
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            continue

        # Preprocess
        input_tensor = preprocess(frame, scale)
        _, _, inf_h, inf_w = input_tensor.shape

        # Inference
        ort_inputs = {input_name: input_tensor}
        ort_outputs = session.run(None, ort_inputs)

        # Postprocess
        stylized = postprocess(ort_outputs[0])

        # FPS calculation
        current_time = time.perf_counter()
        dt = current_time - prev_time
        prev_time = current_time
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

        # HUD
        hud_text = build_hud_text(styles[current_style_idx], scale, inf_w, inf_h, fps, len(styles))
        draw_hud(stylized, hud_text)

        cv2.imshow(window_name, stylized)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            break

        # Style swap: keys 1-9
        if ord("1") <= key <= ord("9"):
            idx = key - ord("1")
            if idx < len(styles) and idx != current_style_idx:
                current_style_idx = idx
                print(f"Switching to '{styles[current_style_idx]}'...")
                session = load_session(args.weights_dir, styles[current_style_idx])
                input_name = session.get_inputs()[0].name

        # Scale adjust
        if key in (ord("+"), ord("=")):
            scale = min(1.0, round(scale + 0.1, 1))
            print(f"Scale: {scale:.1f}x")
        if key == ord("-"):
            scale = max(0.1, round(scale - 0.1, 1))
            print(f"Scale: {scale:.1f}x")

    cv2.destroyAllWindows()
    del camera
    print("Done.")


if __name__ == "__main__":
    main()
