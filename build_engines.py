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
