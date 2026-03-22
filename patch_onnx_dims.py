"""Patch ONNX models to use dynamic spatial dimensions.

The shipped .onnx models declare fixed 1080x1080 input/output dimensions.
The underlying model is fully convolutional and handles arbitrary sizes,
but onnxruntime enforces declared dimensions. This script patches the
dimension declarations so the models accept any height/width.

Usage:
    python patch_onnx_dims.py --weights-dir ./weights
"""

import argparse
import glob
import os

import onnx


def patch_model(model_path):
    """Patch a single ONNX model to use dynamic spatial dimensions."""
    model = onnx.load(model_path)
    graph = model.graph

    for tensor in list(graph.input) + list(graph.output):
        shape = tensor.type.tensor_type.shape
        if shape and len(shape.dim) == 4:
            # dim 0 = batch (already dynamic), dim 1 = channels (fixed 3)
            # dim 2 = height, dim 3 = width — make these dynamic
            shape.dim[2].dim_param = "height"
            shape.dim[2].ClearField("dim_value")
            shape.dim[3].dim_param = "width"
            shape.dim[3].ClearField("dim_value")

    onnx.save(model, model_path)
    print(f"Patched: {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Patch ONNX models for dynamic spatial dims")
    parser.add_argument("--weights-dir", type=str, default="./weights",
                        help="Directory containing .onnx files (default: ./weights)")
    args = parser.parse_args()

    onnx_files = glob.glob(os.path.join(args.weights_dir, "*.onnx"))
    if not onnx_files:
        print(f"No .onnx files found in {args.weights_dir}")
        return

    for path in onnx_files:
        patch_model(path)

    print(f"\nDone. Patched {len(onnx_files)} model(s).")


if __name__ == "__main__":
    main()
