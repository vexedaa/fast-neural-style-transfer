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
