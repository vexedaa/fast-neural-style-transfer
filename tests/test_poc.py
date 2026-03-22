"""Tests for poc.py pure functions."""

import os
import tempfile

import numpy as np
import pytest


def test_round_to_mult4_rounds_down():
    from pipeline import round_to_mult4
    assert round_to_mult4(1081) == 1080


def test_round_to_mult4_rounds_up():
    from pipeline import round_to_mult4
    assert round_to_mult4(1082) == 1084


def test_round_to_mult4_already_aligned():
    from pipeline import round_to_mult4
    assert round_to_mult4(1080) == 1080


def test_round_to_mult4_minimum():
    from pipeline import round_to_mult4
    assert round_to_mult4(1) == 4


def test_preprocess_shape_and_dtype():
    from pipeline import preprocess
    # Simulate a 100x200 BGR frame
    frame = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    result = preprocess(frame, scale=1.0)
    assert result.dtype == np.float32
    assert result.shape == (1, 3, 100, 200)


def test_preprocess_scales_dimensions():
    from pipeline import preprocess
    frame = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
    result = preprocess(frame, scale=0.5)
    # 100*0.5=50 -> round_to_mult4(50)=52, 200*0.5=100 -> round_to_mult4(100)=100
    assert result.shape == (1, 3, 52, 100)


def test_preprocess_rounds_to_mult4():
    from pipeline import preprocess
    # 110*0.5=55 -> round to 56, 210*0.5=105 -> round to 104
    frame = np.random.randint(0, 256, (110, 210, 3), dtype=np.uint8)
    result = preprocess(frame, scale=0.5)
    _, _, h, w = result.shape
    assert h % 4 == 0
    assert w % 4 == 0


def test_preprocess_value_range():
    from pipeline import preprocess
    frame = np.full((100, 200, 3), 128, dtype=np.uint8)
    result = preprocess(frame, scale=1.0)
    assert result.min() >= 0.0
    assert result.max() <= 255.0


def test_preprocess_bgr_to_rgb():
    from pipeline import preprocess
    # Frame with distinct BGR channels
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[:, :, 0] = 10   # B
    frame[:, :, 1] = 20   # G
    frame[:, :, 2] = 30   # R
    result = preprocess(frame, scale=1.0)
    # After BGR->RGB conversion, channel 0 should be R=30
    assert result[0, 0, 0, 0] == 30.0  # R channel
    assert result[0, 1, 0, 0] == 20.0  # G channel
    assert result[0, 2, 0, 0] == 10.0  # B channel


def test_postprocess_shape_and_dtype():
    from pipeline import postprocess
    # Simulate model output: (1, 3, 50, 100) float32
    output = np.random.rand(1, 3, 50, 100).astype(np.float32) * 255
    result = postprocess(output)
    assert result.dtype == np.uint8
    assert result.shape == (50, 100, 3)


def test_postprocess_clamps_values():
    from pipeline import postprocess
    output = np.array([[[[-10.0]], [[300.0]], [[128.0]]]],  dtype=np.float32)
    result = postprocess(output)
    # Input RGB channels: R=-10, G=300, B=128
    # After clamp: R=0, G=255, B=128
    # After RGB->BGR swap: pixel = [B=128, G=255, R=0]
    assert result[0, 0, 0] == 128    # B channel (was RGB B=128)
    assert result[0, 0, 1] == 255    # G channel (was RGB G=300, clamped)
    assert result[0, 0, 2] == 0      # R channel (was RGB R=-10, clamped)


def test_postprocess_rgb_to_bgr():
    from pipeline import postprocess
    output = np.zeros((1, 3, 1, 1), dtype=np.float32)
    output[0, 0, 0, 0] = 30.0   # R
    output[0, 1, 0, 0] = 20.0   # G
    output[0, 2, 0, 0] = 10.0   # B
    result = postprocess(output)
    # After RGB->BGR, pixel should be [B=10, G=20, R=30]
    assert result[0, 0, 0] == 10   # B
    assert result[0, 0, 1] == 20   # G
    assert result[0, 0, 2] == 30   # R


def test_discover_styles_finds_onnx_files():
    from pipeline import discover_styles
    with tempfile.TemporaryDirectory() as tmpdir:
        for name in ["candy.onnx", "mosaic.onnx", "readme.txt"]:
            open(os.path.join(tmpdir, name), "w").close()
        styles = discover_styles(tmpdir)
        assert styles == ["candy", "mosaic"]


def test_discover_styles_empty_dir():
    from pipeline import discover_styles
    with tempfile.TemporaryDirectory() as tmpdir:
        styles = discover_styles(tmpdir)
        assert styles == []


def test_build_hud_text():
    from pipeline import build_hud_text
    text = build_hud_text("candy", 0.5, 1720, 720, 45.2, 4)
    assert "candy" in text
    assert "0.5x" in text
    assert "1720x720" in text
    assert "45" in text
    assert "[1-4]" in text
