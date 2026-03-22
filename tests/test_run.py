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
