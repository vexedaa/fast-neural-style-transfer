# Phase 1: Python Proof of Concept — Design Spec

## Objective

Build a Python script that captures a monitor's display, applies real-time neural style transfer using a pre-trained ONNX model, and displays the stylized output in an OpenCV window. This proves the capture → infer → display pipeline works end-to-end.

## Architecture

Single-threaded loop in a single script (`poc.py`) at the repo root.

```
Init:
  1. Parse CLI args (monitor index, resolution scale, starting style)
  2. Create DXcam screengrab for the selected monitor
  3. Load the initial ONNX model into an onnxruntime InferenceSession (CUDA)
  4. Open an OpenCV display window

Loop:
  1. Grab latest frame from DXcam (numpy array, BGR)
  2. Resize by scale factor
  3. Convert to model input format (float32, [0-255], RGB, NCHW)
  4. Run inference via onnxruntime
  5. Convert output back to displayable format (uint8, BGR, HWC)
  6. Overlay HUD (FPS, style name, keybindings)
  7. cv2.imshow()
  8. Check for key events (style swap, scale adjust, quit)
```

## CLI Interface

```
python poc.py [options]

Options:
  --monitor INT       Monitor index to capture (default: 0)
  --scale FLOAT       Resolution scale factor, 0.0-1.0 (default: 0.5)
  --style STRING      Starting style name: candy, mosaic, rain-princess, udnie (default: candy)
  --weights-dir PATH  Directory containing .onnx files (default: ./weights)
```

## Runtime Controls

| Key       | Action                                    |
|-----------|-------------------------------------------|
| `1`-`4`   | Switch between available styles           |
| `+` / `=` | Increase scale factor by 0.1              |
| `-`       | Decrease scale factor by 0.1              |
| `Q` / ESC | Quit                                      |

Styles are auto-detected from `.onnx` files in the weights directory.

## HUD Overlay

Top-left corner of the OpenCV window, semi-transparent background:

```
Style: candy | Scale: 0.5x (1720x720) | FPS: 45 | [1-4] styles [+/-] scale [Q] quit
```

## Implementation Details

### DXcam Capture

- `dxcam.create(device_idx=0, output_idx=monitor)` to create capture device
- `camera.grab()` for single-frame pulls (not streaming mode)
- Returns BGR numpy arrays `(H, W, 3)` — same format as OpenCV
- If `grab()` returns `None` (screen unchanged), skip iteration

### Preprocessing

1. Resize frame by scale factor using OpenCV
2. Convert BGR → RGB
3. Transpose HWC → CHW
4. Cast to float32 (values already in [0, 255] range)
5. Add batch dimension: `(3, H, W)` → `(1, 3, H, W)`

### ONNX Inference

- `onnxruntime.InferenceSession` with `CUDAExecutionProvider`
- Session created once at startup, reused every frame
- On style swap: create new session with new `.onnx` file (~6.5MB, near-instant reload)
- Input name retrieved from session: `session.get_inputs()[0].name`

### Postprocessing

1. Squeeze batch dimension: `(1, 3, H, W)` → `(3, H, W)`
2. Transpose CHW → HWC
3. Clamp values to [0, 255]
4. Cast to uint8
5. Convert RGB → BGR for OpenCV display

### Scale Adjustment

When `+`/`-` is pressed:
- Clamp new scale to [0.1, 1.0]
- The new scale applies on the next frame — no session reload needed since the ONNX model was exported with dynamic spatial dimensions (only batch axis is dynamic in the current export, but the model architecture is fully convolutional and handles arbitrary spatial dimensions)

## Files Changed

| File               | Action | Purpose                     |
|--------------------|--------|-----------------------------|
| `poc.py`           | Create | Entire PoC script           |
| `requirements.txt` | Edit   | Add `dxcam`                 |

No changes to existing model code. Preprocessing is inlined to keep the PoC self-contained and decoupled from the single-image stylization utilities.

## Error Handling

| Scenario                   | Behavior                                           |
|----------------------------|-----------------------------------------------------|
| No CUDA GPU available      | Exit with clear message                             |
| Invalid monitor index      | Catch DXcam error, list available monitors, exit     |
| No .onnx files in weights  | Exit with message listing expected directory         |
| `grab()` returns None      | Skip iteration, retry next loop pass                 |
| Style key out of range     | Ignore                                              |

## Dependencies

Only one new dependency beyond existing `requirements.txt`:
- `dxcam`

Existing deps used: `onnxruntime`, `opencv-python`, `numpy`.

## Success Criteria

1. Launches, captures the selected monitor, applies a style, displays the result
2. Style swapping via keyboard works without crashing
3. Scale adjustment via keyboard works, visible in HUD
4. HUD displays current style, effective resolution, live FPS, and keybindings
5. `--monitor 1` captures the second display
6. Handles edge cases (static screen, invalid keys) without crashing

## Non-Goals (Phase 1)

- Performance targets (FPS goals are Phase 2+)
- Visual quality tuning
- Simultaneous multi-monitor capture
- Any UI beyond the OpenCV window
- Threading or async pipeline

## Context

- **Hardware targets:** NVIDIA RTX 5080 (dev), minimum RTX 3070/3080 (production)
- **User's monitors:** 3440x1440 + 2560x1440
- **Commercial intent:** Product will be sold on Steam
- **End vision:** Config GUI + system tray icon; overlay runs independently. Not relevant to Phase 1.
- **Phase 2:** ONNX → TensorRT compilation for sub-16ms inference
- **Phase 3:** C++ native app with DXGI + TensorRT, no CPU round-trip
