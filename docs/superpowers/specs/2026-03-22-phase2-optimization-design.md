# Phase 2: TensorRT Optimization — Design Spec

## Objective

Optimize the Phase 1 Python pipeline for maximum FPS using TensorRT inference, FP16 support, and threaded capture. Deliver per-step timing data to inform Phase 3 (C++) decisions. Target: 60+ FPS at 0.5x scale, stretch goal 144 FPS for 144hz display support.

## Architecture

Two-threaded pipeline in `run.py` at the repo root.

```
Capture Thread:
  Loop:
    1. DXcam grab() → BGR frame
    2. Store in shared "latest frame" variable (thread-safe)

Main Thread:
  Init:
    1. Parse CLI args
    2. Build/load TensorRT engines (via onnxruntime TRT provider)
    3. Start capture thread
    4. Open OpenCV window

  Loop:
    1. Read latest frame from shared variable
    2. Preprocess (resize, BGR→RGB, NCHW, float32)
    3. Inference via onnxruntime TRT session
    4. Postprocess (clamp, HWC, BGR)
    5. Overlay HUD (FPS + per-step timing breakdown)
    6. cv2.imshow()
    7. Handle key events
```

The capture thread ensures a fresh frame is always ready. The main thread never blocks waiting for a screen update. The shared frame uses a simple lock — not a queue — because we always want the latest frame, not queued old ones.

## TensorRT Integration

### Session Creation

Same onnxruntime API as Phase 1, with TensorRT provider config:

```python
providers = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": args.fp16,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./engines",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "./engines",
    }),
    ("CUDAExecutionProvider", {}),
    ("CPUExecutionProvider", {}),
]
```

### Engine Caching

First run builds optimized TensorRT engines from the `.onnx` files (~30-60s per model). Engines are saved to `./engines/` and reused on subsequent runs. If an engine exists, startup is near-instant.

### FP16 Toggle

`--fp16` CLI flag enables half-precision inference. TensorRT uses FP16 for most layers, roughly halving inference time with negligible quality loss for style transfer. The engine cache is precision-specific — FP16 and FP32 engines are separate cached files.

Runtime toggle via `F` key rebuilds the session with the opposite precision. If the engine is not cached, the build takes ~30-60s and the UI will freeze during that time (accepted behavior for Phase 2 — the `build_engines.py` script exists to pre-build engines and avoid this). If the engine is cached, the rebuild takes 1-5s.

### Engine Build Script

`build_engines.py` — pre-builds TensorRT engines for all styles. Optional convenience to avoid cold-start penalty on first run. Engines auto-build on first inference if not pre-built.

```
python build_engines.py [options]

Options:
  --weights-dir PATH  ONNX model directory (default: ./weights)
  --engine-dir PATH   Engine cache directory (default: ./engines)
  --fp16              Build FP16 engines (default: FP32)
  --both              Build both FP16 and FP32 engines
```

The script discovers all `.onnx` files via `discover_styles()` (imported from `pipeline.py`), creates the engine directory if needed, and runs a dummy inference per model to trigger engine compilation. Prints progress and timing for each build.

### Fallback Chain

TensorRT → CUDA → CPU. If TensorRT fails for any reason, onnxruntime automatically falls back to CUDA (same as Phase 1).

## Threading Model

```python
class FrameGrabber:
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
```

- **Daemon thread** — dies automatically when main thread exits, no orphan risk
- **Frame copy** — `get_latest()` returns a copy to prevent the capture thread from mutating the array while the main thread processes it
- **Latest-frame semantics** — no queue, no backpressure. Main thread always gets the most recent screen state
- `grab()` returns `None` when screen hasn't changed — thread simply retries, lock is never held for long
- Main loop calls `get_latest()` — if it returns `None` (no frame yet), skip that iteration

## CLI Interface

```
python run.py [options]

Options:
  --monitor INT       Monitor index (default: 0)
  --scale FLOAT       Resolution scale 0.1-1.0 (default: 0.5)
  --style STRING      Starting style (default: candy)
  --weights-dir PATH  ONNX model directory (default: ./weights)
  --fp16              Enable FP16 inference (default: FP32)
  --engine-dir PATH   TensorRT engine cache directory (default: ./engines)
```

## Runtime Controls

| Key       | Action                                |
|-----------|---------------------------------------|
| `1`-`9`   | Switch style                          |
| `+` / `=` | Increase scale by 0.1                 |
| `-`       | Decrease scale by 0.1                 |
| `F`       | Toggle FP16/FP32 (rebuilds if needed) |
| `Q` / ESC | Quit                                  |

## HUD Overlay

Two-line HUD with per-step timing:

```
Style: candy (FP16) | Scale: 0.5x (1720x720) | FPS: 120
Capture: 1.2ms | Pre: 0.8ms | Infer: 3.1ms | Post: 0.5ms | Display: 1.0ms
```

The per-step timing breakdown is the key Phase 2 deliverable — it tells you exactly where time is being spent, directly informing Phase 3 decisions.

## Files Changed

| File              | Action | Purpose                                           |
|-------------------|--------|---------------------------------------------------|
| `run.py`          | Create | Optimized pipeline: TRT, threading, FP16, timing  |
| `build_engines.py`| Create | Pre-build TensorRT engines for all styles          |
| `.gitignore`      | Edit   | Add `engines/` directory                           |

Extract shared pure functions (`preprocess`, `postprocess`, `round_to_mult4`, `discover_styles`, `build_hud_text`) from `poc.py` into a new `pipeline.py` module. Update `poc.py` to import from `pipeline.py` instead of defining them inline. Both `poc.py` and `run.py` import from `pipeline.py`.

| File              | Action | Purpose                                           |
|-------------------|--------|---------------------------------------------------|
| `pipeline.py`     | Create | Shared pure functions extracted from poc.py        |
| `poc.py`          | Modify | Import from pipeline.py instead of inline defs     |

## Dependencies

None new. `onnxruntime-gpu` 1.24.4 already includes `TensorrtExecutionProvider`. Pin `onnxruntime-gpu>=1.24.0` in `requirements.txt` to ensure TensorRT provider options are available.

| File              | Action | Purpose                                           |
|-------------------|--------|---------------------------------------------------|
| `requirements.txt`| Modify | Pin `onnxruntime-gpu>=1.24.0`                     |

## Error Handling

| Scenario                        | Behavior                                                    |
|---------------------------------|-------------------------------------------------------------|
| TensorRT engine build fails     | Fall back to CUDAExecutionProvider, print warning           |
| No CUDA GPU available           | Exit with clear message                                     |
| Invalid monitor index           | Catch DXcam error, list available monitors, exit            |
| No .onnx files in weights       | Exit with message                                           |
| Engine cache dir not writable   | Fall back to non-cached TensorRT or CUDA provider           |
| FP16 toggle during runtime      | Print "Building engine..." message, session rebuild inline  |
| `grab()` returns None           | Skip iteration, retain last displayed frame                 |

## Success Criteria

1. TensorRT inference works with engine caching in `./engines/`
2. FP16 toggle works via `--fp16` flag and runtime `F` key
3. Threaded capture runs independently from inference, measurably higher FPS than Phase 1
4. Per-step timing in HUD: capture, preprocess, inference, postprocess, display individually timed
5. Engine builder works: `build_engines.py` pre-builds engines for all styles
6. Significant FPS improvement over Phase 1 (target: 60+ FPS at 0.5x scale, stretch goal: 144 FPS)
7. All Phase 1 features preserved: style swap, scale adjust, monitor selection, error handling

## Non-Goals (Phase 2)

- Hitting 144 FPS at all resolutions (may require Phase 3's GPU-native pipeline)
- Visual quality tuning between FP16/FP32
- Any UI beyond the OpenCV window
- Model pruning or architecture changes
- GPU-native capture or display (Phase 3)

## Context

- **Hardware targets:** NVIDIA RTX 5080 (dev), minimum RTX 3070/3080 (production)
- **User's monitors:** 3440x1440 (144hz) + 2560x1440
- **Commercial intent:** Product will be sold on Steam
- **Phase 1 baseline:** ~10-20 FPS at 0.5x scale with CUDAExecutionProvider
- **Phase 3 (next):** C++ native app with DXGI + TensorRT, no CPU round-trip
- **CUDA setup:** Toolkit 12.9 + cuDNN 9.20, paths must be on PATH
