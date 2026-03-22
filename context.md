# Project: Real-Time Display Style Transfer

## Objective
Enable real-time (60+ FPS) style transfer of an entire computer display using an ultra-lightweight, feedforward Convolutional Neural Network (CNN), bypassing standard OS overhead.

## Chosen Architecture & Stack
* **Model Architecture:** Fast Neural Style Transfer (Johnson et al.) - A single-pass feedforward network capable of millisecond inference.
* **Target Repository:** `yakhyo/fast-neural-style-transfer`
  * *Reasoning:* Modernized Python/PyTorch implementation, includes built-in ONNX export capabilities, and provides pre-trained ONNX weights out of the box.
* **Prototyping Stack (Python):** `DXcam` (for DXGI screen capture), `onnxruntime` or `torch`, `OpenCV` (for display).
* **Production Stack (C++):** C++ build environment, Windows SDK (DXGI Desktop Duplication API), ONNX, TensorRT (for NVIDIA hardware) or DirectML.

## Implementation Roadmap

### Phase 1: Python Proof of Concept (The Sandbox)
* **Goal:** Prove the pipeline works end-to-end, regardless of framerate.
* **Action:** Write a Python script using `DXcam` to grab a desktop frame, pass it through the pre-trained ONNX model using `onnxruntime`, and display the stylized array using `OpenCV`.

### Phase 2: Optimization and Export (The Bridge)
* **Goal:** Achieve sub-16ms inference times required for 60 FPS.
* **Action:** Export the chosen PyTorch `.pth` models into `.onnx` format (using the provided CLI tools in the yakhyo repo). Compile the ONNX model into a highly optimized TensorRT engine (`.trt` or `.engine`) specifically for the target GPU hardware to fuse network layers and optimize memory.

### Phase 3: C++ Native Application (Production)
* **Goal:** System-wide stylization with near-zero latency.
* **Action:** Rewrite the capture and inference pipeline in C++. Use the DXGI Desktop Duplication API to capture the screen buffer directly on the GPU. Pass the GPU memory pointer directly to the TensorRT C++ API to avoid copying to system RAM. Render the output to a borderless, transparent full-screen DirectX window overlaid on the desktop.

## Contextual Notes for AI Agents
* Avoid raw PyTorch or standard Convolutional Neural Networks for the final driver-level implementation; they introduce too much overhead.
* Diffusion models (even Turbo/LCM) are too slow for this 60 FPS screen-space requirement.
* Memory management between DirectX and the AI execution engine is the primary technical hurdle in Phase 3.