# ExecuTorch Tutorials (WSL/Linux + Android)

This repository is a set of hands-on **ExecuTorch** tutorials and mini-projects:

- **Model export / lowering** (PyTorch → `.pte`)
- **Graph partitioning** (delegation to backends such as XNNPACK)
- **Building from source** (host + Android builds)
- **LLM on-device** (Llama 3.2 Instruct, CPU vs Vulkan speed comparison)

**Created by Taemin Kim, an M.S. student at the Mobile Embedded Systems Lab, Korea University.**

---

## Repo structure

- `Running_Llama/`
  - Llama 3.2 Instruct: export `.pte`, build `llama_chat` (host + Android), and run.
  - Includes a CPU vs Vulkan(GPU) benchmarking flow using `PyTorchObserver`.
- `Building_from_Source/`
  - C++ apps that build ExecuTorch from source via `add_subdirectory(...)`.
  - Includes a small CPU/GPU comparison app (`comparison_cpu_gpu/`).
- `Getting_Started_with_ExecuTorch/`
  - Minimal “hello ExecuTorch” on desktop (export + run).
- `Model_Export_and_Lowering/`
  - Minimal examples showing `torch.export` → `to_edge_transform_and_lower` → `.pte`.
  - Includes an example with **external constants** (`.ptd`) for separating weights.
- `Graph_Partitioning/`
  - Tiny scripts illustrating partitioning/delegation (e.g., MobileNetV2 → XNNPACK).

---

## What is (not) tracked in git?

This repo intentionally **does not commit large artifacts**, including:

- Model binaries: `*.pte`, `*.pt2`, `*.ptd`
- Checkpoints/weights: `*.pth`, `*.safetensors`
- Build outputs: `build/`, `build-android/`, `cmake-out*/`

See `example/.gitignore`.

---

## Prerequisites (typical setup)

- **ExecuTorch source** checked out separately (sibling to this repo), e.g.:
  - `/home/tm0118/Desktop/executorch`
- **Conda** env (example: `basic`)
  - `cmake >= 3.29` (needed by ExecuTorch builds)
  - Python deps depending on which tutorial you run (torch, torchvision, omegaconf, etc.)
- **Android** (optional)
  - Android NDK installed (example path: `/home/tm0118/android-ndk-r27d`)
  - `adb` working and device connected

---

## How to navigate

Start from the README of the folder you care about:

- `Running_Llama/README.md`
- `Building_from_Source/README.md`
- `Getting_Started_with_ExecuTorch/README.md`
- `Model_Export_and_Lowering/README.md`
- `Graph_Partitioning/README.md`


