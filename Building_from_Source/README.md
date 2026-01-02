# Building_from_Source

This folder contains small C++ apps that **build ExecuTorch from source** via CMake `add_subdirectory(...)` and run `.pte` models.

Most subprojects are designed to be built on:

- **Host (WSL/Linux)**
- **Android (NDK toolchain)**

> Large model files (`*.pte`) and build outputs are ignored by git (see `example/.gitignore`).

---

## `comparison_cpu_gpu/`

**Goal**: run **two different `.pte` files** (GPU-delegated vs CPU-delegated) and print timing statistics (avg/min/p50/p95/max/stddev + speedup).

- **Entry points**
  - `main.cpp`: loads `gpu_pte` and `cpu_pte` and benchmarks `Module.forward(...)`
  - `export_model.py`: exports/produces the two `.pte` files (not committed)
  - `CMakeLists.txt`: supports host vs Android builds

---

## `running_on_devices_cpu/`

**Goal**: minimal “run on device” app targeting CPU backends.

- **Entry points**
  - `main.cpp`: single forward pass (minimal)
  - `bench_low_level.cpp`: simple timing loop for `Module.forward(...)`
  - `export_model.py`: produce `model.pte` (ignored by git)

---

## `running_on_devices_gpu/`

**Goal**: minimal “run on device” app with GPU backends enabled in the build (e.g., Vulkan).

- **Entry points**
  - `main.cpp`: minimal forward pass on `model.pte`
  - `export_model.py`: produce `model.pte` (ignored by git)
  - `CMakeLists.txt`: sets `EXECUTORCH_BUILD_VULKAN ON`





