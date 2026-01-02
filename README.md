# ExecuTorch Examples & Benchmarking Toolkit

This repository provides practical examples and tools for exporting, running, and benchmarking various deep learning models (LLMs, Vision, etc.) using **ExecuTorch** on both local PCs (WSL/Linux) and Android devices.

---

## 📂 Project Structure

Each directory focuses on specific ExecuTorch features or model execution scenarios.

### 1. Running LLMs
Export and run the latest lightweight language models with a chat interface.
- **[Running_Llama](./Running_Llama)**: Running Llama-3.2-1B-Instruct on PC and Android. Includes CPU vs. Vulkan performance comparison.
- **[Running_SmolLM2](./Running_SmolLM2)**: SmolLM2-135M-Instruct example. Includes ChatML format handling and EOS token configuration.
- **[Running_Llava](./Running_Llava)**: Contains PTE files related to LLaVA (Vision-Language Model).

### 2. Core Workflows
Examples for learning the basic model conversion and processing flow of ExecuTorch.
- **[Getting_Started_with_ExecuTorch](./Getting_Started_with_ExecuTorch)**: A minimal "Hello World" example from model export to Python runtime execution.
- **[Model_Export_and_Lowering](./Model_Export_and_Lowering)**: Detailed steps of `torch.export`, Backend Partitioning, Lowering, and external constant (PTD) management.
- **[Building_from_Source](./Building_from_Source)**: How to build C++ apps by including ExecuTorch source directly using `add_subdirectory()`. Includes CPU/GPU benchmarking tools.

### 3. Profiling & Optimization
Measure model performance and analyze graph structures.
- **[Profiling](./Profiling)**: Measure layer-by-layer performance of models like MobileNetV2 using `ETDump` and `ETRecord`.
- **[Graph_Partitioning](./Graph_Partitioning)**: Analyze how model operations are delegated to various backends (XNNPACK, Vulkan, Exynos) and check accelerator assignment.

---

## 🛠️ Environment Setup

The examples in this project assume the following environment:

- **OS**: Linux (Ubuntu/WSL2)
- **Python**: Conda environment (e.g., `basic`) recommended
- **ExecuTorch Source**: `/home/tm0118/Desktop/executorch` (Local source reference)
- **Build System**: CMake 3.29+, Ninja
- **Android Support**: Android NDK (r27d recommended), ADB (Android Debug Bridge)

---

## 🚀 Quick Start Guide (Llama-3.2 Example)

1. **Export Model**
   ```bash
   cd Running_Llama
   python export_llama.py
   ```
2. **Build C++ Runner**
   ```bash
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ```
3. **Run (PC)**
   ```bash
   ./build/llama_chat --model_path llama3_2_instruct_bf16.pte --tokenizer_path <path_to_tokenizer> --prompt "Hello!"
   ```

---

## 👤 Author
**Taemin Kim**
M.S. Student at Mobile Embedded Systems Lab, Korea University.

---

> **Note**: Build artifacts like `.pte`, `.pth`, and `build/` directories, as well as large model files, are not tracked by Git (see `.gitignore`). They must be generated locally.

<br><br>

---
---

# ExecuTorch 예제 및 벤치마킹 툴킷 (Korean)

이 저장소는 **ExecuTorch**를 활용하여 다양한 딥러닝 모델(LLM, Vision 등)을 내보내고(Export), 로컬 PC(WSL/Linux) 및 안드로이드 기기에서 실행 및 벤치마킹하는 실전 예제들을 모아놓은 프로젝트입니다.

---

## 📂 프로젝트 구조

각 폴더는 ExecuTorch의 특정 기능이나 모델 실행 시나리오를 담당합니다.

### 1. LLM 실행 (Running LLMs)
최신 경량 언어 모델을 ExecuTorch로 변환하고 채팅 인터페이스로 실행합니다.
- **[Running_Llama](./Running_Llama)**: Llama-3.2-1B-Instruct 모델을 PC 및 안드로이드에서 실행. (CPU vs Vulkan 속도 비교 포함)
- **[Running_SmolLM2](./Running_SmolLM2)**: SmolLM2-135M-Instruct 모델 실행 예제. ChatML 포맷 처리 및 EOS 토큰 설정 포함.
- **[Running_Llava](./Running_Llava)**: LLaVA (Vision-Language Model) 관련 PTE 파일 포함.

### 2. 핵심 워크플로우 (Core Workflows)
ExecuTorch의 기본적인 모델 변환 및 처리 과정을 익히기 위한 예제입니다.
- **[Getting_Started_with_ExecuTorch](./Getting_Started_with_ExecuTorch)**: 모델 Export부터 Python 런타임 실행까지의 최소 단위 "Hello World" 예제.
- **[Model_Export_and_Lowering](./Model_Export_and_Lowering)**: `torch.export`, Backend Partitioning, Lowering 과정의 상세 단계 및 외부 상수(PTD) 관리 예제.
- **[Building_from_Source](./Building_from_Source)**: `add_subdirectory()` 방식을 사용하여 ExecuTorch 소스를 프로젝트에 직접 포함시켜 C++ 앱을 빌드하는 방법. (CPU/GPU 성능 비교 툴 포함)

### 3. 분석 및 최적화 (Profiling & Optimization)
모델의 성능을 측정하고 그래프 구조를 분석합니다.
- **[Profiling](./Profiling)**: `ETDump`, `ETRecord`를 활용하여 MobileNetV2 등의 모델을 레이어 단위(Layer-by-layer)로 성능 측정.
- **[Graph_Partitioning](./Graph_Partitioning)**: XNNPACK, Vulkan, Exynos 등 다양한 백엔드로 모델 연산이 어떻게 분산(Delegation)되는지 분석하고 가속기 할당 현황 확인.

---

## 🛠️ 주요 환경 설정

이 프로젝트의 예제들은 공통적으로 아래 환경을 전제로 합니다.

- **OS**: Linux (Ubuntu/WSL2)
- **Python**: Conda 환경 (`basic` 등) 권장
- **ExecuTorch Source**: `/home/tm0118/Desktop/executorch` (로컬 소스 참조)
- **Build System**: CMake 3.29 이상, Ninja
- **Android Support**: Android NDK (r27d 권장), ADB (Android Debug Bridge)

---

## 🚀 빠른 시작 가이드 (Llama-3.2 예시)

1. **모델 Export**
   ```bash
   cd Running_Llama
   python export_llama.py
   ```
2. **C++ 러너 빌드**
   ```bash
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j
   ```
3. **실행 (PC)**
   ```bash
   ./build/llama_chat --model_path llama3_2_instruct_bf16.pte --tokenizer_path <path_to_tokenizer> --prompt "Hello!"
   ```

---

## 👤 Author
**Taemin Kim**
M.S. Student at Mobile Embedded Systems Lab, Korea University.

---

> **Note**: `.pte`, `.pth`, `build/` 폴더 등 빌드 결과물 및 대용량 모델 파일은 `.gitignore`에 의해 관리되지 않습니다. 로컬에서 직접 생성해야 합니다.
