# Running_Llama (ExecuTorch) — PC(WSL)에서 Llama-3.2-1B-Instruct 실행

이 폴더는 **ExecuTorch의 `examples/models/llama` C++ 러너를 그대로 빌드해서** 로컬(WSL/Linux)에서 Llama PTE를 실행합니다.

현재 이 프로젝트는 **`add_subdirectory()` 방식**으로 `executorch/` 소스를 함께 빌드합니다. (즉 `find_package(executorch)`용 install 패키지가 없어도 됩니다.)

---

## 0) 전제 조건

- **폴더 구조**
  - `/home/tm0118/Desktop/executorch` 가 존재해야 함 (ExecuTorch 소스)
  - 이 프로젝트는 `/home/tm0118/Desktop/example/Running_Llama`
- **Conda 환경**
  - `conda activate basic`
  - 이 환경에서 `cmake`는 **3.29+** 이어야 함 (예: 3.31.10)
  - Python 쪽은 `torch`, `omegaconf` 등이 설치되어 있어야 함 (export에 필요)

---

## 1) 빌드 (llama_chat 생성)

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic

cd /home/tm0118/Desktop/example/Running_Llama

rm -rf build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

빌드 성공 시 실행 파일:

- `./build/llama_chat`

---

## 2) 모델 export (Llama-3.2-1B-Instruct → PTE)

이 폴더에는 이미 meta 포맷 체크포인트가 들어있다고 가정합니다:

- `Llama-3.2-1B-Instruct/original/consolidated.00.pth`
- `Llama-3.2-1B-Instruct/original/params.json`
- `Llama-3.2-1B-Instruct/original/tokenizer.model`

export 스크립트:

- `export_llama.py` 는 위 경로를 읽어서 `llama3_2_instruct_bf16.pte` 를 생성하도록 설정돼 있음

실행:

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic

cd /home/tm0118/Desktop/example/Running_Llama
python export_llama.py
```

생성 결과:

- `./llama3_2_instruct_bf16.pte`

---

## 3) 실행 (Instruct 채팅 템플릿 프롬프트)

권장: Llama-3 Chat 템플릿 토큰을 포함한 prompt를 전달해야 “Instruct 답변”이 잘 나옵니다.

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic

cd /home/tm0118/Desktop/example/Running_Llama

./build/llama_chat \
  --model_path=./llama3_2_instruct_bf16.pte \
  --tokenizer_path=./Llama-3.2-1B-Instruct/original/tokenizer.model \
  --prompt=$'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTell me one fun fact about penguins.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' \
  --temperature=0.6 \
  --max_new_tokens=96 \
  --warmup=true
```

---

## 4) 자주 만나는 문제/해결

### (A) `CMake 3.29 or higher is required ... running 3.22.1`

- conda 환경을 안 켠 상태에서 `/usr/bin/cmake` (3.22.1)이 잡혀서 나는 에러입니다.
- 아래처럼 `basic`을 켠 뒤 `cmake --version`이 3.29+인지 확인하세요.

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic
cmake --version
which cmake
```

### (B) `EXECUTORCH_BUILD_EXTENSION_MODULE requires ... NAMED_DATA_MAP`

- ExecuTorch preset 체크가 걸려서 나는 에러입니다.
- 이 프로젝트는 `CMakeLists.txt`에서 `EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON`을 켜서 해결했습니다.

### (C) 실행 로그에 `hf_tokenizer.cpp:82 Error parsing json file`가 뜸

- `tokenizer.model`을 로더가 여러 포맷으로 “시도(probing)”하는 과정에서 생기는 로그가 섞여 보일 수 있습니다.
- **실행/생성 결과가 정상이라면 무시해도 됩니다.**

---

## 5) 파일 정리

- 빌드 출력: `build/`
- export 결과: `llama3_2_instruct_bf16.pte`
- 실행에 필요한 토크나이저: `Llama-3.2-1B-Instruct/original/tokenizer.model`

---

## 6) Android(핸드폰)에서 실행하는 방법

핵심은 3단계입니다:

- **(1) Android용 `llama_chat` 크로스컴파일**
- **(2) `adb push`로 바이너리 + `.pte` + `tokenizer.model`을 기기에 복사**
- **(3) `adb shell`에서 실행**

### 6.1) Android용 빌드 (arm64-v8a)

NDK 경로를 먼저 준비하세요:

- 예) `ANDROID_NDK=/home/tm0118/android-ndk-r27d`

빌드:

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic

cd /home/tm0118/Desktop/example/Running_Llama

rm -rf build-android
cmake -S . -B build-android -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=/home/tm0118/android-ndk-r27d/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34

cmake --build build-android -j
```

빌드 성공 시 실행 파일:

- `./build-android/llama_chat`

> 참고: Android에서 **bf16이 안 돌아가는 기기/커널 조합**도 있을 수 있습니다.
> 그 경우에는 export를 fp16/fp32로 다시 해서 시도하세요. (export 스크립트/옵션만 바꾸면 됩니다.)

### 6.2) 기기로 파일 push

기기에서 쓸 폴더를 하나 만들고(예: `/data/local/tmp/running_llama`) 바이너리/모델/토크나이저를 올립니다:

```bash
cd /home/tm0118/Desktop/example/Running_Llama

adb shell "mkdir -p /data/local/tmp/running_llama"

adb push ./build-android/llama_chat /data/local/tmp/running_llama/
adb push ./llama3_2_instruct_bf16.pte /data/local/tmp/running_llama/
adb push ./Llama-3.2-1B-Instruct/original/tokenizer.model /data/local/tmp/running_llama/

adb shell "chmod +x /data/local/tmp/running_llama/llama_chat"
```

#### (선택) libc++_shared.so가 필요할 때

만약 실행 시 `CANNOT LINK EXECUTABLE ... libc++_shared.so` 같은 에러가 나오면,
NDK에서 `libc++_shared.so`를 찾아 같이 push 해주세요:

```bash
# 경로는 NDK 버전에 따라 다를 수 있음 (아래 둘 중 하나에 존재하는 경우가 많음)
adb push /home/tm0118/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so /data/local/tmp/running_llama/ || true
adb push /home/tm0118/android-ndk-r27d/sources/cxx-stl/llvm-libc++/libs/arm64-v8a/libc++_shared.so /data/local/tmp/running_llama/ || true
```

그 다음 실행할 때 `LD_LIBRARY_PATH`를 같이 줍니다:

```bash
adb shell "cd /data/local/tmp/running_llama && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:. && ./llama_chat --help"
```

### 6.3) Android에서 실행

```bash
adb shell "cd /data/local/tmp/running_llama && ./llama_chat \
  --model_path=./llama3_2_instruct_bf16.pte \
  --tokenizer_path=./tokenizer.model \
  --prompt=\$'<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nTell me one fun fact about penguins.\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' \
  --temperature=0.6 \
  --max_new_tokens=96"
```

---

## 7) CPU vs GPU(Vulkan) 속도 비교 (tokens/sec)

### 7.1) 왜 export를 2개 해야 하나?

- **CPU용 PTE**: 보통 XNNPACK/optimized kernels로 CPU delegate
- **GPU(Vulkan)용 PTE**: Vulkan partitioner로 그래프 일부를 Vulkan backend로 delegate

즉, **같은 모델이라도 “어느 backend로 내릴지”가 export 단계에서 결정**됩니다. 그래서 PTE를 2개 만들어야 합니다.

> 중요: ExecuTorch Vulkan LLM 경로는 현재 **dtype=fp32만 지원**이라서, Vulkan PTE는 fp32로 export해야 합니다.

### 7.2) CPU용 PTE export (XNNPACK + 8da4w 예시)

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic

cd /home/tm0118/Desktop/executorch

python -m examples.models.llama.export_llama \
  -c /home/tm0118/Desktop/example/Running_Llama/Llama-3.2-1B-Instruct/original/consolidated.00.pth \
  -p /home/tm0118/Desktop/example/Running_Llama/Llama-3.2-1B-Instruct/original/params.json \
  -d fp32 --xnnpack --xnnpack_extended_ops \
  -qmode 8da4w -G 64 \
  --max_seq_length 2048 --max_context_length 2048 \
  -kv --use_sdpa_with_kv_cache \
  --metadata '{"append_eos_to_prompt": 0, "get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --model llama3_2 \
  --output_name /home/tm0118/Desktop/example/Running_Llama/llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte
```

### 7.3) GPU(Vulkan)용 PTE export (Vulkan + 8da4w 예시)

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic

cd /home/tm0118/Desktop/executorch

python -m examples.models.llama.export_llama \
  -c /home/tm0118/Desktop/example/Running_Llama/Llama-3.2-1B-Instruct/original/consolidated.00.pth \
  -p /home/tm0118/Desktop/example/Running_Llama/Llama-3.2-1B-Instruct/original/params.json \
  -d fp32 --vulkan --vulkan-force-fp16 \
  -qmode 8da4w -G 64 \
  --max_seq_length 2048 --max_context_length 2048 \
  -kv --use_sdpa_with_kv_cache \
  --metadata '{"append_eos_to_prompt": 0, "get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --model llama3_2 \
  --output_name /home/tm0118/Desktop/example/Running_Llama/llama3_2_instruct_vulkan_8da4w_g64_c2048.pte
```

### 7.4) Android 러너 빌드 시 Vulkan backend 포함하기

이 프로젝트 `CMakeLists.txt`에는 `LLAMA_BUILD_VULKAN` 옵션이 있습니다.
Vulkan PTE를 실행하려면 Android 빌드할 때 이 옵션을 켜세요.

```bash
source /home/tm0118/anaconda3/etc/profile.d/conda.sh
conda activate basic

cd /home/tm0118/Desktop/example/Running_Llama

rm -rf build-android
cmake -S . -B build-android -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_VULKAN=ON \
  -DCMAKE_TOOLCHAIN_FILE=/home/tm0118/android-ndk-r27d/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34

cmake --build build-android -j
```

### 7.5) 폰에서 CPU PTE / Vulkan PTE를 각각 실행

둘 다 같은 `llama_chat` 바이너리로 실행하고, `--model_path`만 바꿔서 비교합니다.

```bash
adb shell "mkdir -p /data/local/tmp/running_llama"

adb push /home/tm0118/Desktop/example/Running_Llama/build-android/llama_chat /data/local/tmp/running_llama/
adb push /home/tm0118/Desktop/example/Running_Llama/Llama-3.2-1B-Instruct/original/tokenizer.model /data/local/tmp/running_llama/
adb push /home/tm0118/Desktop/example/Running_Llama/llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte /data/local/tmp/running_llama/
adb push /home/tm0118/Desktop/example/Running_Llama/llama3_2_instruct_vulkan_8da4w_g64_c2048.pte /data/local/tmp/running_llama/
adb shell "chmod +x /data/local/tmp/running_llama/llama_chat"
```

CPU 실행:

```bash
adb shell "cd /data/local/tmp/running_llama && export LD_LIBRARY_PATH=\\$LD_LIBRARY_PATH:. && ./llama_chat \
  --model_path=./llama3_2_instruct_cpu_xnnpack_8da4w_g64_c2048.pte \
  --tokenizer_path=./tokenizer.model \
  --temperature=0 --max_new_tokens=128 --warmup \
  --prompt='<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Write one sentence about penguins.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'"
```

Vulkan 실행:

```bash
adb shell "cd /data/local/tmp/running_llama && export LD_LIBRARY_PATH=\\$LD_LIBRARY_PATH:. && ./llama_chat \
  --model_path=./llama3_2_instruct_vulkan_8da4w_g64_c2048.pte \
  --tokenizer_path=./tokenizer.model \
  --temperature=0 --max_new_tokens=128 --warmup \
  --prompt='<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>Write one sentence about penguins.<|eot_id|><|start_header_id|>assistant<|end_header_id|>'"
```

### 7.6) tokens/sec 계산 방법

`llama_chat`는 마지막에 `PyTorchObserver {...}` JSON을 출력합니다.
여기서 아래 값을 사용하면 됩니다:

- `generated_tokens`
- `prompt_eval_end_ms`
- `inference_end_ms`

생성 구간 tokens/sec:

\( \text{tps} = \frac{\text{generated\_tokens}}{(\text{inference\_end\_ms}-\text{prompt\_eval\_end\_ms})/1000} \)


