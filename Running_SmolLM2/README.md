# Running SmolLM2-135M-Instruct (ExecuTorch)

이 프로젝트는 **SmolLM2-135M-Instruct** 모델을 ExecuTorch용 `.pte` 파일로 변환하고, C++ 러너(`llama_chat`)를 통해 PC(WSL/Linux) 및 안드로이드에서 실행하는 예제입니다.

---

## 0) 전제 조건

- **ExecuTorch 소스**: `/home/tm0118/Desktop/executorch`
- **Conda 환경**: `basic` 환경 (CMake 3.29+ 필수)
  ```bash
  source /home/tm0118/anaconda3/etc/profile.d/conda.sh
  conda activate basic
  ```
- **모델 가중치**: `SmolLM2-135M-Instruct/` 폴더 내에 `model.safetensors`가 포함되어 있어야 합니다.

---

## 1) 모델 Export (PTE 생성)

`export_smollm.py`는 `safetensors` 가중치를 `.pth`로 자동 변환한 뒤 최종적으로 `.pte` 파일을 생성합니다.

```bash
# Running_SmolLM2 폴더에서 실행
python export_smollm.py
```

- **생성물**: `smollm2_instruct_135M_bf16.pte`
- **특징**: 메타데이터에 SmolLM2 전용 종료 토큰(ID: 2, 0)이 포함되어 답변이 끝나면 자동으로 실행이 멈춥니다.

---

## 2) C++ 러너 빌드 (llama_chat)

ExecuTorch의 Llama 예제 엔진을 사용하여 빌드합니다.

### PC (WSL/Linux) 빌드
```bash
mkdir -p build && cd build
cmake -G Ninja .. -DCMAKE_BUILD_TYPE=Release
ninja
cd ..
```

### 안드로이드 (arm64-v8a) 빌드
```bash
rm -rf build-android
cmake -S . -B build-android -G Ninja -DHOST_BUILD=OFF
cmake --build build-android -j
```

---

## 3) 실행 명령어

### PC (WSL/Linux)에서 실행
SmolLM2는 ChatML 포맷을 사용하므로 `assistant` 태그 뒤에 **줄바꿈(`\n`)**을 넣는 것이 매우 중요합니다.

```bash
./build/llama_chat \
  --model_path=./smollm2_instruct_135M_bf16.pte \
  --tokenizer_path=./SmolLM2-135M-Instruct/tokenizer.json \
  --prompt="<|im_start|>user
Where is a capital of Korea?<|im_end|>
<|im_start|>assistant
" \
  --max_new_tokens=100 \
  --temperature=0
```

### 안드로이드 (adb shell)에서 실행
```bash
# 파일 전송
adb shell "mkdir -p /data/local/tmp/smollm2"
adb push ./build-android/llama_chat /data/local/tmp/smollm2/
adb push ./smollm2_instruct_135M_bf16.pte /data/local/tmp/smollm2/
adb push ./SmolLM2-135M-Instruct/tokenizer.json /data/local/tmp/smollm2/

# 권한 부여 및 실행
adb shell "chmod +x /data/local/tmp/smollm2/llama_chat"
adb shell "cd /data/local/tmp/smollm2 && ./llama_chat \
  --model_path=./smollm2_instruct_135M_bf16.pte \
  --tokenizer_path=./tokenizer.json \
  --prompt='<|im_start|>user
Where is a capital of Korea?<|im_end|>
<|im_start|>assistant
'"
```

---

## 4) 주요 분석 및 참고 사항

- **토크나이저 호환성**: `tokenizer.json` 로드 시 `Loaded 0 BPE merge rules` 로그가 뜰 수 있습니다. 이는 C++ 엔진의 한계이나 단어장(Vocab)은 로드되므로 일반적인 대화에는 지장이 없습니다.
- **자동 종료(EOS)**: 모델이 `<|im_end|>`를 출력하면 `llama_chat`이 이를 감지하고 루프를 즉시 종료합니다.
- **속도 비교**: `PyTorchObserver` 로그의 `generated_tokens`와 시간을 사용하여 CPU vs GPU 성능을 측정할 수 있습니다.

---
**This project was created by Taemin Kim, an M.S. student at the Mobile Embedded Systems Lab, Korea University.**

