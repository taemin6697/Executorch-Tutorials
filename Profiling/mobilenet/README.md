# MobileNetV2 Profiling Example

이 예제는 ExecuTorch를 사용하여 MobileNetV2 모델의 **연산별(Layer-by-layer) 실행 시간**을 정밀하게 측정하고 분석하는 방법을 보여줍니다.

## 주요 구성 파일

- `export_model.py`: MobileNetV2 모델을 XNNPACK 가속용으로 내보내고, 프로파일링 매핑 데이터인 `etrecord.bin`을 생성합니다.
- `main.py`: 모델을 실행하여 성능 데이터(`etdump.etdp`)를 수집하고, 이를 분석하여 `profiling_results.csv` 리포트를 생성합니다.
- `profiling_results.csv`: 각 레이어별 평균 실행 시간 및 백분위수(p10, p50, p90) 정보가 담긴 결과 파일입니다.

## 사전 준비 사항

정밀한 프로파일링 데이터를 수집하려면 ExecuTorch 런타임이 성능 측정 기능(`Event Tracer`)이 활성화된 상태로 빌드되어야 합니다.

```bash
# ExecuTorch 소스 폴더에서 프로파일링 활성화 빌드 예시
cd executorch
mkdir -p cmake-out && cd cmake-out
cmake .. -DEXECUTORCH_ENABLE_EVENT_TRACER=ON -DEXECUTORCH_BUILD_XNNPACK=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON
make -j4 executor_runner
```

## 실행 방법

### 1. 모델 및 메타데이터 생성
```bash
python3 export_model.py
```
- 결과물: `model.pte`, `etrecord.bin`

### 2. 성능 데이터 수집 (전용 런타임 사용 권장)
빌드된 `executor_runner`를 사용하여 실제 실행 데이터를 뽑아냅니다.
```bash
/path/to/executor_runner --model_path model.pte --etdump_path etdump.etdp
```
- 결과물: `etdump.etdp`

### 3. 결과 분석 및 CSV 리포트 생성
```bash
python3 main.py
```
- 결과물: `profiling_results.csv` 생성 및 터미널 요약 출력

## 리포트 지표 읽는 법

- **avg (ms)**: 각 연산의 평균 실행 시간입니다.
- **p50 (ms)**: 중간값 성능으로, 실제 체감되는 가장 표준적인 지표입니다.
- **p90 (ms)**: 하위 10% 성능(Tail Latency)으로, 서비스의 안정성을 확인하는 지표입니다.
- **is_delegated_op**: `True`인 경우 XNNPACK 가속기에서 실행되었음을 의미합니다.





