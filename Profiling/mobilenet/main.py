from executorch.devtools import Inspector
import os
import pandas as pd

# 1. 파일 경로 설정
etdump_path = "etdump.etdp"
etrecord_path = "etrecord.bin"

# 2. 파일 존재 확인
if not os.path.exists(etdump_path) or not os.path.exists(etrecord_path):
    print("Error: etdump.etdp or etrecord.bin missing.")
    exit(1)

# 3. Inspector 로드
print("Analyzing performance with Inspector...")
inspector = Inspector(etdump_path=etdump_path, etrecord=etrecord_path)

# 4. 데이터프레임 추출
df = inspector.to_dataframe()

# 컬럼명이 다를 수 있으므로 현재 존재하는 컬럼 확인 및 매핑
print(f"Available columns: {df.columns.tolist()}")

# 최대한 많은 정보를 담기 위해 모든 컬럼 저장 시도
# 다만 가독성을 해치는 너무 큰 컬럼(raw data 등)은 제외
excluded = ['raw', 'stack_traces', 'module_hierarchy', 'debug_data']
cols = [c for col in df.columns if (c := str(col)) not in excluded]

# 가독성을 위해 순서 재배치 (존재하는 것만)
preferred_order = ['name', 'op_type', 'average_latency_ms', 'p90_latency_ms', 'is_delegated_op', 'backend_name']
final_cols = [c for c in preferred_order if c in df.columns]
# 나머지 컬럼들도 뒤에 붙임
final_cols += [c for c in cols if c not in final_cols]

# CSV 저장
csv_output_path = "profiling_results.csv"
df[final_cols].to_csv(csv_output_path, index=False)

print(f"\nSuccess! Detailed profiling results saved to: {os.path.abspath(csv_output_path)}")

# 터미널에도 요약 출력
print("\n=== TOP 5 BOTTLENECKS ===")
if 'average_latency_ms' in df.columns:
    print(df.sort_values(by='average_latency_ms', ascending=False)[['name', 'average_latency_ms']].head(5))
else:
    # 컬럼명이 다른 경우 (예: 'p90_latency_ms'만 있는 경우 등) 처리
    latency_col = [c for c in df.columns if 'latency' in str(c)]
    if latency_col:
        print(df.sort_values(by=latency_col[0], ascending=False)[['name', latency_col[0]]].head(5))
