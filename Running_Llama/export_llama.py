import os
import sys

# 1. executorch 경로를 시스템 경로에 추가 (모듈 인식 문제 해결)
EXECUTORCH_ROOT = os.path.expanduser("~/Desktop/executorch")
sys.path.append(EXECUTORCH_ROOT)

from omegaconf import OmegaConf
from executorch.extension.llm.export.config.llm_config import LlmConfig
from executorch.examples.models.llama.export_llama_lib import export_llama

def run_export():
    # 2. 경로 설정
    checkpoint_path = os.path.expanduser(
        "~/Desktop/example/Running_Llama/Llama-3.2-1B-Instruct/original/consolidated.00.pth"
    )
    params_path = os.path.expanduser(
        "~/Desktop/example/Running_Llama/Llama-3.2-1B-Instruct/original/params.json"
    )
    config_file = os.path.join(EXECUTORCH_ROOT, "examples/models/llama/config/llama_bf16.yaml")

    print(f"Loading config from: {config_file}")
    
    # 3. 설정 구성 (기본 구조 + YAML 로드)
    structured_config = OmegaConf.structured(LlmConfig)
    yaml_config = OmegaConf.load(config_file)

    # 4. 사용자 오버라이드 (터미널에서 입력하던 값들)
    overrides = {
        "base": {
            "model_class": "llama3_2",
            "checkpoint": checkpoint_path,
            "params": params_path,
        }
        ,
        "export": {
            "output_dir": os.path.expanduser("~/Desktop/example/Running_Llama"),
            "output_name": "llama3_2_instruct_bf16.pte",
        },
    }
    override_config = OmegaConf.create(overrides)

    # 5. 모든 설정 병합
    merged_config = OmegaConf.merge(structured_config, yaml_config, override_config)
    print(merged_config)
    # 6. Export 실행
    print("Starting Export... (This may take a few minutes)")
    llm_config_obj = OmegaConf.to_object(merged_config)
    export_llama(llm_config_obj)
    
    print("\n✅ Success! llama3_2.pte file has been created.")

if __name__ == "__main__":
    run_export()