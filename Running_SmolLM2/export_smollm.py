import os
import sys
import torch

# 1. executorch 경로 설정
EXECUTORCH_ROOT = os.path.expanduser("~/Desktop/executorch")
sys.path.insert(0, EXECUTORCH_ROOT)

from omegaconf import OmegaConf
from executorch.extension.llm.export.config.llm_config import LlmConfig
from executorch.examples.models.llama.export_llama_lib import export_llama
from executorch.examples.models.smollm2.convert_weights import convert_weights

def run_export():
    # 2. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    hf_ckpt_dir = os.path.join(current_dir, "SmolLM2-135M-Instruct")
    converted_ckpt_path = os.path.join(current_dir, "smollm2_converted.pth")
    params_path = os.path.join(EXECUTORCH_ROOT, "examples/models/smollm2/135M_config.json")
    config_file = os.path.join(EXECUTORCH_ROOT, "examples/models/llama/config/llama_bf16.yaml")

    # 3. 가중치 변환
    if not os.path.exists(converted_ckpt_path):
        print(f"🔄 Converting weights...")
        convert_weights(hf_ckpt_dir, converted_ckpt_path)

    # 4. 설정 및 오버라이드
    structured_config = OmegaConf.structured(LlmConfig)
    yaml_config = OmegaConf.load(config_file)

    overrides = {
        "base": {
            "model_class": "smollm2",
            "checkpoint": converted_ckpt_path,
            "params": params_path,
            # [핵심] 토크나이저 경로를 지정해야 모델이 종료 토큰(EOS) 정보를 저장합니다.
            "tokenizer_path": os.path.join(hf_ckpt_dir, "tokenizer.json"),
            # [핵심] SmolLM2-Instruct의 종료 토큰 ID를 명시적으로 입력합니다.
            "metadata": '{"get_bos_id":1, "get_eos_ids":[2, 0]}',
        },
        "export": {
            "output_dir": current_dir,
            "output_name": "smollm2_instruct_135M_bf16.pte",
        },
    }
    
    # 5. 실행
    merged_config = OmegaConf.merge(structured_config, yaml_config, OmegaConf.create(overrides))
    print("🚀 Re-exporting with Tokenizer info...")
    export_llama(OmegaConf.to_object(merged_config))
    print(f"\n✅ Done! File created: {overrides['export']['output_name']}")

if __name__ == "__main__":
    run_export()