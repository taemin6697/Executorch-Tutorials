import torch
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

def main() -> None:
    # 1. 모델 준비 (ViT-B/16)
    print("Preparing ViT-B/16 model...")
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    # 2. ExecuTorch 내보내기 (XNNPACK 파티셔너 적용)
    print("Exporting model to ExecuTorch with XNNPACK...")
    exported = torch.export.export(model, sample_inputs)
    et_program = (
        to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        .to_executorch()
    )

    # 3. .pte 파일 저장
    output_path = "model.pte"
    with open(output_path, "wb") as f:
        f.write(et_program.buffer)

    print(f"Success! Model saved to {output_path}")

if __name__ == "__main__":
    main()

