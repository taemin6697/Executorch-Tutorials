import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

# 1. 모델 준비
model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# 2. 모델 Export (내부적으로 generate_etrecord=True 설정)
print("Exporting model with XNNPACK partitioner...")
# 이 옵션을 사용하면 ExecuTorch가 가장 안정적인 시점에 ETRecord를 자동으로 캡처합니다.
edge_program = to_edge_transform_and_lower(
    torch.export.export(model, sample_inputs),
    partitioner=[XnnpackPartitioner()],
    generate_etrecord=True 
)
et_program = edge_program.to_executorch()

# 3. 자동 생성된 ETRecord 저장
print("Saving ETRecord...")
etrecord_path = "etrecord.bin"
etrecord = et_program.get_etrecord()
if etrecord:
    etrecord.save(etrecord_path)
    print(f"ETRecord saved to {etrecord_path}")
else:
    print("Warning: ETRecord generation failed inside the manager.")

# 4. 모델 저장
with open("model.pte", "wb") as f:
    f.write(et_program.buffer)

print("Model saved to model.pte")
