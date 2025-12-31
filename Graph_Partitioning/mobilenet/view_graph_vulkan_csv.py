import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.vulkan import VulkanPartitioner
from executorch.exir import to_edge
import csv
import os

# 1. 모델 준비
print("Loading MobileNetV2 model...")
model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# 2. PyTorch Export 및 Edge Dialect 변환
print("Exporting to Edge Dialect...")
exported_program = torch.export.export(model, sample_inputs)
edge_program = to_edge(exported_program)

# 3. Vulkan Partitioner를 사용하여 백엔드 할당 정보 파악
print("Analyzing Backend Partitioning for Vulkan (GPU)...")
# VulkanPartitioner는 옵션 설정이 필요할 수 있으나 기본값으로 생성합니다.
partitioner = VulkanPartitioner()
partition_result = partitioner.partition(edge_program.exported_program())
tagged_program = partition_result.tagged_exported_program

# 백엔드로 넘어갈 노드들을 세트로 저장
delegated_nodes = set()
for node in tagged_program.graph_module.graph.nodes:
    if "delegation_tag" in node.meta:
        delegated_nodes.add(node)

# 4. CSV 저장 (모든 개별 연산 상세 출력)
csv_file = "vulkan_backend_assignment.csv"
print(f"Generating FULL CSV for Vulkan: {csv_file}")

with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # 헤더 작성
    writer.writerow(["Order", "Node Name", "ATen Operator", "Backend Assignment", "Is Delegated?"])
    
    order = 1
    for node in tagged_program.graph_module.graph.nodes:
        # 입출력 노드는 제외하고 실제 연산만 기록
        if node.op in ["placeholder", "output"]:
            continue
            
        is_delegated = node in delegated_nodes
        backend = "Vulkan (GPU Accelerator)" if is_delegated else "ExecuTorch (Portable CPU)"
        
        # 연산 타입 정리
        op_target = str(node.target)
        
        writer.writerow([
            order, 
            node.name, 
            op_target, 
            backend, 
            "YES" if is_delegated else "NO"
        ])
        order += 1

print(f"Success! Vulkan report saved to: {os.path.abspath(csv_file)}")
print(f"Total nodes analyzed: {order - 1}")

