import torch
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge
import csv
import os

# 1. 모델 준비
print("Loading ResNet50 model...")
model = models.resnet50(weights=ResNet50_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# 2. PyTorch Export 및 Edge Dialect 변환
print("Exporting to Edge Dialect...")
exported_program = torch.export.export(model, sample_inputs)
edge_program = to_edge(exported_program)

# 3. Partitioner 분석
print("Analyzing Backend Partitioning for XNNPACK (CPU)...")
partitioner = XnnpackPartitioner()
partition_result = partitioner.partition(edge_program.exported_program())
tagged_program = partition_result.tagged_exported_program

# 백엔드로 넘어갈 노드들을 세트로 저장
delegated_nodes = set()
for node in tagged_program.graph_module.graph.nodes:
    if "delegation_tag" in node.meta:
        delegated_nodes.add(node)

# 4. CSV 저장
csv_file = "xnnpack_backend_assignment.csv"
print(f"Generating FULL CSV for XNNPACK: {csv_file}")

with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Order", "Node Name", "ATen Operator", "Backend Assignment", "Is Delegated?"])
    
    order = 1
    for node in tagged_program.graph_module.graph.nodes:
        if node.op in ["placeholder", "output"]:
            continue
            
        is_delegated = node in delegated_nodes
        backend = "XNNPACK (Accelerator)" if is_delegated else "ExecuTorch (Portable CPU)"
        
        op_target = str(node.target)
        writer.writerow([order, node.name, op_target, backend, "YES" if is_delegated else "NO"])
        order += 1

print(f"Success! XNNPACK report saved to: {os.path.abspath(csv_file)}")
print(f"Total nodes analyzed: {order - 1}")

