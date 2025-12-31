import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower

# 1. 모델 준비
print("Loading MobileNetV2 model...")
model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# 2. PyTorch Export 및 Edge Dialect 변환
print("Exporting to Edge Dialect...")
exported_program = torch.export.export(model, sample_inputs)
edge_program = to_edge(exported_program)

# 3. Partitioner를 사용하여 백엔드 할당 정보 파악
print("Analyzing Backend Partitioning...")
partitioner = XnnpackPartitioner()
partition_result = partitioner.partition(edge_program.exported_program())
tagged_program = partition_result.tagged_exported_program

# 4. 상세 노드별 백엔드 할당 리포트 출력
print("\n" + "="*160)
print(f"{'Node Name':<60} | {'ATen Operator':<60} | {'Assigned Backend'}")
print("-" * 160)

assigned_nodes_count = 0
total_nodes_count = 0

delegated_nodes = set()
for node in tagged_program.graph_module.graph.nodes:
    if node.op in ["placeholder", "output"]:
        continue
    total_nodes_count += 1
    backend_name = "ExecuTorch (Portable)"
    if "delegation_tag" in node.meta:
        backend_name = "\033[92mXNNPACK (Delegated)\033[0m"
        assigned_nodes_count += 1
        delegated_nodes.add(node)
    
    print(f"{node.name:<60} | {str(node.target):<60} | {backend_name}")

print("-" * 160)
print(f"SUMMARY: {assigned_nodes_count} / {total_nodes_count} nodes delegated to XNNPACK ({assigned_nodes_count/total_nodes_count*100:.1f}%)")
print("="*160 + "\n")

# 5. 실제 Lowering 후의 압축된 그래프 구조 요약
final_edge_program = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()])
final_gm = final_edge_program.exported_program().graph_module

print("=== Final Execution Graph (Collapsed) ===")
for node in final_gm.graph.nodes:
    if node.op == "call_function":
        target = str(node.target)
        if "call_delegate" in target:
            print(f"Node: {node.name:30} -> [DELEGATED SUBGRAPH] (Backend: XNNPACK)")
        else:
            print(f"Node: {node.name:30} -> {target}")

# 6. 시각화 (Graphviz)
print("\nGenerating graph visualization...")
try:
    from torch.fx.passes.graph_drawer import FxGraphDrawer
    import os

    # 최종 압축된 그래프 시각화
    drawer = FxGraphDrawer(final_gm, "MobileNetV2_ExecuTorch")
    dot_graph = drawer.get_dot_graph()
    
    # .dot 파일 저장
    dot_file = "model_graph.dot"
    with open(dot_file, "w") as f:
        f.write(dot_graph.to_string())
    print(f"Graph definition saved to '{dot_file}'")

    # PNG 변환 시도 (시스템에 'dot' 명령어가 있어야 함)
    png_file = "model_graph.png"
    try:
        dot_graph.write_png(png_file)
        print(f"Success! Visualization saved as '{png_file}'")
    except Exception as e:
        print(f"Could not generate PNG (perhaps 'dot' command is missing): {e}")
        print(f"Tip: You can copy the content of '{dot_file}' to https://dreampuf.github.io/GraphvizOnline/ to see the graph.")

except ImportError:
    print("Visualization requires 'graphviz' python package. (pip install graphviz)")
except Exception as e:
    print(f"Visualization failed: {e}")
