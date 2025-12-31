import torch
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge, to_edge_transform_and_lower
import networkx as nx
import matplotlib.pyplot as plt
import os

# 1. 모델 준비
print("Loading ViT-B/16 model...")
model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()
sample_inputs = (torch.randn(1, 3, 224, 224), )

# 2. Export
exported_program = torch.export.export(model, sample_inputs)

# 3. 최종 압축 그래프 생성 (XNNPACK 기준)
final_edge_program = to_edge_transform_and_lower(exported_program, partitioner=[XnnpackPartitioner()])
final_gm = final_edge_program.exported_program().graph_module

# 4. 시각화
print("\nGenerating graph visualization with Matplotlib...")
try:
    G = nx.DiGraph()
    color_map = []
    labels = {}

    for node in final_gm.graph.nodes:
        G.add_node(node.name)
        for user in node.users:
            G.add_edge(node.name, user.name)
        
        target_str = str(node.target)
        if node.op == "placeholder":
            color_map.append("#E3F2FD")
            labels[node.name] = f"IN\n{node.name}"
        elif node.op == "output":
            color_map.append("#FBE9E7")
            labels[node.name] = "OUT"
        elif "call_delegate" in target_str:
            color_map.append("#A5D6A7")
            labels[node.name] = f"XNNPACK\n{node.name}"
        else:
            color_map.append("#FFCDD2")
            clean_name = target_str.split('.')[-2] if '.' in target_str else target_str
            labels[node.name] = clean_name[:15]

    plt.figure(figsize=(15, 25))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(G, pos, labels=labels, with_labels=True, node_color=color_map, 
            node_size=4000, node_shape="s", font_size=7, font_weight="bold",
            edge_color="#666666", width=1.5, arrows=True, arrowsize=25)

    plt.title("ViT-B/16 ExecuTorch Execution Graph (XNNPACK)", fontsize=20)
    png_file = "vit_graph_plt.png"
    plt.savefig(png_file, bbox_inches='tight', dpi=100)
    print(f"Success! Visualization saved as '{os.path.abspath(png_file)}'")
    plt.close()

except Exception as e:
    print(f"Visualization failed: {e}")

