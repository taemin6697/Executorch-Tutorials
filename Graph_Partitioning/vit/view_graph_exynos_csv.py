import torch
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights
from executorch.backends.samsung.partition.enn_partitioner import EnnPartitioner
from executorch.backends.samsung.serialization.compile_options import gen_samsung_backend_compile_spec
from executorch.exir import to_edge
import csv
import os

def generate_csv():
    # 1. 모델 준비
    print("Loading ViT-B/16 model...")
    model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224), )

    # 2. Edge Dialect 변환
    exported_program = torch.export.export(model, sample_inputs)
    edge_program = to_edge(exported_program)

    # 3. Exynos Partitioner 분석
    print("Analyzing Backend Partitioning for Exynos (NPU)...")
    chipset = "E9955"
    compile_specs = [gen_samsung_backend_compile_spec(chipset)]
    partitioner = EnnPartitioner(compile_specs)
    
    try:
        partition_result = partitioner.partition(edge_program.exported_program())
        tagged_program = partition_result.tagged_exported_program

        delegated_nodes = set()
        for node in tagged_program.graph_module.graph.nodes:
            if "delegation_tag" in node.meta:
                delegated_nodes.add(node)

        # 4. CSV 저장
        csv_file = "exynos_backend_assignment.csv"
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Order", "Node Name", "ATen Operator", "Backend Assignment", "Is Delegated?"])
            
            order = 1
            for node in tagged_program.graph_module.graph.nodes:
                if node.op in ["placeholder", "output"]: continue
                is_delegated = node in delegated_nodes
                backend = f"Samsung ENN (NPU - {chipset})" if is_delegated else "ExecuTorch (Portable CPU)"
                writer.writerow([order, node.name, str(node.target), backend, "YES" if is_delegated else "NO"])
                order += 1

        print(f"Success! Exynos report saved to: {os.path.abspath(csv_file)}")
    except Exception as e:
        print(f"Exynos partitioning failed: {e}")

if __name__ == "__main__":
    generate_csv()

