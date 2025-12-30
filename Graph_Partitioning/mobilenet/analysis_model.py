"""
Minimal example: export a torchvision MobileNetV2 to ExecuTorch and delegate ops
to XNNPACK via graph partitioning.

Output:
  - model.pte (ignored by repo-wide .gitignore)
"""

import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower


def main() -> None:
    model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    sample_inputs = (torch.randn(1, 3, 224, 224),)

    exported = torch.export.export(model, sample_inputs)
    et_program = (
        to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
        .to_executorch()
    )

    with open("model.pte", "wb") as f:
        f.write(et_program.buffer)

    print("Wrote model.pte")


if __name__ == "__main__":
    main()


