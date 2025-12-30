import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.passes.external_constants_pass import (
    delegate_external_constants_pass_unlifted,
)
from torch.export import Dim, export

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1))
        )
        self.linear = torch.nn.Linear(16, 10)

    def forward(self, x):
        y = self.seq(x)
        y = torch.flatten(y, 1)
        y = self.linear(y)
        return y

model = Model().eval()
inputs = (torch.randn(1,1,16,16),)
dynamic_shapes = {
    "x": {
        2: Dim("h", min=16, max=1024),
        3: Dim("w", min=16, max=1024),
    }
}

exported_program = export(model, inputs, dynamic_shapes=dynamic_shapes)

# Tag constants/weights as external so they are saved into a separate .ptd file.
tagged_module = exported_program.module()
delegate_external_constants_pass_unlifted(
    module=tagged_module,
    gen_tag_fn=lambda x: "model",  # weights will be saved as "model.ptd"
)

# Re-export after tagging to get an ExportedProgram reflecting external constants.
exported_program = export(tagged_module, inputs, dynamic_shapes=dynamic_shapes)

executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [XnnpackPartitioner()]
).to_executorch()

with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)

# Save weights/constants to model.ptd in the current directory.
executorch_program.write_tensor_data_to_file(".")