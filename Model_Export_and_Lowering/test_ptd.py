import torch
from executorch.extension.pybindings import portable_lib

input_tensor = torch.randn(1, 3, 32, 32)
module = portable_lib._load_for_executorch("model.pte", "model.ptd")
outputs = module.forward([input_tensor])
print(outputs)