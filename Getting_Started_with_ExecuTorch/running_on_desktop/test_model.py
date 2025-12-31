import torch
from executorch.runtime import Runtime
from typing import List
import time

from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
import torchvision.models as models

runtime = Runtime.get()

input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
program = runtime.load_program("./model.pte")
method = program.load_method("forward")

warmup_iters = 5
iters = 50

for _ in range(warmup_iters):
    _ = method.execute([input_tensor])

times_ms: List[float] = []
for _ in range(iters):
    t0 = time.perf_counter()
    output: List[torch.Tensor] = method.execute([input_tensor])
    t1 = time.perf_counter()
    times_ms.append((t1 - t0) * 1000.0)

avg_ms = sum(times_ms) / len(times_ms)
min_ms = min(times_ms)
max_ms = max(times_ms)
print(f"ExecuTorch forward: avg/min/max = {avg_ms:.3f}/{min_ms:.3f}/{max_ms:.3f} ms over {iters} iters (warmup {warmup_iters})")

eager_reference_model = models.mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
with torch.no_grad():
    eager_reference_output = eager_reference_model(input_tensor)

print("Comparing against original PyTorch module")
print(torch.allclose(output[0], eager_reference_output, rtol=1e-3, atol=1e-5))