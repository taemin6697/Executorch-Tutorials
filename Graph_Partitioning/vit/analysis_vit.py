import torch
from executorch.runtime import Runtime
from typing import List
import time
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights

def main():
    runtime = Runtime.get()

    # 1. 입력 텐서 준비
    input_tensor: torch.Tensor = torch.randn(1, 3, 224, 224)
    
    # 2. ExecuTorch 프로그램 로드
    print("Loading ExecuTorch program...")
    program = runtime.load_program("model.pte")
    method = program.load_method("forward")

    # 3. 성능 측정 (Benchmarking)
    warmup_iters = 3
    iters = 10
    
    print(f"Starting warmup ({warmup_iters} iters)...")
    for _ in range(warmup_iters):
        _ = method.execute([input_tensor])

    print(f"Starting inference ({iters} iters)...")
    times_ms: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        output: List[torch.Tensor] = method.execute([input_tensor])
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    print(f"ExecuTorch ViT forward: avg = {avg_ms:.3f} ms over {iters} iters")

    # 4. 결과 검증 (Correctness Check)
    print("Verifying against original PyTorch module...")
    eager_model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()
    with torch.no_grad():
        eager_output = eager_model(input_tensor)

    is_close = torch.allclose(output[0], eager_output, rtol=1e-3, atol=1e-4)
    print(f"Correctness Check: {'PASS' if is_close else 'FAIL'}")

if __name__ == "__main__":
    main()

