# Getting_Started_with_ExecuTorch

Beginner-friendly, minimal examples to export and run a model with ExecuTorch.

> Note: generated artifacts like `model.pte` are ignored by git (see `example/.gitignore`).

---

## `running_on_desktop/`

**What it does**

- `export_model.py`: exports a small model to `model.pte`
- `test_model.py`: loads `model.pte` via `executorch.runtime.Runtime` and runs/benchmarks it

**Typical flow**

```bash
cd /home/tm0118/Desktop/example/Getting_Started_with_ExecuTorch/running_on_desktop
python export_model.py
python test_model.py
```

If you see import errors, install the missing deps in your Python environment (torch/torchvision/executorch runtime bindings, etc.).


