# Model_Export_and_Lowering

Small scripts demonstrating the core ExecuTorch export pipeline:

1) `torch.export` (PyTorch graph capture)
2) `to_edge_transform_and_lower(...)` (apply transforms + backend lowering)
3) produce an ExecuTorch program buffer (`.pte`)

> This folder may contain local `model.pte` / `model.ptd` artifacts, but they are ignored by git (see `example/.gitignore`).

---

## Files

- `export_model.py`
  - Minimal export to `model.pte`
  - Uses `XnnpackPartitioner()` to delegate a subset of ops to XNNPACK
- `export_model_with_ptd.py`
  - Same as above but demonstrates **external constants**:
    - Writes the program to `model.pte`
    - Writes weights/constants to a separate `model.ptd`
- `test_pte.py`
  - Loads `model.pte` with `executorch.runtime.Runtime` and runs `forward`
- `test_ptd.py`
  - Loads `model.pte` + `model.ptd` with `portable_lib._load_for_executorch(...)`

---

## Quick start

```bash
cd /home/tm0118/Desktop/example/Model_Export_and_Lowering

# Export (choose one)
python export_model.py
# python export_model_with_ptd.py

# Run (match to what you exported)
python test_pte.py
# python test_ptd.py
```





