# Graph_Partitioning

Tiny scripts focused on **graph partitioning** (a.k.a. delegation):

- Export a model with `torch.export`
- Apply `to_edge_transform_and_lower(..., partitioner=[...])`
- Verify/observe what parts of the graph are delegated to a backend (e.g., XNNPACK)

> Generated `.pte` files are ignored by git (see `example/.gitignore`).

---

## `mobilenet/`

- `analysis_model.py`
  - Exports torchvision MobileNetV2
  - Lowers/delegates with `XnnpackPartitioner()`
  - Writes `model.pte`

Run:

```bash
cd /home/tm0118/Desktop/example/Graph_Partitioning/mobilenet
python analysis_model.py
```





