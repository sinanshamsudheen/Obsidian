# Top 5 GSoC Preparation Issues for `huggingface/datasets`

  

Here are the selected high-value issues for a strong GSoC profile, focusing on core library features (streaming, Arrow integration, formats) and distributed training.

  

## 1. Memory leak when streaming

- **Issue URL:** [https://github.com/huggingface/datasets/issues/7269](https://github.com/huggingface/datasets/issues/7269)

- **Labels:** `bug` (implicit), `streaming`

- **Area:** Core / Streaming

- **Why this issue is high-value for GSoC:** Memory management in streaming datasets is critical for training on infinite data. Investigating this demonstrates you can debug complex iterator interactions, memory pinning, and potentially C++ Arrow bindings.

- **Estimated effort:** Medium-High

- **Risk level:** Medium. Memory leaks can be elusive; you might need to use `memray` or `tracemalloc` to isolate whether it's Python object retention or an underlying Arrow issue.

- **Recommended first action:** Start implementation immediately (Reproduce the leak with the provided script, then profile it).

  

## 2. Dataset load_from_disk is too slow

- **Issue URL:** [https://github.com/huggingface/datasets/issues/2547](https://github.com/huggingface/datasets/issues/2547)

- **Labels:** `bug`, `performance`

- **Area:** Core / IO

- **Why this issue is high-value for GSoC:** Touches the fundamental "load" path. The reporter notes inefficient CPU usage (1 core vs 96). Improvements here (e.g., better multi-threading in `ArrowDataset.load_from_disk` or optimizing memory mapping) affect every user of the library.

- **Estimated effort:** Medium-High

- **Risk level:** Medium. Requires benchmarking and understanding the Arrow serialization format deeply.

- **Recommended first action:** Comment with approach (analyze where the bottleneck is—I/O wait vs GIL—and propose a specific optimization like threaded reading).

  

## 3. Support Apache TsFile Datasets

- **Issue URL:** [https://github.com/huggingface/datasets/issues/7922](https://github.com/huggingface/datasets/issues/7922)

- **Labels:** `enhancement`, `new-format`

- **Area:** Formats / Time-Series

- **Why this issue is high-value for GSoC:** A self-contained "New Feature" task. You will implement a new format reader (`TsFile`) and likely a builder. This shows you understand the library's extension points (`BuilderConfig`, `ArrowWriter`). The maintainers and community (compal) are already supportive.

- **Estimated effort:** Medium

- **Risk level:** Low. The specification is clear, and it's additive code rather than modifying existing logic.

- **Recommended first action:** Comment with approach (ask for assignment and propose a `TsFileBuilder`).

  

## 4. Using Stateful Dataloader with Split Dataset By Node and DCP for DDP

- **Issue URL:** [https://github.com/huggingface/datasets/issues/7927](https://github.com/huggingface/datasets/issues/7927)

- **Labels:** `distributed`, `pytorch`

- **Area:** Integrations / PyTorch

- **Why this issue is high-value for GSoC:** Solves a problem at the intersection of `datasets` and modern PyTorch distributed training (`torch.distributed.checkpoint`). Implementing `state_dict` for split datasets is complex but crucial for fault-tolerant training.

- **Estimated effort:** High

- **Risk level:** Medium. Requires a solid setup for testing distributed code (DDP).

- **Recommended first action:** Comment with approach (confirm the expected behavior for saving state when data is sharded by node).

  

## 5. Support cast_kwargs in cast_columns

- **Issue URL:** [https://github.com/huggingface/datasets/issues/7909](https://github.com/huggingface/datasets/issues/7909)

- **Labels:** `enhancement`, `usability`

- **Area:** Core / Schema

- **Why this issue is high-value for GSoC:** Consistency fix. `cast()` supports arguments (like `batch_size` to avoid OOMs on large rows), but `cast_column` does not expose them. Fixing this involves updating the API surface and ensuring backward compatibility.

- **Estimated effort:** Low-Medium

- **Risk level:** Low. clear path to success.

- **Recommended first action:** Start implementation immediately.