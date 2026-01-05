# Top 5 GSoC Preparation Issues for `mlflow/mlflow`

Here are the selected high-value issues for a strong GSoC profile.

## 1. [BUG] Traces are not nested when using Prompt Optimization with async predict_fn
- **Issue URL:** [https://github.com/mlflow/mlflow/issues/19536](https://github.com/mlflow/mlflow/issues/19536)
- **Labels:** `area/tracing`, `area/evaluation`, `bug`
- **Area:** Tracing / GenAI
- **Why this issue is high-value for GSoC:** This task requires diving into MLflow's modern GenAI tracing internals and understanding `asyncio` context propagation. Fixing this demonstrates strong debugging skills in concurrent Python code and understanding of distributed tracing spans.
- **Estimated effort:** Medium-High
- **Risk level:** Medium. Debugging async context leakage can be subtle and might require deep investigation into `nest_asyncio` or MLflow's span context manager.
- **Recommended first action:** Comment with approach (propose a reproduction with a simplified async context to isolate the break).

## 2. [FR] Extend S3 presigned URL support to all file sizes with server-side enforcement
- **Issue URL:** [https://github.com/mlflow/mlflow/issues/19731](https://github.com/mlflow/mlflow/issues/19731)
- **Labels:** `area/tracking`, `domain/platform`, `enhancement`
- **Area:** Artifacts / Server Infrastructure
- **Why this issue is high-value for GSoC:** It touches `s3_artifact_repo.py` and the core artifact handling logic, critical for MLflow's scalability. Implementing this proves you understand how MLflow authenticates and delegates storage operations for enterprise-scale data.
- **Estimated effort:** Medium
- **Risk level:** Low. The pattern already exists for multipart uploads; this is a logical extension to single-file uploads.
- **Recommended first action:** Comment with approach (acknowledge the reporter's design and propose the specific changes to `S3ArtifactRepository`).

## 3. mlflow ui crashes with assert scope["type"] == "http" for all tracking URIs
- **Issue URL:** [https://github.com/mlflow/mlflow/issues/19643](https://github.com/mlflow/mlflow/issues/19643)
- **Labels:** `bug`, `area/server-infra`, `area/uiux`
- **Area:** Server Infrastructure
- **Why this issue is high-value for GSoC:** A critical crash preventing UI launch. Solving this requires understanding MLflow's ASGI/Gateway server layer (`server/handlers.py` or similar). It shows you can debug fundamental infrastructure issues that affect all users.
- **Estimated effort:** Medium
- **Risk level:** Medium. The root cause might be an external dependency update (e.g., Starlette/FastAPI) or an environment mismatch, requiring careful isolation.
- **Recommended first action:** Start implementation immediately (Reproduce locally first, then push a fix).

## 4. [FR] Can't specify params when running inference on MLServer
- **Issue URL:** [https://github.com/mlflow/mlflow/issues/19667](https://github.com/mlflow/mlflow/issues/19667)
- **Labels:** `area/scoring`, `enhancement`
- **Area:** Scoring / Deployment
- **Why this issue is high-value for GSoC:** Gap in the MLServer integration. Fixing this requires modifying how MLflow translates inference requests and validates inputs (checking `proto_json_utils`). It demonstrates understanding of model serving protocols.
- **Estimated effort:** Low-Medium
- **Risk level:** Low. The error message explicitly points to validation logic that is too strict (`One of "instances" and "inputs" must be specified`).
- **Recommended first action:** Start implementation immediately.

## 5. Add Tracking of System Metrics of the current Process
- **Issue URL:** [https://github.com/mlflow/mlflow/issues/12916](https://github.com/mlflow/mlflow/issues/12916)
- **Labels:** `area/tracking`, `enhancement`
- **Area:** Tracking / Observability
- **Why this issue is high-value for GSoC:** Enhances the system metrics monitoring thread to capture process-specific stats (CPU/Mem of *this* run). Requires working with `psutil` within MLflow's background metric collection loop, a core "robustness" feature.
- **Estimated effort:** Medium
- **Risk level:** Low. The scope is well-defined and isolated to the metrics collection module.
- **Recommended first action:** Comment with approach (Confirm metric naming capability to avoid collision with global metrics).