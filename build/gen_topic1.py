"""Generate Topic 1 notebook: Distributed Training in Databricks."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from nbbuild import md, code, write_notebook

cells = []

cells.append(md(r"""
# Distributed Training in Databricks
### Workshop — Topic 1 of 6

**Goal:** Understand *when* and *how* to scale deep-learning training beyond a single machine on Databricks, and run the core patterns end to end.

> ⚙️ **Recommended compute for this notebook:** a **GPU-enabled cluster** running **Databricks Runtime 14.0 ML or later** (GPU). A single-node GPU cluster is enough for Parts 1–3; a multi-node GPU cluster is needed for the multi-worker cells in Part 4.

---

### What you'll learn
1. The **single-node-first** principle — and why distributed code is a last resort, not a default.
2. **Data parallelism vs. model parallelism** — the two ways to split a training job.
3. **`TorchDistributor`** — Databricks' native way to run distributed PyTorch on Spark (single-node multi-GPU *and* multi-node).
4. **`DeepSpeedTorchDistributor`** — for models too big to fit in GPU memory (ZeRO).
5. **Ray Train** — the alternative distributor for Python-native scaling.
6. How to pick the right tool, size a GPU cluster, and track everything in **MLflow**.

*Source: Databricks docs — [Distributed training](https://docs.databricks.com/machine-learning/train-model/distributed-training/), [Deep learning](https://docs.databricks.com/machine-learning/train-model/deep-learning), [GPU compute](https://docs.databricks.com/compute/gpu).*
"""))

cells.append(md(r"""
## 1. The mental model: start single-node, scale only when forced

Databricks' explicit guidance:

> *"Databricks recommends single-machine neural network training when feasible. Distributed code for training and inference is more complex than single-machine code and slower due to communication overhead."*

So the decision flow is:

| Situation | What to do |
|---|---|
| Model + a reasonable batch fit on **1 GPU** | Train **single-node, single-GPU**. Simplest, fastest to iterate. |
| Model fits on 1 GPU but you want to **train faster / use a bigger batch** | **Data parallelism** on **1 node with multiple GPUs** (`TorchDistributor`, `local_mode=True`). |
| One node's GPUs aren't enough (data or epochs too large) | **Data parallelism across multiple nodes** (`TorchDistributor`, `local_mode=False`). |
| **Model itself doesn't fit** in a single GPU's memory | **`DeepSpeedTorchDistributor`** (ZeRO sharding) or model/pipeline parallelism. |

**Rule of thumb:** scale up (bigger GPU) before you scale out (more GPUs), and scale out on one node before going multi-node. Every hop adds communication overhead.
"""))

cells.append(md(r"""
## 2. Environment check

Confirm the runtime, GPU availability, and how many GPUs are attached to the driver. Run this first — the rest of the notebook branches on what you have.
"""))

cells.append(code(r"""
import os
import sys
import platform

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"GPUs on driver: {n}")
        for i in range(n):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU detected — TorchDistributor cells will fall back to CPU (gloo backend).")
except ImportError:
    print("PyTorch not installed. Use a Databricks Runtime ML (GPU) cluster.")
"""))

cells.append(code(r"""
# On a real Databricks cluster, inspect the worker fleet to know your total GPU budget.
# (Safe no-op when run outside Databricks.)
try:
    workers = spark.sparkContext.getConf().get("spark.databricks.clusterUsageTags.clusterWorkers", "unknown")
    print("Spark master:", spark.sparkContext.master)
    print("Default parallelism (total cores):", spark.sparkContext.defaultParallelism)
except NameError:
    print("`spark` not defined — running outside Databricks. That's fine for reading along.")
"""))

cells.append(md(r"""
## 3. Two kinds of parallelism

Before any code, get the vocabulary straight — it drives every tool choice.

### Data parallelism (the common case)
- **Every worker holds a full copy of the model.**
- The **data batch is split** across workers; each computes gradients on its shard.
- Gradients are **averaged (all-reduce)** every step so all replicas stay in sync.
- Scales throughput almost linearly until communication dominates.
- ➡️ This is what `TorchDistributor` and `Ray Train` give you by default.

### Model parallelism (the "model won't fit" case)
- **The model itself is split** across devices (by layer = *pipeline parallelism*, or by tensor = *tensor parallelism*).
- Needed when a single GPU cannot even hold the model weights + optimizer state.
- More complex; higher communication cost.
- ➡️ This is where **DeepSpeed / ZeRO** comes in — it shards optimizer state, gradients, and (optionally) parameters across GPUs.

> Most teams need **data parallelism**. Reach for model parallelism / DeepSpeed only when a single GPU can't hold the model.
"""))

cells.append(md(r"""
## 4. The training function (shared by every approach below)

The key discipline for distributed PyTorch on Databricks:

1. **Put all imports *inside* the function** — the function is pickled and shipped to workers; top-level imports won't travel and cause errors.
2. Read rank/world from environment variables that the distributor sets: `LOCAL_RANK`, `RANK`, `WORLD_SIZE`.
3. Initialize the process group with **`nccl`** (GPU) or **`gloo`** (CPU).
4. Wrap the model in **`DistributedDataParallel` (DDP)**.
5. Use a **`DistributedSampler`** so each worker sees a different data shard.
6. **Checkpoint from rank 0** with `torch.save(...)`, saving **`model.module.state_dict()`** — the `.module` unwrap gets the *underlying* model out of the DDP wrapper.
7. **Log metrics / params / the model from rank 0 only** (`if global_rank == 0:`) so N workers don't create N duplicate MLflow runs. Use `mlflow.pytorch.log_model` for the final model.
8. Run a **held-out evaluation pass on rank 0** after training and log the test metric.
9. **Clean up** the process group at the end (`dist.destroy_process_group()`).

The cell below defines a self-contained training function — mirroring the official Databricks end-to-end TorchDistributor notebook — that we'll reuse across every approach.
"""))

cells.append(code(r'''
def save_checkpoint(log_dir, ddp_model, optimizer, epoch):
    """Save a DDP checkpoint. NOTE the `.module` — it unwraps the underlying
    model from the DistributedDataParallel wrapper so the state_dict keys are clean."""
    import os
    import torch
    filepath = os.path.join(log_dir, f"checkpoint-{epoch}.pth.tar")
    state = {
        "model": ddp_model.module.state_dict(),   # <-- .module unwraps DDP
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, filepath)
    return filepath


def load_checkpoint(log_dir, epoch):
    """Reload a checkpoint saved by save_checkpoint()."""
    import os
    import torch
    filepath = os.path.join(log_dir, f"checkpoint-{epoch}.pth.tar")
    return torch.load(filepath)


def train_fn(learning_rate=1e-3, epochs=3, batch_size=64, use_gpu=True, log_dir="/tmp/torchdist"):
    """Self-contained distributed training + checkpoint + eval + MLflow logging.

    Runs identically on 1 GPU, N GPUs, or N nodes. Mirrors the official Databricks
    end-to-end TorchDistributor notebook.

    IMPORTANT: every import lives inside this function so it pickles cleanly to workers.
    """
    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    # So worker processes can reach the MLflow tracking server (see the callout below).
    # os.environ["DATABRICKS_HOST"] = db_host
    # os.environ["DATABRICKS_TOKEN"] = db_token
    try:
        import mlflow
        # mlflow.set_experiment(experiment_path)   # point workers at a known experiment
        _mlflow_ok = True
    except Exception:
        _mlflow_ok = False

    os.makedirs(log_dir, exist_ok=True)

    # --- 1. Set up the distributed process group -------------------------------
    backend = "nccl" if use_gpu else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = torch.device(f"cuda:{local_rank}" if use_gpu else "cpu")
    if use_gpu:
        torch.cuda.set_device(local_rank)

    if global_rank == 0:
        print(f"World size: {world_size} | backend: {backend} | device: {device}")
        # Log params ONCE, from rank 0 only — avoids N duplicate MLflow runs.
        if _mlflow_ok:
            mlflow.log_params({"lr": learning_rate, "epochs": epochs,
                               "batch_size": batch_size, "world_size": world_size,
                               "trainer": "TorchDistributor"})

    # --- 2. Dataset with a DistributedSampler + a held-out split ----------------
    torch.manual_seed(42)
    X = torch.randn(4096, 32)
    y = (X.sum(dim=1) > 0).long()
    split = 3584
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=global_rank)
    loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)

    # --- 3. Model wrapped in DDP ------------------------------------------------
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if use_gpu else None)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # --- 4. Training loop -------------------------------------------------------
    last_loss = None
    for epoch in range(epochs):
        sampler.set_epoch(epoch)  # reshuffle shards each epoch
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(ddp_model(xb), yb)
            loss.backward()
            optimizer.step()
        last_loss = loss.item()
        # RANK == 0 guard: only the chief logs metrics + writes checkpoints.
        if global_rank == 0:
            print(f"epoch {epoch}: loss={last_loss:.4f}")
            if _mlflow_ok:
                mlflow.log_metric("train_loss", last_loss, step=epoch)
            save_checkpoint(log_dir, ddp_model, optimizer, epoch)

    # --- 5. Evaluation + model logging on rank 0 only ---------------------------
    test_loss = None
    if global_rank == 0:
        ddp_model.eval()
        test_loader = DataLoader(test_ds, batch_size=batch_size)
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                total += loss_fn(ddp_model(xb), yb).item() * len(yb)
                n += len(yb)
        test_loss = total / max(n, 1)
        print(f"Average test loss: {test_loss:.4f}")
        if _mlflow_ok:
            mlflow.log_metric("test_loss", test_loss)
            # Log the trained model (unwrapped) so it can be reloaded from the registry.
            mlflow.pytorch.log_model(ddp_model.module, "model")

    # --- 6. Clean up ------------------------------------------------------------
    dist.destroy_process_group()

    # Only rank 0 returns the artifact you care about.
    return {"final_loss": last_loss, "test_loss": test_loss,
            "world_size": world_size, "log_dir": log_dir} if global_rank == 0 else None
'''))

cells.append(md(r"""
> ### ⚙️ Operational notes for real clusters
>
> A few things the official Databricks TorchDistributor notebook calls out that matter in production:
>
> - **Use a Single User (dedicated) access mode cluster.** Distributed PyTorch training is only supported on **Single User** clusters — not on Shared/No-isolation clusters. If the cluster must be shared, talk to your Databricks account team.
> - **Workers need MLflow credentials.** Each worker process is a *separate* process that must reach the tracking server. Set these **inside** the training function so they travel to the workers:
>   ```python
>   os.environ["DATABRICKS_HOST"]  = db_host      # e.g. "https://<workspace>.cloud.databricks.com"
>   os.environ["DATABRICKS_TOKEN"] = db_token     # a workspace PAT
>   mlflow.set_experiment(experiment_path)        # a path all workers can resolve
>   ```
>   Create the experiment **once on the driver** so you know its ID, then reference it from the workers.
> - **Split large projects with `%run`.** For anything beyond a demo, keep the model/data code in a separate notebook and pull it in with `%run ./model_def` (or a `.py` module + `%run`). Databricks recommends this to keep distributed training code manageable — the driver notebook then just wires up `TorchDistributor(...).run(main_fn, ...)`.
> - **`mlflow.pytorch.autolog()` targets PyTorch Lightning**, not native PyTorch — with native PyTorch, log metrics/models explicitly (as `train_fn` does above).
"""))

cells.append(md(r"""
## 5. Approach A — Single-node baseline (always start here)

Before distributing anything, prove the model trains on **one process**. If it doesn't work here, distributing it just multiplies the bug.

You can run the exact same `train_fn` in-process (world size 1) to sanity-check logic.
"""))

cells.append(code(r"""
# Quick single-process smoke test (works on CPU or a single GPU).
# We set the env vars a distributor would normally provide.
import os

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

try:
    import torch
    result = train_fn(learning_rate=1e-3, epochs=3, use_gpu=torch.cuda.is_available(),
                      log_dir="/tmp/torchdist_baseline")
    print("Baseline result:", result)
except Exception as e:
    print("Baseline run skipped/failed:", repr(e))
"""))

cells.append(md(r"""
### Reloading a saved checkpoint

`train_fn` wrote a checkpoint per epoch via `save_checkpoint` (using `model.module.state_dict()`).
To resume or serve, rebuild the *bare* model architecture and load the saved weights with
`load_state_dict`. This is the reload half of the checkpoint lifecycle.
"""))

cells.append(code(r"""
# Rebuild the architecture and load the last checkpoint's weights.
try:
    import torch
    import torch.nn as nn

    reloaded = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2))
    ckpt = load_checkpoint("/tmp/torchdist_baseline", epoch=2)   # last epoch index
    reloaded.load_state_dict(ckpt["model"])
    reloaded.eval()
    print("Reloaded model from checkpoint. Sample logits:",
          reloaded(torch.randn(1, 32)).tolist())
except Exception as e:
    print("Checkpoint reload skipped (run the baseline cell first). Error:", repr(e))
"""))

cells.append(md(r"""
## 6. Approach B — `TorchDistributor` on a single node (multi-GPU)

`TorchDistributor` is an **open-source PySpark module** that runs distributed PyTorch on a Spark cluster. It initializes the environment and worker communication for you, wrapping `torch.distributed.run` under the hood.

**Constructor parameters:**
- `num_processes` — total number of training processes (≈ total GPUs you want to use).
- `local_mode` — `True` = run all processes on the **driver node** (single-node multi-GPU); `False` = distribute across **workers** (multi-node).
- `use_gpu` — `True` to assign one GPU per process.

Requirements: **Spark 3.4+**, **Databricks Runtime 13.0 ML or higher**.
"""))

cells.append(code(r'''
# Single-node, multi-GPU: local_mode=True runs num_processes on the driver's GPUs.
# Set num_processes to the number of GPUs on your driver node.
try:
    from pyspark.ml.torch.distributor import TorchDistributor
    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Launching TorchDistributor with {n_gpus} process(es) in local_mode...")

    # Positional args after train_fn map to: learning_rate, epochs, batch_size, use_gpu, log_dir
    result = TorchDistributor(
        num_processes=max(n_gpus, 1),
        local_mode=True,               # all processes on the driver node
        use_gpu=torch.cuda.is_available(),
    ).run(train_fn, 1e-3, 3, 64, torch.cuda.is_available(), "/dbfs/ml/torchdist_singlenode")

    print("TorchDistributor (single-node) result:", result)
except Exception as e:
    print("Single-node distributor cell needs a Databricks GPU cluster. Error:", repr(e))
'''))

cells.append(md(r"""
## 7. Approach C — `TorchDistributor` across multiple nodes

The **only change** to go multi-node is `local_mode=False` and a `num_processes` that spans workers. This is the power of the abstraction: *the same `train_fn` scales from 1 GPU to a whole cluster.*

- `num_processes` = total GPUs across all workers (e.g., 4 workers × 4 GPUs = 16).
- Databricks schedules one process per GPU across the worker fleet.
- Under the hood it uses the `nccl` backend for GPU all-reduce.
"""))

cells.append(code(r'''
# Multi-node data-parallel training. Requires a multi-worker GPU cluster.
# Example: 4 processes across workers.
try:
    from pyspark.ml.torch.distributor import TorchDistributor

    TOTAL_GPUS = 4  # <-- set to (num_workers * gpus_per_worker) on your cluster

    result = TorchDistributor(
        num_processes=TOTAL_GPUS,
        local_mode=False,              # distribute across worker nodes
        use_gpu=True,
    ).run(train_fn, 1e-3, 3, 64, True, "/dbfs/ml/torchdist_multinode")

    print("TorchDistributor (multi-node) result:", result)
except Exception as e:
    print("Multi-node cell needs a multi-worker GPU cluster. Error:", repr(e))
'''))

cells.append(md(r"""
### File-based training (production pattern)

For real jobs, keep training logic in a **`.py` file** rather than a notebook function. `TorchDistributor` can launch a file directly with CLI args — this mirrors how you'd run it in a Databricks Job.

```python
distributor = TorchDistributor(num_processes=8, local_mode=False, use_gpu=True)
distributor.run(
    "/Workspace/Repos/team/project/train.py",
    "--learning_rate=0.001",
    "--batch_size=64",
    "--epochs=10",
)
```

The `train.py` file parses args, reads data (typically from a **Delta table**), and writes checkpoints to a **Unity Catalog Volume** or **DBFS**.
"""))

cells.append(md(r"""
## 8. Approach D — `DeepSpeedTorchDistributor` (when the model won't fit)

When a model is **too large for a single GPU's memory**, data parallelism alone won't help — every replica still needs a full copy. **DeepSpeed** (from Microsoft) solves this with **ZeRO**, which *shards* the optimizer state, gradients, and optionally the parameters across GPUs.

`DeepSpeedTorchDistributor` is **built on top of `TorchDistributor`** and ships in **Databricks Runtime 14.0 ML+**. Use it for:
- **Low GPU memory** relative to model size.
- **Large model training** (e.g., fine-tuning multi-billion-parameter LLMs).
- **Large input data** in batch inference.

**ZeRO stages (memory vs. communication trade-off):**

| Stage | What it shards | Memory saved | Comm cost |
|---|---|---|---|
| ZeRO-1 | Optimizer states | Moderate | Low |
| ZeRO-2 | + Gradients | High | Medium |
| ZeRO-3 | + Parameters | Highest (fits huge models) | Highest |
"""))

cells.append(code(r'''
# DeepSpeedTorchDistributor: same launch ergonomics as TorchDistributor,
# plus a deepspeed config controlling ZeRO stage & offload.
try:
    from pyspark.ml.deepspeed.deepspeed_distributor import DeepspeedTorchDistributor

    deepspeed_config = {
        "train_micro_batch_size_per_gpu": 8,
        "zero_optimization": {
            "stage": 2,                 # shard optimizer state + gradients
            "offload_optimizer": {"device": "cpu"},   # spill to CPU RAM if needed
        },
        "fp16": {"enabled": True},      # mixed precision to cut memory further
    }

    distributor = DeepspeedTorchDistributor(
        numGpus=1,                      # GPUs per node
        nnodes=1,                       # number of nodes
        localMode=True,
        useGpu=True,
        deepspeedConfig=deepspeed_config,
    )
    # distributor.run(train_fn, ...)   # a DeepSpeed-aware train fn (uses deepspeed.initialize)
    print("DeepspeedTorchDistributor configured. Attach a deepspeed-aware train fn to .run().")
    print("See: Fine-tune Llama 2 7B Chat with DeepspeedTorchDistributor (Databricks docs).")
except Exception as e:
    print("DeepSpeed distributor needs DBR 14.0 ML+ on a GPU cluster. Error:", repr(e))
'''))

cells.append(md(r"""
## 9. Approach E — Ray Train (the Python-native alternative)

**Ray** is an open-source framework for scaling Python. On Databricks, **Ray clusters run in the same environment as your Spark cluster** (Ray 2.3.0+), so Ray and Spark are complementary:

- **Spark** → data parallelism for ETL/analytics (DataFrames).
- **Ray** → logical parallelism for dynamic, compute-heavy tasks (RL, complex tuning, custom distributed loops).

The Ray ecosystem:
- **Ray Train** — distributed deep-learning training (wraps PyTorch/TF).
- **Ray Tune** — distributed hyperparameter search (covered in Topic 2).
- **Ray Data** — distributed data loading/preprocessing.
- **Ray Serve** — model serving.

Use Ray Train when your workload is more naturally expressed in Ray, or you're already using Ray Tune/Data.
"""))

cells.append(code(r'''
# Start a Ray cluster on top of Spark, then run a Ray Train TorchTrainer.
try:
    import ray
    from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

    # Provision Ray workers on the Spark cluster (tune to your fleet).
    setup_ray_cluster(
        num_worker_nodes=2,
        num_gpus_worker_node=1,
        num_cpus_worker_node=4,
    )
    ray.init(ignore_reinit_error=True)
    print("Ray cluster up:", ray.cluster_resources())

    # --- Ray Train sketch (data-parallel PyTorch) ---------------------------
    # from ray.train.torch import TorchTrainer
    # from ray.train import ScalingConfig
    #
    # def ray_train_loop(config):
    #     import ray.train.torch
    #     # ... standard torch loop; ray.train.torch.prepare_model/prepare_data_loader
    #     # handle DDP wrapping and sharding automatically.
    #
    # trainer = TorchTrainer(
    #     ray_train_loop,
    #     scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
    # )
    # result = trainer.fit()

    shutdown_ray_cluster()
    print("Ray cluster shut down.")
except Exception as e:
    print("Ray cell needs Ray on a Databricks ML cluster. Error:", repr(e))
'''))

cells.append(md(r"""
## 10. Track everything with MLflow

MLflow tracking is the backbone of iterative DL work. Our `train_fn` already logs params, per-epoch `train_loss`, `test_loss`, and the final model **from rank 0 only** — the pattern below is the *driver-side* wrapper that starts the run and captures the returned metrics.

```python
import mlflow

# Create the experiment on the driver so its ID is known to all workers.
experiment = mlflow.set_experiment(f"/Users/{username}/pytorch-distributor")

with mlflow.start_run(run_name="torchdist-4gpu") as run:
    mlflow.log_param("run_type", "multi_node")
    result = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True).run(
        train_fn, 1e-3, 3, 64, True, "/dbfs/ml/torchdist_run"
    )
    # train_fn logged train/test loss + the model on rank 0; capture the summary too.
    mlflow.log_metric("final_train_loss", result["final_loss"])
    mlflow.log_metric("final_test_loss", result["test_loss"])
```

**Two rules that keep distributed MLflow clean:**
- **Log from rank 0 only** (`if global_rank == 0:`) — otherwise every worker opens a duplicate run.
- **Log the *unwrapped* model** — `mlflow.pytorch.log_model(ddp_model.module, "model")`, not the DDP wrapper, so it reloads without a distributed context.

> `mlflow.pytorch.autolog()` is built for **PyTorch Lightning** and won't capture native-PyTorch training automatically — log explicitly, as `train_fn` does.
"""))

cells.append(md(r"""
## 11. Sizing & configuring the GPU cluster

From the Databricks GPU compute guidance:

**Instance families (AWS examples):**
| Family | GPU | Typical use |
|---|---|---|
| `P5` | H100 (up to 8) | Largest LLM training |
| `P4d` | A100 (8) | LLMs, NLP, recsys, object detection |
| `G6e` | L40S (1–8) | Modern mid-range training/inference |
| `G5` | A10G (1–8) | Cost-effective training/inference |
| `G4dn` | T4 (1–4) | Light inference, dev |

**Configuration checklist:**
- ✅ Enable the **Machine Learning** runtime — the GPU ML version auto-selects from the worker type.
- ❌ Leave **Photon unchecked** — it's incompatible with GPU instances.
- For **training**, set `spark.task.resource.gpu.amount` = GPUs per worker so one task owns all GPUs on a node (minimizes cross-task contention).
- For **inference**, use *fractional* values (`0.5`, `0.33`, `0.25`) to pack multiple tasks per GPU.
- **Single-node (driver-only) GPU cluster** is typically the fastest & most cost-effective for development. Note: GPU *scheduling* is unavailable on single-node clusters.
- Prefer **on-demand** over spot for GPUs (spot availability is volatile).
- The runtime bundles CUDA, cuDNN, and NCCL — you don't install drivers yourself.
"""))

cells.append(md(r"""
## 12. Decision guide — which tool, when

```
Does the model fit on ONE GPU?
├─ YES ─ Do you need more speed / bigger effective batch?
│        ├─ NO  → Single-node, single-GPU  (simplest)
│        ├─ YES, one node's GPUs suffice → TorchDistributor(local_mode=True)
│        └─ YES, need many nodes          → TorchDistributor(local_mode=False)
│
└─ NO (model too big for one GPU)
         → DeepSpeedTorchDistributor (ZeRO-2/3)  or model/pipeline parallelism

Already Ray-native, or need dynamic/complex orchestration?
         → Ray Train (data-parallel) — complements Spark, same cluster
```

### Key takeaways
- **Single-node first.** Distribution adds complexity + communication overhead.
- **Same `train_fn`, one flag** (`local_mode`) takes you from multi-GPU to multi-node with `TorchDistributor`.
- **DeepSpeed / ZeRO** is the answer to *"the model doesn't fit"*, not *"I want it faster"*.
- **Ray** complements Spark for Python-native scaling.
- **MLflow** tracks it all; log from rank 0.
"""))

cells.append(md(r"""
## References
- [Distributed training — Databricks docs](https://docs.databricks.com/machine-learning/train-model/distributed-training/)
- [Distributed training with TorchDistributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/spark-pytorch-distributor)
- [Distributed training with DeepSpeed distributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/deepspeed)
- [Deep learning on Databricks](https://docs.databricks.com/machine-learning/train-model/deep-learning)
- [Best practices for deep learning](https://docs.databricks.com/machine-learning/train-model/dl-best-practices)
- [Ray on Databricks](https://docs.databricks.com/machine-learning/ray/)
- [GPU-enabled compute](https://docs.databricks.com/compute/gpu)
- [`TorchDistributor` API (PySpark)](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.torch.distributor.TorchDistributor.html)
"""))

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "01_distributed_training.ipynb")
write_notebook(os.path.abspath(out), cells)
