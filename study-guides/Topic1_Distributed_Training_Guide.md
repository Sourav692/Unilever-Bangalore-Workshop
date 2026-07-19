# Topic 1 — Distributed Training in Databricks
### Presenter study guide (deep-dive for delivering the topic)

This is your single reference for delivering Topic 1. It mirrors the notebook
(`notebooks/01_distributed_training.ipynb`) and the slide deck
(`generated-slides/01_distributed_training.pptx`), and explains the *why* behind
each concept so you can answer questions confidently.

**Companion files**
- Notebook: `notebooks/01_distributed_training.ipynb`
- Deck: `generated-slides/01_distributed_training.pptx`
- Presenter flow & timing: `presenter-guide/README.md`

---

## 0. The one-sentence summary
> On Databricks you scale deep-learning training in stages — single GPU → multi-GPU on one node → multiple nodes → model sharding — and `TorchDistributor` lets the **same PyTorch code** move up that ladder by changing one flag, while DeepSpeed and Ray handle the harder cases.

If the audience remembers only that, the session succeeded.

---

## 1. Why this topic matters

Deep-learning models are **data- and compute-intensive**. As models and datasets grow,
a single machine eventually can't train them in acceptable time — or can't hold them at all.
Distributed training spreads that work across many GPUs / many machines.

But there's a catch Databricks states plainly:

> *"Distributed code for training and inference is more complex than single-machine code and slower due to communication overhead."*

So the guiding principle is **single-node first**: only distribute when a single machine
genuinely can't do the job. Distribution is a cost you pay to overcome a hard limit, not a
default setting.

**Talking point:** "More GPUs is not automatically faster. Every time you add a worker,
gradients have to be synchronized over the network. You distribute to break a ceiling —
memory or time — not because bigger is better."

---

## 2. The mental model: the scaling ladder

Present this as a ladder you climb only when forced up a rung:

| Rung | Situation | What to do | Tool / setting |
|---|---|---|---|
| 1 | Model + a reasonable batch fit on **one GPU** | Train single-node, single-GPU | Plain PyTorch |
| 2 | Fits on 1 GPU, but you want **more speed / bigger batch** | Data parallelism on **one node, multiple GPUs** | `TorchDistributor(local_mode=True)` |
| 3 | One node's GPUs **aren't enough** | Data parallelism **across nodes** | `TorchDistributor(local_mode=False)` |
| 4 | The **model itself doesn't fit** in one GPU's memory | Shard the model | `DeepSpeedTorchDistributor` (ZeRO) |

**Two rules of thumb:**
- **Scale up before you scale out** — a bigger GPU (A10G → A100 → H100) before more GPUs.
- **Scale out on one node before going multi-node** — intra-node GPU links (NVLink) are far faster than the network between nodes.

---

## 3. The two kinds of parallelism (core concept)

This is the conceptual backbone. Make sure the audience can distinguish these.

### Data parallelism — the common case (~90% of needs)
- **Every worker keeps a *full copy* of the model.**
- The **data batch is split** across workers; each computes gradients on its slice.
- Gradients are **averaged across all workers every step** (an "all-reduce" operation) so all model copies stay identical.
- Throughput scales almost linearly — until network communication starts to dominate.
- ➡️ This is what **`TorchDistributor`** and **Ray Train** give you by default.

**Analogy:** 4 students each grade 1/4 of the exams using the *same* rubric, then meet to average their adjustments to the rubric. Everyone ends with the same rubric; the grading finished 4× faster.

### Model parallelism — the "model won't fit" case
- **The model itself is split** across devices:
  - **Pipeline parallelism** — different *layers* on different GPUs.
  - **Tensor parallelism** — a single layer's math split *within* itself across GPUs.
- Needed when a single GPU can't even hold the weights + optimizer state (e.g., a multi-billion-parameter LLM).
- More complex, higher communication cost.
- ➡️ This is where **DeepSpeed / ZeRO** comes in.

**Analogy:** the textbook is too heavy for one person to carry, so you split the chapters among several people. Now they have to coordinate constantly because the book is spread out.

> **Key line:** "Most teams need *data* parallelism. Reach for *model* parallelism only when a single GPU can't hold the model."

---

## 4. TorchDistributor — the star of the topic

### What it is
An **open-source PySpark module** (`pyspark.ml.torch.distributor.TorchDistributor`) that runs
distributed PyTorch on a Spark cluster. It sets up the environment and inter-worker
communication for you — wrapping PyTorch's own `torch.distributed.run` (a.k.a. `torchrun`)
under the hood. You define a `train()` function in your notebook and hand it to the distributor.

**Requirements:** Spark 3.4+, Databricks Runtime **13.0 ML or higher**.

### The three constructor parameters (know these cold)

| Parameter | Meaning | Values |
|---|---|---|
| `num_processes` | Total number of training processes to launch | ≈ total GPUs you want to use (e.g. `workers × GPUs/worker`) |
| `local_mode` | Where the processes run | `True` = all on the **driver** node (single-node multi-GPU); `False` = spread across **worker** nodes (multi-node) |
| `use_gpu` | Assign one GPU per process | `True` (GPU, `nccl` backend) / `False` (CPU, `gloo` backend) |

### The headline message
> **Single-node multi-GPU → multi-node is a one-flag change.**

```python
# Single node, multiple GPUs (all on the driver):
TorchDistributor(num_processes=4,  local_mode=True,  use_gpu=True).run(train_fn, ...)

# Multiple nodes — ONLY local_mode changes:
TorchDistributor(num_processes=16, local_mode=False, use_gpu=True).run(train_fn, ...)
```

That portability is the single most impressive thing to demo. The *training code doesn't change* — only the launch configuration.

### How you invoke training — two patterns
1. **Function-based (interactive / notebook):** `distributor.run(train_fn, arg1, arg2, ...)` — args after the function are passed straight to it.
2. **File-based (production / Jobs):**
   ```python
   distributor.run("/Workspace/Repos/team/project/train.py",
                   "--learning_rate=0.001", "--batch_size=64", "--epochs=10")
   ```
   The `.py` file parses args, reads data (usually from a **Delta table**), and writes checkpoints to a **Unity Catalog Volume** or **DBFS**.

---

## 5. Anatomy of a distributed training function

This is what to walk through in the notebook. Every distributed PyTorch job on Databricks
has this shape. The nine disciplines:

1. **All imports go *inside* the function.** The function is pickled and shipped to workers;
   top-level imports don't travel and cause errors. *(Most common beginner mistake.)*
2. **Read rank/world from environment variables** the distributor sets:
   - `LOCAL_RANK` — which GPU on *this* node (0..GPUs-per-node-1).
   - `RANK` (global rank) — which process across the *whole* job (0..world_size-1).
   - `WORLD_SIZE` — total number of processes.
3. **Init the process group:** `dist.init_process_group("nccl")` for GPU, `"gloo"` for CPU.
   - `nccl` = NVIDIA's fast GPU-to-GPU communication library.
4. **Wrap the model in DDP:** `DDP(model, device_ids=[local_rank])`. DDP is what synchronizes
   gradients across replicas automatically.
5. **Use a `DistributedSampler`** on the DataLoader so each worker sees a *different shard* of the data (and call `sampler.set_epoch(epoch)` to reshuffle each epoch).
6. **Checkpoint from rank 0** with `torch.save`, saving `model.module.state_dict()` — the
   `.module` unwraps the *underlying* model from the DDP wrapper (see §6).
7. **Log metrics/params/model from rank 0 only** (`if global_rank == 0:`) so N workers don't
   create N duplicate MLflow runs.
8. **Run a held-out evaluation on rank 0** after training and log the test metric.
9. **Clean up:** `dist.destroy_process_group()` at the end.

### Why "rank 0 only" for logging?
Every process runs the *same* code. Without a guard, all N workers would each open an MLflow
run, each write a checkpoint, each print — producing N copies of everything. By convention the
**chief worker (global rank 0)** owns side effects; the others just compute.

---

## 6. Checkpointing & the `.module` detail (a favorite gotcha)

When you wrap a model in `DDP(model)`, the real model becomes accessible as `ddp_model.module`.
So to save clean weights:

```python
def save_checkpoint(log_dir, ddp_model, optimizer, epoch):
    state = {
        "model": ddp_model.module.state_dict(),   # .module unwraps DDP
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, f"{log_dir}/checkpoint-{epoch}.pth.tar")
```

To reload — rebuild the **bare architecture** and load the weights:

```python
model = Net()                                  # the plain, un-wrapped model
ckpt = torch.load(f"{log_dir}/checkpoint-{epoch}.pth.tar")
model.load_state_dict(ckpt["model"])
model.eval()
```

**Why it matters:** if you save `ddp_model.state_dict()` (without `.module`), every key is
prefixed with `module.` and the weights won't load into a plain model without surgery.
Saving `.module.state_dict()` keeps the checkpoint portable.

---

## 7. DeepSpeed distributor — when the model won't fit

### The problem it solves
Data parallelism doesn't help if the model is too big for one GPU, because *every replica still
needs a full copy*. You need to **shard the model's state** across GPUs. That's **DeepSpeed**
(from Microsoft), exposed on Databricks as **`DeepSpeedTorchDistributor`**, **built on top of
`TorchDistributor`**. Available in **Databricks Runtime 14.0 ML+**.

Use it for:
- **Low GPU memory** relative to model size.
- **Large model training** — e.g. fine-tuning multi-billion-parameter LLMs (the docs reference a "Fine-tune Llama 2 7B Chat" example).
- **Large input data** in batch inference.

### ZeRO — the core idea (Zero Redundancy Optimizer)
Normally each GPU redundantly stores the full optimizer state, gradients, and parameters.
ZeRO **partitions** these across GPUs so each holds only a slice — trading a bit more
communication for a large memory saving.

| ZeRO stage | What it shards | Memory saved | Communication cost |
|---|---|---|---|
| **Stage 1** | Optimizer states | Moderate | Low |
| **Stage 2** | + Gradients | High | Medium |
| **Stage 3** | + Parameters | Highest (fits huge models) | Highest |

Plus **CPU offload** (spill state to CPU RAM) and **mixed precision (fp16)** to cut memory further.

**Talking point:** "TorchDistributor answers *'make it faster'*. DeepSpeed answers *'it doesn't
even fit'*. Higher ZeRO stage = more memory saved but more network chatter — climb only as high
as you need."

---

## 8. Ray Train — the Python-native alternative

### What Ray is
An **open-source framework for scaling Python**. On Databricks (Ray **2.3.0+**), a Ray cluster
runs **in the same environment as your Spark cluster** — so you can use both together.

### Ray and Spark are complementary (not competitors)
| Spark | Ray |
|---|---|
| **Data parallelism** for ETL & analytics | **Logical parallelism** for dynamic, compute-heavy tasks |
| DataFrame / SQL workloads | RL, complex tuning, custom distributed loops |
| Feeds data to training | Runs the training / tuning logic |

### The Ray ecosystem (name-drop these)
- **Ray Train** — distributed deep learning (wraps PyTorch/TF; handles DDP for you).
- **Ray Tune** — distributed hyperparameter search *(this reappears in Topic 2)*.
- **Ray Data** — distributed data loading / preprocessing.
- **Ray Serve** — model serving.

### When to choose Ray Train over TorchDistributor
- You're **already Ray-native** or using Ray Tune / Data / Serve together.
- Your workload needs **dynamic or complex orchestration** beyond straightforward DataFrame ops.
- Otherwise, if you're "PyTorch on Spark," **TorchDistributor is the more direct path.**

---

## 9. GPU clusters — sizing & configuration

### Instance families (AWS examples — the pattern is the same across clouds)
| Family | GPU | Best for |
|---|---|---|
| `P5` | H100 (up to 8) | Largest LLM training |
| `P4d` | A100 (8) | LLMs, NLP, recsys, object detection |
| `G6e` | L40S (1–8) | Modern mid-range training + inference |
| `G5` | A10G (1–8) | Cost-effective training + inference |
| `G4dn` | T4 (1–4) | Light inference, dev |

### Configuration checklist
- ✅ **Enable the Machine Learning runtime** — the GPU ML version auto-selects from the worker type.
- ❌ **Leave Photon unchecked** — it's incompatible with GPU instances.
- **Training:** set `spark.task.resource.gpu.amount` = **GPUs per worker**, so one task owns all a node's GPUs (minimizes cross-task contention). *(Set this in the cluster's Spark config before creating it.)*
- **Inference:** use *fractional* values (`0.5`, `0.33`, `0.25`) to pack 2–4 tasks per GPU.
- A **single-node (driver-only) GPU cluster** is typically fastest & most cost-effective for development. Note: GPU *scheduling* isn't available on single-node clusters.
- Prefer **on-demand over spot** for GPUs (spot availability is volatile).
- CUDA, cuDNN, and NCCL are **bundled** in the runtime — you don't install drivers.

---

## 10. MLflow tracking

MLflow is the backbone of iterative DL work — it tracks params, metrics, and models across runs.

**Two rules that keep distributed MLflow clean:**
- **Log from rank 0 only** (`if global_rank == 0:`) — else every worker opens a duplicate run.
- **Log the *unwrapped* model** — `mlflow.pytorch.log_model(ddp_model.module, "model")` — so it reloads without needing a distributed context.

**Driver-side wrapper pattern:**
```python
experiment = mlflow.set_experiment(f"/Users/{username}/pytorch-distributor")
with mlflow.start_run(run_name="torchdist-4gpu"):
    mlflow.log_param("run_type", "multi_node")
    result = TorchDistributor(num_processes=4, local_mode=False, use_gpu=True).run(
        train_fn, 1e-3, 3, 64, True, "/dbfs/ml/torchdist_run")
    mlflow.log_metric("final_test_loss", result["test_loss"])
```

> ⚠️ **`mlflow.pytorch.autolog()` is built for PyTorch Lightning**, not native PyTorch —
> with native PyTorch, log explicitly.

---

## 11. Operational notes for real clusters (production gotchas)

These come straight from the official Databricks end-to-end notebook and are great "voice of
experience" points:

- **Single User (dedicated) access mode is required.** Distributed PyTorch training is *not*
  supported on Shared / No-isolation clusters. If a cluster must be shared, involve your
  Databricks account team.
- **Workers need MLflow credentials.** Each worker is a separate process that must reach the
  tracking server. Set these **inside** the training function so they travel to workers:
  ```python
  os.environ["DATABRICKS_HOST"]  = db_host    # https://<workspace>.cloud.databricks.com
  os.environ["DATABRICKS_TOKEN"] = db_token   # a workspace PAT
  mlflow.set_experiment(experiment_path)      # a path all workers can resolve
  ```
  Create the experiment **once on the driver** so you know its ID, then reference it from workers.
- **Split large projects with `%run`.** Keep model/data code in a separate notebook or `.py`
  module and pull it in with `%run ./model_def`; the driver notebook just wires up
  `TorchDistributor(...).run(main_fn, ...)`.

---

## 12. The decision guide (recap slide)

```
Does the model fit on ONE GPU?
├─ YES ─ Need more speed / bigger effective batch?
│        ├─ NO  → Single-node, single-GPU  (simplest)
│        ├─ YES, one node's GPUs suffice → TorchDistributor(local_mode=True)
│        └─ YES, need many nodes          → TorchDistributor(local_mode=False)
│
└─ NO (model too big for one GPU)
         → DeepSpeedTorchDistributor (ZeRO-2/3)  or model/pipeline parallelism

Already Ray-native, or need dynamic/complex orchestration?
         → Ray Train (data-parallel) — complements Spark, same cluster

ALWAYS → track with MLflow (log from rank 0)
```

---

## 13. Which framework, when (quick table)

| Framework / library | Use it when | Key requirement |
|---|---|---|
| **PyTorch / TF single-node** | Model + batch fit on one GPU; you're iterating | DBR ML |
| **TorchDistributor** | Model fits, you want data parallelism / speed; single- or multi-node | DBR 13.0 ML+, Spark 3.4+ |
| **DeepSpeedTorchDistributor** | Model does *not* fit one GPU; large LLM fine-tuning | DBR 14.0 ML+ |
| **Ray Train** | Python-native/dynamic workloads; already using Ray Tune/Data/Serve | Ray 2.3.0+ |
| **Spark ML (`pyspark.ml.connect`)** | Classic ML at scale on distributed DataFrames | DBR 17.0+ (Standard) for `.connect` |
| **MLflow** | Always — track params/metrics/models | DBR ML |

---

## 14. Notebook demo walkthrough (what to run, in order)

The notebook is `notebooks/01_distributed_training.ipynb`. Suggested live flow:

1. **Environment check** — show runtime + GPU count. Everything branches on this.
2. **`train_fn` + checkpoint helpers** — walk the 9 disciplines; point out imports-inside, the rank guards, `.module` on the checkpoint.
3. **Single-process baseline** — prove it trains on one process. "Always start here."
4. **Checkpoint reload cell** — show `load_state_dict` restoring saved weights.
5. **`TorchDistributor(local_mode=True)`** — single-node multi-GPU. The core moment.
6. **`TorchDistributor(local_mode=False)`** — flip **one flag** for multi-node. Say it out loud. *(If single-node only, read it aloud and skip execution.)*
7. **DeepSpeed cell** — walk the ZeRO config as a configured example.
8. **Ray Train cell** — show the setup/shutdown of a Ray cluster on Spark.
9. **Decision guide markdown** — recap.

> **Runnability note:** the notebook uses a small **synthetic dataset** so it runs end-to-end
> without downloading data. Cells degrade gracefully (print a clear message) when run off
> Databricks or without GPUs, so nothing errors out mid-demo.

---

## 15. Likely audience questions (with answers)

**Q: Do I have to rewrite my PyTorch code to distribute it?**
A: No — wrap the model in DDP, put imports inside the function, and read rank from env vars.
`TorchDistributor` handles launch and communication. The training loop is standard PyTorch.

**Q: When DeepSpeed vs. TorchDistributor?**
A: TorchDistributor for **data parallelism** when the model fits one GPU (you want speed).
DeepSpeed/ZeRO when the **model doesn't fit** one GPU (you need to shard it).

**Q: Ray or TorchDistributor?**
A: TorchDistributor if you're "PyTorch on Spark" and want the direct path. Ray if you're
Ray-native or need Ray Tune/Data/Serve and dynamic orchestration. They coexist on the same cluster.

**Q: Why is my distributed run slower per-epoch than single-GPU?**
A: Communication overhead — gradients sync every step. You distribute to overcome a memory/time
*ceiling* or to shrink *total* wall-clock on huge data, not to speed up a job that already fits
comfortably on one GPU.

**Q: What's `local_mode` really doing?**
A: `True` runs all processes on the **driver** node (uses the driver's GPUs — single machine).
`False` spreads processes across **worker** nodes (true multi-machine).

**Q: Why `model.module.state_dict()` and not `model.state_dict()`?**
A: DDP wraps your model; `.module` is the real model underneath. Saving `.module` keeps
checkpoint keys clean so they reload into a plain model.

**Q: Can I use a shared cluster?**
A: Distributed training needs **Single User** access mode. For shared setups, talk to your
Databricks account team.

---

## 16. Glossary (quick definitions to have ready)

- **DDP (DistributedDataParallel):** PyTorch wrapper that keeps model replicas in sync by averaging gradients each step.
- **All-reduce:** the collective operation that averages gradients across all workers.
- **Rank / global rank:** a process's unique ID across the whole job (0 = chief).
- **Local rank:** a process's GPU index on its own node.
- **World size:** total number of processes.
- **NCCL:** NVIDIA's GPU-to-GPU communication library (the GPU backend).
- **Gloo:** the CPU communication backend.
- **ZeRO:** DeepSpeed's technique for sharding optimizer state / gradients / parameters across GPUs.
- **Pipeline vs. tensor parallelism:** splitting a model by layers vs. splitting within a layer.

---

## Sources (current Databricks documentation)
- [Distributed training](https://docs.databricks.com/machine-learning/train-model/distributed-training/)
- [TorchDistributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/spark-pytorch-distributor)
- [End-to-end TorchDistributor example notebook](https://docs.databricks.com/aws/en/notebooks/source/deep-learning/torch-distributor-notebook.html)
- [DeepSpeed distributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/deepspeed)
- [Deep learning on Databricks](https://docs.databricks.com/machine-learning/train-model/deep-learning) · [DL best practices](https://docs.databricks.com/machine-learning/train-model/dl-best-practices)
- [Ray on Databricks](https://docs.databricks.com/machine-learning/ray/)
- [GPU-enabled compute](https://docs.databricks.com/compute/gpu)
