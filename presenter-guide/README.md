# Databricks Workshop — Presenter Guide (Topics 1–4)

Live walkthrough + demo materials for the Bangalore workshop. This guide ties the
slide decks and notebooks together with a talk track, timing, and demo flow.

> **Audience:** Data scientists / ML engineers.
> **Style:** Concept slides → live, runnable notebook demo per topic.

---

## Materials in this folder

| Topic | Slides (`.pptx`) | Notebook (`.ipynb`) |
|-------|------------------|---------------------|
| 1 — Distributed Training | `generated-slides/01_distributed_training.pptx` | `notebooks/01_distributed_training.ipynb` |
| 2 — Parallelization Techniques | `generated-slides/02_parallelization_techniques.pptx` | `notebooks/02_parallelization_techniques.ipynb` |
| 3 — Feature Store Concepts | `generated-slides/03_feature_store_concepts.pptx` | `notebooks/03_feature_store_concepts.ipynb` |
| 4 — Data Leakage Prevention | `generated-slides/04_data_leakage_prevention.pptx` | `notebooks/04_data_leakage_prevention.ipynb` |

> Study guides for every topic live in `study-guides/TopicN_*_Guide.md` — read these to prepare.

- **Slides** are Databricks-branded PowerPoint. Import into Google Slides via
  *File → Open → Upload*, or present directly from PowerPoint / Keynote.
- **Notebooks** are Jupyter `.ipynb`. Import into a Databricks workspace
  (*Workspace → Import → File*) — they carry Databricks notebook metadata and run cell-by-cell.

---

## Before the session — setup checklist

- [ ] **Cluster for Topic 1:** GPU-enabled, **Databricks Runtime 14.0 ML (GPU) or later**.
      - A **single-node GPU** cluster covers Parts 1–6 (baseline + single-node multi-GPU).
      - A **multi-node GPU** cluster (e.g., 2–4 workers) is needed for the multi-node `TorchDistributor` cell (Part 7). If you only have single-node, present that cell as a read-along.
- [ ] **Cluster for Topic 2:** multi-node **DBR ML 14.3 LTS+** (GPU optional). More workers = more visible parallelism.
- [ ] **Cluster for Topics 3 & 4:** **UC-enabled DBR ML 15.4 LTS+** (GPU not needed). Pick a catalog/schema you can write to and set `CATALOG`/`SCHEMA` at the top of each notebook.
- [ ] Import the notebooks into the workspace and attach them to the cluster.
- [ ] Dry-run the notebooks top to bottom once — first Spark job and Ray cluster startup take a minute.
- [ ] Open each deck and confirm branding/fonts render (import to Google Slides if presenting from there).
- [ ] Have the [Databricks distributed training docs](https://docs.databricks.com/machine-learning/train-model/distributed-training/) open in a tab for Q&A.

> **Cost tip:** GPU clusters are expensive. Start them ~10 min before, and **terminate immediately after**. Prefer on-demand over spot (GPU spot availability is volatile).

---

## Suggested timing (≈45 min total, adjustable)

| Segment | Time |
|---|---|
| Topic 1 slides (mental model → tools) | 12 min |
| Topic 1 notebook demo | 10 min |
| Topic 2 slides (patterns) | 10 min |
| Topic 2 notebook demo | 10 min |
| Q&A / buffer | 3 min |

---

## Topic 1 — Distributed Training

### Talk track (slides)
1. **Open with the mental model.** The single most important message: *single-node first.*
   Distributed code is more complex and slower per step from communication overhead.
2. **The scaling ladder** (timeline slide): single GPU → multi-GPU one node → multi-node → model sharding.
   Emphasize you move down the ladder *only when forced*.
3. **Data vs. model parallelism.** Most teams need **data parallelism**. Model parallelism / DeepSpeed
   is specifically for *"the model won't fit in one GPU."*
4. **TorchDistributor** is the star: three params (`num_processes`, `local_mode`, `use_gpu`), and the
   *same training function* scales from multi-GPU to multi-node by flipping `local_mode`.
5. **DeepSpeed / ZeRO** — the answer to memory limits, not speed. Walk the ZeRO stage table.
6. **Ray Train** — Python-native alternative; Ray and Spark are complementary on the same cluster.
7. **Close** with GPU sizing, task scheduling, and MLflow (log from rank 0).

### Demo flow (`01_distributed_training.ipynb`)
- **Cell: Environment check** — show the runtime + GPU count live. Sets up everything after.
- **Cell: single-process baseline** — prove the model trains on one process first. *"Always start here."*
- **Cell: TorchDistributor `local_mode=True`** — the multi-GPU run on the driver. This is the money moment.
- **Cell: TorchDistributor `local_mode=False`** — flip **one flag** for multi-node. Call this out explicitly.
  *(If single-node only, read it aloud and skip execution.)*
- **DeepSpeed & Ray cells** — walk through as configured examples; running them requires DBR 14.0 ML+ / Ray.
- **Decision guide (final markdown)** — recap: which tool, when.

### Anticipated questions
- *"Do I have to rewrite my PyTorch code?"* → No — wrap the model in DDP, put imports inside the function, read rank from env. `TorchDistributor` handles launch + comms.
- *"When DeepSpeed vs. TorchDistributor?"* → TorchDistributor for data parallelism (model fits). DeepSpeed/ZeRO when the model doesn't fit one GPU.
- *"Ray or TorchDistributor?"* → TorchDistributor if you're PyTorch-on-Spark. Ray if you're Ray-native or need dynamic orchestration + Ray Tune/Data.

---

## Topic 2 — Parallelization Techniques

### Talk track (slides)
1. **The framing:** a cluster is only useful if you fill it. Show the **workload shape → pattern** grid.
2. **Anti-pattern callout:** `.collect()` / `.toPandas()` + a Python loop uses one core. This lands well — most people have done it.
3. **Spark foundation:** partitions → tasks → cores. The lever you control is *number of partitions* (~2–4× cores).
4. **Pandas Function APIs** — the heart of the topic:
   - `applyInPandas` = **one model per group** (per store / SKU / sensor), using sklearn *inside* the function.
   - `mapInPandas` = **distributed batch inference**, model loaded once per partition.
5. **Hyperparameter tuning:** Optuna (+ Joblib) and **Ray Tune**. **Call out that Hyperopt is deprecated**
   (removed from DBR ML after 16.4 LTS) — this is a common "gotcha" for people with older code.
6. **joblib-spark:** distribute scikit-learn grid search with just a context manager.
7. **MLlib / `pyspark.ml.connect`** for one big model; **GPU task scheduling** (full GPU for training, fractions for inference).
8. **Close** on the utilization checklist.

### Demo flow (`02_parallelization_techniques.ipynb`)
- **Partitions cell** — show default parallelism and repartitioning live.
- **`applyInPandas` cell** — the highlight: train **50 models in parallel**, one per group. Show the results table.
- **`mapInPandas` cell** — score 1M rows across the cluster; stress "model loaded once per partition."
- **Optuna cell** then **joblib-spark / Ray Tune cells** — 20-trial search; show best params.
- **MLlib pipeline cell** — one distributed LogisticRegression over 200k rows.
- **Utilization checklist (final markdown)** — recap.

### Anticipated questions
- *"applyInPandas vs a Pandas UDF?"* → UDF is column→column (feature transforms). `applyInPandas` gives you a whole group as a DataFrame for per-group model training.
- *"What about skew?"* → Real risk — the whole group loads in memory. Salt keys or split hot groups.
- *"We still use Hyperopt/SparkTrials."* → Migrate to Optuna or Ray Tune; Hyperopt is unmaintained and gone from DBR ML after 16.4 LTS.

---

## Topic 3 — Feature Store Concepts

**Cluster:** UC-enabled **DBR ML 15.4 LTS+**. Pick a catalog/schema you can write to; set `CATALOG`/`SCHEMA` at the top of the notebook. GPU not needed.

### Talk track (slides)
1. **Open with the two problems:** feature *reuse* (everyone rebuilds the same features) and *train/serve skew* (features computed differently in training vs. serving). Everything else is the solution to these.
2. **Feature table = a Delta table in UC with a primary key.** Demystify it — no special engine.
3. **The workflow timeline:** `create_table` → `FeatureLookup`/`create_training_set` → `fe.log_model` → `fe.score_batch`.
4. **The key insight (callout):** `fe.log_model` packages feature metadata *into* the model, so `score_batch` needs only keys and fetches features itself — that's what kills skew.
5. **Offline vs. online:** same definition backs batch and real-time (Lakebase).
6. **Close on governance:** discovery + lineage + UC permissions are what make features truly reusable.

### Demo flow (`03_feature_store_concepts.ipynb`)
- **create_table** — build customer features, register with a primary key.
- **FeatureLookup + create_training_set** — labels have *no* features; show the joined training DataFrame. This is "reuse."
- **fe.log_model** — stress `training_set=` packaging metadata into the model.
- **score_batch** — the payoff: score with keys only; features fetched automatically. This is "consistency."
- **Governance markdown** — discovery, lineage, tags.

### Anticipated questions
- *"Is a feature table special storage?"* → No — a Delta table in UC with a primary key.
- *"Do I pass features at inference?"* → No — `score_batch` looks them up from the keys.
- *"Is this the old Workspace Feature Store?"* → Use Feature Engineering in Unity Catalog; the workspace-local one is deprecated for new work.

---

## Topic 4 — Data Leakage Prevention

**Cluster:** UC-enabled **DBR ML 15.4 LTS+**. The leakage-illustration cell runs on any cluster (plain Spark); the feature-table cells need UC.

### Talk track (slides)
1. **Define leakage:** using feature values that weren't available when the label was recorded. Stress it makes offline metrics look *better* — that's why it's dangerous.
2. **The 08:50 / 08:52 example** — the concrete "time-travel bug." Say the correct value out loud.
3. **Point-in-time correctness** is the fix: use the value known *at or before* the label time.
4. **Time series feature table** = Delta + a declared `TIMESERIES` timestamp key → enables **AS OF** joins.
5. **The one parameter:** `timestamp_lookup_key` on the `FeatureLookup` makes the training set leak-free. `lookback_window` bounds staleness.
6. **Consistency carries to scoring:** `score_batch` re-applies the same point-in-time logic — leak-free training *and* skew-free serving.

### Demo flow (`04_data_leakage_prevention.ipynb`)
- **Leakage illustration** — labels + a feature history with a future reading; state the correct value (12.0, not 99.0).
- **create_table with `timeseries_columns`** — the time series feature table (+ SQL form).
- **Point-in-time training set** — add `timestamp_lookup_key`; show AS OF picks the at-or-before value.
- **lookback_window** — bound history.
- **Guardrails markdown** — split-before-engineering, exclude target, label windows.

### Anticipated questions
- *"Why do point-in-time joins fix leakage?"* → They join the value known at/before the label time; no future data enters.
- *"My AUC is 0.99 — great?"* → Be suspicious; near-perfect offline metrics are a classic leakage symptom.
- *"Do I redo this at inference?"* → No — `score_batch` re-applies it if the scoring DataFrame carries the same timestamp column.

---

## Content provenance

All concepts are grounded in current Databricks documentation (retrieved for this workshop):

- [Distributed training](https://docs.databricks.com/machine-learning/train-model/distributed-training/)
- [TorchDistributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/spark-pytorch-distributor)
- [DeepSpeed distributor](https://docs.databricks.com/machine-learning/train-model/distributed-training/deepspeed)
- [Deep learning](https://docs.databricks.com/machine-learning/train-model/deep-learning) · [DL best practices](https://docs.databricks.com/machine-learning/train-model/dl-best-practices)
- [Ray on Databricks](https://docs.databricks.com/machine-learning/ray/)
- [GPU-enabled compute](https://docs.databricks.com/compute/gpu)
- [Pandas function APIs](https://docs.databricks.com/pandas/pandas-function-apis)
- [Hyperparameter tuning](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/)

---

## Regenerating the materials

Notebooks (no external deps — `.ipynb` is built as JSON):
```bash
python3 build/gen_topic1.py
python3 build/gen_topic2.py
```

Slides (uses the `databricks-slides` skill generator + `python-pptx`):
```bash
SKILL_DIR="$HOME/.claude/plugins/marketplaces/plugin-marketplace/experimental/general/databricks-slides/skills/slide-deck"
python3 "$SKILL_DIR/scripts/generate-pptx.py" --input /tmp/slides-topic1.json --output ./generated-slides/01_distributed_training.pptx
python3 "$SKILL_DIR/scripts/generate-pptx.py" --input /tmp/slides-topic2.json --output ./generated-slides/02_parallelization_techniques.pptx
```
(Slide content JSON is saved at `presenter-guide/slides-topic1.json` and `slides-topic2.json`.)
