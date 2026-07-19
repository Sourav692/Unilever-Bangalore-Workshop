# Topic 2 — Parallelization Techniques to Utilize Clusters Effectively
### Presenter study guide (deep-dive for delivering the topic)

This is your single reference for delivering Topic 2. It mirrors the notebook
(`notebooks/02_parallelization_techniques.ipynb`) and the slide deck
(`generated-slides/02_parallelization_techniques.pptx`), and explains the *why* behind
each pattern so you can field questions confidently.

**Companion files**
- Notebook: `notebooks/02_parallelization_techniques.ipynb`
- Deck: `generated-slides/02_parallelization_techniques.pptx`
- Presenter flow & timing: `presenter-guide/README.md`

---

## 0. The one-sentence summary
> A cluster is only as useful as your ability to *fill* it — so the whole skill is matching the **right parallelization pattern to the *shape* of your workload**, from Spark data parallelism to per-group model training, distributed tuning, and distributed ML.

---

## 1. Why this topic matters

Distributed training (Topic 1) is about *one big model*. This topic is broader: **how do you
keep every core and GPU on the cluster busy** across *all* the ML work you do — feature
engineering, training many small models, hyperparameter search, batch scoring?

The failure mode this topic prevents: **you rent a 100-core cluster and your job uses 1 core**
because the code accidentally funnels everything through the driver. The right pattern turns
N nodes into ~N× throughput; the wrong one leaves the cluster idle while you wait.

**Talking point:** "Paying for a big cluster doesn't make your code parallel. You have to
*hand the work to the cluster* in a shape it can spread out. That's what these patterns do."

---

## 2. The core idea: workload shape → pattern

This is the spine of the whole talk. Everything else is an instance of it.

| Workload shape | Best pattern | Tool |
|---|---|---|
| Transform / aggregate big data | **Data parallelism** (partitions) | Spark DataFrame / SQL |
| Train **one model per group** (per store, per SKU, per region) | **Grouped map** | `groupBy().applyInPandas()` |
| Score a big table with a model in batches | **Map partitions** | `mapInPandas()` / Pandas UDF |
| Search **many hyperparameter configs** for one model | **Distributed tuning** | Optuna + Joblib, **Ray Tune** |
| Distribute **scikit-learn** work (grid search, ensembles) | **Joblib Spark backend** | `joblib-spark` |
| Train **one big model** on big data with MLlib | **Distributed ML** | `pyspark.ml` / `pyspark.ml.connect` |

### The anti-pattern to call out explicitly
> **Never `.collect()` or `.toPandas()` a big dataset and loop over it in plain Python.**
> That pulls all the data to the driver and uses **one core** while every executor sits idle.

This lands well with practitioners — almost everyone has done it. Frame the rest of the talk as
"here's what to do *instead*."

---

## 3. Foundation — Spark data parallelism

Everything else rides on Spark's execution model, so establish it first:

- A DataFrame is split into **partitions**.
- Each partition becomes a **task**.
- Tasks run on **executor cores** in parallel — the scheduler keeps all cores fed.

### The lever you control most: number of partitions
- **Too few partitions** → idle cores (you can't run more tasks than partitions).
- **Too many** → scheduling overhead dominates.
- **Heuristic:** aim for **2–4× total cluster cores**; repartition after heavy filters/joins that shrink the data.

### Adaptive Query Execution (AQE)
Leave it **on** (default). AQE auto-tunes the number of shuffle partitions *at runtime* based on
actual data sizes — so it fixes a lot of partition-sizing mistakes for you.

**Demo cell:** show `spark.sparkContext.defaultParallelism` (total cores), the default partition
count, and a `repartition(cores * 3)`.

---

## 4. Pattern A — one model per group with `applyInPandas`

### The scenario
"Train **thousands of small models in parallel**" — a demand model **per store**, a forecast
**per SKU**, an anomaly detector **per sensor**. Very common in retail/CPG (relevant for a
Unilever audience: per-product, per-region models).

### The mechanism: split-apply-combine
1. **Split:** `df.groupBy("group_key")` partitions rows by group.
2. **Apply:** your Python function receives **each group as a pandas DataFrame** and returns a pandas DataFrame.
3. **Combine:** Spark stitches the per-group outputs back into one DataFrame.

Each group is processed **on an executor**, so hundreds of groups train **concurrently** across
the cluster — using ordinary single-node libraries (**scikit-learn, statsmodels, prophet**)
*inside* the function.

### Code shape
```python
result_schema = "group_id long, coef_x1 double, coef_x2 double, intercept double, n_rows long"

def fit_group(pdf):                     # runs on an executor, once per group
    from sklearn.linear_model import LinearRegression
    model = LinearRegression().fit(pdf[["x1", "x2"]], pdf["y"])
    return pd.DataFrame([{ "group_id": pdf["group_id"].iloc[0], ... }])

models_df = df.groupBy("group_id").applyInPandas(fit_group, schema=result_schema)
```

### Two things to stress
- **You must declare the output schema up front** — Spark needs to know the result shape before it runs the function.
- ⚠️ **Skew risk:** the *entire* group loads into memory at once. A few huge groups can OOM an
  executor while others sit idle. Mitigate by salting keys or splitting hot groups.

**Talking point:** "This is the magic trick — you write *normal single-machine scikit-learn
code*, and Spark runs one copy per group across the whole cluster. No distributed-ML rewrite."

---

## 5. Pattern B — distributed batch inference with `mapInPandas`

### The scenario
You have a trained model and need to **score a huge table**.

### The mechanism
`mapInPandas` streams **batches of rows** (as pandas DataFrames) through your function on each
partition. Compared to a row-at-a-time UDF you get:
- **Load the model once per partition**, not once per row.
- **Vectorized** prediction over a whole batch.
- **Memory control** — the input is an *iterator* of DataFrames, so you process batch by batch.

### Code shape
```python
def predict_batches(iterator):          # runs on an executor
    model = load_model(...)             # loaded ONCE per partition, outside the loop
    for pdf in iterator:                # each pdf is a batch of rows
        yield pdf.assign(prediction=model.predict(pdf[features]))

scored = df.mapInPandas(predict_batches, schema=out_schema)
```

**Key line:** "The model loads once per partition, then scores many batches. That's the
difference between fast and painfully slow batch inference."

---

## 6. The pandas API family (comparison to keep straight)

| API | Input → Output | Use for |
|---|---|---|
| **Pandas UDF** (Series→Series) | column → column | Vectorized feature transforms |
| **`mapInPandas`** | iterator of DataFrames → iterator of DataFrames | Batch inference, flexible row counts |
| **`applyInPandas`** (grouped) | one group DataFrame → one DataFrame | Per-group model training / aggregation |
| **`applyInPandas`** (cogrouped) | two group DataFrames → one DataFrame | Per-key joins (e.g., `pd.merge_asof` on time) |

All four are **"pandas function APIs"** — they let you run pandas/Python code on partitions of a
Spark DataFrame, so single-node libraries scale out.

---

## 7. Pattern C — distributed hyperparameter tuning

### Why it parallelizes so well
Tuning one model = training it many times with different configs. Those trials are
**independent** — "embarrassingly parallel." Perfect for a cluster.

### The current recommended stack
- **Optuna** — lightweight, supports **dynamic search spaces**; parallelize trials with
  **Joblib**; integrates with **MLflow**. Good for moderate parallelism.
- **Ray Tune** — the **distributed** tuning library; uses Ray as the backend to spread trials
  across the whole cluster. Built into Databricks Runtime ML. **Recommended for large-scale search.**

### ⚠️ Deprecation to call out (important, common gotcha)
> **Hyperopt is deprecated.** The open-source project is no longer maintained, and it was
> **removed from Databricks Runtime ML after 16.4 LTS**. If you have `SparkTrials` / Hyperopt
> code, **migrate to Optuna or Ray Tune.**

This is a genuinely useful "did you know" — many teams still have Hyperopt code and will hit this.

### Code shapes
**Optuna (single-driver, multi-trial):**
```python
def objective(trial):
    params = {"n_estimators": trial.suggest_int("n_estimators", 50, 300), ...}
    return cross_val_score(RandomForestClassifier(**params), X, y, cv=3).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)
```

**Optuna distributed across the cluster (Joblib-Spark backend):**
```python
from joblibspark import register_spark
register_spark()
with joblib.parallel_backend("spark", n_jobs=-1):
    study.optimize(objective, n_trials=20, n_jobs=-1)
```

**Ray Tune (the distributed tuner):**
```python
tuner = tune.Tuner(trainable,
    param_space={"n_estimators": tune.randint(50, 300), "max_depth": tune.randint(2, 16)},
    tune_config=tune.TuneConfig(num_samples=20, metric="accuracy", mode="max"))
results = tuner.fit()
```

---

## 8. Pattern D — distribute scikit-learn with `joblib-spark`

### The mechanism
Many scikit-learn tools (`GridSearchCV`, `RandomForestClassifier`, cross-validation) parallelize
internally through **Joblib**. The **`joblib-spark`** backend **redirects that parallelism onto
the Spark cluster** — so a grid search that would run on the driver's cores instead fans out
across executors, with **no code rewrite** beyond a context manager.

```python
from joblibspark import register_spark
register_spark()
grid = GridSearchCV(SVC(), param_grid, cv=3)
with joblib.parallel_backend("spark", n_jobs=-1):
    grid.fit(X, y)          # each (param combo × CV fold) becomes an executor task
```

**Talking point:** "If your scikit-learn code already uses `n_jobs=-1`, this is almost a
free lunch — swap the Joblib backend and the same search runs cluster-wide."

---

## 9. Pattern E — distributed ML with Spark MLlib

### The scenario
Train **one model over data too big for a single machine**. Here the *algorithm itself* is
distributed (unlike `applyInPandas`, where each group is single-node).

- **`pyspark.ml`** — the classic DataFrame-based MLlib: pipelines, feature transformers,
  estimators (LogisticRegression, GBTs, ALS, etc.). Distributed by design.
- **`pyspark.ml.connect`** — a newer module for distributed training/inference that works with
  **Spark Connect**, available **by default on Databricks Runtime 17.0+ Standard compute**.

```python
assembler = VectorAssembler(inputCols=["f1", "f2"], outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label")
model = Pipeline(stages=[assembler, lr]).fit(train)   # training distributed across the cluster
```

**When to use which:**
- **MLlib** → one big model on distributed data, classic ML algorithms.
- **`applyInPandas`** → many small models, one per group, using single-node libraries.
- Different tools for different shapes — don't confuse them.

---

## 10. Using GPUs efficiently (task scheduling)

If the cluster has GPUs, how tasks map to GPUs decides utilization (from the GPU compute docs):

| Goal | Setting | Effect |
|---|---|---|
| **Training** | `spark.task.resource.gpu.amount` = **GPUs per worker** | One task owns all a node's GPUs → minimizes communication overhead |
| **Inference** | `spark.task.resource.gpu.amount` = **fraction** (`0.5`, `0.33`, `0.25`) | 2–4 tasks **share** each GPU → pack more concurrent scoring tasks |

- GPU scheduling is **not available on single-node clusters** (no separate executor tasks to schedule).

```python
spark.conf.set("spark.task.resource.gpu.amount", "0.5")   # 2 inference tasks per GPU
```

---

## 11. Cluster-utilization checklist (the recap)

- **Right-size partitions** — target ~2–4× total cores; repartition after heavy filters/joins.
- **Avoid the driver bottleneck** — never `.collect()` / `.toPandas()` a big dataset to loop in Python. Push work into `applyInPandas` / `mapInPandas`.
- **Beware skew** — uneven group sizes leave some executors idle and others OOM. Salt keys / split hot groups.
- **Leave AQE on** — it auto-tunes shuffle partitions at runtime.
- **Match the pattern to the shape** — per-group → grouped map; many configs → distributed tuning; one big model → MLlib.
- **Autoscale for bursty jobs**, fixed size for steady ones; use cluster policies to standardize.
- **Watch the Spark UI / metrics** — look for stragglers, spill, and idle cores.

---

## 12. Notebook demo walkthrough (what to run, in order)

The notebook is `notebooks/02_parallelization_techniques.ipynb`. Suggested live flow:

1. **Partitions cell** — show default parallelism and `repartition`.
2. **`applyInPandas` cell** — the highlight: **train 50 models in parallel**, one per group. Show the results table with 50 rows of coefficients.
3. **`mapInPandas` cell** — score 1M rows across the cluster; stress "model loaded once per partition."
4. **Optuna cell**, then **joblib-spark / Ray Tune cells** — a 20-trial search; show best params.
5. **MLlib pipeline cell** — one distributed LogisticRegression over 200k rows.
6. **Utilization checklist markdown** — recap.

> **Runnability note:** cells build **synthetic Spark DataFrames**, so they run on any DBR ML
> cluster (GPU optional). Cells degrade gracefully (print a message) when `spark` isn't defined
> or a library is missing, so nothing errors out mid-demo.

---

## 13. Likely audience questions (with answers)

**Q: `applyInPandas` vs. a Pandas UDF — what's the difference?**
A: A Pandas UDF is column → column (feature transforms). `applyInPandas` hands you a **whole
group as a DataFrame**, so you can train a per-group model or do group-level aggregation.

**Q: What about skew — won't one giant group blow up?**
A: Real risk. The whole group loads into executor memory at once. Salt the key, split hot
groups, or filter outliers. Watch the Spark UI for a straggler task.

**Q: We still use Hyperopt / SparkTrials — is that fine?**
A: Migrate. Hyperopt is unmaintained and was removed from Databricks Runtime ML after 16.4 LTS.
Use **Optuna** (with the Joblib-Spark backend) or **Ray Tune**.

**Q: When do I use MLlib vs. `applyInPandas`?**
A: MLlib trains **one model** on distributed data (the algorithm is distributed). `applyInPandas`
trains **many small models**, one per group, each on a single executor with normal libraries.

**Q: Optuna or Ray Tune?**
A: Optuna for lightweight/moderate parallelism with dynamic search spaces. Ray Tune when you
need to spread a large search across the whole cluster (and it pairs with Ray Train / Data).

**Q: Does `joblib-spark` change my scikit-learn code?**
A: Barely — wrap the fit in `with joblib.parallel_backend("spark", n_jobs=-1):`. The trials/folds
become executor tasks.

**Q: Why is my cluster underutilized even though the job is "parallel"?**
A: Usually too few partitions (fewer than cores), a driver-side loop, or skew. Check partition
count vs. cores, and the Spark UI for idle executors.

---

## 14. How Topics 1 and 2 connect (transition line)

- **Topic 1** = scaling **one big model** (distributed *training* — TorchDistributor, DeepSpeed, Ray Train).
- **Topic 2** = scaling **everything else** — filling the cluster across feature engineering,
  many-small-models, tuning, and batch scoring.
- **Shared thread:** both are about *matching the parallelism strategy to the problem* rather
  than throwing hardware at it. **Ray** appears in both (Ray Train for DL; Ray Tune for tuning).

---

## 15. Glossary (quick definitions to have ready)

- **Partition:** a chunk of a DataFrame; the unit of parallel work in Spark.
- **Task:** the execution of one partition on one core.
- **Executor:** a worker process holding cores + memory that runs tasks.
- **Split-apply-combine:** group → process each group → recombine (what `applyInPandas` does).
- **Pandas function API:** family of methods (`applyInPandas`, `mapInPandas`, cogroup) that run pandas code on Spark partitions.
- **AQE (Adaptive Query Execution):** runtime optimizer that auto-tunes shuffle partitions.
- **Skew:** uneven partition/group sizes causing some tasks to dominate runtime.
- **Embarrassingly parallel:** work with no dependencies between units (e.g., HPO trials).
- **Trial:** one hyperparameter configuration evaluated during tuning.
- **Joblib backend:** the engine Joblib uses to run parallel jobs; `joblib-spark` makes it the cluster.

---

## Sources (current Databricks documentation)
- [Pandas function APIs (`applyInPandas`, `mapInPandas`, cogroup)](https://docs.databricks.com/pandas/pandas-function-apis)
- [Pandas user-defined functions](https://docs.databricks.com/udf/pandas)
- [Hyperparameter tuning](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/)
- [Optuna on Databricks](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/optuna)
- [Ray / Ray Tune on Databricks](https://docs.databricks.com/machine-learning/ray/)
- [Apache Spark MLlib on Databricks](https://docs.databricks.com/machine-learning/train-model/mllib)
- [GPU-enabled compute & task scheduling](https://docs.databricks.com/compute/gpu)
