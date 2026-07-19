"""Generate Topic 2 notebook: Parallelization techniques to utilize clusters effectively."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from nbbuild import md, code, write_notebook

cells = []

cells.append(md(r"""
# Parallelization Techniques to Utilize Clusters Effectively
### Workshop — Topic 2 of 6

**Goal:** Learn how to keep every core (and GPU) on a Databricks cluster busy — matching the *right* parallelization pattern to the *shape* of your workload.

> ⚙️ **Recommended compute:** a **multi-node Databricks Runtime ML cluster** (14.3 LTS ML or later). GPU is optional for this topic — most patterns here are CPU-parallel. Serverless/Standard compute works for the Spark ML section.

---

### The core idea
A cluster is only as useful as your ability to *fill* it. The wrong pattern leaves workers idle; the right one turns N nodes into ~N× throughput. This notebook walks through the main patterns, from coarse to fine:

1. **Spark data parallelism** — the default engine (partitions → tasks → cores).
2. **Pandas Function APIs** — `applyInPandas` / `mapInPandas` for per-group model training & batch scoring.
3. **Distributed hyperparameter tuning** — Optuna + Joblib, and Ray Tune.
4. **`joblib-spark`** — distribute scikit-learn / grid search across the cluster.
5. **Distributed Spark ML** — `pyspark.ml` and `pyspark.ml.connect`.
6. **Cluster utilization tuning** — partitions, GPU task scheduling, avoiding skew.

*Source: Databricks docs — [Pandas function APIs](https://docs.databricks.com/pandas/pandas-function-apis), [Hyperparameter tuning](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/), [GPU compute](https://docs.databricks.com/compute/gpu).*
"""))

cells.append(md(r"""
## 1. Know your workload shape → pick the pattern

| Workload shape | Best parallelization pattern | Tool |
|---|---|---|
| Transform / aggregate big data | **Data parallelism** (partitions) | Spark DataFrame / SQL |
| Train **one model per group** (per store, per SKU, per region) | **Grouped map** | `groupBy().applyInPandas()` |
| Score a big table with a model in batches | **Map partitions** | `mapInPandas()` / Pandas UDF |
| Search **many hyperparameter configs** for one model | **Distributed tuning** | Optuna+Joblib, **Ray Tune** |
| Distribute **scikit-learn** work (grid search, ensembles) | **Joblib Spark backend** | `joblib-spark` |
| Train **one big model** on big data with MLlib | **Distributed ML** | `pyspark.ml` / `pyspark.ml.connect` |

The mistake to avoid: pulling data to the driver with `.collect()` / `.toPandas()` and running a plain Python loop. That uses **one core** while the rest of the cluster sits idle.
"""))

cells.append(md(r"""
## 2. Foundation — Spark data parallelism

Spark's execution model is the base layer every other pattern rides on:

- A DataFrame is split into **partitions**.
- Each partition becomes a **task**.
- Tasks run on **executor cores** in parallel — the scheduler keeps all cores fed.

**The lever you control most often: number of partitions.**
- Too few → idle cores (parallelism capped below core count).
- Too many → scheduling overhead dominates.
- Heuristic: aim for **2–4× total cluster cores**, and repartition after heavy filters.
"""))

cells.append(code(r"""
# Inspect and control parallelism. Safe to run on any Databricks cluster.
try:
    total_cores = spark.sparkContext.defaultParallelism
    print("Total executor cores (default parallelism):", total_cores)

    df = spark.range(0, 10_000_000)
    print("Default partitions:", df.rdd.getNumPartitions())

    # Right-size partitions to ~3x cores for a compute-heavy stage.
    target = total_cores * 3
    df = df.repartition(target)
    print("Repartitioned to:", df.rdd.getNumPartitions())

    # AQE (Adaptive Query Execution) auto-tunes shuffle partitions at runtime.
    print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
except NameError:
    print("`spark` not defined — running outside Databricks. Read along; the patterns still apply.")
"""))

cells.append(md(r"""
## 3. Pattern A — One model per group with `applyInPandas`

This is the workhorse for **"train thousands of small models in parallel"** — a per-store demand model, a per-SKU forecast, a per-sensor anomaly detector. It's the *split-apply-combine* pattern:

1. **Split**: `df.groupBy("group_key")` partitions rows by group.
2. **Apply**: your Python function receives **each group as a pandas DataFrame** and returns a pandas DataFrame.
3. **Combine**: Spark stitches the outputs back into one DataFrame.

Each group is processed **on an executor**, so hundreds of groups train concurrently across the cluster — using ordinary single-node libraries (scikit-learn, statsmodels, prophet) *inside* the function.

> ⚠️ **Watch for skew:** all rows for a group load into memory at once. A few huge groups can OOM an executor while others idle.
"""))

cells.append(code(r'''
# Train one scikit-learn model PER GROUP, all groups in parallel across the cluster.
try:
    import pandas as pd
    from pyspark.sql.functions import rand, when, col

    # --- Build a demo dataset: 50 groups, ~2k rows each -------------------------
    base = (spark.range(0, 100_000)
                 .withColumn("group_id", (col("id") % 50))
                 .withColumn("x1", rand(seed=1))
                 .withColumn("x2", rand(seed=2))
                 .withColumn("y", (col("x1") * 2 + col("x2") + rand(seed=3) * 0.1)))

    # --- Define the per-group training function ---------------------------------
    # Output schema must be declared up front (Spark needs it before running).
    result_schema = "group_id long, coef_x1 double, coef_x2 double, intercept double, n_rows long"

    def fit_group(pdf: "pd.DataFrame") -> "pd.DataFrame":
        # This runs on an EXECUTOR, once per group. Import inside for safety.
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(pdf[["x1", "x2"]], pdf["y"])
        return pd.DataFrame([{
            "group_id": int(pdf["group_id"].iloc[0]),
            "coef_x1": float(model.coef_[0]),
            "coef_x2": float(model.coef_[1]),
            "intercept": float(model.intercept_),
            "n_rows": len(pdf),
        }])

    # --- Split-apply-combine: 50 models trained in parallel ---------------------
    models_df = base.groupBy("group_id").applyInPandas(fit_group, schema=result_schema)
    models_df.orderBy("group_id").show(10)
    print("Trained", models_df.count(), "models in parallel — one per group.")
except NameError:
    print("`spark` not defined — run on a Databricks cluster to execute this cell.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 4. Pattern B — Distributed batch inference with `mapInPandas`

When you need to **score a large table** with an already-trained model, `mapInPandas` streams **batches of rows** (as pandas DataFrames) through your function on each partition. Compared to a row-at-a-time UDF, you:

- Load the model **once per partition** (not once per row).
- Vectorize prediction over a whole batch.
- Control memory by iterating batch-by-batch (the input is an *iterator* of DataFrames).

This keeps every executor core busy scoring its share of the data.
"""))

cells.append(code(r'''
# Distributed batch scoring: load model once per partition, predict in vectorized batches.
try:
    from pyspark.sql.functions import rand, col

    score_df = (spark.range(0, 1_000_000)
                     .withColumn("x1", rand(seed=10))
                     .withColumn("x2", rand(seed=11)))

    def predict_batches(iterator):
        # Runs on an executor. Load the model ONCE here, outside the per-batch loop.
        import pandas as pd
        # In practice: model = mlflow.sklearn.load_model("models:/my_model/Production")
        # Demo: a fixed linear rule stands in for a loaded model.
        w1, w2, b = 2.0, 1.0, 0.05
        for pdf in iterator:                     # each pdf is a batch of rows
            preds = pdf["x1"] * w1 + pdf["x2"] * w2 + b
            pdf = pdf.assign(prediction=preds)
            yield pdf

    out_schema = "id long, x1 double, x2 double, prediction double"
    scored = score_df.mapInPandas(predict_batches, schema=out_schema)
    scored.show(5)
    print("Scored", scored.count(), "rows across the cluster (model loaded once per partition).")
except NameError:
    print("`spark` not defined — run on a Databricks cluster to execute this cell.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
### `mapInPandas` vs `applyInPandas` vs Pandas UDF

| API | Input → Output | Use for |
|---|---|---|
| **Pandas UDF** (Series→Series) | column → column | Vectorized feature transforms |
| **`mapInPandas`** | iterator of DataFrames → iterator of DataFrames | Batch inference, flexible row counts |
| **`applyInPandas`** (grouped) | one group DataFrame → one DataFrame | Per-group model training / aggregation |
| **`applyInPandas`** (cogrouped) | two group DataFrames → one DataFrame | Per-key joins (e.g., `merge_asof`) |
"""))

cells.append(md(r"""
## 5. Pattern C — Distributed hyperparameter tuning

Tuning one model = training it many times with different configs. Those trials are **embarrassingly parallel**. Databricks' current recommendations:

- **Optuna** — lightweight, dynamic search spaces; parallelize trials with **Joblib**; track with MLflow. Great for single-node-heavy or moderate parallelism.
- **Ray Tune** — the distributed tuning library; uses Ray as the backend to spread trials across the whole cluster. Built into Databricks Runtime ML.

> 📌 **Deprecation note:** **Hyperopt is deprecated** — the open-source project is no longer maintained and it was **removed from Databricks Runtime ML after 16.4 LTS**. Migrate `SparkTrials`/Hyperopt code to **Optuna** or **Ray Tune**.
"""))

cells.append(code(r'''
# Optuna hyperparameter search with MLflow tracking (single-driver, multi-trial).
try:
    import optuna
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(n_samples=2000, n_features=20, random_state=0)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 16),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        }
        clf = RandomForestClassifier(**params, random_state=0, n_jobs=-1)
        return cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best accuracy:", round(study.best_value, 4))
    print("Best params:", study.best_params)
except ImportError:
    print("Optuna not installed here. It ships with Databricks Runtime ML.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(code(r'''
# Scale Optuna trials ACROSS the cluster with the Joblib-Spark backend.
try:
    import optuna
    import joblib
    from joblibspark import register_spark

    register_spark()  # registers "spark" as a Joblib backend

    # With the spark backend, independent trials are dispatched to executors.
    # study.optimize can be wrapped so its internal parallel calls use Spark:
    with joblib.parallel_backend("spark", n_jobs=-1):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, n_jobs=-1)
    print("Distributed best:", round(study.best_value, 4), study.best_params)
except ImportError:
    print("joblib-spark not installed. `pip install joblibspark` or use Ray Tune (below).")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(code(r'''
# Ray Tune: the recommended DISTRIBUTED tuner on Databricks.
try:
    import ray
    from ray import tune
    from ray.util.spark import setup_ray_cluster, shutdown_ray_cluster

    setup_ray_cluster(num_worker_nodes=2, num_cpus_worker_node=4)
    ray.init(ignore_reinit_error=True)

    def trainable(config):
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
        clf = RandomForestClassifier(
            n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=0)
        score = cross_val_score(clf, X, y, cv=3).mean()
        tune.report(accuracy=score)

    tuner = tune.Tuner(
        trainable,
        param_space={
            "n_estimators": tune.randint(50, 300),
            "max_depth": tune.randint(2, 16),
        },
        tune_config=tune.TuneConfig(num_samples=20, metric="accuracy", mode="max"),
    )
    results = tuner.fit()
    print("Best config:", results.get_best_result().config)

    shutdown_ray_cluster()
except ImportError:
    print("Ray not available here. Ray + Ray Tune ship with Databricks Runtime ML.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 6. Pattern D — Distribute scikit-learn with `joblib-spark`

Many scikit-learn tools (e.g., `GridSearchCV`, `RandomForestClassifier`, cross-validation) parallelize through **Joblib**. The **`joblib-spark`** backend redirects that parallelism onto the **Spark cluster** — so a grid search that would run on the driver's cores instead fans out across executors, no code rewrite beyond a context manager.
"""))

cells.append(code(r'''
# GridSearchCV distributed over the cluster via the Spark Joblib backend.
try:
    from sklearn.datasets import make_classification
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from joblibspark import register_spark
    import joblib

    register_spark()
    X, y = make_classification(n_samples=1500, n_features=15, random_state=0)

    param_grid = {"C": [0.1, 1, 10, 100], "gamma": [1, 0.1, 0.01, 0.001], "kernel": ["rbf"]}
    grid = GridSearchCV(SVC(), param_grid, cv=3)

    # Each (param combo × fold) is a task dispatched to an executor.
    with joblib.parallel_backend("spark", n_jobs=-1):
        grid.fit(X, y)

    print("Best score:", round(grid.best_score_, 4))
    print("Best params:", grid.best_params_)
except ImportError:
    print("Install joblibspark to run this: `pip install joblibspark`.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 7. Pattern E — Distributed ML with Spark MLlib

When you want to train **one model over data that doesn't fit on a single machine**, use Spark's own ML library — the algorithms are distributed by design:

- **`pyspark.ml`** — the classic DataFrame-based MLlib (pipelines, feature transformers, estimators).
- **`pyspark.ml.connect`** — a newer module for distributed training/inference that works with **Spark Connect** and is available by default on **Databricks Runtime 17.0+ Standard compute**.

The training data stays distributed across the cluster; the algorithm coordinates across partitions.
"""))

cells.append(code(r'''
# Distributed logistic regression with Spark MLlib pipelines.
try:
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml import Pipeline
    from pyspark.sql.functions import rand, when, col

    data = (spark.range(0, 200_000)
                 .withColumn("f1", rand(seed=1))
                 .withColumn("f2", rand(seed=2))
                 .withColumn("label", when(col("f1") + col("f2") > 1, 1.0).otherwise(0.0)))

    assembler = VectorAssembler(inputCols=["f1", "f2"], outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    pipeline = Pipeline(stages=[assembler, lr])

    train, test = data.randomSplit([0.8, 0.2], seed=42)
    model = pipeline.fit(train)              # training is distributed across the cluster
    preds = model.transform(test)
    acc = preds.filter(col("label") == col("prediction")).count() / preds.count()
    print(f"Distributed LogisticRegression accuracy: {acc:.4f}")
except NameError:
    print("`spark` not defined — run on a Databricks cluster.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 8. Utilizing GPUs efficiently (task scheduling)

If your cluster has GPUs, controlling how tasks map to GPUs is what separates full utilization from waste (from the GPU compute docs):

- **Training** — set `spark.task.resource.gpu.amount` = **number of GPUs per worker**, so a single task owns all the node's GPUs (minimizes communication overhead).
- **Inference** — set it to a **fraction** (`0.5`, `0.33`, `0.25`) so **2–4 tasks share each GPU**, packing more concurrent scoring tasks onto the hardware.
- GPU scheduling is **not available on single-node clusters** (there are no separate executor tasks to schedule).

```python
# Example: multiple inference tasks per GPU (set in cluster Spark config)
spark.conf.set("spark.task.resource.gpu.amount", "0.5")   # 2 tasks per GPU
```
"""))

cells.append(md(r"""
## 9. Cluster-utilization checklist

Keep the whole cluster busy:

- **Right-size partitions** — target ~2–4× total cores; repartition after heavy filters/joins.
- **Avoid the driver bottleneck** — never `.collect()` / `.toPandas()` a big dataset to loop in plain Python. Push work into `applyInPandas` / `mapInPandas`.
- **Beware skew** — uneven group sizes leave some executors idle and others OOM. Salt keys or split hot groups.
- **Leave AQE on** — Adaptive Query Execution auto-tunes shuffle partitions at runtime.
- **Match the pattern to the shape** — per-group → grouped map; many configs → distributed tuning; big single model → MLlib.
- **Autoscale for bursty jobs**, fixed size for steady ones; use cluster policies to standardize.
- **Watch the Spark UI / Ganglia metrics** — look for stragglers, spill, and idle cores.

### Key takeaways
- Parallelism is about **filling the cluster** — pick the pattern that matches the workload's shape.
- **`applyInPandas`** turns single-node libraries into per-group distributed training.
- **`mapInPandas`** gives efficient distributed batch inference.
- **Optuna + Ray Tune** are the current tuning stack (**Hyperopt is deprecated**).
- **`joblib-spark`** distributes scikit-learn with almost no code change.
- **MLlib / `pyspark.ml.connect`** trains one big model on distributed data.
"""))

cells.append(md(r"""
## References
- [Pandas function APIs (`applyInPandas`, `mapInPandas`, cogroup)](https://docs.databricks.com/pandas/pandas-function-apis)
- [Pandas user-defined functions](https://docs.databricks.com/udf/pandas)
- [Hyperparameter tuning](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/)
- [Optuna on Databricks](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/optuna)
- [Ray Tune on Databricks](https://docs.databricks.com/machine-learning/ray/)
- [`joblib-spark` — distribute scikit-learn](https://docs.databricks.com/machine-learning/train-model/distributed-training/)
- [Apache Spark MLlib on Databricks](https://docs.databricks.com/machine-learning/train-model/mllib)
- [GPU-enabled compute & task scheduling](https://docs.databricks.com/compute/gpu)
"""))

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "02_parallelization_techniques.ipynb")
write_notebook(os.path.abspath(out), cells)
