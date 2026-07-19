"""Generate Topic 3 notebook: Feature Store concepts — reusability & consistency."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from nbbuild import md, code, write_notebook

cells = []

cells.append(md(r"""
# Feature Store: Reusability & Consistency Across Models
### Workshop — Topic 3 of 6

**Goal:** Understand how the Databricks Feature Store (Feature Engineering in Unity Catalog) makes
features **reusable across teams and models** and **consistent between training and serving** — and
run the end-to-end workflow.

> ⚙️ **Recommended compute:** a **Databricks Runtime ML** cluster (15.4 LTS ML or later) in a
> **Unity Catalog-enabled** workspace. Feature Engineering in UC needs the `databricks-feature-engineering` package (bundled in DBR ML).

---

### What you'll learn
1. What a feature store is and the two problems it solves: **reuse** and **train/serve skew**.
2. **Feature tables** in Unity Catalog — just Delta tables with a **primary key**.
3. The `FeatureEngineeringClient` end-to-end flow: `create_table` → `create_training_set` → `log_model` → `score_batch`.
4. How **packaging feature metadata with the model** removes the need to recompute features at inference.
5. **Discovery, lineage, and governance** that make features reusable across the org.

*Source: Databricks docs — [Feature Store](https://docs.databricks.com/machine-learning/feature-store/), [Train models with Feature Store](https://docs.databricks.com/machine-learning/feature-store/train-models-with-feature-store).*
"""))

cells.append(md(r"""
## 1. Why a feature store? Two problems it solves

A **feature store is a central registry for the features used in your AI/ML models.** On Databricks
this is **Feature Engineering in Unity Catalog** — features live in governed Delta tables.

### Problem 1 — Reinventing features (no reuse)
Without a feature store, every project recomputes "customer 30-day spend," "days since last order,"
etc. from scratch. The logic drifts between teams, wastes effort, and produces subtly different
numbers for the "same" feature.

### Problem 2 — Train/serve skew (no consistency)
The classic production failure: features are computed **one way in the training pipeline** (batch SQL
over a warehouse) and **another way at serving time** (hand-written app code). The two drift apart,
so the model sees different inputs in production than it trained on — and quietly degrades.

> **The feature store fixes both:** features are computed **once**, stored in a governed table,
> **discovered and reused** by any model, and **looked up identically** at training and inference —
> eliminating skew.
"""))

cells.append(md(r"""
## 2. Setup — the FeatureEngineeringClient

All feature operations go through the `FeatureEngineeringClient` from the
`databricks-feature-engineering` package (preinstalled on Databricks Runtime ML).
"""))

cells.append(code(r'''
# The Feature Engineering client is the entry point for all feature-store operations.
try:
    from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
    fe = FeatureEngineeringClient()
    print("FeatureEngineeringClient ready.")
except Exception as e:
    print("Runs on Databricks Runtime ML with Unity Catalog. Error:", repr(e))

# Naming: features live in a UC three-level namespace -> catalog.schema.table
CATALOG = "main"
SCHEMA = "feature_workshop"
CUSTOMER_FT = f"{CATALOG}.{SCHEMA}.customer_features"
'''))

cells.append(md(r"""
## 3. Create a feature table

A **feature table is just a Delta table in Unity Catalog with a primary key.** That primary key is
what lets any model **look up** features by an entity id (e.g. `customer_id`).

Key points:
- Define one or more **primary keys** (the entity identifier).
- The table is governed by Unity Catalog — permissions, lineage, and discovery come for free.
- You write to it like any Delta table; you can recompute/refresh on a schedule.
"""))

cells.append(code(r'''
# Build a small feature DataFrame, then register it as a feature table.
try:
    from pyspark.sql import functions as F

    # Create the schema if needed (UC).
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

    # Compute some example customer features.
    customer_features = (
        spark.range(0, 1000)
             .withColumnRenamed("id", "customer_id")
             .withColumn("total_purchases_30d", (F.rand(seed=1) * 500).cast("double"))
             .withColumn("days_since_last_order", (F.rand(seed=2) * 60).cast("int"))
             .withColumn("avg_basket_size", (F.rand(seed=3) * 8 + 1).cast("double"))
    )

    # Register as a feature table with a primary key.
    fe.create_table(
        name=CUSTOMER_FT,
        primary_keys=["customer_id"],
        df=customer_features,
        description="Customer behavioral features (30-day window). Reusable across models.",
    )
    print(f"Created feature table: {CUSTOMER_FT}")
except NameError:
    print("`spark`/`fe` not defined — run on a Databricks ML cluster with UC.")
except Exception as e:
    print("If the table already exists, use fe.write_table(...) to update it. Error:", repr(e))
'''))

cells.append(md(r"""
### Updating features

To refresh feature values (e.g. a nightly job recomputes the 30-day window), write to the same
table with `mode="merge"` — downstream models automatically pick up the governed, single source of truth.

```python
fe.write_table(name=CUSTOMER_FT, df=updated_features, mode="merge")
```
"""))

cells.append(md(r"""
## 4. Reuse features in training — `FeatureLookup` + `create_training_set`

Here's the reuse mechanism. Your **training labels** live in one DataFrame keyed by `customer_id`.
Instead of re-joining feature logic by hand, you declare **`FeatureLookup`** objects and let the
Feature Store perform the join. The result is a **`TrainingSet`** that knows exactly which features
(and versions) fed the model.
"""))

cells.append(code(r'''
# Declare which features to pull, then build a training set by lookup.
try:
    from pyspark.sql import functions as F

    # Raw training events: the label + the entity key (NO features pre-joined).
    labels_df = (
        spark.range(0, 1000)
             .withColumnRenamed("id", "customer_id")
             .withColumn("churned", (F.rand(seed=9) > 0.7).cast("int"))
    )

    feature_lookups = [
        FeatureLookup(
            table_name=CUSTOMER_FT,
            feature_names=["total_purchases_30d", "days_since_last_order", "avg_basket_size"],
            lookup_key="customer_id",     # matches the key column in labels_df
        )
    ]

    training_set = fe.create_training_set(
        df=labels_df,
        feature_lookups=feature_lookups,
        label="churned",
        exclude_columns=["customer_id"],   # drop the key from the model inputs
    )

    training_df = training_set.load_df()   # materialized, feature-joined training data
    training_df.show(5)
except NameError:
    print("`spark`/`fe` not defined — run on a Databricks ML cluster with UC.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
**Why this matters for reuse:** three different models (churn, LTV, next-best-offer) can all declare
a `FeatureLookup` against `customer_features`. They share the *same* definitions and values — no
copy-paste feature code, no drift.
"""))

cells.append(md(r"""
## 5. Train and log the model **with feature metadata**

This is the step that guarantees **consistency**. Instead of `mlflow.sklearn.log_model`, use
**`fe.log_model`** and pass the `training_set`. This **packages the feature lookup metadata *into*
the model**: the model now "knows" which feature tables and columns it needs.
"""))

cells.append(code(r'''
# Train a simple model, then log it WITH the feature metadata via fe.log_model.
try:
    import mlflow
    from sklearn.ensemble import RandomForestClassifier

    pdf = training_df.toPandas()
    X = pdf[["total_purchases_30d", "days_since_last_order", "avg_basket_size"]]
    y = pdf["churned"]

    model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)

    fe.log_model(
        model=model,
        artifact_path="churn_model",
        flavor=mlflow.sklearn,
        training_set=training_set,              # <-- packages feature lookups with the model
        registered_model_name=f"{CATALOG}.{SCHEMA}.churn_model",
    )
    print("Model logged with feature metadata — it now carries its own feature lookups.")
except NameError:
    print("Run on a Databricks ML cluster with UC.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 6. Inference — `score_batch` with automatic feature lookup

Because the model was logged **with** its feature metadata, at inference you only supply the
**primary keys**. `fe.score_batch` automatically:
1. Reads the primary keys from your input DataFrame.
2. **Looks up the current feature values** from the feature table(s).
3. Assembles the exact feature vector the model expects and predicts.

**This is what kills train/serve skew** — the *same lookup logic* runs at training and inference, so
the features are identical by construction. You never re-implement feature code in the serving path.
"""))

cells.append(code(r'''
# Score new customers by KEY ONLY — features are fetched automatically.
try:
    scoring_df = spark.range(0, 20).withColumnRenamed("id", "customer_id")

    model_uri = f"models:/{CATALOG}.{SCHEMA}.churn_model/1"
    predictions = fe.score_batch(model_uri=model_uri, df=scoring_df)
    predictions.select("customer_id", "prediction").show()
    print("Note: scoring_df had NO features — score_batch looked them up from the feature table.")
except NameError:
    print("Run on a Databricks ML cluster with UC.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 7. Offline vs. online — real-time serving

- **Offline store** = the Delta feature table. Used for **training** and **batch inference** (what we did above).
- **Online store** = a low-latency store (Databricks **Online Feature Store**, powered by **Lakebase**)
  that serves the same features to **real-time model serving** endpoints with **millisecond latency**.

The important part: **the same feature table definition** backs both. Publish the offline table to an
online store and a served model does the identical lookup online that it did offline — again, no skew.

```python
# Sketch: publish features to an online store for real-time serving
# fe.publish_table(name=CUSTOMER_FT, online_store=<online_store_spec>)
```
"""))

cells.append(md(r"""
## 8. Discovery, lineage & governance — what makes features *reusable*

Reusability isn't just the API — it's the governance layer around it:

- **Discovery** — feature tables appear in **Catalog Explorer** and the **Features UI**, so a data
  scientist can *find* "customer_features" instead of rebuilding it. Add **tags** (key-value) to categorize.
- **Lineage** — Unity Catalog tracks which **models** consumed which **features**, and which
  **upstream tables** produced them. You can answer "what breaks if I change this column?"
- **Governance** — UC permissions control who can read/write each feature table; sharing works
  **across workspaces**.
- **Automatic lineage on models** — a model logged with `fe.log_model` records the exact features it used.

> Reuse = *discoverable* + *governed* + *lineage-tracked*. That's why a feature store beats a folder of feature-engineering notebooks.
"""))

cells.append(md(r"""
## 9. Key takeaways

- A **feature store** solves two problems: **feature reuse** (one definition, many models) and
  **train/serve consistency** (identical lookups everywhere).
- A **feature table** is a **Delta table in UC with a primary key** — governed, discoverable, lineage-tracked.
- The workflow: **`create_table` → `FeatureLookup` + `create_training_set` → `fe.log_model` → `fe.score_batch`.**
- **`fe.log_model` packages feature metadata with the model**, so inference needs only the **keys** —
  `score_batch` fetches features automatically. This is what eliminates skew.
- **Offline (training/batch)** and **online (real-time)** stores share the same definitions.
- **Discovery + lineage + governance** are what make features genuinely reusable across the org.

> Next topic: how the *time* dimension of feature lookups prevents **data leakage** (point-in-time correctness).
"""))

cells.append(md(r"""
## References
- [Feature Store (Feature Engineering in Unity Catalog)](https://docs.databricks.com/machine-learning/feature-store/)
- [Feature tables in Unity Catalog](https://docs.databricks.com/machine-learning/feature-store/uc/feature-tables-uc)
- [Train models with Feature Store](https://docs.databricks.com/machine-learning/feature-store/train-models-with-feature-store)
- [Feature Engineering Python API](https://docs.databricks.com/machine-learning/feature-store/python-api)
- [Online Feature Store](https://docs.databricks.com/machine-learning/feature-store/online-feature-store)
"""))

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "03_feature_store_concepts.ipynb")
write_notebook(os.path.abspath(out), cells)
