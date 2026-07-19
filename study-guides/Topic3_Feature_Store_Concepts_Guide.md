# Topic 3 — Feature Store: Reusability & Consistency Across Models
### Presenter study guide (deep-dive for delivering the topic)

Your single reference for delivering Topic 3. Mirrors the notebook
(`notebooks/03_feature_store_concepts.ipynb`) and deck
(`generated-slides/03_feature_store_concepts.pptx`).

**Companion files**
- Notebook: `notebooks/03_feature_store_concepts.ipynb`
- Deck: `generated-slides/03_feature_store_concepts.pptx`
- Presenter flow & timing: `presenter-guide/README.md`

---

## 0. The one-sentence summary
> A feature store lets you **compute a feature once**, store it in a **governed Delta table**, and have **any model reuse it** with the **exact same values at training and serving** — killing duplicated work and train/serve skew.

---

## 1. Why this topic matters

Feature engineering is where most ML time goes — and where most silent bugs live. Two recurring
pains, both expensive:

1. **Everyone rebuilds the same features.** Three teams each write their own "customer 30-day spend."
   Three slightly different definitions, three maintenance burdens, three sources of "why don't our
   numbers match?"
2. **Training and serving disagree.** The training pipeline computes features in batch SQL; the
   serving app recomputes them in application code. They drift. The model performs worse in
   production than in evaluation, and no one knows why.

**Talking point:** "The model is usually not the hard part — the features are. A feature store turns
features into a *governed, shared asset* instead of code copy-pasted between notebooks."

On Databricks this is **Feature Engineering in Unity Catalog** (the modern successor to the older
Workspace Feature Store, which is deprecated for new work).

---

## 2. The mental model

| Without a feature store | With a feature store |
|---|---|
| Feature logic duplicated per project | One definition in a governed table |
| Training vs. serving computed separately → skew | Identical lookup at train & serve |
| "Where is this feature? Does it exist?" | Discoverable in Catalog Explorer / Features UI |
| No idea which models use which features | Automatic lineage in Unity Catalog |

Think of a feature table as a **library book** everyone borrows, versus everyone **rewriting the book**
from memory each time they need it.

---

## 3. Core concept 1 — the feature table

> **A feature table is just a Delta table in Unity Catalog with a primary key.**

- The **primary key** is the entity id (e.g. `customer_id`) — it's what a model uses to *look up*
  features later.
- Because it's a UC table, it inherits **permissions, lineage, discovery, and tags** for free.
- You write/refresh it like any Delta table (e.g. a nightly job recomputes the 30-day window).

```python
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
fe = FeatureEngineeringClient()

fe.create_table(
    name="main.feature_workshop.customer_features",
    primary_keys=["customer_id"],
    df=customer_features,
    description="Customer behavioral features (30-day window). Reusable across models.",
)
# refresh later:
fe.write_table(name="...customer_features", df=updated, mode="merge")
```

**Key line:** "There's no special storage engine — it's a Delta table plus a declared primary key.
That's what makes it approachable."

---

## 4. Core concept 2 — reuse via FeatureLookup + create_training_set

This is the **reuse** mechanism. Your labels live in a DataFrame keyed by the entity id, *without*
features pre-joined. You declare **which features** to pull and let the store do the join.

```python
feature_lookups = [
    FeatureLookup(
        table_name="main.feature_workshop.customer_features",
        feature_names=["total_purchases_30d", "days_since_last_order", "avg_basket_size"],
        lookup_key="customer_id",
    )
]
training_set = fe.create_training_set(
    df=labels_df, feature_lookups=feature_lookups,
    label="churned", exclude_columns=["customer_id"],
)
training_df = training_set.load_df()   # feature-joined training data
```

**Why it drives reuse:** a churn model, an LTV model, and a next-best-offer model can all declare a
`FeatureLookup` against the *same* `customer_features` table. Same definitions, same values, zero
copy-paste. The `TrainingSet` object also **remembers** exactly which features fed the model.

---

## 5. Core concept 3 — consistency via fe.log_model

This is the step that guarantees **train/serve consistency**. Instead of `mlflow.sklearn.log_model`,
use **`fe.log_model`** and pass the `training_set`:

```python
fe.log_model(
    model=model,
    artifact_path="churn_model",
    flavor=mlflow.sklearn,
    training_set=training_set,                 # packages the feature lookups INTO the model
    registered_model_name="main.feature_workshop.churn_model",
)
```

The model now **carries its own feature lookup metadata** — it knows which feature tables and columns
it needs. That's the hinge the next step swings on.

**Key line:** "The model is logged *with* its shopping list of features. At inference it can go get
them itself."

---

## 6. Core concept 4 — inference via score_batch (skew eliminated)

Because the feature metadata travels with the model, at inference you supply **only the primary keys**:

```python
predictions = fe.score_batch(
    model_uri="models:/main.feature_workshop.churn_model/1",
    df=scoring_df,      # only needs customer_id — NO features
)
```

`score_batch` reads the keys, **looks up the current feature values** from the feature table(s),
assembles the exact vector the model expects, and predicts.

> **This is the anti-skew guarantee:** the *same lookup logic* runs at training and inference, so the
> features are identical by construction. You never re-implement feature code in the serving path.

---

## 7. Offline vs. online serving

| | Offline store | Online store |
|---|---|---|
| **What** | The Delta feature table | Low-latency store (Databricks Online Feature Store, powered by **Lakebase**) |
| **Used for** | Training & batch inference | Real-time model serving |
| **Latency** | High throughput | **Millisecond** lookups |

The crucial point: **the same feature table definition backs both.** Publish the offline table to an
online store, and a served model does the identical lookup online that it did offline — no skew across
the batch/real-time boundary either.

---

## 8. What actually makes features "reusable" — governance

Reusability isn't only the API; it's the governance layer:

- **Discovery** — feature tables show up in **Catalog Explorer** and the **Features UI**; add **tags**
  (key-value) so people can *find* them instead of rebuilding.
- **Lineage** — UC tracks **model → features** and **features → upstream tables**. You can answer
  "what breaks if I change this column?"
- **Governance** — UC permissions gate read/write; sharing works **across workspaces**.
- **Automatic model lineage** — a model logged via `fe.log_model` records exactly the features it used.

**Key line:** "Reuse = discoverable + governed + lineage-tracked. That's the difference between a
feature store and a folder of feature notebooks."

---

## 9. Notebook demo walkthrough (what to run, in order)

Notebook: `notebooks/03_feature_store_concepts.ipynb`.

1. **Client setup** — `FeatureEngineeringClient()`, and the `catalog.schema.table` naming.
2. **create_table** — build customer features, register with a primary key. Point out it's a Delta table.
3. **FeatureLookup + create_training_set** — show labels with *no* features, then the joined `training_df`. This is "reuse."
4. **fe.log_model** — stress `training_set=` packaging metadata into the model.
5. **score_batch** — the payoff: score with keys only; features fetched automatically. This is "consistency."
6. **Offline vs online** markdown — same definition, two stores.
7. **Governance** markdown — discovery, lineage, tags.
8. **Key takeaways.**

> **Runnability note:** cells need a UC-enabled DBR ML cluster to execute; off-platform they print a
> clear message. If demoing live, pre-create the schema and pick a catalog you can write to
> (change `CATALOG`/`SCHEMA` at the top).

---

## 10. Likely audience questions (with answers)

**Q: Is a feature table something other than a Delta table?**
A: No — it's a Delta table in Unity Catalog with a declared primary key. That's what makes it usable
as a lookup source.

**Q: How is this different from just joining a features table myself?**
A: Three things: (1) the join logic is standardized and reused via `FeatureLookup`; (2) the model is
logged *with* that metadata so inference re-does the identical lookup (no skew); (3) you get discovery,
lineage, and governance automatically.

**Q: Do I have to pass features at inference?**
A: No — with `fe.log_model` + `score_batch`, you pass only the primary keys and the store looks up the
features. That's the whole point.

**Q: What's train/serve skew, concretely?**
A: When the features a model sees in production are computed differently from those it trained on
(different code, different timing). It silently degrades the model. The feature store removes it by
using the same lookup both times.

**Q: Offline vs online store — do I define features twice?**
A: No. Same table definition. Publish the offline table to an online store for real-time serving.

**Q: Is this the old Workspace Feature Store?**
A: The modern path is **Feature Engineering in Unity Catalog**. The workspace-local feature store is
deprecated for new work; UC gives you cross-workspace governance and lineage.

**Q: Can multiple models share one feature table?**
A: Yes — that's the reuse story. Each model just declares its own `FeatureLookup` against the shared table.

**Q: How do I refresh features?**
A: Write to the same table (e.g. `fe.write_table(..., mode="merge")`), typically on a schedule. Models
pick up the governed source of truth.

**Q: How does this connect to the next topic?**
A: Topic 4 adds the *time* dimension — point-in-time lookups on time series feature tables to prevent
data leakage.

---

## 11. Glossary

- **Feature store:** central, governed registry of features for ML models.
- **Feature table:** a Delta table in UC with a primary key, used as a feature lookup source.
- **FeatureEngineeringClient (`fe`):** the Python entry point for feature operations.
- **FeatureLookup:** a declaration of which features to pull from which table, by key.
- **TrainingSet:** the object produced by `create_training_set`; remembers the feature lookups.
- **`fe.log_model`:** logs a model packaged with its feature metadata.
- **`fe.score_batch`:** batch inference that auto-looks-up features from keys.
- **Train/serve skew:** features differing between training and serving; a top production failure mode.
- **Offline / online store:** batch feature table vs. low-latency real-time store (Lakebase).
- **Lineage:** UC's tracking of which models/tables depend on which features.

---

## Sources (current Databricks documentation)
- [Feature Store (Feature Engineering in Unity Catalog)](https://docs.databricks.com/machine-learning/feature-store/)
- [Feature tables in Unity Catalog](https://docs.databricks.com/machine-learning/feature-store/uc/feature-tables-uc)
- [Train models with Feature Store](https://docs.databricks.com/machine-learning/feature-store/train-models-with-feature-store)
- [Feature Engineering Python API](https://docs.databricks.com/machine-learning/feature-store/python-api)
- [Online Feature Store](https://docs.databricks.com/machine-learning/feature-store/online-feature-store)
