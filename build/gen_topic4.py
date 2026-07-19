"""Generate Topic 4 notebook: Data leakage prevention in feature stores."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from nbbuild import md, code, write_notebook

cells = []

cells.append(md(r"""
# Data Leakage Prevention in Feature Stores
### Workshop — Topic 4 of 6

**Goal:** Understand **data leakage** in ML features and how Databricks **point-in-time lookups** and
**time series feature tables** prevent it — so your training data only ever "knows" what was actually
known when the label was recorded.

> ⚙️ **Recommended compute:** a **Databricks Runtime ML** cluster (15.4 LTS ML or later) in a
> **Unity Catalog-enabled** workspace.

---

### What you'll learn
1. What **data leakage** is and why it silently wrecks models.
2. **Point-in-time correctness** — the fix.
3. **Time series feature tables** — Delta tables with a **timeseries key**.
4. `FeatureLookup(timestamp_lookup_key=...)` and the **AS OF** join.
5. `lookback_window` to bound how far back features are pulled.
6. How the same point-in-time logic is re-applied automatically at scoring time.

*Source: Databricks docs — [Point-in-time feature joins / time series feature tables](https://docs.databricks.com/machine-learning/feature-store/time-series).*
"""))

cells.append(md(r"""
## 1. What is data leakage?

**Data leakage is when your training data includes information that would not have been available at
the moment the label was recorded.** The model learns from "the future," scores brilliantly in
offline evaluation, then **fails in production** where the future isn't available yet.

Databricks defines it precisely:

> *Data leakage happens "when you use feature values for model training that were not available at the time the label was recorded."*

### A concrete example
You're predicting whether a sensor event is anomalous. The **label** was recorded at **08:50**.
Your feature table has a CO₂ reading from **08:52**. If you naively join "latest reading" onto the
label, the model trains on a reading from **2 minutes into the future**. In production at 08:50 that
reading doesn't exist yet — so the model's real-world accuracy collapses.

> **The trap:** leakage makes offline metrics look *great*, so it's easy to miss until production. It's one of the most common and costly ML bugs.
"""))

cells.append(md(r"""
## 2. Where leakage sneaks in with features

Any time you join features to labels **by entity id alone**, you get the *current* (or latest)
feature value — not the value **as of the label's timestamp**. For any feature that changes over
time (account balance, 30-day spend, rolling averages, sensor readings), that's leakage.

The fix is **point-in-time correctness**: for each labeled event at time *t*, join the feature value
that was current **at or before *t*** — never after.
"""))

cells.append(code(r'''
# Illustrate the leakage vs. point-in-time difference with plain data (runs anywhere with Spark).
try:
    from pyspark.sql import functions as F

    # Labels: one event per user at a specific time.
    labels = spark.createDataFrame(
        [(1, "2024-01-10 08:50:00", 1),
         (2, "2024-01-10 09:15:00", 0)],
        ["user_id", "event_ts", "label"]
    ).withColumn("event_ts", F.to_timestamp("event_ts"))

    # Feature history: the SAME feature changes over time.
    feature_history = spark.createDataFrame(
        [(1, "2024-01-10 08:00:00", 10.0),
         (1, "2024-01-10 08:45:00", 12.0),   # <- valid as of 08:50
         (1, "2024-01-10 08:52:00", 99.0),   # <- FUTURE relative to 08:50 (leakage if used)
         (2, "2024-01-10 09:00:00", 5.0),
         (2, "2024-01-10 09:30:00", 7.0)],   # <- future relative to 09:15
        ["user_id", "feature_ts", "reading"]
    ).withColumn("feature_ts", F.to_timestamp("feature_ts"))

    print("For user 1 labeled at 08:50, the correct 'reading' is 12.0 (08:45), NOT 99.0 (08:52).")
    labels.show(truncate=False)
    feature_history.show(truncate=False)
except NameError:
    print("`spark` not defined — run on a Databricks cluster.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 3. Time series feature tables — the primitive

The Feature Store solves this with **time series feature tables**: a Delta table in Unity Catalog
that has, in addition to its primary key, a **timeseries column** (a `DATE`/`TIMESTAMP`).

- Each row represents the **latest known feature values as of that timestamp**.
- The timestamp column is declared as a **`TIMESERIES`** key.
- Databricks recommends **no more than two primary key columns** for performance, and time series
  tables **cannot have partition columns**.
"""))

cells.append(code(r'''
# Create a TIME SERIES feature table: primary key + a declared timeseries column.
try:
    from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
    from pyspark.sql import functions as F

    fe = FeatureEngineeringClient()
    CATALOG, SCHEMA = "main", "feature_workshop"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    TS_FT = f"{CATALOG}.{SCHEMA}.user_reading_features"

    fe.create_table(
        name=TS_FT,
        primary_keys=["user_id", "feature_ts"],   # entity key + timestamp
        timeseries_columns="feature_ts",          # <-- declares point-in-time semantics
        df=feature_history.withColumnRenamed("reading", "reading_value"),
        description="Time series features; supports point-in-time (AS OF) lookups.",
    )
    print(f"Created time series feature table: {TS_FT}")
except NameError:
    print("Run on a Databricks ML cluster with UC.")
except Exception as e:
    print("If it exists already, skip creation. Error:", repr(e))
'''))

cells.append(md(r"""
### Or in SQL

```sql
CREATE TABLE main.feature_workshop.user_reading_features (
    user_id     STRING    NOT NULL,
    feature_ts  TIMESTAMP NOT NULL,
    reading_value DOUBLE,
    CONSTRAINT pk_user_reading PRIMARY KEY (user_id, feature_ts TIMESERIES)
);
```

The `TIMESERIES` keyword on the timestamp column is what enables AS OF joins.
"""))

cells.append(md(r"""
## 4. Point-in-time training set — `timestamp_lookup_key`

Now the payoff. Add **`timestamp_lookup_key`** to the `FeatureLookup`. The Feature Store performs an
**AS OF join**:

> For each labeled row, match by primary key and take the **latest feature timestamp that does not
> exceed the label's timestamp**. If no prior value exists, return `null`.

That single parameter is the difference between leaky and correct training data.
"""))

cells.append(code(r'''
# Build a POINT-IN-TIME correct training set.
try:
    feature_lookups = [
        FeatureLookup(
            table_name=TS_FT,
            feature_names=["reading_value"],
            lookup_key="user_id",              # entity match
            timestamp_lookup_key="event_ts",   # AS OF the label's timestamp -> no future data
        )
    ]

    training_set = fe.create_training_set(
        df=labels,
        feature_lookups=feature_lookups,
        label="label",
        exclude_columns=["user_id"],
    )
    training_df = training_set.load_df()
    # user 1 (08:50) should get 12.0, NOT 99.0; user 2 (09:15) should get 5.0, NOT 7.0
    training_df.show(truncate=False)
    print("AS OF join returned the value known at/before each label time — leakage prevented.")
except NameError:
    print("Run on a Databricks ML cluster with UC.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
## 5. Bounding history — `lookback_window`

Sometimes even *old* valid features are wrong to use — a feature from 6 months ago may be stale.
`lookback_window` excludes feature values older than a cutoff **relative to the label timestamp**.
"""))

cells.append(code(r'''
# Only consider features within 7 days before each label.
try:
    from datetime import timedelta

    bounded_lookups = [
        FeatureLookup(
            table_name=TS_FT,
            feature_names=["reading_value"],
            lookup_key="user_id",
            timestamp_lookup_key="event_ts",
            lookback_window=timedelta(days=7),   # ignore features older than 7 days
        )
    ]
    bounded_set = fe.create_training_set(df=labels, feature_lookups=bounded_lookups, label="label",
                                         exclude_columns=["user_id"])
    bounded_set.load_df().show(truncate=False)
    print("Features older than the window are treated as unavailable (null).")
except NameError:
    print("Run on a Databricks ML cluster with UC.")
except Exception as e:
    print("Error:", repr(e))
'''))

cells.append(md(r"""
> **Note on inference:** `lookback_window` applies to **training and batch inference**. **Online
> (real-time) inference always uses the latest feature value** — which is correct, because at serving
> time "now" *is* the point in time.
"""))

cells.append(md(r"""
## 6. Consistency carries to scoring automatically

The point-in-time logic isn't something you re-implement for prediction. When you log the model with
`fe.log_model(training_set=...)` (Topic 3) and later call `fe.score_batch`, the Feature Store
**re-applies the same AS OF logic** during scoring.

Requirement: the scoring DataFrame must include a **timestamp column with the same name and type** as
the training `timestamp_lookup_key`. Then batch scoring reproduces exactly the point-in-time behavior
used in training — **no leakage at train time, and no skew at serve time.**

```python
# scoring_df must carry the same 'event_ts' column used as timestamp_lookup_key
predictions = fe.score_batch(model_uri=model_uri, df=scoring_df)
```
"""))

cells.append(md(r"""
## 7. Practical guardrails against leakage

Point-in-time joins fix *temporal* leakage. A few more habits prevent the rest:

- **Split before you engineer.** Do train/test splitting first; never fit scalers/encoders on the
  full dataset (that leaks test statistics into training).
- **Never include the target** (or a proxy computed from it) among features.
- **Watch label windows.** If a label reflects "churned within 30 days," don't use features computed
  during that same window — they encode the outcome.
- **Prefer time series feature tables** for anything time-varying, and always set `timestamp_lookup_key`.
- **Be suspicious of "too good" offline metrics** — near-perfect scores usually mean leakage.
- **Use `lookback_window`** to exclude stale features when appropriate.
"""))

cells.append(md(r"""
## 8. Key takeaways

- **Data leakage** = training on information not available when the label was recorded → great offline
  metrics, broken production.
- **Point-in-time correctness** is the fix: for each label at time *t*, use the feature value known
  **at or before *t***.
- **Time series feature tables** (Delta + a declared **`TIMESERIES`** column) enable **AS OF** joins.
- **`FeatureLookup(timestamp_lookup_key=...)`** produces a leak-free training set; **`lookback_window`**
  bounds how far back to look.
- The **same point-in-time logic is re-applied at scoring** via `fe.score_batch` — leakage-free
  training *and* skew-free serving.
- Combine with classic hygiene: split before engineering, exclude the target, respect label windows.
"""))

cells.append(md(r"""
## References
- [Point-in-time feature joins & time series feature tables](https://docs.databricks.com/machine-learning/feature-store/time-series)
- [Train models with Feature Store](https://docs.databricks.com/machine-learning/feature-store/train-models-with-feature-store)
- [Feature Engineering Python API](https://docs.databricks.com/machine-learning/feature-store/python-api)
- [Feature Store overview](https://docs.databricks.com/machine-learning/feature-store/)
"""))

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "04_data_leakage_prevention.ipynb")
write_notebook(os.path.abspath(out), cells)
