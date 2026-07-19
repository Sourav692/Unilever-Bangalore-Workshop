# Topic 4 — Data Leakage Prevention in Feature Stores
### Presenter study guide (deep-dive for delivering the topic)

Your single reference for delivering Topic 4. Mirrors the notebook
(`notebooks/04_data_leakage_prevention.ipynb`) and deck
(`generated-slides/04_data_leakage_prevention.pptx`).

**Companion files**
- Notebook: `notebooks/04_data_leakage_prevention.ipynb`
- Deck: `generated-slides/04_data_leakage_prevention.pptx`
- Presenter flow & timing: `presenter-guide/README.md`

---

## 0. The one-sentence summary
> Data leakage is training on information that wasn't available when the label was recorded; Databricks prevents it with **point-in-time (AS OF) lookups** on **time series feature tables**, so each training row only ever sees features known **at or before** its label's timestamp.

---

## 1. Why this topic matters

This is the topic that saves someone's model from a humiliating production failure. Leakage is
**common, costly, and hard to spot** — because it makes your offline metrics look *better*, not worse.

**Talking point:** "If your validation accuracy is suspiciously great, your first suspicion should be
leakage, not genius. A leaky model aces the exam because it saw the answer key."

It builds directly on Topic 3: the same Feature Store that gives reuse and consistency also gives the
*time-correct* lookups that prevent leakage.

---

## 2. The mental model

Think of every labeled training row as a **snapshot taken at a specific moment**. The only features
you're allowed to use are the ones a camera at that moment could have captured.

| Naive join (leaky) | Point-in-time join (correct) |
|---|---|
| Join features by entity id → "latest" value | Join features **as of** the label's timestamp |
| Can pull values from *after* the label | Only values known **at or before** the label |
| Offline metrics look inflated | Offline metrics reflect reality |
| Model fails in production | Model behaves in production as in eval |

The whole topic is one move: **stop joining "the latest feature," start joining "the feature as of
when the label happened."** On Databricks that move is a single parameter — `timestamp_lookup_key` —
on a time series feature table.

---

## 3. Core concept — what data leakage is

> **Data leakage: using feature values for training that were not available at the time the label was recorded.**

The model learns from "the future," so it looks brilliant offline — then fails in production where the
future genuinely isn't available yet.

### The canonical example (use this in the room)
- A label is recorded at **08:50**.
- The feature table has a sensor reading from **08:52**.
- A naive "latest value" join attaches the **08:52** reading to the **08:50** label.
- The model trains on data **2 minutes into the future**.
- In production at 08:50, that reading doesn't exist → real accuracy collapses.

**Key line:** "Leakage is a time-travel bug. The feature came from *after* the moment you're
pretending to predict."

---

## 4. Where leakage sneaks in with features

Any join **by entity id alone** gives you the *current/latest* feature value, not the value **as of
the label's time**. For anything that changes over time — account balance, rolling 30-day spend,
sensor readings — that's leakage.

The fix is **point-in-time correctness**: for a label at time *t*, use the feature value current
**at or before *t***, never after.

---

## 5. The mechanic — time series feature tables + AS OF join

### Time series feature table
A Delta table in UC that, besides its primary key, has a **timeseries column** (`DATE`/`TIMESTAMP`)
declared as a **`TIMESERIES`** key. Each row = the latest known feature value **as of that timestamp**.

```python
fe.create_table(
    name="main.feature_workshop.user_reading_features",
    primary_keys=["user_id", "feature_ts"],
    timeseries_columns="feature_ts",         # declares point-in-time semantics
    df=feature_history,
)
```
```sql
CREATE TABLE ... (
  user_id STRING NOT NULL,
  feature_ts TIMESTAMP NOT NULL,
  reading_value DOUBLE,
  CONSTRAINT pk PRIMARY KEY (user_id, feature_ts TIMESERIES)
);
```

### The AS OF join (how the store does it)
For each label row, the Feature Store:
1. matches by **primary key** (excluding the timestamp),
2. takes the **latest feature timestamp that does *not* exceed** the label's timestamp,
3. returns **`null`** if no prior value exists.

So user 1 labeled at 08:50 gets the **08:45** reading, **not** 08:52.

**Constraints to know:** time series tables **can't have partition columns**; Databricks recommends
**≤ 2 primary key columns** for performance; writes must supply all feature values (reduces sparsity).

---

## 6. Point-in-time training set — the one parameter that matters

Add **`timestamp_lookup_key`** to the `FeatureLookup`. That single parameter turns a leaky lookup into
a leak-free one.

```python
feature_lookups = [
    FeatureLookup(
        table_name=TS_FT,
        feature_names=["reading_value"],
        lookup_key="user_id",
        timestamp_lookup_key="event_ts",   # AS OF the label's timestamp -> no future data
    )
]
training_set = fe.create_training_set(df=labels, feature_lookups=feature_lookups, label="label")
```

**Key line:** "`lookup_key` says *which* entity; `timestamp_lookup_key` says *as of when*. Add the
second one and the future can't leak in."

---

## 7. Bounding history — lookback_window

Even old *valid* features can be wrong to use (a 6-month-old value may be stale). `lookback_window`
excludes features older than a cutoff relative to the label timestamp:

```python
from datetime import timedelta
FeatureLookup(..., timestamp_lookup_key="event_ts", lookback_window=timedelta(days=7))
```

- Applies to **training and batch inference**.
- **Online (real-time) inference always uses the latest value** — correct, because at serving time
  "now" *is* the point in time.

---

## 8. Consistency carried to scoring

You don't re-implement point-in-time logic for prediction. A model logged with
`fe.log_model(training_set=...)` (Topic 3) will, on `fe.score_batch`, **re-apply the same AS OF logic**.

- Requirement: the scoring DataFrame must include a **timestamp column with the same name and type**
  as the training `timestamp_lookup_key`.
- Result: **leak-free at train time AND skew-free at serve time** — the two guarantees compose.

---

## 9. Practical guardrails (beyond point-in-time)

Point-in-time joins fix *temporal* leakage. Round it out:

- **Split before you engineer.** Fit scalers/encoders on the training split only — fitting on the full
  dataset leaks test statistics.
- **Never include the target** (or a proxy computed from it) among features.
- **Respect label windows.** If the label is "churned within 30 days," don't use features computed
  during that same window — they encode the outcome.
- **Prefer time series tables** for anything time-varying; always set `timestamp_lookup_key`.
- **Distrust "too good" metrics** — near-perfect offline scores usually mean leakage.
- **Use `lookback_window`** to drop stale features.

---

## 10. Notebook demo walkthrough (what to run, in order)

Notebook: `notebooks/04_data_leakage_prevention.ipynb`.

1. **Leakage illustration cell** — plain Spark data: labels at 08:50/09:15 and a feature history with
   a future reading. State out loud the correct value (12.0, not 99.0). This makes leakage concrete.
2. **create_table with `timeseries_columns`** — the time series feature table (+ the SQL equivalent).
3. **Point-in-time training set** — add `timestamp_lookup_key`; show the AS OF result picks the
   at-or-before value.
4. **lookback_window cell** — bound the history.
5. **score_batch markdown** — same logic re-applied at inference.
6. **Guardrails + key takeaways.**

> **Runnability note:** cells need a UC-enabled DBR ML cluster; off-platform they print a clear
> message. The leakage-illustration cell uses only Spark, so it runs on any cluster.

---

## 11. Likely audience questions (with answers)

**Q: What exactly is data leakage?**
A: Training on information not available when the label was recorded — most often a feature value from
*after* the label's timestamp. Great offline metrics, poor production performance.

**Q: Why do point-in-time joins fix it?**
A: They join, for each label at time *t*, the feature value known at or before *t* (an AS OF join),
so no future information enters training.

**Q: What makes a table a "time series feature table"?**
A: A primary key plus a timestamp column declared as a `TIMESERIES` key. Then the store can do AS OF
lookups against it.

**Q: What does `timestamp_lookup_key` do?**
A: It tells the `FeatureLookup` which column in your label DataFrame holds the event time, so the join
is done as of that time rather than "latest."

**Q: What's `lookback_window` for?**
A: To exclude features older than a cutoff (e.g. 7 days) relative to the label — dropping stale values.
It applies to training and batch inference; online serving always uses the latest.

**Q: Do I redo this logic at inference?**
A: No — `score_batch` re-applies the same point-in-time logic automatically, as long as the scoring
DataFrame carries the same timestamp column used in training.

**Q: My model scores 0.99 AUC in validation — good, right?**
A: Be suspicious. Near-perfect offline metrics are a classic leakage symptom. Check whether any
feature encodes the future or the target.

**Q: Is point-in-time only about feature stores?**
A: The temporal part is handled elegantly by time series feature tables, but leakage also comes from
splitting after engineering, target proxies, and label-window overlap — guard those too.

**Q: Any limits on time series tables?**
A: No partition columns; ≤ 2 primary keys recommended for performance; writes should supply all
feature values.

---

## 12. Glossary

- **Data leakage:** using information at training time that wasn't available when the label was recorded.
- **Point-in-time correctness:** using the feature value known at or before the label's timestamp.
- **Time series feature table:** a feature table with a declared `TIMESERIES` timestamp key.
- **AS OF join:** join that picks the latest feature row not exceeding the lookup timestamp.
- **`timestamp_lookup_key`:** FeatureLookup parameter naming the label DataFrame's time column.
- **`lookback_window`:** bound on how far back (relative to the label) features may be pulled.
- **Label window:** the time span a label summarizes (e.g. "churn in next 30 days").
- **Train/serve skew:** features differing between training and serving (see Topic 3).

---

## Sources (current Databricks documentation)
- [Point-in-time feature joins & time series feature tables](https://docs.databricks.com/machine-learning/feature-store/time-series)
- [Train models with Feature Store](https://docs.databricks.com/machine-learning/feature-store/train-models-with-feature-store)
- [Feature Engineering Python API](https://docs.databricks.com/machine-learning/feature-store/python-api)
- [Feature Store overview](https://docs.databricks.com/machine-learning/feature-store/)
