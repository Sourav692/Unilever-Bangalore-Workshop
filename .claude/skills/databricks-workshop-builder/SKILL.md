---
name: databricks-workshop-builder
description: Build a complete set of Databricks workshop deliverables for a single topic — a runnable Jupyter notebook, a Databricks-branded PPTX deck, a presenter study guide, and presenter notes. Use when the user wants to create workshop / tutorial / training / demo material for a Databricks topic, grounded in the latest Databricks documentation. Follows a research → develop → validate → create loop.
---

# Databricks Workshop Builder

Produce four consistent, doc-grounded deliverables per workshop topic:

1. **Notebook** — `notebooks/NN_topic_slug.ipynb` — runnable, GPU/cluster-ready Jupyter notebook.
2. **Slide deck** — `generated-slides/NN_topic_slug.pptx` — Databricks-branded PowerPoint.
3. **Study guide** — `study-guides/TopicN_Title_Guide.md` — presenter deep-dive (learn it, then teach it).
4. **Presenter notes** — a per-topic section appended to `presenter-guide/README.md` (talk track, timing, demo flow, Q&A).

This skill is the **develop** half of a develop → validate → create loop. After building, always
run the companion **`databricks-workshop-review`** skill to validate, then fix and regenerate
until it passes. Only then present the topic as done.

## When to use

- The user asks to create a workshop / tutorial / training / demo / walkthrough on a Databricks topic.
- The user references `topics.md` (or similar) and asks to cover one or more topics.
- The user wants slides + notebook + study material as a set, consistent with earlier topics.

## Required inputs (ask if not known)

Confirm these once, then reuse for all topics in the same workshop:

- **Audience** (e.g. data scientists / ML engineers; mixed technical + leadership; platform engineers).
- **Notebook format** — default **Jupyter `.ipynb`** (imports cleanly to Databricks *and* opens locally).
- **Code style** — **runnable demo** (real, executable code, degrades gracefully off-Databricks) vs. illustrative snippets.
- **Topic number & title** — drives file naming (`NN` zero-padded, matching `topics.md` order).

If a previous topic in this workshop already established these, reuse them silently.

---

## The build loop (per topic)

### Step 1 — RESEARCH (always, before writing anything)

Ground every concept in **current** Databricks docs — do not rely on memory for product details,
API names, deprecations, or version gating.

1. Load the `databricks-docs` skill and fetch `https://docs.databricks.com/llms.txt` to find the
   relevant doc URLs for the topic.
2. `WebFetch` the specific pages. Pull: what the feature is, when to use it, exact API/class names,
   code patterns, version/runtime requirements, and any **deprecations or "recommended instead"** notes.
3. If the topic names a specific framework/library and the user asked for "latest," fetch the
   deep pages too (e.g. the API reference, the example notebook). Decode Databricks example
   notebooks if needed (they embed a base64+URL-encoded `__DATABRICKS_NOTEBOOK_MODEL` blob — see
   `references/decode-databricks-notebook.md`).
4. Note anything that changed recently (deprecated tools, new default modules, new runtime gating) —
   these make the strongest teaching points and must be accurate.

> `WebSearch` in this environment can be unreliable — prefer `WebFetch` on known docs.databricks.com URLs.

### Step 2 — DEVELOP the notebook

Write a **generator script** `build/gen_topicN.py` that assembles cells via the helper
`scripts/nbbuild.py` (copy it to the workshop's `build/` dir if not present). Do **not** hand-write
raw `.ipynb` JSON, and do **not** depend on `nbformat` (it may be unavailable; `nbbuild.py` needs no deps).

Notebook content rules:
- Open with a **title markdown cell**: topic name, workshop position, goal, recommended compute
  (runtime version + cluster type), a numbered "what you'll learn" list, and a doc-source line.
- **Concept before code.** Each section: a markdown explainer, then a runnable code cell.
- **Runnable + graceful degradation.** Every code cell that needs Databricks/Spark/GPU must be
  wrapped so it prints a clear message instead of crashing when run off-platform
  (`try/except`, `NameError` for missing `spark`, `torch.cuda.is_available()` guards, synthetic
  data instead of external downloads).
- **Imports inside functions** that get shipped to workers (pickling).
- Match the **latest** APIs from Step 1 exactly; call out deprecations inline.
- Include a **decision guide / "which tool when"** section and a **key takeaways** recap.
- End with a **References** markdown cell linking the docs used.
- Write to `notebooks/NN_topic_slug.ipynb`.

### Step 3 — DEVELOP the slide deck

Use the **`databricks-slides:slide-deck`** skill's generator. Author a content JSON and save the
source to `presenter-guide/slides-topicN.json` (so it's regenerable and version-controlled).

Deck design rules:
- **Match slide type to the *shape* of the content** (see the slide-deck skill's shape table). Avoid
  long runs of bullet slides — vary `section`, `callout`, `timeline`, `two-column`, `cards`,
  `icon-grid`, `checklist`, `comparison`, `pros-cons`, `three-column`, `card-left/right/full`.
- **Rhythm:** dense slides need breathing room (a `callout`/`section`/`big-number` between them).
- Include, for a technical topic: title, agenda, a "mental model" section, the core concepts,
  a **"which framework/library — use each when"** slide (cards or comparison), config/best-practice
  checklist, key-takeaways `content` slide, and a `closing`.
- Keep titles ≤ 8 words, bullets ≤ 12 words, 3–5 bullets per slide, one idea per slide.
- Generate to `generated-slides/NN_topic_slug.pptx`. Validate the JSON parses before generating.

Generation commands:
```bash
SKILL_DIR="$HOME/.claude/plugins/marketplaces/plugin-marketplace/experimental/general/databricks-slides/skills/slide-deck"
python3 "$SKILL_DIR/scripts/generate-pptx.py" --input presenter-guide/slides-topicN.json --output ./generated-slides/NN_topic_slug.pptx
```
(The generator reads any path; pass the repo path directly. `python-pptx` is required.)

### Step 4 — DEVELOP the study guide

Write `study-guides/TopicN_Title_Guide.md` — the presenter's deep-dive. Required sections:

- **One-sentence summary** ("if they remember only this…").
- **Why this topic matters** (with a plain-language talking point).
- **The mental model** (a table or ladder).
- **Core concepts**, each with a short analogy where it helps.
- **Which framework/library, when** table.
- **Config / best-practice** details.
- **Notebook demo walkthrough** — what to run, in order, and what to say.
- **Likely audience questions with answers** (≥ 8).
- **Glossary** of terms.
- **Sources** — the doc links used.

Tone: explain the *why*, give "key lines" to say aloud, and cross-reference the notebook + deck.

### Step 5 — DEVELOP presenter notes

Append (or create) a topic section in `presenter-guide/README.md`: materials table row, setup
checklist, suggested timing, talk track, demo flow, and anticipated Q&A. Keep the file's existing
structure; add the new topic consistently with prior ones.

### Step 6 — VALIDATE (hand off to the review skill)

Run the **`databricks-workshop-review`** skill on the topic. Fix every ❌ and reconsider every ⚠️,
then **regenerate** (re-run `gen_topicN.py`, re-run the PPTX generator) and re-review. Loop until
the review passes. Never mark a topic done on an unreviewed or failing build.

### Step 7 — CREATE (finalize & report)

- Confirm the four files exist and are valid.
- Give the user a short summary: what each deliverable covers, any deprecations/gotchas surfaced,
  and the doc sources. Offer auto-upload of the deck to Google Slides (see the slide-deck skill).

---

## Conventions

- **File naming:** `NN` = zero-padded topic number from `topics.md`; `topic_slug` = lower_snake_case;
  `TopicN_Title_Guide.md` = TitleCase with underscores.
- **Directory layout** (create if missing):
  ```
  notebooks/            NN_topic_slug.ipynb
  generated-slides/     NN_topic_slug.pptx
  study-guides/         TopicN_Title_Guide.md
  presenter-guide/      README.md, slides-topicN.json
  build/                nbbuild.py, gen_topicN.py
  ```
- **Consistency across topics:** reuse the same audience, format, tone, and section structure as
  earlier topics in the same workshop so the set feels like one course.
- **Accuracy over fluency:** if the docs contradict prior knowledge, the docs win. Flag deprecations.

## Reference files

- `scripts/nbbuild.py` — dependency-free `.ipynb` builder (`md()`, `code()`, `write_notebook()`).
- `references/deliverable-templates.md` — skeleton outlines for each of the four deliverables.
- `references/decode-databricks-notebook.md` — how to decode a docs.databricks.com example notebook.
