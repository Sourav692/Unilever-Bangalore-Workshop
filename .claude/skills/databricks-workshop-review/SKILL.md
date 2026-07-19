---
name: databricks-workshop-review
description: Review and validate the workshop deliverables produced by databricks-workshop-builder (notebook, PPTX deck, study guide, presenter notes) for one Databricks topic. Runs automated structural checks and a human-judgment quality checklist covering accuracy, doc-grounding, pedagogy, and consistency, then drives a fix loop. Use after building or editing workshop material, or when the user asks to review/validate a topic's tutorial materials.
---

# Databricks Workshop Review

The **validate** half of the develop → validate → create loop. Given a built topic, this skill
verifies the four deliverables are complete, correct, runnable, doc-grounded, and internally
consistent — then reports findings ranked so the builder can fix and regenerate.

## When to use

- Right after `databricks-workshop-builder` produces a topic (always review before calling it done).
- When the user asks to review, validate, QA, or check workshop / tutorial material.
- After any manual edit to a notebook, deck, or study guide, to re-verify.

## Inputs

- `--root` — the workshop directory (contains `notebooks/`, `generated-slides/`, `study-guides/`, `presenter-guide/`).
- Topic `--num` (zero-padded, e.g. `03`), `--slug` (lower_snake_case), and `--guide` (study-guide filename without `.md`).

If unknown, infer them from the files present (`ls notebooks/`, `ls study-guides/`).

---

## Step 1 — Run the automated checks

```bash
python3 {baseDir}/scripts/validate.py --root <workshop_dir> --num NN --slug topic_slug --guide TopicN_Title_Guide
```

The script checks, and exits non-zero on any ERROR:

- **Notebook:** valid JSON, nbformat 4, cell count, has markdown + code, first cell is a title,
  **every code cell parses as valid Python**, uses off-platform guard patterns, links docs.
- **Deck:** source JSON is version-controlled and valid, first slide is `title`, has a `closing`,
  good slide-type variety (≥5 distinct), likely has a "which framework/when" slide; **PPTX is a
  valid package with slides**.
- **Study guide:** substantial (≥1200 words), covers all required sections, has enough Q&A.
- **Presenter notes:** README references the topic.

Fix every ✗ ERROR. Treat each ⚠ WARNING as a prompt to look closer (some are acceptable — use judgment).

## Step 2 — Human-judgment review (the checklist the script can't do)

Read the actual content and score each dimension. The script proves the files are *well-formed*;
you must confirm they are *correct and good*.

### A. Accuracy & doc-grounding (highest priority)
- [ ] Every API/class name, parameter, and default matches **current** Databricks docs (re-fetch if unsure).
- [ ] Version/runtime gating is correct (e.g. "DBR 14.0 ML+", "17.0+ Standard compute").
- [ ] **Deprecations are flagged** and no deprecated tool is taught as current (e.g. Hyperopt, Workspace Feature Store).
- [ ] Code cells use the APIs the way the docs show; no invented parameters or methods.
- [ ] Claims (perf, behavior) are supported by the docs, not embellished.

### B. Pedagogy
- [ ] Concept-before-code: each section explains the *why* before showing code.
- [ ] There's a clear **mental model** and a **decision guide / "which tool when"**.
- [ ] Analogies are accurate (an analogy that misleads is worse than none).
- [ ] Difficulty ramps sensibly; no unexplained jargon before its glossary/first use.
- [ ] Key takeaways recap the essentials.

### C. Runnability
- [ ] Code is genuinely runnable on the stated compute; off-platform cells degrade gracefully.
- [ ] No external downloads that would break a live demo (synthetic data preferred).
- [ ] Function signatures used in `.run(...)`/calls match their definitions (arg count/order).
- [ ] Imports that ship to workers are inside the function.

### D. Deck quality
- [ ] Slide type matches the *shape* of each slide's content; no long runs of identical layouts.
- [ ] Titles ≤ 8 words; bullets ≤ 12 words; ≤ ~5 bullets/slide; one idea per slide.
- [ ] Rhythm: dense slides separated by section/callout/big-number breathing room.
- [ ] Deck and notebook tell the *same* story with the *same* terminology.

### E. Consistency across the set
- [ ] Notebook, deck, study guide, and presenter notes agree on facts, names, and structure.
- [ ] Matches the audience, format, and tone of earlier topics in the same workshop.
- [ ] File naming and directory layout follow the builder conventions.

### F. Cross-check against the official example (when one exists)
- [ ] If Databricks publishes an example notebook for this topic, decode it (see the builder skill's
      `references/decode-databricks-notebook.md`) and confirm the deliverables cover its core
      concepts. Report any gaps explicitly.

## Step 3 — Report findings

Produce a ranked list, most-severe first. For each: **file · location · what's wrong · why it
matters · suggested fix**. Group by severity:

- **❌ ERROR** — factually wrong, deprecated-as-current, broken code, missing deliverable. Must fix.
- **⚠️ IMPROVE** — accurate but weak: thin explanation, poor slide-type choice, missing Q&A, inconsistency.
- **✅ STRENGTH** — note what's good so it's preserved on regeneration.

End with a one-line verdict: **PASS** (ship it) or **NEEDS FIXES** (list the blocking ERRORs).

## Step 4 — Drive the fix loop

If NEEDS FIXES: hand the ERROR/IMPROVE list back to the build step, apply fixes at the **source**
(edit `build/gen_topicN.py` and `presenter-guide/slides-topicN.json`, not the generated files),
**regenerate**, and re-run this review. Loop until the automated checks pass and no ERROR-level
judgment findings remain. Never sign off on an unreviewed or failing build.

## Notes

- Prefer editing generators/sources and regenerating over hand-patching `.ipynb`/`.pptx`.
- The automated script is necessary but not sufficient — a deck can pass every structural check and
  still be inaccurate. Accuracy and doc-grounding are the reviewer's job.
