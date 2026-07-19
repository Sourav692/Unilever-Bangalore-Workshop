# Deliverable skeletons

Copy and adapt these outlines. They match the structure used for Topics 1 & 2 so the set stays consistent.

---

## 1. Notebook generator (`build/gen_topicN.py`)

```python
"""Generate Topic N notebook: <Title>."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from nbbuild import md, code, write_notebook

cells = []

cells.append(md(r"""
# <Title>
### Workshop — Topic N of M

**Goal:** <one line>

> ⚙️ **Recommended compute:** <runtime + cluster type>

### What you'll learn
1. ...
2. ...

*Source: Databricks docs — [links].*
"""))

# ... concept markdown + runnable code cells, each guarded for off-platform use ...

cells.append(md(r"""
## References
- [doc links]
"""))

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "NN_topic_slug.ipynb")
write_notebook(os.path.abspath(out), cells)
```

Guard patterns for runnable-but-safe code cells:
```python
try:
    ...spark/torch/mlflow work...
except NameError:
    print("`spark` not defined — running outside Databricks. Read along; the pattern still applies.")
except Exception as e:
    print("Needs a Databricks cluster / library. Error:", repr(e))
```

---

## 2. Slide content JSON (`presenter-guide/slides-topicN.json`)

```json
{
  "title": "<Title>",
  "author": "Databricks Workshop — <Location>",
  "date": "<Month Year>",
  "slides": [
    {"type": "title", "title": "<Title>", "subtitle": "<sub>", "author": "...", "date": "..."},
    {"type": "agenda", "title": "What we'll cover", "items": ["...", "..."]},
    {"type": "section", "title": "<Mental model>", "subtitle": "..."},
    {"type": "callout", "text": "<the one key idea>", "source": "Databricks documentation"},
    {"type": "timeline", "title": "...", "steps": [{"title": "...", "description": "..."}]},
    {"type": "two-column", "title": "...", "left_header": "...", "left": ["..."], "right_header": "...", "right": ["..."]},
    {"type": "cards", "title": "Which framework — use each when", "cards": [{"header": "...", "items": ["..."]}]},
    {"type": "icon-grid", "title": "...", "items": [{"icon": "🔥", "title": "...", "description": "..."}]},
    {"type": "checklist", "title": "...", "items": [{"text": "...", "checked": true}]},
    {"type": "content", "title": "Key takeaways", "bullets": ["..."]},
    {"type": "closing", "title": "Let's run the notebook →"}
  ]
}
```

Valid slide types: title, section, section-description, content, one-column, closing,
two-column, two-column-icons, three-column, three-column-icons, cards, card-right, card-left,
card-full, big-number, stat-row, comparison, pros-cons, agenda, timeline, icon-grid, checklist,
quote, callout, logos.

---

## 3. Study guide (`study-guides/TopicN_Title_Guide.md`)

```markdown
# Topic N — <Title>
### Presenter study guide

**Companion files:** notebook, deck, presenter README (list paths).

## 0. The one-sentence summary
## 1. Why this topic matters        (+ talking point)
## 2. The mental model              (table / ladder)
## 3. Core concepts                 (each with an analogy where useful)
## ...topic-specific sections...
## N. Which framework/library, when (table)
## N. Notebook demo walkthrough     (what to run, in order)
## N. Likely audience questions     (>= 8 Q&A)
## N. Glossary
## Sources
```

---

## 4. Presenter notes (append to `presenter-guide/README.md`)

- Add a row to the materials table.
- Add a `## Topic N — <Title>` section with: talk track (slides), demo flow (notebook cells in order),
  anticipated questions. Keep formatting consistent with existing topics.
