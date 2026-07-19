#!/usr/bin/env python3
"""Automated validator for Databricks workshop deliverables (one topic).

Runs the mechanical, objective checks so the reviewer can focus on judgment calls
(accuracy, pedagogy, doc-grounding). Exit code is non-zero if any ERROR-level check fails.

Usage:
    python3 validate.py --root <workshop_dir> --num 03 --slug feature_store_concepts --guide Topic3_Feature_Store_Concepts_Guide

All four deliverables are located from --root:
    notebooks/<num>_<slug>.ipynb
    generated-slides/<num>_<slug>.pptx
    study-guides/<guide>.md
    presenter-guide/README.md   (must mention the topic)
    presenter-guide/slides-topic<N>.json  (deck source; N = int(num))
"""
import argparse, ast, json, os, sys, zipfile

ERRORS, WARNINGS, OKS = [], [], []

def err(m): ERRORS.append(m)
def warn(m): WARNINGS.append(m)
def ok(m): OKS.append(m)


def check_notebook(path):
    if not os.path.exists(path):
        err(f"Notebook missing: {path}")
        return
    try:
        nb = json.load(open(path))
    except Exception as e:
        err(f"Notebook is not valid JSON: {e}")
        return
    ok(f"Notebook is valid JSON: {path}")

    cells = nb.get("cells", [])
    if nb.get("nbformat") != 4:
        warn("Notebook nbformat is not 4")
    md_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    if len(cells) < 8:
        warn(f"Notebook has only {len(cells)} cells — likely too thin for a workshop topic")
    else:
        ok(f"Notebook has {len(cells)} cells ({len(md_cells)} md, {len(code_cells)} code)")
    if not md_cells:
        err("Notebook has no markdown cells (concepts must be explained)")
    if not code_cells:
        err("Notebook has no code cells")

    # First cell should be a markdown title
    if cells and cells[0].get("cell_type") != "markdown":
        warn("First cell is not markdown (expected a title/intro cell)")

    # Every code cell must compile
    bad = 0
    for i, c in enumerate(code_cells):
        src = "".join(c.get("source", []))
        # strip notebook magics / shell lines that aren't valid python
        pysrc = "\n".join(
            ("" if l.strip().startswith(("%", "!", "display(")) else l)
            for l in src.splitlines()
        )
        try:
            ast.parse(pysrc)
        except SyntaxError as e:
            bad += 1
            err(f"Code cell #{i} has a syntax error: {e}")
    if bad == 0:
        ok(f"All {len(code_cells)} code cells parse as valid Python")

    text = json.dumps(nb).lower()
    # Runnable-but-safe: expect some guarding for off-platform execution
    if not any(g in text for g in ["try:", "nameerror", "is_available", "except"]):
        warn("No guard patterns (try/except, NameError, is_available) — cells may crash off-platform")
    else:
        ok("Notebook uses guard patterns for off-platform execution")
    # References
    if "references" not in text and "docs.databricks.com" not in text:
        warn("Notebook has no References section / doc links")
    else:
        ok("Notebook links Databricks documentation")


def check_deck(pptx_path, json_path, topic_int):
    if not os.path.exists(json_path):
        err(f"Deck source JSON missing: {json_path} (should be version-controlled)")
    else:
        try:
            content = json.load(open(json_path))
            slides = content.get("slides", [])
            ok(f"Deck source JSON valid with {len(slides)} slides")
            types = [s.get("type") for s in slides]
            if types and types[0] != "title":
                warn("First slide is not a 'title' slide")
            if "closing" not in types:
                warn("No 'closing' slide")
            if len(set(types)) < 5:
                warn(f"Low slide-type variety ({len(set(types))} distinct) — avoid bullet monotony")
            else:
                ok(f"Good slide-type variety: {len(set(types))} distinct types")
            # encourage a 'which framework/library when' slide for technical topics
            blob = json.dumps(content).lower()
            if not any(k in blob for k in ["use each when", "which framework", "which library", "when to use", "use it when"]):
                warn("Deck may lack a 'which framework/library — use each when' slide")
        except Exception as e:
            err(f"Deck source JSON invalid: {e}")

    if not os.path.exists(pptx_path):
        err(f"PPTX missing: {pptx_path} (run the generator)")
        return
    # A .pptx is a zip; verify it opens and has slides
    try:
        with zipfile.ZipFile(pptx_path) as z:
            names = z.namelist()
            n_slides = len([n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")])
        if n_slides == 0:
            err("PPTX contains no slides")
        else:
            ok(f"PPTX is a valid package with {n_slides} slides")
    except zipfile.BadZipFile:
        err("PPTX is not a valid Office Open XML (zip) file")


def check_guide(path):
    if not os.path.exists(path):
        err(f"Study guide missing: {path}")
        return
    txt = open(path, encoding="utf-8").read()
    words = len(txt.split())
    if words < 1200:
        warn(f"Study guide is short ({words} words) — aim for a real deep-dive")
    else:
        ok(f"Study guide is substantial ({words} words)")
    low = txt.lower()
    required = {
        "one-sentence summary": ["one-sentence summary", "one sentence summary"],
        "why it matters": ["why this topic matters", "why it matters"],
        "mental model": ["mental model"],
        "which framework/when": ["when", "use it when", "which"],
        "demo walkthrough": ["demo walkthrough", "what to run", "notebook demo"],
        "audience questions": ["audience question", "likely audience", "q:", "questions"],
        "glossary": ["glossary"],
        "sources": ["sources", "docs.databricks.com"],
    }
    for label, needles in required.items():
        if any(n in low for n in needles):
            ok(f"Study guide covers: {label}")
        else:
            warn(f"Study guide may be missing section: {label}")
    # Count Q&A
    qa = low.count("**q:") + low.count("q: ")
    if qa and qa < 6:
        warn(f"Only ~{qa} audience Q&A found — aim for >= 8")


def check_presenter(readme_path, topic_int, slug):
    if not os.path.exists(readme_path):
        err(f"Presenter README missing: {readme_path}")
        return
    txt = open(readme_path, encoding="utf-8").read().lower()
    if f"topic {topic_int}" in txt or slug.replace("_", " ") in txt or slug in txt:
        ok("Presenter README references this topic")
    else:
        warn("Presenter README does not clearly reference this topic (add a section)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--num", required=True, help="zero-padded topic number, e.g. 03")
    ap.add_argument("--slug", required=True, help="lower_snake_case slug")
    ap.add_argument("--guide", required=True, help="study guide filename without .md")
    args = ap.parse_args()

    root = args.root
    topic_int = int(args.num)
    nb = os.path.join(root, "notebooks", f"{args.num}_{args.slug}.ipynb")
    pptx = os.path.join(root, "generated-slides", f"{args.num}_{args.slug}.pptx")
    djson = os.path.join(root, "presenter-guide", f"slides-topic{topic_int}.json")
    guide = os.path.join(root, "study-guides", f"{args.guide}.md")
    readme = os.path.join(root, "presenter-guide", "README.md")

    print(f"Validating Topic {topic_int}: {args.slug}\n" + "=" * 50)
    check_notebook(nb)
    check_deck(pptx, djson, topic_int)
    check_guide(guide)
    check_presenter(readme, topic_int, args.slug)

    print("\n--- PASS ---")
    for m in OKS: print(f"  ✓ {m}")
    if WARNINGS:
        print("\n--- WARNINGS (use judgment) ---")
        for m in WARNINGS: print(f"  ⚠ {m}")
    if ERRORS:
        print("\n--- ERRORS (must fix) ---")
        for m in ERRORS: print(f"  ✗ {m}")

    print("\n" + "=" * 50)
    print(f"RESULT: {len(OKS)} passed, {len(WARNINGS)} warnings, {len(ERRORS)} errors")
    sys.exit(1 if ERRORS else 0)


if __name__ == "__main__":
    main()
