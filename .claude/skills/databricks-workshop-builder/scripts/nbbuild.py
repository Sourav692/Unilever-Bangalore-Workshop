"""Minimal Jupyter notebook (.ipynb v4) builder — no external deps.

Notebooks are just JSON, so we construct the nbformat v4 structure by hand.
The resulting files open in Jupyter/VS Code and import cleanly into Databricks.
"""
import json


def md(text):
    """Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(text),
    }


def code(text):
    """Code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


def _lines(text):
    """Split into a list of lines, each keeping its trailing newline (nbformat convention)."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [""]


def write_notebook(path, cells, kernel_name="python3", display_name="Python 3"):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": display_name,
                "language": "python",
                "name": kernel_name,
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "application/vnd.databricks.v1+notebook": {
                "notebookName": path.split("/")[-1].replace(".ipynb", ""),
                "language": "python",
                "dashboards": [],
                "notebookMetadata": {"pythonIndentUnit": 4},
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {path} ({len(cells)} cells)")
