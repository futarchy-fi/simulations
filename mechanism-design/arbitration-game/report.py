"""Result I/O: results/<name>.json + compact markdown tables."""

from __future__ import annotations

import json
import os

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def write_json(name, params, rows):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name + ".json")
    with open(path, "w") as f:
        json.dump({"params": params, "rows": rows}, f, indent=2,
                  sort_keys=True)
        f.write("\n")
    return path


def _fmt(x):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return "%.4g" % x
    return str(x)


def print_table(rows, cols=None):
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    grid = [cols] + [[_fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(g[i]) for g in grid) for i in range(len(cols))]
    lines = ["| " + " | ".join(v.ljust(w) for v, w in zip(g, widths)) + " |"
             for g in grid]
    lines.insert(1, "|" + "|".join("-" * (w + 2) for w in widths) + "|")
    print("\n".join(lines))
