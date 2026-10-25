"""
attribution_distribution.py

Distribution of responsibility-attribution labels across the annotated corpus
(the first results section of the paper). The script reports three things:

  1. the overall label distribution - how often each label is assigned,
  2. the single- vs multi-label split, and how many posts assign a responsible
     agent at all versus merely describing poverty,
  3. the most frequent pairwise co-occurrences between labels.

It is driven by one file: the 10k posts the model annotated. Point the path
below at your copy and run.

Expected columns:
  - a label column holding one post per cell as a comma-separated list,
    e.g. "State, Descriptive/No Agent".

Dependencies: pandas.
"""

import os
import itertools
from collections import Counter
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and settings. These are the only things you normally need to touch.
# ---------------------------------------------------------------------------
ANNOTATED_FILE = "annotated_fanar_arapoverty - annotated_fanar_10000posts_natural.csv"

OUTPUT_DIR = "output"
MODEL_LABEL_COL = "llm_label"

# The five labels of the scheme. State, Community and Religious/Spiritual are the
# three that actually assign responsibility to an agent.
AGENT_LABELS = ["State", "Community", "Religious/Spiritual"]
ALL_LABELS = AGENT_LABELS + ["Descriptive/No Agent", "Other/Unclear"]


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------
def load_table(path):
    """Read the annotated CSV. BOM-safe for the Arabic exports."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def split_labels(cell):
    """Turn a 'State, Descriptive/No Agent' cell into a set of labels."""
    if not isinstance(cell, str):
        return set()
    return {part.strip() for part in cell.split(",") if part.strip()}


# ---------------------------------------------------------------------------
# 1. Overall label distribution.
# ---------------------------------------------------------------------------
def label_distribution(label_sets, n):
    """
    Count how many posts carry each label. Because a post can hold several
    labels, the shares sum to more than 100%.
    """
    counts = Counter()
    for labels in label_sets:
        for label in labels:
            counts[label] += 1

    rows = [{"label": label, "posts": counts.get(label, 0),
             "share_%": round(100 * counts.get(label, 0) / n, 1)}
            for label in ALL_LABELS]
    table = pd.DataFrame(rows).sort_values("posts", ascending=False, ignore_index=True)
    table.to_csv(os.path.join(OUTPUT_DIR, "label_distribution.csv"), index=False)
    return table


# ---------------------------------------------------------------------------
# 2. Single vs multi-label, and agent vs description.
# ---------------------------------------------------------------------------
def label_counts_breakdown(label_sets, n):
    """
    How many labels posts carry, and whether they assign a responsible agent.
    Everything is printed; nothing to save.
    """
    sizes = Counter(len(labels) for labels in label_sets)
    single = sizes.get(1, 0)
    two = sizes.get(2, 0)
    three_plus = sum(v for k, v in sizes.items() if k >= 3)
    multi = n - single

    print("\n  Labels per post:")
    print(f"    single-label        : {single:5d} ({100 * single / n:.1f}%)")
    print(f"    multi-label         : {multi:5d} ({100 * multi / n:.1f}%)")
    print(f"      - two labels      : {two:5d} ({100 * two / n:.1f}%)")
    print(f"      - three or four   : {three_plus:5d} ({100 * three_plus / n:.1f}%)")

    # A post either names at least one responsible agent, or only describes.
    agents = set(AGENT_LABELS)
    with_agent = sum(1 for labels in label_sets if labels & agents)
    descriptive_only = sum(1 for labels in label_sets if labels == {"Descriptive/No Agent"})

    print("\n  Responsibility vs description:")
    print(f"    assigns >=1 agent   : {with_agent:5d} ({100 * with_agent / n:.1f}%)")
    print(f"    Descriptive-only    : {descriptive_only:5d} ({100 * descriptive_only / n:.1f}%)")


# ---------------------------------------------------------------------------
# 3. Pairwise co-occurrence.
# ---------------------------------------------------------------------------
def cooccurrence(label_sets):
    """
    Count how often each pair of labels appears together on the same post.
    Other/Unclear is excluded since it is used on its own by definition.
    """
    labels_for_pairs = AGENT_LABELS + ["Descriptive/No Agent"]
    rows = []
    for a, b in itertools.combinations(labels_for_pairs, 2):
        together = sum(1 for labels in label_sets if a in labels and b in labels)
        rows.append({"label_pair": f"{a} + {b}", "posts": together})
    table = pd.DataFrame(rows).sort_values("posts", ascending=False, ignore_index=True)
    table.to_csv(os.path.join(OUTPUT_DIR, "cooccurrence.csv"), index=False)
    return table


# ---------------------------------------------------------------------------
def run_distribution():
    print("\n[1] Responsibility-attribution distribution")

    data = load_table(ANNOTATED_FILE)
    label_sets = [split_labels(cell) for cell in data[MODEL_LABEL_COL]]
    n = len(label_sets)
    print(f"  {n} annotated posts")

    table = label_distribution(label_sets, n)
    print("\n  Overall label distribution (shares sum to >100%, multi-label):")
    print(table.to_string(index=False))

    label_counts_breakdown(label_sets, n)

    pairs = cooccurrence(label_sets)
    print("\n  Most frequent label co-occurrences:")
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_distribution()
    print(f"\nDone. Tables are in '{OUTPUT_DIR}/'.")
