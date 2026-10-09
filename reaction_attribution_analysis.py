"""
reaction_attribution_analysis.py

Analysis and figures for the responsibility-attribution study on Arabic poverty
discourse. The script reproduces two parts of the paper:

  1. Reaction profiling and full-corpus projection
     - the average reaction profile of each responsibility label,
     - the significance tests behind those profiles,
     - and the year-by-year and country-by-country comparison of the full
       corpus against those profiles.

  2. Human validation of the annotation pipeline
     - overall agreement between the model and the human annotator,
     - per-label precision / recall / F1 and Cohen's kappa,
     - and a short breakdown of where the two annotators disagree.

Everything is driven by two files: the 10k posts the model annotated, and the
full relevance-filtered corpus. Point the paths below at your copies and run.

Expected columns:
  - a label column (comma-separated labels, e.g. "State, Descriptive/No Agent"),
  - the six reaction counts: Love, Wow, Haha, Sad, Angry, Care,
  - "Post Created Date" and "Page Admin Top Country" for the projection,
  - a second human-label column in the validation file.

Dependencies: pandas, numpy, scipy, matplotlib.
"""

import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # write files, don't open windows
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu


# ---------------------------------------------------------------------------
# Paths and settings. These are the only things you normally need to touch.
# ---------------------------------------------------------------------------
ANNOTATED_FILE = "annotated_fanar_arapoverty - annotated_fanar_10000posts_natural.csv"      # 10k posts with model labels
FULL_CORPUS_FILES = [                             # the three filtered datasets
    "الفقر OR مكافحة الفقر OR التنمية الاقتصادية_filtered.csv",
    "اللامساواة الاقتصادية OR الفقر OR التفاوت الطبقي_filtered.csv",
    "الفقر المدقع OR الحد من الفقر OR التنمية المستدامة_filtered.csv",
]
VALIDATION_FILE = "annotated_fanar_arapoverty_10k.xlsx"  # has both label columns

OUTPUT_DIR = "output"
FIGURE_DPI = 300

# Column names, kept in one place so they are easy to change for a new dataset.
MODEL_LABEL_COL = "llm_label"
HUMAN_LABEL_COL = "Human_Annotation"
DATE_COL = "Post Created Date"
COUNTRY_COL = "Page Admin Top Country"

REACTIONS = ["Love", "Wow", "Haha", "Sad", "Angry", "Care"]

# The three labels that actually assign responsibility. Descriptive/No Agent and
# Other/Unclear are handled separately where that distinction matters.
AGENT_LABELS = ["State", "Community", "Religious/Spiritual"]
PROFILE_LABELS = ["State", "Community", "Religious/Spiritual", "Descriptive/No Agent"]
ALL_LABELS = PROFILE_LABELS + ["Other/Unclear"]

# A muted palette, one fixed colour per reaction so every figure agrees.
REACTION_COLORS = {
    "Love": "#4C72B0", "Wow": "#55A868", "Haha": "#C44E52",
    "Sad": "#8172B3", "Angry": "#CCB974", "Care": "#64B5CD",
}


# ---------------------------------------------------------------------------
# Small helpers shared by both parts of the analysis.
# ---------------------------------------------------------------------------
def load_table(path):
    """Read a CSV or Excel file. BOM-safe for the Arabic CSV exports."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def split_labels(cell):
    """Turn a 'State, Descriptive/No Agent' cell into a set of labels."""
    if not isinstance(cell, str):
        return set()
    return {part.strip() for part in cell.split(",") if part.strip()}


def reaction_shares(df):
    """
    Convert the six raw reaction counts into per-post proportions.

    A post with no reactions at all carries no profile, so we drop it rather
    than divide by zero. Returns the share matrix and the mask of kept rows so
    callers can line their other columns back up.
    """
    counts = df[REACTIONS].apply(pd.to_numeric, errors="coerce").fillna(0)
    totals = counts.sum(axis=1)
    kept = totals > 0
    shares = counts[kept].div(totals[kept], axis=0)
    return shares, kept


def save_figure(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


# ===========================================================================
# PART 1 - Reaction profiling and full-corpus projection
# ===========================================================================
def build_reference_profiles(annotated):
    """
    For each responsibility label, average the reaction shares of the posts that
    carry it. These averages are the 'reference signatures' the projection later
    compares the full corpus against.
    """
    shares, kept = reaction_shares(annotated)
    labels = annotated.loc[kept, MODEL_LABEL_COL].map(split_labels).reset_index(drop=True)
    shares = shares.reset_index(drop=True)

    rows = []
    for label in PROFILE_LABELS:
        has_label = labels.map(lambda s: label in s)
        mean_share = shares[has_label.values].mean()
        rows.append({"label": label, "n": int(has_label.sum()),
                     **{r: round(mean_share[r] * 100, 1) for r in REACTIONS}})
    profiles = pd.DataFrame(rows)
    profiles.to_csv(os.path.join(OUTPUT_DIR, "reference_profiles.csv"), index=False)
    return profiles


def cliffs_delta(x, y):
    """
    Cliff's delta effect size: the chance that a value drawn from x exceeds one
    from y, minus the reverse. Runs from -1 to +1; 0 means the two groups overlap
    completely. Computed with a sort-and-search pass so it stays fast on the full
    samples rather than comparing every pair directly.
    """
    x = np.asarray(x, float)
    y_sorted = np.sort(np.asarray(y, float))
    n, m = len(x), len(y_sorted)
    greater = less = 0
    for value in x:
        below = np.searchsorted(y_sorted, value, side="left")
        above = m - np.searchsorted(y_sorted, value, side="right")
        less += below       # y < value  ->  value is the larger one
        greater += above    # y > value  ->  value is the smaller one
    return (less - greater) / (n * m)


def delta_magnitude(delta):
    """The usual words for a Cliff's delta size (Romano et al. thresholds)."""
    size = abs(delta)
    if size < 0.147:
        return "negligible"
    if size < 0.33:
        return "small"
    if size < 0.474:
        return "medium"
    return "large"


def clr_transform(counts):
    """
    Centered-log-ratio transform of one post's six reaction counts. A small
    pseudo-count keeps the log defined when a reaction is zero. This moves the
    shares off the sum-to-one simplex so a test run on the result does not depend
    on the closure, which is the compositional robustness check.
    """
    values = np.asarray(counts, float) + 0.5
    proportions = values / values.sum()
    log_props = np.log(proportions)
    return log_props - log_props.mean()


def test_profile_differences(annotated):
    """
    The full set of tests behind the reaction-profile table.

    Reaction shares are bounded and skewed, so every test here is rank-based
    rather than assuming a normal distribution. There are three steps:

      - Kruskal-Wallis on the raw shares, asking whether each reaction differs
        across the four labels at all;
      - the same test on CLR-transformed values, to confirm the differences
        survive a proper compositional treatment and are not an artifact of the
        shares summing to one;
      - pairwise Mann-Whitney U between the three responsibility agents, reported
        with Cliff's delta effect sizes and a Bonferroni correction for the
        number of comparisons. The effect sizes are written to effect_sizes.csv.
    """
    shares, kept = reaction_shares(annotated)
    labels = annotated.loc[kept, MODEL_LABEL_COL].map(split_labels).reset_index(drop=True)
    shares = shares.reset_index(drop=True)

    # one share matrix per label, reused by every test below
    grouped = {label: shares[labels.map(lambda s, l=label: l in s).values]
               for label in PROFILE_LABELS}

    print("\n  Kruskal-Wallis across the four labels (raw shares):")
    for r in REACTIONS:
        stat, p = kruskal(*[grouped[l][r] for l in PROFILE_LABELS])
        print(f"    {r:6s} H={stat:8.1f}  p={p:.2e}")

    # Compositional robustness: repeat the test on CLR-transformed reactions,
    # using the same posts (those with at least one reaction) as above.
    raw_counts = (annotated.loc[kept, REACTIONS]
                  .apply(pd.to_numeric, errors="coerce").fillna(0)
                  .reset_index(drop=True))
    clr = np.vstack([clr_transform(row) for row in raw_counts.values])
    clr = pd.DataFrame(clr, columns=REACTIONS)
    print("\n  Kruskal-Wallis on CLR-transformed reactions (compositional check):")
    for r in REACTIONS:
        groups_clr = [clr[r][labels.map(lambda s, l=l: l in s).values] for l in PROFILE_LABELS]
        stat, p = kruskal(*groups_clr)
        print(f"    {r:6s} H={stat:8.1f}  p={p:.2e}")

    # Pairwise agent comparisons with Bonferroni correction and effect sizes.
    n_tests = len(list(itertools.combinations(AGENT_LABELS, 2))) * len(REACTIONS)
    alpha = 0.05 / n_tests
    print(f"\n  Pairwise Mann-Whitney U + Cliff's delta "
          f"(Bonferroni alpha = 0.05/{n_tests} = {alpha:.4f}):")
    effect_rows = []
    for a, b in itertools.combinations(AGENT_LABELS, 2):
        for r in REACTIONS:
            _, p = mannwhitneyu(grouped[a][r], grouped[b][r], alternative="two-sided")
            p_bonf = min(p * n_tests, 1.0)
            delta = cliffs_delta(grouped[a][r], grouped[b][r])
            effect_rows.append({"contrast": f"{a} vs {b}", "reaction": r,
                                "p_raw": p, "p_bonferroni": p_bonf,
                                "cliffs_delta": round(delta, 3),
                                "magnitude": delta_magnitude(delta),
                                "significant_bonf": p_bonf < 0.05})
            flag = "" if p_bonf < 0.05 else "   (n.s. after correction)"
            print(f"    {a[:4]} vs {b[:4]}  {r:6s}  "
                  f"d={delta:+.3f} ({delta_magnitude(delta)})  p_bonf={p_bonf:.2e}{flag}")

    effects = pd.DataFrame(effect_rows)
    effects.to_csv(os.path.join(OUTPUT_DIR, "effect_sizes.csv"), index=False)
    n_sig = int(effects["significant_bonf"].sum())
    print(f"\n  {n_sig} of {n_tests} comparisons significant after Bonferroni correction.")


def plot_reference_profiles(profiles):
    """Grouped bar chart: one cluster of six reaction bars per label."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(profiles))
    width = 0.13
    for i, r in enumerate(REACTIONS):
        ax.bar(x + (i - 2.5) * width, profiles[r], width,
               label=r, color=REACTION_COLORS[r], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("/", "/\n") for l in profiles["label"]])
    ax.set_ylabel("Share of reactions (%)")
    ax.set_title("Reaction profiles by responsibility label")
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "reference_profiles.png")


def cosine(a, b):
    """Plain cosine similarity between two reaction-share vectors."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not a.any() or not b.any():
        return np.nan
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def project_corpus(full, profiles, group_col, out_name, min_posts=100, top_n=None):
    """
    Compare the full corpus against the reference signatures.

    For each group (a year, or a country) we average its reaction shares, then
    ask which reference profile that average sits closest to. This is a
    descriptive comparison, not a classifier: it never labels an individual
    post, it only reports which signature a group's aggregate behaviour most
    resembles. Groups below `min_posts` are dropped to avoid noisy averages.
    """
    shares, kept = reaction_shares(full)
    groups = full.loc[kept, group_col].reset_index(drop=True)
    shares = shares.reset_index(drop=True)
    shares[group_col] = groups

    agg = shares.groupby(group_col)[REACTIONS].mean()
    sizes = shares.groupby(group_col).size()
    agg = agg[sizes >= min_posts]
    sizes = sizes[sizes >= min_posts]
    if top_n:
        keep = sizes.sort_values(ascending=False).head(top_n).index
        agg = agg.loc[keep]

    ref_vectors = {row["label"]: np.array([row[r] for r in REACTIONS], float)
                   for _, row in profiles.iterrows()}

    records = []
    for name, profile in agg.iterrows():
        group_vec = profile[REACTIONS].values * 100  # same scale as the reference table
        sims = {label: cosine(group_vec, vec) for label, vec in ref_vectors.items()}
        closest = max(sims, key=sims.get)
        records.append({group_col: name, "n_posts": int(sizes[name]),
                        **{r: round(profile[r] * 100, 1) for r in REACTIONS},
                        **{f"sim_{l}": round(s, 3) for l, s in sims.items()},
                        "closest": closest})
    result = pd.DataFrame(records)
    result.to_csv(os.path.join(OUTPUT_DIR, f"{out_name}.csv"), index=False)
    return result


def plot_yearly_profiles(yearly):
    """Line chart of the six reaction shares across the years."""
    yearly = yearly.sort_values(DATE_COL if DATE_COL in yearly else yearly.columns[0])
    years = yearly.iloc[:, 0].astype(int)
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in REACTIONS:
        ax.plot(years, yearly[r], marker="o", ms=4, lw=1.8,
                label=r, color=REACTION_COLORS[r])
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of reactions (%)")
    ax.set_title("Corpus-wide reaction profile by year")
    ax.set_xticks(years)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.10), frameon=False)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "yearly_profiles.png")


def plot_country_similarity(country):
    """Heatmap of each country's cosine similarity to the four signatures."""
    sim_cols = [f"sim_{l}" for l in PROFILE_LABELS]
    matrix = country[sim_cols].astype(float).values
    labels = country[COUNTRY_COL].tolist()

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(PROFILE_LABELS)))
    ax.set_xticklabels([l.replace("/", "/\n") for l in PROFILE_LABELS])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Country profile vs. reference signatures (cosine similarity)")

    # annotate each cell, flipping text colour so it stays readable
    cutoff = matrix.min() + 0.6 * (matrix.max() - matrix.min())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color="white" if matrix[i, j] < cutoff else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    fig.tight_layout()
    save_figure(fig, "country_similarity.png")


def run_reaction_profiling():
    print("\n[1] Reaction profiling and full-corpus projection")

    annotated = load_table(ANNOTATED_FILE)
    profiles = build_reference_profiles(annotated)
    print("\n  Reference profiles (% of each reaction):")
    print(profiles.to_string(index=False))
    test_profile_differences(annotated)
    plot_reference_profiles(profiles)

    # Load the full corpus once and reuse it for both projections.
    frames = [load_table(p) for p in FULL_CORPUS_FILES if os.path.exists(p)]
    if not frames:
        print("\n  Full-corpus files not found; skipping the projection.")
        return
    full = pd.concat(frames, ignore_index=True)
    full["_year"] = pd.to_datetime(full[DATE_COL], errors="coerce").dt.year

    yearly = project_corpus(full, profiles, "_year", "yearly_projection")
    plot_yearly_profiles(yearly)

    country = project_corpus(full, profiles, COUNTRY_COL, "country_projection",
                             top_n=15)
    plot_country_similarity(country)


# ===========================================================================
# PART 2 - Human validation of the annotation pipeline
# ===========================================================================
def cohens_kappa_binary(pairs, label):
    """
    Cohen's kappa for one label, treated as a yes/no decision (does this post
    carry the label or not). Works from the model/human pairs directly so it
    handles the multi-label setting one label at a time.
    """
    tp = fp = fn = tn = 0
    for model_set, human_set in pairs:
        in_model, in_human = label in model_set, label in human_set
        if in_model and in_human:
            tp += 1
        elif in_model and not in_human:
            fp += 1
        elif not in_model and in_human:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    observed = (tp + tn) / n
    expected = (((tp + fp) / n) * ((tp + fn) / n)
                + ((fn + tn) / n) * ((fp + tn) / n))
    kappa = (observed - expected) / (1 - expected) if expected != 1 else float("nan")

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall)
          else float("nan"))
    return {"label": label, "support": tp + fn, "precision": precision,
            "recall": recall, "f1": f1, "kappa": kappa}


def run_validation():
    print("\n[2] Human validation of the annotation pipeline")

    data = load_table(VALIDATION_FILE)
    # Keep only the rows the human actually annotated. Blank cells come through
    # as NaN from Excel, so test for real values rather than empty strings.
    data = data[data[HUMAN_LABEL_COL].notna()
                & (data[HUMAN_LABEL_COL].astype(str).str.strip() != "")]
    pairs = [(split_labels(m), split_labels(h))
             for m, h in zip(data[MODEL_LABEL_COL], data[HUMAN_LABEL_COL])]
    n = len(pairs)
    print(f"  {n} human-annotated posts")

    # Overall agreement: exact set match, plus two softer set-overlap measures.
    exact = sum(1 for m, h in pairs if m == h)
    jaccard = np.mean([len(m & h) / len(m | h) if (m | h) else 1.0 for m, h in pairs])
    hamming = np.mean([sum((l in m) == (l in h) for l in ALL_LABELS) / len(ALL_LABELS)
                       for m, h in pairs])
    print(f"  exact-set agreement : {100 * exact / n:.1f}%")
    print(f"  mean Jaccard        : {jaccard:.3f}")
    print(f"  mean Hamming        : {hamming:.3f}")

    # Per-label scores against the human annotation as the gold standard.
    per_label = pd.DataFrame([cohens_kappa_binary(pairs, l) for l in ALL_LABELS])
    per_label = per_label.round(3)
    per_label.to_csv(os.path.join(OUTPUT_DIR, "per_label_agreement.csv"), index=False)
    print("\n  Per-label agreement:")
    print(per_label.to_string(index=False))

    plot_per_label_agreement(per_label)

    # Where do the two annotators actually disagree?
    mismatches = [(m, h) for m, h in pairs if m != h]
    if mismatches:
        desc = sum(1 for m, h in mismatches
                   if ("Descriptive/No Agent" in m) != ("Descriptive/No Agent" in h))
        other = sum(1 for m, h in mismatches
                    if ("Other/Unclear" in m) != ("Other/Unclear" in h))
        print(f"\n  {len(mismatches)} posts disagree:")
        print(f"    {100 * other / len(mismatches):.0f}% involve the Other/Unclear boundary")
        print(f"    {100 * desc / len(mismatches):.0f}% involve the Descriptive/No Agent boundary")


def plot_per_label_agreement(per_label):
    """Grouped bars of precision / recall / F1 per label."""
    metrics = ["precision", "recall", "f1"]
    colors = {"precision": "#4C72B0", "recall": "#55A868", "f1": "#C44E52"}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(per_label))
    width = 0.25
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1) * width, per_label[metric].fillna(0), width,
               label=metric.capitalize(), color=colors[metric],
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("/", "/\n") for l in per_label["label"]])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-label agreement with the human gold standard")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "per_label_agreement.png")


# ===========================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    run_reaction_profiling()
    run_validation()
    print(f"\nDone. Tables and figures are in '{OUTPUT_DIR}/'.")
