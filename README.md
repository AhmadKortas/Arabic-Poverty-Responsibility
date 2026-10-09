# Reaction Attribution Analysis

Analysis code for the study of responsibility attribution in Arabic poverty
discourse on Facebook. The script reproduces the reaction-profiling, full-corpus
projection, and human-validation results, along with the figures that go with
them.

## What it does

The script runs in two parts.

**Reaction profiling and full-corpus projection.** It computes the average
reaction profile (the share of each of the six Facebook reactions — Love, Wow,
Haha, Sad, Angry, Care) for every responsibility label and tests whether those
profiles differ. The tests are all rank-based: Kruskal–Wallis across the four
labels, the same test on centered-log-ratio (CLR) transformed values as a
compositional robustness check, and pairwise Mann–Whitney U between the three
responsibility agents, reported with Cliff's δ effect sizes and a Bonferroni
correction. It then compares the full corpus against the profiles year by year
and country by country using cosine similarity: for each year or country it
reports which reference profile the aggregate reaction behaviour most closely
resembles. This is a descriptive comparison — it never assigns a label to an
individual post.

**Human validation.** It measures how closely the model-assigned labels match a
blind human annotation of a 1,000-post subset: overall exact-set agreement,
Jaccard and Hamming overlap, and per-label precision, recall, F1 and Cohen's
kappa. It also reports where the two annotators disagree most.

## Outputs

Everything is written to `output/`:

| File | Contents |
|------|----------|
| `reference_profiles.csv` / `.png` | average reaction profile per label |
| `effect_sizes.csv` | Cliff's δ and Bonferroni-corrected p-values per agent contrast |
| `yearly_projection.csv` | each year's profile and its closest reference |
| `yearly_profiles.png` | reaction shares across the years |
| `country_projection.csv` | each country's profile and its closest reference |
| `country_similarity.png` | country-vs-reference similarity heatmap |
| `per_label_agreement.csv` / `.png` | precision, recall, F1, kappa per label |

The statistical test results (p-values, etc.) are printed to the console.

## Input files

Set the paths at the top of the script:

- `ANNOTATED_FILE` — the annotated posts, with a label column and the six
  reaction counts.
- `FULL_CORPUS_FILES` — the relevance-filtered datasets, each with the reaction
  counts, `Post Created Date`, and `Page Admin Top Country`. If these are not
  available the projection step is skipped and the rest still runs.
- `VALIDATION_FILE` — a file containing both the model label column and the
  human label column, for the validated subset.

Labels are stored one post per cell as a comma-separated list, for example
`State, Descriptive/No Agent`. CSV and Excel inputs are both accepted.

## Running

```bash
pip install pandas numpy scipy matplotlib openpyxl
python reaction_attribution_analysis.py
```
