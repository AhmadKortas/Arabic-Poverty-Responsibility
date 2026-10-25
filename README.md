# Arabic Poverty Responsibility

Analysis code for the study of responsibility attribution in Arabic poverty
discourse on Facebook. The two scripts reproduce the paper's three results
sections, along with the tables and figures that go with them.

## Scripts

### `attribution_distribution.py`

The distribution of responsibility-attribution labels across the annotated
corpus. It reports how often each label is assigned, the single- versus
multi-label split (and how many posts assign a responsible agent at all rather
than merely describing poverty), and the most frequent pairwise co-occurrences
between labels.

### `reaction_attribution_analysis.py`

Two parts. The first computes the average reaction profile (the share of each of
the six Facebook reactions — Love, Wow, Haha, Sad, Angry, Care) for every
responsibility label and tests whether those profiles differ. The tests are all
rank-based: Kruskal–Wallis across the four labels, the same test on
centered-log-ratio (CLR) transformed values as a compositional robustness check,
and pairwise Mann–Whitney U between the three responsibility agents, reported
with Cliff's δ effect sizes and a Bonferroni correction. It then compares the
full corpus against the profiles year by year and country by country using
cosine similarity: for each year or country it reports which reference profile
the aggregate reaction behaviour most closely resembles. This is a descriptive
comparison — it never assigns a label to an individual post. The second part
measures how closely the model-assigned labels match a blind human annotation of
a 1,000-post subset: overall exact-set agreement, Jaccard and Hamming overlap,
and per-label precision, recall, F1 and Cohen's kappa, plus a breakdown of where
the two annotators disagree.

## Outputs

Everything is written to `output/`:

| File | Contents |
|------|----------|
| `label_distribution.csv` | share of posts carrying each label |
| `cooccurrence.csv` | most frequent pairwise label co-occurrences |
| `reference_profiles.csv` / `.png` | average reaction profile per label |
| `effect_sizes.csv` | Cliff's δ and Bonferroni-corrected p-values per agent contrast |
| `yearly_projection.csv` | each year's profile and its closest reference |
| `yearly_profiles.png` | reaction shares across the years |
| `country_projection.csv` | each country's profile and its closest reference |
| `country_similarity.png` | country-vs-reference similarity heatmap |
| `per_label_agreement.csv` / `.png` | precision, recall, F1, kappa per label |

The statistical test results (p-values, etc.) are printed to the console.

## Input files

Set the paths at the top of each script:

- The annotated posts, with a label column and the six reaction counts. Labels
  are stored one post per cell as a comma-separated list, for example
  `State, Descriptive/No Agent`.
- The full corpus datasets, each with the reaction counts, `Post Created Date`,
  and `Page Admin Top Country`, for the projection in
  `reaction_attribution_analysis.py`. If these are not available that script
  skips the projection step and the rest still runs.
- A file containing both the model label column and the human label column, for
  the validated subset.

CSV and Excel inputs are both accepted.

## Running

```bash
pip install pandas numpy scipy matplotlib openpyxl
python attribution_distribution.py
python reaction_attribution_analysis.py
```
