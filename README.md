# Fairness and Dependency Preservation in Synthetic Credit Data: expanded code

Follow-up experiments for my paper **[Fairness and Dependency Preservation in Synthetic Credit Data](https://nhsjs.com/2026/fairness-and-dependency-preservation-in-synthetic-credit-data/)** (*The National High School Journal of Science*, April 2026). They test the questions the paper left open.

**See the results:** [research.rayanupadhyay.com](https://research.rayanupadhyay.com) shows these experiments interactively, next to the published results.

> These are new runs, not the published results, and they have not been peer reviewed. Where the paper doesn't pin down a detail, the code makes its own choice and writes it down (see [Definitions](#definitions)), so the absolute numbers aren't directly comparable with the paper's.

## What it runs

Every method is trained and tested on the same 20 stratified 80/20 train/test splits (seeds 0–19). That makes the comparisons paired.

| Method key | What it is |
|---|---|
| `baseline` | Original training data, no balancing |
| `baseline_weighted` | Original data with XGBoost class weighting (`scale_pos_weight`) |
| `baseline_oversampled` | Original data with random oversampling to 50/50 |
| `reweighing` | Original data with reweighing by sex × label, a fairness-aware preprocessing method (Kamiran & Calders; the paper's recommendation) |
| `smote` | SMOTE on one-hot data, as in the paper |
| `smotenc` | SMOTE-NC, the categorical-aware variant |
| `ctgan`, `tvae`, `gaussian_copula` | SDV generators adding synthetic bad-credit rows until the classes are 50/50 (CTGAN/TVAE: 300 epochs, batch 500, as in the paper) |
| `mix_<generator>_<25/50/75/100>` | The "synthetic slider": that percentage of the real training rows replaced with synthetic rows |

For each run it records:
- **Performance:** ROC AUC.
- **Fairness by sex:** Disparate Impact (`di`), Equalized Odds difference (`eo`) and Predictive Parity gap (`pp`).
- **Fairness by age group** (25 or under vs older): `di_age`, `eo_age`, `pp_age`.
- **Dependencies:** the Dependency Preservation Score (`dps`, with numeric and categorical parts) and the Frobenius distance between Spearman matrices (`frob`).
- **Privacy:** distance to closest record for synthetic vs real held-out rows (`dcr_*`), and the share of synthetic rows that exactly copy a training row.
- **Validity:** whether a generator produced degenerate columns.
- **Significance:** paired Wilcoxon tests of every method against `baseline`, on AUC and the three sex-based fairness metrics.

## Repository contents

```
experiments/
  experiments.py        everything: data, generators, model, metrics, summary
  requirements.txt      exact library versions used
  data/GermanCredit.csv the dataset (1,000 rows)
  data/DATA-LICENSE.md  dataset credit and licence
  results/raw.jsonl     one line per seed × method, every metric
  results/summary.json  means, standard deviations, p-values and the definitions
```

## Running it

Python 3.11 was used. CTGAN and TVAE need PyTorch, which `sdv` installs.

```bash
pip install -r experiments/requirements.txt
python experiments/experiments.py 20       # 20 seeds, as on the site; default is 10
python experiments/experiments.py summary  # rebuild summary.json from raw.jsonl only
```

A full 20-seed run takes a while on a CPU, mostly training CTGAN and TVAE. For a quick check, try `python experiments/experiments.py 2`. Runs write `raw.jsonl` as they go, and `summary.json` at the end. If a generator fails on a seed, the error is logged for that method and the run carries on.

## Definitions

The paper leaves some details open, so these are the choices made here. They're also saved inside `summary.json`.

- **Target:** bad credit = 1, good credit = 0.
- **Model:** XGBoost with library defaults. The paper used AutoGluon, which selected XGBoost.
- **Protected attribute:** sex, derived from `personal_status_sex`: female = "female : divorced/separated/married", male = every other value. Age group (25 or under) is an extra attribute.
- **Approval:** predicted good credit (probability of bad credit below 0.5).
- **Disparate Impact:** women's approval rate ÷ men's (closer to 1 is fairer).
- **Equalized Odds:** the larger of the TPR gap and the FPR gap between groups (lower is fairer).
- **Predictive Parity:** absolute gap in precision of approvals (lower is fairer).
- **Dependency Preservation Score:** the average of two parts, numeric and categorical. The numeric part is 1 minus the mean absolute difference between real and synthetic Spearman correlations. The categorical part does the same with Cramér's V. This is a reading of the paper's description, not its exact code.
- **SMOTE:** runs on unscaled one-hot data, as in the paper, so loan amount dominates which neighbours it picks.
- **Generators:** see only the training split.

## Data and licence

Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The copy in `experiments/data/` is a public CSV version with readable category labels ([github.com/selva86/datasets](https://github.com/selva86/datasets)). Its rows are the same as the UCI original, but some of its labels don't exactly match UCI's own codes. The paper also notes known label problems in this dataset.

## Citation

If you use this, please cite the paper:

```bibtex
@article{upadhyay2026fairness,
  title   = {Fairness and Dependency Preservation in Synthetic Credit Data},
  author  = {Upadhyay, Rayan},
  journal = {The National High School Journal of Science},
  year    = {2026},
  url     = {https://nhsjs.com/2026/fairness-and-dependency-preservation-in-synthetic-credit-data/}
}
```

## Author

Rayan Upadhyay · [rayanupadhyay.com](https://rayanupadhyay.com)
