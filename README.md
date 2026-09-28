# HADS-SHAP

Code and data for hierarchical auditing of SHAP attribution changes in correlated feature groups. The analysis separates changes in a group's total normalized attribution share from changes in its members' conditional shares.

## Contents

```text
data/raw/             Two UCI source archives
experiments/          Controlled, Beijing, Gas Sensor, and sensitivity analyses
src/hads_shap/        Shared attribution and permutation-test implementation
results/reference/    Run-level and group-level results used in the manuscript
figures/              Editable manuscript Figures 1–3 (SVG)
scripts/              Data checks, result summary, and Figure 2–3 generation
```

## Environment and data

Use Python 3.13 and install the pinned package dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/verify_data.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. The UCI source files, citations, and licenses are listed in [`data/README.md`](data/README.md).

## Analyses

Run from the repository root. The complete set of analyses uses 1,000 constrained environment-label permutations per comparison:

```bash
python scripts/run_all.py --permutations 1000
```

Each experiment is also independently runnable:

| Script | Manuscript analysis |
| --- | --- |
| `experiments/01_synthetic.py` | Four-state controlled experiment, 100 repeats per state |
| `experiments/02_beijing_main.py` | Three 24 h Beijing prediction targets |
| `experiments/03_gas_main.py` | Gas Sensor XGBoost analysis |
| `experiments/04_beijing_horizon_sensitivity.py` | 6, 12, and 24 h prediction horizons |
| `experiments/05_correlation_threshold_sensitivity.py` | Correlation thresholds 0.80–0.95 |
| `experiments/06_gas_model_sensitivity.py` | LightGBM and Random Forest comparisons |
| `experiments/07_gas_shap_dependence_sensitivity.py` | TreeSHAP feature-dependence settings |

The fixed seed is `20260926`; the primary correlation threshold is `|Spearman rho| >= 0.90`. The experiment scripts write new results to the ignored `results/generated/` directory. Beijing labels are permuted by complete week; Gas Sensor batch labels are permuted within the true-class strata. Feature groups are built from training observations only.

## Manuscript results and figures

`results/reference/` contains the analysis outputs used for the reported tables and figures. To summarize the stored repeat-level and group-level results:

```bash
python scripts/check_results.py
```

Figure 1 is supplied as an editable SVG. Figures 2 and 3 can be regenerated from the stored Beijing group and case results with:

```bash
python scripts/make_figures.py
```

The controlled experiment's stored file contains all 400 repeat-level outcomes from `experiments/01_synthetic.py` using the fixed seed and 1,000 permutations. This run gives correct-identification counts of 92, 97, 94, and 100 for Stable, Mass-only, WGAM-only, and Mixed, respectively. The same run was reproduced under Python 3.13.5 with the pinned dependencies; `scripts/check_results.py` recomputes its confusion matrix from the stored records.

## License

The code is released under the MIT License. The two source datasets remain under their original CC BY 4.0 licenses; see [`data/README.md`](data/README.md).
