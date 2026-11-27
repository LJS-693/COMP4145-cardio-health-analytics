# COMP4145 Cardio Health Analytics

This project combines interactive descriptive analysis with seven-feature
classification models and a retrospective review-prioritization simulation.
The target is the dataset's existing `cardio` label. Scores do not estimate
future disease incidence or establish clinical urgency.

## Project structure

| Folder | Contents |
| --- | --- |
| `data/raw/` | Raw semicolon-delimited dataset |
| `data/processed/` | Cleaned full cohort, development/test splits, normalization parameters and cleaning counts |
| `data_cleaning/` | Executed cleaning notebook, dependencies, methodology and reproducibility verification |
| `descriptive_analytics/dashboard/` | Streamlit descriptive dashboard and its dependencies |
| `predictive_analytics/` | Four executed training/evaluation notebooks and their shared helper |
| `references/` | Data sources and method references |

Notebook outputs include tables, evaluation metrics and figures. Running the
notebooks saves models and row-level predictions under
`predictive_analytics/runs/`, which is ignored by Git.

## Data

The raw dataset contains 70,000 records. Cleaning retains 68,569 records and 20 columns, including ID and target. The stratified
80/20 split contains 54,855 development records and 13,714 test records (seed 42).

| File in `data/processed/` | Purpose |
| --- | --- |
| `cardio_fully_cleaned.csv` | Full cleaned cohort for descriptive analysis |
| `cardio_train_80.csv` | Development data for five-fold CV, tuning and final refitting |
| `cardio_test_20.csv` | Held-out records for scoring and evaluation |
| `denormalization_params.json` | Restores numeric values from the stored MinMax representation |
| `cleaning_stats.json` | Cleaning counts and export information |

The processed CSVs store eight numeric columns using full-cohort MinMax
normalization. The predictive loader restores their measurement units, checks
them against source records, and fits model preprocessing within each
development fold. IDs and `cardio` are excluded from model inputs.

## Run the project

Use Python 3.12 and run these commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r predictive_analytics/requirements-notebooks.txt
python -m jupyter lab
```

1. Run `data_cleaning/data_cleaning.ipynb` to generate the five processed data files.
2. Run `predictive_analytics/01_baseline.ipynb` through `03_optimization_round2.ipynb` in order.
   Set the same new `RUN_NAME` in all notebooks to start a fresh experiment.
3. Run `predictive_analytics/04_test_and_ranking.ipynb` using that run name.
   It loads the saved XGBoost round2 model; it does not train on test records.

For the website:

```bash
python -m pip install -r descriptive_analytics/dashboard/requirements.txt -c data_cleaning/requirements.txt
python -m streamlit run descriptive_analytics/dashboard/app.py
```

For Streamlit deployment, use `descriptive_analytics/dashboard/app.py` as the
main file path.

## Predictive workflow

Seven inputs: age, gender, BMI, systolic BP, diastolic BP, cholesterol and glucose.
All models use the same five development folds, grouped by source health profile
(seed 4145). Baselines compare Logistic Regression, Random Forest and XGBoost.
Round1 tunes LR regularization and tree complexity; round2 retains LR and tunes
RF/XGBoost sampling. Selection uses mean validation ROC-AUC.

The selected XGBoost round2 model has mean validation ROC-AUC 0.8007 and full-test
ROC-AUC 0.7938. Notebook04 reports classification metrics at threshold 0.5,
continuous scores, the ranked test list, and capacity results from @10% to @100%.
True labels are used for evaluation, not score generation or ordering.

Evaluation uses an already-inspected test set, and 10 test records share complete
source profiles with development records. Notebook04 documents these limitations.
The full test set is the main evaluation; profile-excluded sensitivity results
are reported separately. These are retrospective results, not clinical validation.

See `references/sources.md` for attribution.
