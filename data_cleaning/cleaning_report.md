# Cleaning methodology and final data

The final notebook preserves the original cleaning rules and processing order.
Its thresholds are project data-quality rules, not diagnostic criteria.

1. Load the unchanged 70,000-record semicolon-delimited source and retain IDs.
2. Compute age in years as age in days / 365.25.
3. Filter negative/inverted blood pressure and the original broad bounds for
   height (100–230 cm), weight (30–300 kg), and age (18–100 years).
4. Apply the original second-stage bounds: systolic BP 70–250 mmHg,
   diastolic BP 40–150 mmHg, BMI 12–70, and height 130–220 cm.
5. Derive BMI, pulse pressure and mean arterial pressure; encode gender and
   cholesterol/glucose categories as in the original procedure.
6. Apply the original full-cohort MinMax export transformation to eight numeric
   columns and save its inverse parameters. This preserves existing file bytes.
7. Split by stratified `train_test_split(test_size=0.20, random_state=42)`.
8. Export the three CSVs and two JSON files, preserving row order and formatting.

## Final outputs

| Export | Records | Columns |
| --- | ---: | ---: |
| `cardio_fully_cleaned.csv` | 68,569 | 20 |
| `cardio_train_80.csv` | 54,855 | 20 |
| `cardio_test_20.csv` | 13,714 | 20 |

The two JSONs are `denormalization_params.json` and `cleaning_stats.json`.
Individual rule counts can overlap; the notebook reports staged removal totals.

The export normalization precedes splitting in the preserved cleaning algorithm.
For predictive modeling, the shared loader restores original units and checks them
against source records. LR scaling and categorical preprocessing are then fitted
inside development folds; tree numeric inputs pass through. The target and ID
are not model features. Development-fold grouping keeps identical source profiles
together, while notebook04 audits cross-set profile overlap separately.

The data are observational. Cleaning bounds, descriptive associations and model
scores do not establish causality, clinical validity or future disease risk.
