# Descriptive analytics

The Streamlit dashboard is in `dashboard/app.py`.
It reads the shared cleaned CSVs and normalization metadata from `data/processed/`.

From the project root:

```bash
python -m pip install -r descriptive_analytics/dashboard/requirements.txt -c data_cleaning/requirements.txt
python -m streamlit run descriptive_analytics/dashboard/app.py
```

The dashboard offers the full cleaned cohort (68,569) and development cohort
(54,855), interactive cohort filters, distributions, comparisons by cardio label,
Spearman correlations and adjusted odds ratios. Descriptive associations and the
dashboard's indicator counts should not be interpreted as clinical or XGBoost scores.

Deployment entrypoint: `descriptive_analytics/dashboard/app.py`.
