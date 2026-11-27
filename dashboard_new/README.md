# Cardio Health Analytics

From the repository root, run:

    python -m pip install -r dashboard_new/requirements.txt
    python -m streamlit run dashboard_new/app.py

The Dataset selector defaults to Full Cohort (100%). Training Cohort (80%) loads the repository’s cardio_train_80.csv. Filters apply within the selected cohort. The correlation matrix and forest plot use the entire selected cohort. The two PNG previews in figures use the separate 54,855-record development cohort.
