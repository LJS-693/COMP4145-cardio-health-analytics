# Sources and attribution

## Data and original reference project

- Sulianova. Cardiovascular Disease Dataset. Kaggle.
  https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
- Bagchi, S., Chacko, R., and Van Uden, C. Cardiovascular disease diagnosis.
  Original reference repository: https://github.com/caravanuden/cardio
  The supplied `cardio-master` folder was used as a reference. Only one unchanged
  source CSV is retained in `data/raw/`; duplicate datasets and original exploratory
  notebooks are omitted from the final submission. See the original sources for
  applicable terms and provenance.

## Models and evaluation

- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
  Journal of Machine Learning Research, 12, 2825–2830.
  https://www.jmlr.org/papers/v12/pedregosa11a.html
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical
  learning: Data mining, inference, and prediction* (2nd ed.). Springer.
  [Book](https://doi.org/10.1007/978-0-387-84858-7).
  Used for the statistical learning and logistic regression background.
- Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.
  https://doi.org/10.1023/A:1010933404324
- Chen, T., and Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
  Proceedings of KDD, 785–794. https://doi.org/10.1145/2939672.2939785
- Fawcett, T. (2006). An introduction to ROC analysis.
  Pattern Recognition Letters, 27(8), 861–874.
  https://doi.org/10.1016/j.patrec.2005.10.010
- Scikit-learn developers. LogisticRegression.
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Scikit-learn developers. GridSearchCV.
  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
- Scikit-learn developers. Metrics and scoring: Quantifying the quality of predictions.
  https://scikit-learn.org/stable/modules/model_evaluation.html

- IBM. (n.d.). *Cumulative gains and lift charts*. IBM SPSS Statistics documentation.
  [Documentation](https://www.ibm.com/docs/en/spss-statistics/30.0.0?topic=overtraining-cumulative-gains-lift-charts).
  Used to explain cumulative gain, lift and the random-selection baseline in
  ranking evaluation. Project calculations use the test-set scores and labels.

## Descriptive definitions

- American Heart Association. (2025, August 14). *Understanding blood pressure readings*.
  [Blood pressure categories](https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings).
  Used for blood pressure category definitions, including the stage 2 threshold
  of systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg. The dashboard also uses
  project-specific systolic bands for descriptive comparisons.
- World Health Organization. (2026, October 7). *Obesity and overweight*.
  [Fact sheet](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight).
  Used for adult BMI definitions: overweight at BMI ≥25 kg/m² and obesity
  at BMI ≥30 kg/m².

These descriptive definitions do not establish the project's numeric outlier
exclusion bounds; cleaning bounds are documented in `data_cleaning/cleaning_report.md`.

Project results and figures are produced by the group's retained cleaning notebook,
descriptive dashboard and four predictive notebooks.
