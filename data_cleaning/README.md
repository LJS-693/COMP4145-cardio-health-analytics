# Data cleaning

Open `data_cleaning.ipynb` and run all cells in order. Use Python 3.12 and install
`python -m pip install -r data_cleaning/requirements.txt` from the project root.
The notebook locates the root automatically when launched anywhere inside this repository.

Input: `data/raw/cardio_train.csv` (separator `;`). The notebook writes or replaces
the five shared exports in `data/processed/`: three CSV datasets, normalization
parameters and cleaning statistics. The notebook covers filtering, feature
engineering, encoding, normalization and the stratified development/test split.
Saved cell outputs show each stage of the cleaning process.

See `cleaning_report.md` for the method and `verification.json` for reproducibility checks.
