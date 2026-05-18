# Kepler Exoplanet System

Streamlit frontend for the `main.ipynb` Kepler light-curve exoplanet detection
workflow.

## Run the Dashboard

```powershell
.\.venv\Scripts\streamlit.exe run .\app\app.py
```

Then open:

```text
http://localhost:8501
```

## Data Format

The dashboard expects the notebook dataset format:

- `LABEL` column where `1 = no planet`, `2 = exoplanet`, or already mapped
  `0 = no planet`, `1 = exoplanet`
- numeric flux time-step columns for each light curve

You can upload `exo_all_combined.csv` from the notebook. The app also looks for
`exo_all_combined.csv` or `exoTrain.csv` in the project root or `Data` folder.
If no light-curve CSV is found, it opens with synthetic demo data so the
frontend still works.

## Dashboard Features

- Row-wise z-score normalization and Gaussian smoothing
- Notebook-style feature engineering from light curves
- Class-balance and variability EDA
- Random Forest, SVM, and XGBoost model options
- SMOTE on the training split only
- Precision, recall, ROC-AUC, average precision, and confusion matrix
- Single light-curve inspection and prediction
