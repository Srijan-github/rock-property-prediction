# Rock Property Prediction Website — Seven-Feature Version

A responsive browser interface and FastAPI backend for estimating Uniaxial Compressive Strength (UCS) and Young's Modulus (E) using four fitted machine-learning models. All predictions run server-side; model files are not sent to the browser.

## Seven input features

The models require exactly these seven features in this order:

1. Plagioclase feldspar (%)
2. Amphibole (%)
3. Quartz (%)
4. Alkali feldspar (%)
5. Density (g/cm³)
6. Porosity (%)
7. Mica (%)

Targets: UCS (MPa) and Young's Modulus E (GPa).

## Included models

- `XGBoost_UCS.joblib` — XGBoost regression model for UCS
- `XGBoost_E.joblib` — XGBoost regression model for E
- `Sparse_Polynomial_UCS.joblib` — degree-2 polynomial feature pipeline with StandardScaler and LassoCV for UCS
- `Sparse_Polynomial_E.joblib` — degree-2 polynomial feature pipeline with StandardScaler and LassoCV for E

The sparse polynomial models are LassoCV-selected polynomial feature-library models. They are not genetic-programming symbolic regression (PySR/gplearn).

The four model objects were inspected before integration: each reports `n_features_in_ = 7`, and each fitted model's feature names match the seven-feature order above.

## Model evaluation

The supplied `model_performance_summary.csv` is the source of the reported evaluation metrics. The evaluation used 193 complete records, an 80:20 split (154 training / 39 test records), and random state 42.

| Model | Test R² | Test RMSE | Test MAE | 5-fold CV R² (train) |
|---|---:|---:|---:|---:|
| XGBoost — UCS | 0.6408 | 31.7903 MPa | 23.2288 MPa | 0.6107 |
| Sparse polynomial — UCS | 0.6568 | 31.0767 MPa | 22.7184 MPa | 0.4549 |
| XGBoost — E | 0.7675 | 10.2339 GPa | 6.4479 GPa | 0.6976 |
| Sparse polynomial — E | 0.7468 | 10.6792 GPa | 7.4333 GPa | 0.6462 |

The final model objects are the fitted models supplied in `rock_property_4_models_package.zip`. The test metrics describe the supplied evaluation split; they are not guarantees of performance on future samples.

## Run locally

Python 3.10–3.13 recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. API documentation: <http://127.0.0.1:8000/docs>. Health check: `/api/health`.

## Prediction API

`POST /api/predict`

```json
{
  "plagioclase": 30,
  "amphibole": 20,
  "quartz": 15,
  "alkali_feldspar": 15,
  "density": 2.70,
  "porosity": 1.0,
  "mica": 5
}
```

The response contains four values in `predictions`:
- `xgboost_ucs_mpa`
- `xgboost_e_gpa`
- `symbolic_ucs_mpa` (sparse polynomial model)
- `symbolic_e_gpa` (sparse polynomial model)

All seven values are required. Unknown request fields are rejected. NaN and infinite values are rejected.

## Deploy to Render

The included `render.yaml` is configured for a Python web service.

1. Push this project folder to the connected Git repository.
2. In Render, create or update the Web Service using this repository.
3. Build command: `pip install -r requirements.txt`.
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
5. Confirm `/api/health` returns `input_features: 7` and `models_loaded: 4`.

Do not deploy the old six-feature JSON models or old `predict_models.py` alongside this version. This package replaces them with the four seven-feature `.joblib` files.

## Model provenance and supporting files

- `train_rock_models.py` — supplied training script
- `model_settings.json` — supplied training configuration and feature order
- `model_performance_summary.csv` — supplied model metrics
- `test_set_predictions.csv` — supplied held-out predictions
- `rock_model_results.xlsx` — supplied model results workbook
- `model_metadata.json` — deployment metadata for the seven-feature website

The model package did not include an empirical Mica minimum/maximum. The interface therefore accepts Mica from 0–100% but does not claim an observed training range for Mica.

## Research-use limitation

Predictions are research estimates and should not replace laboratory testing, engineering judgement, or validation on independent rock samples. Predictions outside the training data distribution may be unreliable.
