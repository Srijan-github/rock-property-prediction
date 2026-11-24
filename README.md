# Rock Property Prediction Website

A responsive browser interface and FastAPI backend for the four trained rock-property models. The website runs all four models server-side; model files are not sent to the browser.

## Models

- XGBoost — UCS (MPa)
- XGBoost — Young's Modulus E (GPa)
- Sparse polynomial symbolic equation — UCS (MPa)
- Sparse polynomial symbolic equation — E (GPa)

All models use exactly these six inputs, in order: plagioclase feldspar (%), amphibole (%), quartz (%), alkali feldspar (%), density (g/cm³), porosity (%).

**Method disclosure:** the two equation-based models are sparse degree-two polynomial feature-library models selected using LassoCV. They are not genetic-programming symbolic regression (PySR/gplearn). See `model_metadata.json`.

## Run locally

Python 3.10–3.13 recommended.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>; health check: `/api/health`.

## Prediction API

`POST /api/predict`

```json
{
  "plagioclase": 30,
  "amphibole": 20,
  "quartz": 15,
  "alkali_feldspar": 15,
  "density": 2.70,
  "porosity": 1.0
}
```

Returns four predictions in `predictions`: `xgboost_ucs_mpa`, `xgboost_e_gpa`, `symbolic_ucs_mpa`, `symbolic_e_gpa`.

## Deploy to Render

1. Create a new **Web Service** and connect the repository containing this folder (or upload/push the project to a Git repository).
2. Set the service root directory to this project folder if it is inside a larger repository.
3. Runtime: Python. Build command: `pip install -r requirements.txt`.
4. Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
5. Deploy. Render will provide a public HTTPS URL. Add a custom domain in Render/DNS settings if required.

The included `render.yaml` can also be used for Blueprint deployment.

## Validation / scope

The source model metadata records the original held-out 80:20 validation experiment (39 test observations) and the final full-data refit. Test-set R² values: XGBoost UCS 0.616; XGBoost E 0.620; sparse symbolic UCS 0.696; sparse symbolic E 0.745. These are dataset-specific validation metrics, not a guarantee for future samples. Inputs outside the observed training ranges are flagged in the UI but are still passed to the models.

This project is an application package, not a live deployment. A public website requires deployment to a hosting provider and, optionally, a domain.
