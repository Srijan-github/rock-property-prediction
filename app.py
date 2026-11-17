from pathlib import Path
import math
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict

from predict_models import predict

BASE = Path(__file__).resolve().parent
app = FastAPI(
    title="Rock Property Prediction API",
    version="1.0.0",
    description="Predict UCS and Young's Modulus from six mineralogical and physical inputs using four trained models.",
)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

FiniteNumber = Annotated[float, Field(strict=True)]

class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plagioclase: FiniteNumber = Field(description="Plagioclase feldspar (%)")
    amphibole: FiniteNumber = Field(description="Amphibole (%)")
    quartz: FiniteNumber = Field(description="Quartz (%)")
    alkali_feldspar: FiniteNumber = Field(description="Alkali feldspar (%)")
    density: FiniteNumber = Field(description="Density (g/cm³)")
    porosity: FiniteNumber = Field(description="Porosity (%)")

    def validate_finite(self):
        for key, value in self.model_dump().items():
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(BASE / "static" / "index.html")

@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": 4}

@app.post("/api/predict")
def run_prediction(payload: PredictionInput):
    try:
        payload.validate_finite()
        result = predict(**payload.model_dump())
        return {
            "predictions": result,
            "units": {"ucs": "MPa", "youngs_modulus": "GPa"},
            "model_labels": {
                "xgboost_ucs_mpa": "XGBoost · UCS",
                "xgboost_e_gpa": "XGBoost · Young's Modulus",
                "symbolic_ucs_mpa": "Sparse symbolic equation · UCS",
                "symbolic_e_gpa": "Sparse symbolic equation · Young's Modulus",
            },
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
