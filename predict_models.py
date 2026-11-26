"""
Seven-feature rock-property prediction inference module.

Feature order (must match the fitted model objects):
1. Plagioclase feldspar (%)
2. Amphibole (%)
3. Quartz (%)
4. Alkali feldspar (%)
5. Density (g/cm^3)
6. Porosity (%)
7. Mica (%)

Targets are predicted independently: UCS (MPa) and Young's Modulus E (GPa).
"""
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd

_BASE = Path(__file__).resolve().parent
_FEATURES = [
    "Plagioclase feldspar (%)",
    "Amphibole (%)",
    "Quartz (%)",
    "Alkali feldspar (%)",
    "Density (g/cm³)",
    "Porosity (%)",
    "Mica (%)",
]
_INPUT_KEYS = [
    "plagioclase", "amphibole", "quartz", "alkali_feldspar",
    "density", "porosity", "mica",
]

def _load_model(filename):
    model = joblib.load(_BASE / filename)
    n_features = getattr(model, "n_features_in_", None)
    if n_features != len(_FEATURES):
        raise RuntimeError(
            f"{filename} expects {n_features} features; expected {len(_FEATURES)}."
        )
    names = getattr(model, "feature_names_in_", None)
    if names is not None and list(names) != _FEATURES:
        raise RuntimeError(f"Unexpected feature order in {filename}: {list(names)}")
    return model

# Load and validate all four fitted models once at application startup.
_xgb_ucs = _load_model("XGBoost_UCS.joblib")
_xgb_e = _load_model("XGBoost_E.joblib")
_poly_ucs = _load_model("Sparse_Polynomial_UCS.joblib")
_poly_e = _load_model("Sparse_Polynomial_E.joblib")

def predict(plagioclase, amphibole, quartz, alkali_feldspar, density, porosity, mica):
    """Return predictions from all four models; reject non-numeric/non-finite inputs."""
    raw = [plagioclase, amphibole, quartz, alkali_feldspar, density, porosity, mica]
    try:
        row = [float(value) for value in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("All seven input values must be numeric.") from exc

    if not all(math.isfinite(value) for value in row):
        raise ValueError("All seven input values must be finite (not NaN or infinity).")

    frame = pd.DataFrame([row], columns=_FEATURES)
    return {
        "xgboost_ucs_mpa": float(_xgb_ucs.predict(frame)[0]),
        "xgboost_e_gpa": float(_xgb_e.predict(frame)[0]),
        "symbolic_ucs_mpa": float(_poly_ucs.predict(frame)[0]),
        "symbolic_e_gpa": float(_poly_e.predict(frame)[0]),
    }

if __name__ == "__main__":
    # Smoke-test example only; replace with measured sample inputs.
    example = predict(30, 20, 15, 15, 2.70, 1.0, 5)
    print(example)
