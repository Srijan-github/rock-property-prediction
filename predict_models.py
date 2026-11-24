"""
Rock-property prediction inference module.

Input order:
1. Plagioclase feldspar (%)
2. Amphibole (%)
3. Quartz (%)
4. Alkali feldspar (%)
5. Density (g/cm^3)
6. Porosity (%)

The two targets are predicted independently. No target property is used as an input.
"""
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

_BASE = Path(__file__).resolve().parent
_FEATURES = [
    "plagioclase",
    "amphibole",
    "quartz",
    "alkali_feldspar",
    "density",
    "porosity",
]

def _load_symbolic(filename):
    with open(_BASE / filename, "r", encoding="utf-8") as f:
        model = json.load(f)
    if model.get("input_features") != _FEATURES:
        raise ValueError(f"Unexpected feature order in {filename}")
    return model

def _symbolic_predict(model, row):
    value = float(model["intercept"])
    for term in model["terms"]:
        monomial = 1.0
        for x, power in zip(row, term["powers"]):
            if power:
                monomial *= x ** int(power)
        value += float(term["coefficient"]) * monomial
    return float(value)

# Load once when this module is imported.
_xgb_ucs = XGBRegressor()
_xgb_ucs.load_model(str(_BASE / "xgboost_ucs.json"))

_xgb_e = XGBRegressor()
_xgb_e.load_model(str(_BASE / "xgboost_e.json"))

_sym_ucs = _load_symbolic("symbolic_ucs.json")
_sym_e = _load_symbolic("symbolic_e.json")

def predict(plagioclase, amphibole, quartz, alkali_feldspar, density, porosity):
    """
    Return predictions from all four independently trained models.

    Raises TypeError/ValueError for non-numeric, NaN, or infinite inputs.
    Units: mineral percentages (%), density (g/cm^3), porosity (%).
    """
    raw = [plagioclase, amphibole, quartz, alkali_feldspar, density, porosity]
    try:
        row = [float(v) for v in raw]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("All six input values must be numeric.") from exc

    if not all(math.isfinite(v) for v in row):
        raise ValueError("All six input values must be finite (not NaN or infinity).")

    frame = pd.DataFrame([row], columns=_FEATURES)
    xgb_ucs_prediction = float(_xgb_ucs.predict(frame)[0])
    xgb_e_prediction = float(_xgb_e.predict(frame)[0])
    symbolic_ucs_prediction = _symbolic_predict(_sym_ucs, row)
    symbolic_e_prediction = _symbolic_predict(_sym_e, row)

    return {
        "xgboost_ucs_mpa": xgb_ucs_prediction,
        "xgboost_e_gpa": xgb_e_prediction,
        "symbolic_ucs_mpa": symbolic_ucs_prediction,
        "symbolic_e_gpa": symbolic_e_prediction,
    }

if __name__ == "__main__":
    # Example only; replace with measured inputs.
    example = predict(30, 20, 15, 15, 2.70, 1.0)
    print(json.dumps(example, indent=2))
