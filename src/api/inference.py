"""
Inference layer: load trained models and produce calibrated predictions.

Models are loaded once at startup (see `load_all_models`) and held in
module-level state. Each `predict_*` function takes a validated request
schema and returns a PredictionResponse.

The feature-engineering pipeline (raw history -> lagged features -> scaled
features -> point prediction -> conformal interval) is fully encapsulated
here. The API layer (main.py) only sees clean inputs and outputs.
"""

from typing import Optional
import numpy as np
import joblib
import pandas as pd

from src.api.config import (
    DEBUTANIZER_XGB_PATH,
    DEBUTANIZER_SCALER_PATH,
    DEBUTANIZER_CONFORMAL_PATH,
    DEBUTANIZER_RAW_SENSORS,
    DEBUTANIZER_LAGS,
    FERMENTATION_XGB_PATH,
    FERMENTATION_SCALER_PATH,
    FERMENTATION_CONFORMAL_PATH,
    FERMENTATION_RAW_SENSORS,
    FERMENTATION_LAGS,
    FERMENTATION_SPECIAL,
    API_VERSION,
)
from src.api.schemas import (
    DebutanizerRequest,
    FermentationRequest,
    PredictionResponse,
)

# --------------------------------------------------------------------------
# Module-level model cache; populated by load_all_models().
# --------------------------------------------------------------------------
_models: dict = {
    "debutanizer_xgb": None,
    "debutanizer_scaler": None,
    "debutanizer_conformal": None,
    "fermentation_xgb": None,
    "fermentation_scaler": None,
    "fermentation_conformal": None,
}


def load_all_models() -> dict:
    """Load every trained artifact from disk. Returns dict[name -> bool]
    indicating which loaded successfully. Called at API startup."""
    results = {}
    try:
        _models["debutanizer_xgb"] = joblib.load(DEBUTANIZER_XGB_PATH)
        _models["debutanizer_scaler"] = joblib.load(DEBUTANIZER_SCALER_PATH)
        _models["debutanizer_conformal"] = joblib.load(DEBUTANIZER_CONFORMAL_PATH)
        results["debutanizer"] = True
    except Exception as e:
        print(f"[warn] Failed to load debutanizer artifacts: {e}")
        results["debutanizer"] = False

    try:
        _models["fermentation_xgb"] = joblib.load(FERMENTATION_XGB_PATH)
        _models["fermentation_scaler"] = joblib.load(FERMENTATION_SCALER_PATH)
        _models["fermentation_conformal"] = joblib.load(FERMENTATION_CONFORMAL_PATH)
        results["fermentation"] = True
    except Exception as e:
        print(f"[warn] Failed to load fermentation artifacts: {e}")
        results["fermentation"] = False

    return results


def models_loaded() -> dict:
    """Report which models are currently loaded — used by /health."""
    return {
        "debutanizer": all(
            _models[k] is not None
            for k in ["debutanizer_xgb", "debutanizer_scaler", "debutanizer_conformal"]
        ),
        "fermentation": all(
            _models[k] is not None
            for k in [
                "fermentation_xgb",
                "fermentation_scaler",
                "fermentation_conformal",
            ]
        ),
    }


# --------------------------------------------------------------------------
# Feature construction: turn raw history into the lagged feature vector
# the model expects. These must match EXACTLY what we built in the notebooks.
# --------------------------------------------------------------------------


def _build_debutanizer_features(history: dict) -> pd.DataFrame:
    """
    Construct the 77-feature DataFrame for the debutanizer model from raw history.
    Returns a DataFrame so that StandardScaler sees the same column names it was
    fitted on (no sklearn UserWarning about missing feature names).
    """
    feature_values = {}
    # Current values
    for sensor in DEBUTANIZER_RAW_SENSORS:
        feature_values[sensor] = [history[sensor][-1]]
    # Lagged values
    for sensor in DEBUTANIZER_RAW_SENSORS:
        for lag in DEBUTANIZER_LAGS:
            feature_values[f"{sensor}_lag{lag}"] = [history[sensor][-1 - lag]]

    return pd.DataFrame(feature_values)


def _build_fermentation_features(time_h: float, history: dict) -> pd.DataFrame:
    """
    Construct the 37-feature DataFrame for the fermentation model.
    Order matches notebook 05: time_h, [6 current sensors], [6 × 5 lagged].
    """
    feature_values = {"time_h": [time_h]}
    for sensor in FERMENTATION_RAW_SENSORS:
        feature_values[sensor] = [history[sensor][-1]]
    for sensor in FERMENTATION_RAW_SENSORS:
        for lag in FERMENTATION_LAGS:
            feature_values[f"{sensor}_lag{lag}"] = [history[sensor][-1 - lag]]

    return pd.DataFrame(feature_values)


# --------------------------------------------------------------------------
# Prediction entry points
# --------------------------------------------------------------------------


def predict_debutanizer(req: DebutanizerRequest) -> PredictionResponse:
    """Predict C4 concentration with 95% conformal interval."""
    if not models_loaded()["debutanizer"]:
        raise RuntimeError("Debutanizer models not loaded.")

    X = _build_debutanizer_features(req.history)
    X_scaled = _models["debutanizer_scaler"].transform(X)

    # MAPIE prefit conformal predict_interval returns (point, interval)
    y_pred, y_pis = _models["debutanizer_conformal"].predict_interval(X_scaled)
    point = float(y_pred[0])
    lower = float(y_pis[0, 0, 0])
    upper = float(y_pis[0, 1, 0])

    return PredictionResponse(
        prediction=point,
        lower_95=lower,
        upper_95=upper,
        interval_width=upper - lower,
        case_study="debutanizer",
        units="normalized [0,1]",
        model_version=API_VERSION,
    )


def predict_fermentation(req: FermentationRequest) -> PredictionResponse:
    """Predict penicillin concentration (g/L) with 95% conformal interval."""
    if not models_loaded()["fermentation"]:
        raise RuntimeError("Fermentation models not loaded.")

    X = _build_fermentation_features(req.time_h, req.history)
    X_scaled = _models["fermentation_scaler"].transform(X)

    y_pred, y_pis = _models["fermentation_conformal"].predict_interval(X_scaled)
    point = float(y_pred[0])
    lower = float(y_pis[0, 0, 0])
    upper = float(y_pis[0, 1, 0])

    return PredictionResponse(
        prediction=point,
        lower_95=lower,
        upper_95=upper,
        interval_width=upper - lower,
        case_study="fermentation",
        units="g/L",
        model_version=API_VERSION,
    )
