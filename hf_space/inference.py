"""
Standalone inference layer for the soft-sensor HF Spaces deployment.

Loads trained .joblib artifacts from ./models/ and produces calibrated
predictions for the debutanizer and fermentation case studies.

This is a slimmed-down version of src/api/inference.py from the parent
project, with no FastAPI/Pydantic dependencies.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"

# --------------------------------------------------------------------------
# Feature definitions — must match the trained models exactly
# --------------------------------------------------------------------------
DEBUTANIZER_RAW_SENSORS = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
DEBUTANIZER_LAGS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]
DEBUTANIZER_HISTORY_LENGTH = max(DEBUTANIZER_LAGS) + 1  # 31

FERMENTATION_RAW_SENSORS = [
    "feed_rate_Lph",
    "temperature_K",
    "pH",
    "DO_pct",
    "agitator_rpm",
    "volume_L",
]
FERMENTATION_LAGS = [2, 4, 8, 16, 32]
FERMENTATION_HISTORY_LENGTH = max(FERMENTATION_LAGS) + 1  # 33


# --------------------------------------------------------------------------
# Module-level model cache
# --------------------------------------------------------------------------
_models: dict = {}


def load_all_models() -> dict:
    """Load every trained artifact from disk. Returns dict[name -> bool]."""
    results = {}

    # Debug: print what's actually in the models directory
    print(f"[debug] MODELS_DIR: {MODELS_DIR}")
    print(f"[debug] MODELS_DIR exists: {MODELS_DIR.exists()}")
    if MODELS_DIR.exists():
        for f in MODELS_DIR.iterdir():
            print(f"[debug]   {f.name}: {f.stat().st_size} bytes")
            # Read first 100 bytes to see if it's a real binary or a pointer
            with open(f, "rb") as fh:
                head = fh.read(100)
            print(f"[debug]     first 100 bytes: {head[:50]!r}...")

    try:
        _models["debutanizer_xgb"] = joblib.load(
            MODELS_DIR / "xgb_lagged_extended.joblib"
        )
        _models["debutanizer_scaler"] = joblib.load(
            MODELS_DIR / "scaler_lagged_extended.joblib"
        )
        _models["debutanizer_conformal"] = joblib.load(
            MODELS_DIR / "split_conformal_xgb.joblib"
        )
        results["debutanizer"] = True
    except Exception as e:
        print(f"[warn] Failed to load debutanizer artifacts: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["debutanizer"] = False

    try:
        _models["fermentation_xgb"] = joblib.load(
            MODELS_DIR / "xgb_fermentation_tuned.joblib"
        )
        _models["fermentation_scaler"] = joblib.load(
            MODELS_DIR / "scaler_fermentation.joblib"
        )
        _models["fermentation_conformal"] = joblib.load(
            MODELS_DIR / "split_conformal_fermentation.joblib"
        )
        results["fermentation"] = True
    except Exception as e:
        print(f"[warn] Failed to load fermentation artifacts: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        results["fermentation"] = False

    return results


def models_loaded() -> dict:
    """Report which models are currently loaded."""
    return {
        "debutanizer": all(
            _models.get(k) is not None
            for k in ["debutanizer_xgb", "debutanizer_scaler", "debutanizer_conformal"]
        ),
        "fermentation": all(
            _models.get(k) is not None
            for k in [
                "fermentation_xgb",
                "fermentation_scaler",
                "fermentation_conformal",
            ]
        ),
    }


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------


def _build_debutanizer_features(history: dict) -> pd.DataFrame:
    """Construct the 77-feature DataFrame for the debutanizer model."""
    feature_values = {}
    for sensor in DEBUTANIZER_RAW_SENSORS:
        feature_values[sensor] = [history[sensor][-1]]
    for sensor in DEBUTANIZER_RAW_SENSORS:
        for lag in DEBUTANIZER_LAGS:
            feature_values[f"{sensor}_lag{lag}"] = [history[sensor][-1 - lag]]
    return pd.DataFrame(feature_values)


def _build_fermentation_features(time_h: float, history: dict) -> pd.DataFrame:
    """Construct the 37-feature DataFrame for the fermentation model."""
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


def predict_debutanizer(history: dict) -> dict:
    """Predict C4 concentration with 95% conformal interval."""
    if not models_loaded()["debutanizer"]:
        raise RuntimeError("Debutanizer models not loaded.")

    X = _build_debutanizer_features(history)
    X_scaled = _models["debutanizer_scaler"].transform(X)
    y_pred, y_pis = _models["debutanizer_conformal"].predict_interval(X_scaled)

    return {
        "prediction": float(y_pred[0]),
        "lower_95": float(y_pis[0, 0, 0]),
        "upper_95": float(y_pis[0, 1, 0]),
        "interval_width": float(y_pis[0, 1, 0] - y_pis[0, 0, 0]),
        "case_study": "debutanizer",
        "units": "normalized [0,1]",
    }


def predict_fermentation(time_h: float, history: dict) -> dict:
    """Predict penicillin concentration (g/L) with 95% conformal interval."""
    if not models_loaded()["fermentation"]:
        raise RuntimeError("Fermentation models not loaded.")

    X = _build_fermentation_features(time_h, history)
    X_scaled = _models["fermentation_scaler"].transform(X)
    y_pred, y_pis = _models["fermentation_conformal"].predict_interval(X_scaled)

    return {
        "prediction": float(y_pred[0]),
        "lower_95": float(y_pis[0, 0, 0]),
        "upper_95": float(y_pis[0, 1, 0]),
        "interval_width": float(y_pis[0, 1, 0] - y_pis[0, 0, 0]),
        "case_study": "fermentation",
        "units": "g/L",
    }
