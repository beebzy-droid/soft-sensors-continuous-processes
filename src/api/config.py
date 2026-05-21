"""
Configuration constants for the soft-sensor API.

Centralizes file paths to model artifacts so that the inference layer
doesn't hardcode locations. If you move the model files, change them here.
"""

from pathlib import Path

# Project root, two levels up from this file (src/api/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Model artifact paths
MODELS_DIR = PROJECT_ROOT / "src" / "models"

# Debutanizer (petrochemical case study, week 3 conformal model)
DEBUTANIZER_XGB_PATH = MODELS_DIR / "xgb_lagged_extended.joblib"
DEBUTANIZER_SCALER_PATH = MODELS_DIR / "scaler_lagged_extended.joblib"
DEBUTANIZER_CONFORMAL_PATH = MODELS_DIR / "split_conformal_xgb.joblib"

# Fermentation (F&B case study, week 5 conformal model)
FERMENTATION_XGB_PATH = MODELS_DIR / "xgb_fermentation_tuned.joblib"
FERMENTATION_SCALER_PATH = MODELS_DIR / "scaler_fermentation.joblib"
FERMENTATION_CONFORMAL_PATH = MODELS_DIR / "split_conformal_fermentation.joblib"

# Feature definitions — must match exactly what the trained models expect
DEBUTANIZER_RAW_SENSORS = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
DEBUTANIZER_LAGS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

FERMENTATION_RAW_SENSORS = [
    "feed_rate_Lph",
    "temperature_K",
    "pH",
    "DO_pct",
    "agitator_rpm",
    "volume_L",
]
FERMENTATION_LAGS = [2, 4, 8, 16, 32]
FERMENTATION_SPECIAL = ["time_h"]  # not lagged

# API metadata
API_VERSION = "0.1.0"
API_TITLE = "Soft Sensors for Continuous Processes"
API_DESCRIPTION = (
    "Inference API for two industrial soft sensors:\n"
    "- Debutanizer: predicts butane (C4) concentration in distillation bottom flow\n"
    "- Fermentation: predicts penicillin concentration during fed-batch fermentation\n\n"
    "Both endpoints return point predictions plus calibrated 95% conformal intervals."
)
