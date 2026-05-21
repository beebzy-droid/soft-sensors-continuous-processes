"""
Pydantic schemas for API request and response payloads.

These schemas are validated automatically by FastAPI on every request.
Bad inputs (wrong types, missing fields, wrong shapes) produce a clean
422 Unprocessable Entity response, NOT a 500 Internal Server Error.
"""

from typing import Dict, List
from pydantic import BaseModel, Field, field_validator, ConfigDict

from src.api.config import (
    DEBUTANIZER_RAW_SENSORS,
    DEBUTANIZER_LAGS,
    FERMENTATION_RAW_SENSORS,
    FERMENTATION_LAGS,
)

# --------------------------------------------------------------------------
# Required history depths — the caller must send at least this many samples
# of each sensor for us to compute the maximum lag.
# --------------------------------------------------------------------------
DEBUTANIZER_HISTORY_LENGTH = max(DEBUTANIZER_LAGS) + 1  # 31 samples
FERMENTATION_HISTORY_LENGTH = max(FERMENTATION_LAGS) + 1  # 33 samples


# --------------------------------------------------------------------------
# Debutanizer request: dict of 7 sensors, each a list of recent values
# --------------------------------------------------------------------------
class DebutanizerRequest(BaseModel):
    """A single soft-sensor inference request for the debutanizer column.

    Provide the most recent 31 samples of each of the 7 process sensors,
    ordered oldest-to-newest. The last value in each array is "now".
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "history": {
                    sensor: [0.5] * DEBUTANIZER_HISTORY_LENGTH
                    for sensor in DEBUTANIZER_RAW_SENSORS
                }
            }
        }
    )

    history: Dict[str, List[float]] = Field(
        ...,
        description=(
            f"Dict mapping each sensor name to its recent {DEBUTANIZER_HISTORY_LENGTH} samples "
            f"(oldest first). Required sensors: {DEBUTANIZER_RAW_SENSORS}."
        ),
    )

    @field_validator("history")
    @classmethod
    def check_sensors_and_lengths(
        cls, v: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        missing = set(DEBUTANIZER_RAW_SENSORS) - set(v.keys())
        if missing:
            raise ValueError(f"Missing required sensors: {sorted(missing)}")
        extra = set(v.keys()) - set(DEBUTANIZER_RAW_SENSORS)
        if extra:
            raise ValueError(f"Unexpected sensors: {sorted(extra)}")
        for sensor, values in v.items():
            if len(values) != DEBUTANIZER_HISTORY_LENGTH:
                raise ValueError(
                    f"Sensor '{sensor}' has {len(values)} values; expected {DEBUTANIZER_HISTORY_LENGTH}"
                )
        return v


# --------------------------------------------------------------------------
# Fermentation request: 6 online sensors history + current time_h
# --------------------------------------------------------------------------
class FermentationRequest(BaseModel):
    """A single soft-sensor inference request for fed-batch fermentation.

    Provide the most recent 33 samples of each of the 6 online sensors,
    plus the current time-since-batch-start in hours.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "time_h": 100.0,
                "history": {
                    sensor: [0.5] * FERMENTATION_HISTORY_LENGTH
                    for sensor in FERMENTATION_RAW_SENSORS
                },
            }
        }
    )

    time_h: float = Field(
        ...,
        ge=0,
        description="Current time since batch start, in hours.",
    )
    history: Dict[str, List[float]] = Field(
        ...,
        description=(
            f"Dict mapping each online sensor to its recent {FERMENTATION_HISTORY_LENGTH} samples. "
            f"Required sensors: {FERMENTATION_RAW_SENSORS}."
        ),
    )

    @field_validator("history")
    @classmethod
    def check_sensors_and_lengths(
        cls, v: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        missing = set(FERMENTATION_RAW_SENSORS) - set(v.keys())
        if missing:
            raise ValueError(f"Missing required sensors: {sorted(missing)}")
        extra = set(v.keys()) - set(FERMENTATION_RAW_SENSORS)
        if extra:
            raise ValueError(f"Unexpected sensors: {sorted(extra)}")
        for sensor, values in v.items():
            if len(values) != FERMENTATION_HISTORY_LENGTH:
                raise ValueError(
                    f"Sensor '{sensor}' has {len(values)} values; expected {FERMENTATION_HISTORY_LENGTH}"
                )
        return v


# --------------------------------------------------------------------------
# Unified prediction response
# --------------------------------------------------------------------------
class PredictionResponse(BaseModel):
    """Soft-sensor prediction with calibrated 95% interval."""

    prediction: float = Field(..., description="Point prediction in target units.")
    lower_95: float = Field(..., description="Lower bound of 95% conformal interval.")
    upper_95: float = Field(..., description="Upper bound of 95% conformal interval.")
    interval_width: float = Field(
        ..., description="Upper minus lower; in target units."
    )
    case_study: str = Field(..., description="Which model produced this prediction.")
    units: str = Field(
        ..., description="Units of the prediction (e.g. 'g/L', 'normalized')."
    )
    model_version: str = Field(..., description="Model artifact version identifier.")


# --------------------------------------------------------------------------
# Health check response
# --------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if service is healthy.")
    api_version: str
    models_loaded: Dict[str, bool] = Field(
        ..., description="Which models loaded successfully at startup."
    )
