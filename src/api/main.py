"""
FastAPI application for the soft-sensor inference service.

Two prediction endpoints (one per case study) plus a health endpoint.
Models are loaded once at startup via the lifespan handler.

Run locally:
    uvicorn src.api.main:app --reload

Interactive docs:
    http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from src.api.config import API_TITLE, API_DESCRIPTION, API_VERSION
from src.api.inference import (
    load_all_models,
    models_loaded,
    predict_debutanizer,
    predict_fermentation,
)
from src.api.schemas import (
    DebutanizerRequest,
    FermentationRequest,
    PredictionResponse,
    HealthResponse,
)


# --------------------------------------------------------------------------
# Lifespan handler: load models at startup, log cleanup at shutdown.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load all model artifacts. Shutdown: log only (no cleanup needed)."""
    print("[startup] Loading model artifacts...")
    results = load_all_models()
    for case, ok in results.items():
        print(f"  {case}: {'OK' if ok else 'FAILED'}")
    if not any(results.values()):
        print("[startup] WARNING: No models loaded; predictions will fail.")
    yield
    print("[shutdown] Service stopping.")


# --------------------------------------------------------------------------
# App instance
# --------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Health endpoint
# --------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["meta"],
    summary="Service health check",
)
def health() -> HealthResponse:
    """Returns 200 with status info. Useful for load balancers and Docker healthchecks."""
    return HealthResponse(
        status="ok",
        api_version=API_VERSION,
        models_loaded=models_loaded(),
    )


# --------------------------------------------------------------------------
# Root endpoint — friendly redirect to /docs
# --------------------------------------------------------------------------
@app.get("/", tags=["meta"], include_in_schema=False)
def root():
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": ["/predict/debutanizer", "/predict/fermentation", "/health"],
    }


# --------------------------------------------------------------------------
# Prediction endpoints
# --------------------------------------------------------------------------
@app.post(
    "/predict/debutanizer",
    response_model=PredictionResponse,
    tags=["soft-sensor"],
    summary="Predict C4 concentration in debutanizer bottom flow",
    description=(
        "Given the most recent 31 samples of each of 7 process sensors, "
        "returns the predicted butane (C4) concentration in the column's "
        "bottom product, plus a calibrated 95% conformal interval."
    ),
)
def predict_debutanizer_endpoint(req: DebutanizerRequest) -> PredictionResponse:
    if not models_loaded()["debutanizer"]:
        raise HTTPException(
            status_code=503,
            detail="Debutanizer models not loaded.",
        )
    try:
        return predict_debutanizer(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")


@app.post(
    "/predict/fermentation",
    response_model=PredictionResponse,
    tags=["soft-sensor"],
    summary="Predict penicillin concentration in fed-batch fermentation",
    description=(
        "Given the most recent 33 samples of 6 online sensors plus the "
        "current time-since-batch-start, returns the predicted penicillin "
        "concentration (g/L) plus a calibrated 95% conformal interval."
    ),
)
def predict_fermentation_endpoint(req: FermentationRequest) -> PredictionResponse:
    if not models_loaded()["fermentation"]:
        raise HTTPException(
            status_code=503,
            detail="Fermentation models not loaded.",
        )
    try:
        return predict_fermentation(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
