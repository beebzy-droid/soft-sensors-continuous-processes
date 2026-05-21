"""
Integration tests for the soft-sensor API.

Uses FastAPI's TestClient to drive the app in-process — no running server
required. The lifespan handler (which loads models) runs when the TestClient
is constructed inside the fixture.

Run from project root:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.schemas import DEBUTANIZER_HISTORY_LENGTH, FERMENTATION_HISTORY_LENGTH


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    """TestClient with lifespan enabled, so startup loads the models once."""
    with TestClient(app) as c:
        yield c


def _valid_debutanizer_request():
    return {
        "history": {
            s: [0.5] * DEBUTANIZER_HISTORY_LENGTH
            for s in ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
        }
    }


def _valid_fermentation_request():
    return {
        "time_h": 100.0,
        "history": {
            s: [50.0] * FERMENTATION_HISTORY_LENGTH
            for s in [
                "feed_rate_Lph",
                "temperature_K",
                "pH",
                "DO_pct",
                "agitator_rpm",
                "volume_L",
            ]
        },
    }


# --------------------------------------------------------------------------
# Meta endpoints
# --------------------------------------------------------------------------
class TestMetaEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        body = r.json()
        assert "endpoints" in body
        assert "/predict/debutanizer" in body["endpoints"]
        assert "/predict/fermentation" in body["endpoints"]

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["models_loaded"]["debutanizer"] is True
        assert body["models_loaded"]["fermentation"] is True


# --------------------------------------------------------------------------
# Debutanizer endpoint
# --------------------------------------------------------------------------
class TestDebutanizer:
    def test_valid_request(self, client):
        r = client.post("/predict/debutanizer", json=_valid_debutanizer_request())
        assert r.status_code == 200
        body = r.json()
        # Response shape
        for key in [
            "prediction",
            "lower_95",
            "upper_95",
            "interval_width",
            "case_study",
            "units",
            "model_version",
        ]:
            assert key in body
        # Sanity: lower < prediction < upper
        assert body["lower_95"] < body["prediction"] < body["upper_95"]
        # Sanity: split conformal has constant width = 0.4718...
        assert abs(body["interval_width"] - 0.4718) < 0.001
        assert body["case_study"] == "debutanizer"
        assert body["units"] == "normalized [0,1]"

    def test_missing_sensors(self, client):
        bad = {"history": {"u1": [0.5] * DEBUTANIZER_HISTORY_LENGTH}}
        r = client.post("/predict/debutanizer", json=bad)
        assert r.status_code == 422
        # The error message should name the missing sensors
        msg = str(r.json())
        for sensor in ["u2", "u3", "u4", "u5", "u6", "u7"]:
            assert sensor in msg

    def test_wrong_history_length(self, client):
        bad = {
            "history": {
                s: [0.5] * 10 for s in ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
            }
        }
        r = client.post("/predict/debutanizer", json=bad)
        assert r.status_code == 422
        assert "expected" in str(r.json()).lower()

    def test_unexpected_sensor(self, client):
        good = _valid_debutanizer_request()
        good["history"]["u99"] = [0.5] * DEBUTANIZER_HISTORY_LENGTH
        r = client.post("/predict/debutanizer", json=good)
        assert r.status_code == 422
        assert "unexpected" in str(r.json()).lower()

    def test_prediction_deterministic(self, client):
        """Same input → same output. Critical for reproducibility."""
        req = _valid_debutanizer_request()
        r1 = client.post("/predict/debutanizer", json=req).json()
        r2 = client.post("/predict/debutanizer", json=req).json()
        assert r1["prediction"] == r2["prediction"]
        assert r1["lower_95"] == r2["lower_95"]
        assert r1["upper_95"] == r2["upper_95"]


# --------------------------------------------------------------------------
# Fermentation endpoint
# --------------------------------------------------------------------------
class TestFermentation:
    def test_valid_request(self, client):
        r = client.post("/predict/fermentation", json=_valid_fermentation_request())
        assert r.status_code == 200
        body = r.json()
        for key in [
            "prediction",
            "lower_95",
            "upper_95",
            "interval_width",
            "case_study",
            "units",
            "model_version",
        ]:
            assert key in body
        assert body["lower_95"] < body["prediction"] < body["upper_95"]
        # Split conformal width on fermentation ≈ 1.638 g/L
        assert abs(body["interval_width"] - 1.638) < 0.01
        assert body["case_study"] == "fermentation"
        assert body["units"] == "g/L"

    def test_missing_time_h(self, client):
        bad = _valid_fermentation_request()
        del bad["time_h"]
        r = client.post("/predict/fermentation", json=bad)
        assert r.status_code == 422

    def test_negative_time_h(self, client):
        bad = _valid_fermentation_request()
        bad["time_h"] = -5.0
        r = client.post("/predict/fermentation", json=bad)
        assert r.status_code == 422

    def test_missing_sensors(self, client):
        bad = {
            "time_h": 100.0,
            "history": {"feed_rate_Lph": [50.0] * FERMENTATION_HISTORY_LENGTH},
        }
        r = client.post("/predict/fermentation", json=bad)
        assert r.status_code == 422

    def test_realistic_time_dependence(self, client):
        """At time_h=10, penicillin should be near zero; at time_h=150, much higher.
        This is a model-behavior sanity check, not just an API check."""
        early = _valid_fermentation_request()
        early["time_h"] = 10.0
        late = _valid_fermentation_request()
        late["time_h"] = 150.0

        early_pred = client.post("/predict/fermentation", json=early).json()[
            "prediction"
        ]
        late_pred = client.post("/predict/fermentation", json=late).json()["prediction"]

        # The growth-curve signal should make late predictions higher than early
        assert late_pred > early_pred, (
            f"Expected late_pred ({late_pred}) > early_pred ({early_pred}). "
            "The model has lost its time-dependence."
        )
