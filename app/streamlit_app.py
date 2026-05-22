"""
Streamlit dashboard for the soft-sensor service.

Calls the FastAPI service running at API_BASE_URL (default: localhost:8000)
and displays predictions, calibrated 95% intervals, and trust signals
for both the debutanizer and fermentation case studies.

Run locally (FastAPI must be running on port 8000):
    uvicorn src.api.main:app --port 8000   # in one terminal
    streamlit run app/streamlit_app.py     # in another terminal
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt

# --------------------------------------------------------------------------
# Page configuration — must be the first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Soft Sensors — Calibrated Industrial Predictions",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
API_BASE_URL = os.environ.get("SOFTSENSORS_API_URL", "http://localhost:8000")
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEBUTANIZER_SENSORS = ["u1", "u2", "u3", "u4", "u5", "u6", "u7"]
DEBUTANIZER_SENSOR_LABELS = {
    "u1": "Top temperature",
    "u2": "Top pressure",
    "u3": "Reflux flow",
    "u4": "Flow to next process",
    "u5": "6th tray temperature",
    "u6": "Bottom temperature A",
    "u7": "Bottom temperature B",
}
DEBUTANIZER_HISTORY_LEN = 31

FERMENTATION_SENSORS = [
    "feed_rate_Lph",
    "temperature_K",
    "pH",
    "DO_pct",
    "agitator_rpm",
    "volume_L",
]
FERMENTATION_SENSOR_LABELS = {
    "feed_rate_Lph": "Sugar feed rate (L/h)",
    "temperature_K": "Temperature (K)",
    "pH": "pH",
    "DO_pct": "Dissolved O₂ (% sat.)",
    "agitator_rpm": "Agitator RPM",
    "volume_L": "Reactor volume (L)",
}
FERMENTATION_HISTORY_LEN = 33
TEST_BATCHES = [33, 34, 35, 36, 37, 38, 39, 40]


# --------------------------------------------------------------------------
# Data loading — cache so we only read from disk once per session
# --------------------------------------------------------------------------
@st.cache_data
def load_debutanizer_data():
    """Load the raw Debutanizer dataset for replay in the dashboard.

    Looks first in app/data (deployment fixture), then in data/raw (local dev).
    """
    bundled_path = PROJECT_ROOT / "app" / "data" / "Debutanizer_Data.txt"
    local_path = PROJECT_ROOT / "data" / "raw" / "Debutanizer_Data.txt"
    path = bundled_path if bundled_path.exists() else local_path
    df = pd.read_csv(path, sep=r"\s+")
    return df


@st.cache_data
def load_fermentation_data():
    """Load the simulated fermentation campaign (40 batches)."""
    path = PROJECT_ROOT / "data" / "processed" / "indpensim_simulated_40batches.csv"
    if not path.exists():
        # Regenerate if missing — the simulator is reproducible from seed=42
        from src.simulator.fermentation import simulate_campaign

        df = simulate_campaign(n_batches=40, duration_h=200.0, dt=0.5, seed=42)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# API call wrappers
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Prediction wrappers — work in both API mode (default, calls HTTP) and
# direct mode (used on Hugging Face Spaces, calls inference functions in-process)
# --------------------------------------------------------------------------
# Default: HTTP/API mode for local dev.
# When running on Hugging Face Spaces, force direct mode.
if os.environ.get("SPACE_ID"):
    MODE = "direct"
else:
    MODE = os.environ.get("SOFTSENSORS_MODE", "api")

if MODE == "direct":
    # Import the inference layer directly. Models are loaded once when this
    # module is first imported (we call load_all_models() below).
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.api.inference import (
        load_all_models,
        models_loaded as _models_loaded,
        predict_debutanizer as _predict_debutanizer,
        predict_fermentation as _predict_fermentation,
    )
    from src.api.schemas import DebutanizerRequest, FermentationRequest

    # Lazy load — only when first prediction is requested
    _models_initialized = False

    def _ensure_models_loaded():
        global _models_initialized
        if not _models_initialized:
            load_all_models()
            _models_initialized = True

    def call_debutanizer_api(history_dict):
        _ensure_models_loaded()
        req = DebutanizerRequest(history=history_dict)
        return _predict_debutanizer(req).model_dump()

    def call_fermentation_api(time_h, history_dict):
        _ensure_models_loaded()
        req = FermentationRequest(time_h=time_h, history=history_dict)
        return _predict_fermentation(req).model_dump()

    def check_api_health():
        _ensure_models_loaded()
        loaded = _models_loaded()
        return {
            "status": "ok" if all(loaded.values()) else "degraded",
            "api_version": "direct-mode",
            "models_loaded": loaded,
        }

else:
    # HTTP mode (default) — for local development with a running FastAPI server
    def call_debutanizer_api(history_dict):
        response = requests.post(
            f"{API_BASE_URL}/predict/debutanizer",
            json={"history": history_dict},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def call_fermentation_api(time_h, history_dict):
        response = requests.post(
            f"{API_BASE_URL}/predict/fermentation",
            json={"time_h": time_h, "history": history_dict},
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def check_api_health():
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return None


# --------------------------------------------------------------------------
# Sidebar — global controls and API status
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Soft Sensors")
    st.markdown(
        "Calibrated 95% predictions for two industrial processes.\n\n"
        "*Powered by XGBoost + Conformal Prediction.*"
    )
    st.divider()

    if MODE == "direct":
        st.caption("Mode: direct (in-process)")
    else:
        st.caption(f"Mode: HTTP via `{API_BASE_URL}`")

    health = check_api_health()
    if health and health.get("status") == "ok":
        st.success("Models loaded")
    elif health and health.get("status") == "degraded":
        st.warning("Some models failed to load")
    else:
        st.error("API: offline — start the FastAPI server")


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("Industrial Soft Sensors with Calibrated Uncertainty")
st.markdown(
    "Real-time prediction of hard-to-measure process variables, "
    "with calibrated 95% confidence intervals."
)

tab_debutanizer, tab_fermentation = st.tabs(
    [
        "Debutanizer (Petrochemical)",
        "Fermentation (Food & Beverage)",
    ]
)


# ==========================================================================
# DEBUTANIZER TAB
# ==========================================================================
with tab_debutanizer:
    st.header("Distillation column — butane (C4) in bottom flow")
    st.caption(
        "Predicts C4 concentration in the column's bottom product from 7 process sensors. "
        "Replaces a slow, expensive gas chromatograph measurement (~45 min delay)."
    )

    df_debut = load_debutanizer_data()

    # Replay slider: pick a sample index to use as "now"
    col_slider, col_metrics = st.columns([2, 1])
    with col_slider:
        st.subheader("Replay sample")
        st.markdown(
            "Choose a sample index from the historical dataset. The model predicts "
            "the C4 concentration using the previous 31 samples as history."
        )
        max_idx = len(df_debut) - 1
        sample_idx = st.slider(
            "Sample index ('now')",
            min_value=DEBUTANIZER_HISTORY_LEN,
            max_value=max_idx,
            value=2200,
            step=10,
        )

    # Construct the history window: 31 samples ending at sample_idx
    history_window = df_debut.iloc[
        sample_idx - DEBUTANIZER_HISTORY_LEN + 1 : sample_idx + 1
    ]
    history_dict = {s: history_window[s].tolist() for s in DEBUTANIZER_SENSORS}

    # ------------------------------------------------------------
    # What-if sliders — override current values of key sensors
    # ------------------------------------------------------------
    with st.expander("🛠️ What-if: override current sensor values", expanded=False):
        st.markdown(
            "Adjust the **current** value (most recent sample) of operator-relevant sensors. "
            "Historical context is preserved; only 'now' changes."
        )
        whatif_cols = st.columns(3)

        # u5 (6th tray temperature) — week 2 SHAP showed this dominates
        with whatif_cols[0]:
            u5_orig = history_dict["u5"][-1]
            u5_new = st.slider(
                f"{DEBUTANIZER_SENSOR_LABELS['u5']} (u5)",
                min_value=0.0,
                max_value=1.0,
                value=float(u5_orig),
                step=0.01,
                key="whatif_u5",
            )

        # u3 (reflux flow) — second strongest predictor
        with whatif_cols[1]:
            u3_orig = history_dict["u3"][-1]
            u3_new = st.slider(
                f"{DEBUTANIZER_SENSOR_LABELS['u3']} (u3)",
                min_value=0.0,
                max_value=1.0,
                value=float(u3_orig),
                step=0.01,
                key="whatif_u3",
            )

        # u6 (bottom temperature A)
        with whatif_cols[2]:
            u6_orig = history_dict["u6"][-1]
            u6_new = st.slider(
                f"{DEBUTANIZER_SENSOR_LABELS['u6']} (u6)",
                min_value=0.0,
                max_value=1.0,
                value=float(u6_orig),
                step=0.01,
                key="whatif_u6",
            )

        # Replace last value of each adjusted sensor; lagged values stay intact
        history_dict["u5"] = history_dict["u5"][:-1] + [u5_new]
        history_dict["u3"] = history_dict["u3"][:-1] + [u3_new]
        history_dict["u6"] = history_dict["u6"][:-1] + [u6_new]

        # Show whether any slider was nudged from the historical value
        nudged = (u5_new != u5_orig) or (u3_new != u3_orig) or (u6_new != u6_orig)
        if nudged:
            st.info("📊 Prediction below reflects your overridden values.")

    # Hit the API
    try:
        result = call_debutanizer_api(history_dict)
        actual_y = float(df_debut["y"].iloc[sample_idx])
        prediction = result["prediction"]
        lower = result["lower_95"]
        upper = result["upper_95"]
        in_interval = lower <= actual_y <= upper
    except Exception as e:
        st.error(f"Could not reach the prediction API: {e}")
        st.stop()

    # ------------------------------------------------------------
    # Headline metrics
    # ------------------------------------------------------------
    with col_metrics:
        st.subheader("Prediction")
        st.metric(
            label="Predicted C4 (normalized)",
            value=f"{prediction:.3f}",
            delta=f"actual: {actual_y:.3f}",
            delta_color="off",
        )
        st.markdown(
            f"**95% interval:** `[{lower:.3f}, {upper:.3f}]`  \n"
            f"**Width:** `{upper - lower:.3f}`"
        )

    # ------------------------------------------------------------
    # Trust signal
    # ------------------------------------------------------------
    st.divider()
    if in_interval:
        st.success(
            f"**In-distribution prediction.** Actual value ({actual_y:.3f}) "
            f"falls inside the calibrated 95% interval. Trust the prediction."
        )
    else:
        st.warning(
            f"**Coverage failure at this sample.** The actual value ({actual_y:.3f}) "
            f"is outside the 95% interval [{lower:.3f}, {upper:.3f}]. "
            f"By design, this should happen ~5% of the time on in-distribution data — "
            f"but persistent failures suggest the model is operating outside its trusted range."
        )

    # ------------------------------------------------------------
    # Time-series chart — recent context window
    # ------------------------------------------------------------
    st.subheader("Recent history & prediction")
    context_start = max(0, sample_idx - 200)
    context_window = df_debut.iloc[context_start : sample_idx + 1].reset_index(
        drop=True
    )
    context_window["idx"] = range(context_start, sample_idx + 1)

    base = alt.Chart(context_window).encode(
        x=alt.X("idx:Q", title="Sample index"),
    )
    actual_line = base.mark_line(color="black", strokeWidth=1.2).encode(
        y=alt.Y("y:Q", title="C4 concentration (normalized)"),
    )

    pred_df = pd.DataFrame(
        [
            {
                "idx": sample_idx,
                "prediction": prediction,
                "lower": lower,
                "upper": upper,
                "actual": actual_y,
            }
        ]
    )

    pred_band = (
        alt.Chart(pred_df)
        .mark_errorbar(color="crimson", thickness=3)
        .encode(
            x="idx:Q",
            y="lower:Q",
            y2="upper:Q",
        )
    )
    pred_point = (
        alt.Chart(pred_df)
        .mark_point(color="crimson", size=120, filled=True)
        .encode(
            x="idx:Q",
            y="prediction:Q",
        )
    )

    chart = (actual_line + pred_band + pred_point).properties(height=300).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Black line: historical actual C4 concentration. "
        "Crimson dot: current prediction. Crimson bar: 95% conformal interval."
    )


# ==========================================================================
# FERMENTATION TAB
# ==========================================================================
with tab_fermentation:
    st.header("Fed-batch fermentation — penicillin titre")
    st.caption(
        "Predicts penicillin concentration during industrial Penicillium fermentation, "
        "from 6 online sensors and time-since-batch-start. "
        "Replaces an offline HPLC assay (~hours delay)."
    )

    df_ferm = load_fermentation_data()

    # ------------------------------------------------------------
    # Batch and time selectors
    # ------------------------------------------------------------
    col_batch, col_time, col_metrics = st.columns([1, 2, 1])

    with col_batch:
        st.subheader("Replay batch")
        batch_id = st.selectbox(
            "Test batch ID",
            options=TEST_BATCHES,
            index=0,
            help="Held-out batches from week 5 (not seen during training).",
        )

    batch_df = df_ferm[df_ferm["batch_id"] == batch_id].reset_index(drop=True)
    min_time = batch_df["time_h"].iloc[
        FERMENTATION_HISTORY_LEN
    ]  # first usable time after lag
    max_time = batch_df["time_h"].iloc[-1]

    with col_time:
        st.subheader("Time in batch")
        time_h = st.slider(
            "Hours since batch start",
            min_value=float(min_time),
            max_value=float(max_time),
            value=float(min(150.0, max_time)),
            step=0.5,
        )

    # Find the row matching time_h, build history window
    now_idx = (batch_df["time_h"] - time_h).abs().idxmin()
    history_window = batch_df.iloc[now_idx - FERMENTATION_HISTORY_LEN + 1 : now_idx + 1]
    history_dict = {s: history_window[s].tolist() for s in FERMENTATION_SENSORS}
    actual_y = float(batch_df["penicillin_gL"].iloc[now_idx])
    actual_time = float(batch_df["time_h"].iloc[now_idx])

    # ------------------------------------------------------------
    # What-if sliders
    # ------------------------------------------------------------
    with st.expander(
        "🛠️ What-if: override current operator-controlled inputs", expanded=False
    ):
        st.markdown(
            "Adjust the **current** value of operator-controlled inputs. "
            "Historical context preserved; only 'now' changes."
        )
        whatif_cols = st.columns(3)

        with whatif_cols[0]:
            feed_orig = history_dict["feed_rate_Lph"][-1]
            feed_new = st.slider(
                FERMENTATION_SENSOR_LABELS["feed_rate_Lph"],
                min_value=0.0,
                max_value=0.12,
                value=float(feed_orig),
                step=0.005,
                key="whatif_feed",
            )

        with whatif_cols[1]:
            temp_orig = history_dict["temperature_K"][-1]
            temp_new = st.slider(
                FERMENTATION_SENSOR_LABELS["temperature_K"],
                min_value=293.0,
                max_value=303.0,
                value=float(temp_orig),
                step=0.1,
                key="whatif_temp",
            )

        with whatif_cols[2]:
            ph_orig = history_dict["pH"][-1]
            ph_new = st.slider(
                FERMENTATION_SENSOR_LABELS["pH"],
                min_value=5.5,
                max_value=7.5,
                value=float(ph_orig),
                step=0.05,
                key="whatif_ph",
            )

        history_dict["feed_rate_Lph"] = history_dict["feed_rate_Lph"][:-1] + [feed_new]
        history_dict["temperature_K"] = history_dict["temperature_K"][:-1] + [temp_new]
        history_dict["pH"] = history_dict["pH"][:-1] + [ph_new]

        nudged = (
            (feed_new != feed_orig) or (temp_new != temp_orig) or (ph_new != ph_orig)
        )
        if nudged:
            st.info("📊 Prediction below reflects your overridden values.")

    # ------------------------------------------------------------
    # API call
    # ------------------------------------------------------------
    try:
        result = call_fermentation_api(actual_time, history_dict)
        prediction = result["prediction"]
        lower = result["lower_95"]
        upper = result["upper_95"]
        in_interval = lower <= actual_y <= upper
    except Exception as e:
        st.error(f"Could not reach the prediction API: {e}")
        st.stop()

    # ------------------------------------------------------------
    # Headline metric
    # ------------------------------------------------------------
    with col_metrics:
        st.subheader("Prediction")
        st.metric(
            label="Predicted penicillin (g/L)",
            value=f"{prediction:.2f}",
            delta=f"actual: {actual_y:.2f}",
            delta_color="off",
        )
        st.markdown(
            f"**95% interval:** `[{lower:.2f}, {upper:.2f}]` g/L  \n"
            f"**Width:** `{upper - lower:.2f}` g/L"
        )

    # ------------------------------------------------------------
    # Trust signal
    # ------------------------------------------------------------
    st.divider()
    if in_interval:
        st.success(
            f"**In-distribution prediction.** Actual ({actual_y:.2f} g/L) "
            f"is inside the calibrated 95% interval."
        )
    else:
        st.warning(
            f"**Coverage failure at this point in batch {batch_id}.** "
            f"Actual ({actual_y:.2f} g/L) is outside [{lower:.2f}, {upper:.2f}] g/L. "
            f"Some batches systematically deviate from the campaign average — "
            f"a known limitation of constant-width conformal."
        )

    # ------------------------------------------------------------
    # Time-series chart — full batch trajectory + current prediction
    # ------------------------------------------------------------
    st.subheader(f"Batch {batch_id} trajectory")
    chart_df = batch_df[["time_h", "penicillin_gL"]].copy()

    base = alt.Chart(chart_df).encode(
        x=alt.X("time_h:Q", title="Time (h)"),
    )
    actual_line = base.mark_line(color="black", strokeWidth=1.2).encode(
        y=alt.Y("penicillin_gL:Q", title="Penicillin (g/L)"),
    )

    pred_df = pd.DataFrame(
        [
            {
                "time_h": actual_time,
                "prediction": prediction,
                "lower": lower,
                "upper": upper,
            }
        ]
    )
    pred_band = (
        alt.Chart(pred_df)
        .mark_errorbar(color="crimson", thickness=3)
        .encode(
            x="time_h:Q",
            y="lower:Q",
            y2="upper:Q",
        )
    )
    pred_point = (
        alt.Chart(pred_df)
        .mark_point(color="crimson", size=120, filled=True)
        .encode(
            x="time_h:Q",
            y="prediction:Q",
        )
    )

    chart = (actual_line + pred_band + pred_point).properties(height=300).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption(
        "Black line: actual penicillin titre over this batch. "
        "Crimson dot: model prediction at the selected time. Crimson bar: 95% interval."
    )
