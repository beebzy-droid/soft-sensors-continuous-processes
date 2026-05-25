---
title: Soft Sensors for Continuous Processes
emoji: 🏭
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: 1.57.0
app_file: app.py
pinned: false
license: mit
short_description: Calibrated industrial soft sensors with conformal bounds
---

# Soft Sensors for Continuous Processes

Calibrated industrial soft sensors with 95% conformal prediction intervals, across two distinct continuous processes.

**Live demo:** this is the deployment. Use the tabs above to switch between case studies; the sliders to replay historical samples; the what-if panels to see how the predictions respond to operator input.

## What's running here

- **Debutanizer column** — predicts butane (C4) concentration in the bottoms of a distillation column, from 7 process sensors. Replaces a 45-minute gas chromatograph measurement.
- **Fermentation** — predicts penicillin titre during a fed-batch fermentation, from 6 online sensors plus time-since-batch-start. Replaces an HPLC assay that takes hours.

Both predictions come with a calibrated 95% conformal interval, plus a trust signal (green = in-distribution, yellow = coverage failure at this point).

## How it works

- **Models:** XGBoost regressors trained with lagged sensor features (lags up to 30 samples for Debutanizer, 32 samples for fermentation)
- **Uncertainty:** Split conformal prediction with constant-width intervals calibrated on held-out data
- **Interpretability** (not shown in the dashboard, but documented in the source repo): SHAP attribution identified physically-defensible features in both cases — mid-column tray temperature for the Debutanizer, cumulative metabolized substrate for the fermentation

## Source code & write-up

[GitHub repository →](https://github.com/beebzy-droid/soft-sensors-continuous-processes)

The repo contains the full project: notebooks for each case study, FastAPI inference service with pytest tests, the fermentation simulator (Monod + Luedeking-Piret kinetics), and result figures.

## Status

⚠️ **Not the active deployment.** The live demo at https://beebzy-droid-soft-sensors.hf.space/ is currently served from the project root (`app/streamlit_app.py` + root `README.md`'s YAML frontmatter), not from this folder.

This folder was created during a planned restructure to match the cleaner per-folder deployment pattern used in other projects (e.g., [predictive-maintenance-cmapss](https://github.com/beebzy-droid/predictive-maintenance-cmapss)), but the migration hit a Hugging Face Spaces constraint: model artifacts above ~100 KB must be stored via Xet/LFS, but Xet pointers were not being resolved at runtime in the Streamlit SDK container.

The folder is preserved as portfolio signal showing the intended architecture, and as a base for future migration once the runtime issue is resolved (likely via `hf_hub_download` from a separate model repo).

## What's inside

- `app.py` — Standalone Streamlit dashboard (no FastAPI dependency, calls inference functions directly)
- `inference.py` — Slim inference layer (loads `.joblib` artifacts, applies feature engineering and conformal prediction)
- `models/` — Trained model artifacts (XGBoost + scaler + conformal wrapper, per case study)
- `example_data/Debutanizer_Data.txt` — Bundled fixture for the Debutanizer replay slider
- `simulator/` — Custom Monod + Luedeking-Piret fermentation simulator (generates fermentation campaign on first run)
- `requirements.txt` — Pinned, slimmer dependency tree (no FastAPI/Pydantic, just what the dashboard needs)
- `README.md` — This file, with HF Spaces frontmatter at the top

## Running locally

```bash
cd hf_space
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. The fermentation tab will take ~10 seconds on first load while the simulator generates 40 batches.

---

For the full project (notebooks, FastAPI service, pytest tests, source data), see the parent directory or the [GitHub repository](https://github.com/beebzy-droid/soft-sensors-continuous-processes).