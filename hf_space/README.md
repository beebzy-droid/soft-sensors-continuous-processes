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
short_description: Calibrated industrial soft sensors with 95% conformal intervals
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