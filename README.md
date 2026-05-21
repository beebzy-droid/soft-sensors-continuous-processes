# Soft Sensors for Continuous Processes

A dual case study in **calibrated industrial soft sensors**, covering a petrochemical distillation column and a pharmaceutical/F&B fermentation process.

## Status

🚧 **In progress** — see [project plan](./PROJECT_PLAN.md) for the 10-week roadmap.

| Week | Milestone | Status |
|------|-----------|--------|
| 1 | EDA + PLS baseline (Debutanizer) | ✅ |
| 2 | XGBoost + SHAP + lag discovery | ✅ |
| 3 | Conformal prediction (3 methods compared) | ✅ |
| 4 | Fermentation simulator + EDA | ✅ |
| 5 | Fermentation modeling + cross-case findings | ✅ |
| 6 | FastAPI inference service + tests | ✅ |
| 7 | Streamlit dashboard | ⏳ |
| 8 | Deployment to Hugging Face Spaces | ⏳ |
| 9 | Technical write-up | ⏳ |
| 10 | Polish & outreach | ⏳ |

## Headline results

| Case Study | Process | PLS R² | XGBoost R² | Conformal Coverage |
|------------|---------|-------:|-----------:|-------------------:|
| Debutanizer (petrochem) | Continuous distillation | 0.15 | **0.52** | 93.2% (target 95%) |
| Fermentation (F&B) | Fed-batch bioreactor | 0.95 | **0.97** | 88.6% (target 95%) |

![Cross-case comparison](docs/figures/cross_case_summary.png)

## Key findings

1. **Same pipeline, two industries, different headline numbers — same uncertainty failure mode.** Split conformal achieved approximate marginal coverage on both case studies but failed conditional coverage in regimes systematically harder than the calibration distribution. For industrial deployment, conformal intervals must be complemented with runtime drift detection.
2. **Data-driven discovery of process physics.** SHAP attribution rediscovered known engineering principles in both case studies: mid-column tray temperature with transport delay drives debutanizer bottoms composition; cumulative metabolized substrate (lagged reactor volume) drives penicillin production.
3. **Adaptive conformal methods (CQR, residual-normalized) underperformed constant-width split conformal on the Debutanizer.** CQR's interval width was anti-correlated (r = −0.19) with model error: the bands were tightest in exactly the regions where the model was most wrong, because the quantile sub-models could not extrapolate.

## Repository structure

```
src/
├── api/           FastAPI inference service
├── simulator/     Custom fermentation simulator (Monod + Luedeking-Piret)
└── models/        Trained XGBoost + conformal artifacts (.joblib)

notebooks/
├── 01_debutanizer_eda.ipynb
├── 02_debutanizer_modeling.ipynb
├── 03_debutanizer_conformal.ipynb
├── 04_indpensim_eda.ipynb
└── 05_indpensim_modeling.ipynb

tests/             pytest integration tests for the API
docs/              Result JSONs, comparison tables, figures
```

## Local setup

```bash
conda create -n softsensors python=3.11 -y
conda activate softsensors
pip install -r requirements.txt   # to be added in week 10
```

Run the API:

```bash
uvicorn src.api.main:app --reload --port 8000
# Then open http://localhost:8000/docs for the interactive Swagger UI
```

Run the tests:

```bash
pytest tests/ -v
```

## Datasets

- **Debutanizer column** — Fortuna et al. (2007), included via [softsensors/soft-sensor-data](https://github.com/softsensors/soft-sensor-data). 2,394 samples, 7 inputs.
- **Fermentation** — Custom Monod + Luedeking-Piret simulator, inspired by IndPenSim (Goldrick et al. 2015, 2019). 40 batches × 401 samples, fully reproducible from `seed=42` via `src/simulator/fermentation.py`. The canonical IndPenSim dataset is available at [Mendeley Data](https://data.mendeley.com/datasets/pdnjz7zz5x/2); our methodology applies directly to it.

## Future work

- Containerization (Dockerfile) for one-command deployment
- Stress-test campaign with higher-disturbance fermentation simulator
- Runtime drift detection to complement static conformal intervals
- LSTM/Transformer sequence models as a comparison to XGBoost

## License

MIT