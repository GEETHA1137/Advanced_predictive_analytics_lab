# Lab 08 – Agricultural Predictive Analytics: Crop Yield Prediction
**Course:** MDI3003 – Advanced Predictive Analytics
**Student:** Geetha Priya S | **Roll No:** 23MID0021
**Date:** 2025-09-15

---

## Overview
This lab builds a regression pipeline to predict district-level **rice yield (t/ha)** across
10 Indian states using only [state, district, season, year] features. Four models are compared
under a strict chronological train/validation/test split. The best model (selected by validation
MAE) is saved as a sklearn Pipeline for reproducible inference.

---

## Files

| File | Description |
|---|---|
| `lab08.py` | Reference implementation (Appendix A) — validate + test stages |
| `lab08_pipeline.py` | Master pipeline: data generation, figures, notebook, report |
| `build_lab08.py` | Builds Word (.docx) + PDF report |
| `data/rice_canonical.csv` | Synthetic Indian rice yield dataset (canonical schema) |
| `config.json` | Pipeline configuration |
| `23MID0021_Lab08_CropYield.ipynb` | Jupyter notebook with full workflow |
| `23MID0021_Lab08_Report.pdf` | Full PDF lab report |
| `23MID0021_Lab08_Report.docx` | Word document report |
| `23MID0021_Lab08_Validation_Results.csv` | Predictions for validation years |
| `23MID0021_Lab08_Test_Results.csv` | Predictions for test years |
| `23MID0021_Lab08_Error_Analysis.csv` | 5 worst-case test predictions |
| `23MID0021_Lab08_README.md` | This file |
| `models/` | Saved joblib models (all candidates + selected bundle) |
| `lab08_figs/` | 12 figures embedded in the report |
| `outputs/core/` | Full pipeline output directory |

---

## Dataset Card

| Field | Value |
|---|---|
| Source | Synthetic (ICRISAT DOI 10.17632/ywp3y5j9vv.1 schema) |
| Crop | Rice (Oryza sativa) |
| States / Districts | 10 / 40 |
| Seasons | Kharif, Rabi |
| Years | 2010–2019 (10 unique) |
| Train years | 2010–2015 |
| Validation years | 2016–2017 |
| Test years | 2018–2019 |

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib joblib nbformat
# Full pipeline (data gen + validate + test + figures + notebook + report markdown)
python lab08_pipeline.py
# Build Word + PDF report
python build_lab08.py
```

Or run individual stages:
```bash
python lab08.py --config config.json --stage validate
python lab08.py --config config.json --stage test
```

---

## Load the Saved Model

```python
import joblib, pandas as pd

bundle = joblib.load('models/23MID0021_Lab08_selected_bundle.joblib')
pipe   = bundle['pipeline']

# Score a new observation
query = pd.DataFrame([{
    'state': 'Punjab', 'district': 'Ludhiana', 'season': 'Kharif', 'year': 2024
}])
pred = pipe.predict(query)
print(f"Predicted yield: {pred[0]:.3f} t/ha")
```

---

## Evaluation Summary

| Model | Val MAE | Val R² |
|---|---|---|
| Median Baseline | 0.6963 | -0.1140 |
| Ridge Trend | 0.1888 | 0.9140 |
| Decision Tree | 0.2763 | 0.8156 |
| **ridge_trend** | **0.1888** | **0.9140** |

**Test set (one-shot):** MAE = 0.1950 t/ha | RMSE = 0.2471 | R² = 0.9057

---

## Dependencies

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
matplotlib>=3.4
joblib>=1.1
nbformat>=5.0
python-docx>=0.8
pywin32 (for PDF export)
```
