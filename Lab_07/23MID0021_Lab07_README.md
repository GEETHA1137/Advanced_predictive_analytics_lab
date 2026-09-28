# Lab 07 – Product Recommendation System (Random Forest Candidate Scoring)
**Course:** MDI3003 – Advanced Predictive Analytics  
**Student:** Geetha Priya S | **Roll No:** 23MID0021  
**Date:** 2025-09-07

---

## Overview
This lab implements a supervised recommendation system on a synthetic UCI Online Retail-schema dataset (1,798 customers, 389 items, ~34K clean rows). A **Random Forest** classifier scores user-item candidate pairs — represented by 17 engineered features covering customer RFM, item statistics, and pair-level interaction history — to rank product recommendations. The pipeline uses a strict **chronological train / val / test split** and evaluates with Precision@K, Recall@K, HitRate@K, NDCG@K, and MAP@K (K = 5, 10, 20). The RF outperforms both a popularity baseline and item-item cosine CF.

---

## Files

| File | Description |
|---|---|
| `lab07_pipeline.py` | Full pipeline: data generation, cleaning, split, features, RF training, evaluation, figures |
| `lab07_extra_figs.py` | Supplementary figure generation (fig13–fig18) for expanded report |
| `dataset.csv` | Synthetic transaction dataset (39,266 raw rows, 34,553 clean) |
| `random_forest.joblib` | Fitted RandomForestClassifier (300 trees, seed=42, ~166 MB) |
| `feature_schema.json` | 17-feature names and order used at train/inference time |
| `split_manifest.json` | Chronological split timestamps (t1=train_end, t2=val_end) |
| `candidate_policy.json` | Candidate catalog parameters (min_buyers=3, size=361, recall=97.99%) |
| `23MID0021_Lab07_Report.pdf` | Full PDF lab report |
| `23MID0021_Lab07_Report.docx` | Word document report |
| `lab07_figs/` | 18 figures embedded in the report |
| `lab07_outputs/` | CSV artefacts: Ranking_Metrics, Recommendations, Error_Analysis, Candidate_Recall |

---

## Dataset Card

| Field | Value |
|---|---|
| Source | Synthetic (UCI Online Retail schema) |
| SHA-256 (first 32 hex) | `15784294b859350de69375e7f7b60740` |
| Raw rows | 39,266 |
| Clean rows | 34,553 |
| Customers | 1,798 |
| Items | 389 |
| Date range | 2010-12-01 → 2011-11-30 |
| Cancellations removed | 2,908 |

---

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib joblib
# Full pipeline (takes ~5-10 min due to RF training + scoring)
python -u lab07_pipeline.py

# Extra figures only (requires pipeline outputs to exist)
python -u lab07_extra_figs.py
```

---

## Load the Saved Model

```python
import joblib, numpy as np, json

rf = joblib.load('random_forest.joblib')
schema = json.load(open('feature_schema.json'))
print("Features:", schema['feature_names'])

# Score a single user-item pair (17 features in order)
# [recency_days, cust_txns, cust_items, cust_spend, cust_uniq, pref_cat_enc,
#  item_txns, item_buyers, item_avg_price, item_recency_days, item_cat_enc, item_pop_rank,
#  pair_purchases, pair_qty, pair_spend, pair_days_since, cat_match]
x = np.array([[15, 8, 42, 320.5, 12, 2,
               55, 30, 4.25, 10, 2, 0.15,
               2, 5, 8.50, 7, 1]])
prob = rf.predict_proba(x)[0][1]
print(f"P(purchase) = {prob:.4f}")
```

---

## Evaluation Results (Test Set)

| Model | P@5 | P@10 | P@20 | R@10 | HR@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| Popularity Baseline | 0.0734 | 0.0622 | 0.0486 | 0.1345 | 0.4540 | 0.1123 |
| Item-Item CF | 0.1060 | 0.0883 | 0.0680 | 0.1799 | 0.5902 | 0.1561 |
| **Random Forest** | **0.2243** | **0.1617** | **0.1152** | **0.2798** | **0.7374** | **0.2809** |

RF achieves **3.1× higher NDCG@10** and **1.6× higher HR@10** vs the popularity baseline.

---

## Chronological Split

| Split | Rows | Date Range |
|---|---|---|
| Train | 24,258 | 2010-12-01 → 2011-08-17 |
| Validation | 5,144 | 2011-08-18 → 2011-10-03 |
| Test | 5,151 | 2011-10-04 → 2011-11-30 |

---

## Dependencies

```
pandas>=1.3
numpy>=1.21
scikit-learn>=1.0
matplotlib>=3.4
joblib>=1.1
```
