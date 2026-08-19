# Lab 02 — Decision Tree Classifier

**Registration Number:** 23MID0021  
**Lab:** Machine Learning — Decision Tree with Cross Validation & Model Comparison

---

## Overview

This project implements a Decision Tree Classifier on three medical datasets:
- Breast Cancer
- Heart Disease
- Early Stage Diabetes

It covers EDA, preprocessing, model training, CCP alpha pruning, cross validation, and comparison with other classifiers.

---

## Project Structure

```
Lab02_DecisionTree/
├── RegistrationNumber_Lab02_DecisionTree.ipynb   # Main notebook
├── lab02_pipeline.py                              # Full ML pipeline script
├── create_notebook.py                             # Execute notebook script
├── create_datasets.py                             # Dataset generation script
├── generate_docx_report.py                        # DOCX report generator
├── generate_pdf_report.py                         # PDF report generator
│
├── Datasets
│   ├── breast_cancer_dataset.csv
│   ├── early_stage_diabetes_dataset.csv
│   └── heart_disease_dataset.csv
│
├── Cross Validation Results
│   ├── RegistrationNumber_Lab02_CV_Results.csv
│   └── heart_disease_cv_results.csv
│
├── Model Comparison
│   ├── breast_cancer_model_comparison.csv
│   ├── diabetes_model_comparison.csv
│   └── heart_disease_model_comparison.csv
│
├── RegistrationNumber_Lab02_Test_Metrics.csv      # Test set metrics
├── RegistrationNumber_Lab02_Model.joblib          # Saved trained model
├── RegistrationNumber_Lab02_Report.md             # Lab report
│
└── figures/
    ├── fig1_class_distribution.png
    ├── fig2_eda_distributions.png
    ├── fig3_feature_correlations.png
    ├── fig4_unconstrained_tree.png
    ├── fig5_ccp_alpha_pruning.png
    ├── fig6_confusion_matrix.png
    ├── fig7_roc_pr_curves.png
    ├── fig8_pruned_tree_structure.png
    ├── fig9_feature_importance.png
    ├── fig10_3datasets_synthesis.png
    └── fig10_model_comparison.png
```

---

## How to Run

1. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn joblib
   ```

2. Run the full pipeline:
   ```bash
   python lab02_pipeline.py
   ```

3. Or open and run the notebook:
   ```
   RegistrationNumber_Lab02_DecisionTree.ipynb
   ```

---

## Key Results

- Best model saved as `RegistrationNumber_Lab02_Model.joblib`
- Cross validation results in `RegistrationNumber_Lab02_CV_Results.csv`
- Model comparison across 3 datasets in `*_model_comparison.csv` files
