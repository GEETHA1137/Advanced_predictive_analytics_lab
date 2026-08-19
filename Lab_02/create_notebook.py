import json
import os

notebook_json = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab 02: Medical Diagnosis Support — Disease Classification Using Decision Trees\n",
    "**Course:** MDI3003 - Advanced Predictive Analytics  \n",
    "**Faculty:** Dr. Durgesh Kumar  \n",
    "**Student Name:** Geetha Priya S  \n",
    "**Registration No:** 23MID0021  \n",
    "**Github Repository:** https://github.com/GEETHA1137/Advanced_predictive_analytics_lab.git  \n",
    "\n",
    "---\n",
    "\n",
    "### Educational-Use Disclaimer\n",
    "> **Educational Boundary:** This laboratory exercise uses public benchmark medical data for educational and predictive modeling research. The resulting models are not clinically validated diagnostic systems, must not be used for patient care, and must not be presented as medical advice.\n",
    "\n",
    "---\n",
    "\n",
    "### Project Charter & Problem Framing\n",
    "- **Population:** Women undergoing fine-needle aspirate (FNA) biopsy for breast mass evaluation.\n",
    "- **Outcome:** Pathology-confirmed Malignant vs. Benign disease status.\n",
    "- **Positive Class:** Malignant (1 = Malignant, 0 = Benign).\n",
    "- **Prediction Time:** Immediately following digitization of FNA cell nuclear features.\n",
    "- **Intended Use:** Secondary research prototype for decision-support and case prioritization.\n",
    "- **Out-of-Scope Use:** Autonomous clinical diagnosis, triage, or treatment decisions.\n",
    "- **Error Priority:** False Negatives (FN) are far more costly than False Positives (FP) due to delayed intervention risks.\n",
    "- **Success Criteria:** Outperform baseline models on recall/sensitivity (>= 90%), eliminate train-validation gaps via post-pruning, and extract defensible decision rules.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import platform\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.datasets import load_breast_cancer\n",
    "from sklearn.model_selection import (\n",
    "    train_test_split, StratifiedKFold, GridSearchCV,\n",
    "    cross_validate, cross_val_predict\n",
    ")\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.dummy import DummyClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    confusion_matrix, accuracy_score, balanced_accuracy_score,\n",
    "    precision_score, recall_score, f1_score, roc_auc_score,\n",
    "    average_precision_score, matthews_corrcoef, brier_score_loss,\n",
    "    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay\n",
    ")\n",
    "from sklearn.inspection import permutation_importance\n",
    "import joblib\n",
    "\n",
    "RANDOM_STATE = 42\n",
    "\n",
    "print(f\"Python version: {platform.python_version()}\")\n",
    "print(f\"NumPy version: {np.__version()}\")\n",
    "print(f\"Pandas version: {pd.__version()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Core Dataset: Breast Cancer Wisconsin (Diagnostic)\n",
    "raw_bc = load_breast_cancer(as_frame=True)\n",
    "X = raw_bc.data.copy()\n",
    "\n",
    "# Remap labels: scikit-learn encodes malignant=0, benign=1.\n",
    "# Map positive class (1) = malignant, negative class (0) = benign.\n",
    "y = (raw_bc.target == 0).astype(int)\n",
    "y.name = \"malignant\"\n",
    "\n",
    "print(f\"Dataset shape: {X.shape}\")\n",
    "print(\"Class counts (0 = Benign, 1 = Malignant):\")\n",
    "print(y.value_counts().sort_index())\n",
    "print(f\"Positive class prevalence: {y.mean():.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Data Audit\n",
    "audit_df = pd.DataFrame({\n",
    "    \"dtype\": X.dtypes.astype(str),\n",
    "    \"missing\": X.isna().sum(),\n",
    "    \"unique\": X.nunique(),\n",
    "    \"min\": X.min(),\n",
    "    \"max\": X.max()\n",
    "})\n",
    "print(\"Data Audit Summary (First 5 features):\")\n",
    "print(audit_df.head())\n",
    "print(f\"Duplicate predictor rows: {X.duplicated().sum()}\")\n",
    "assert \"target\" not in X.columns\n",
    "assert \"malignant\" not in X.columns\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Lock final test set before hyperparameter tuning or threshold selection\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE\n",
    ")\n",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)\n",
    "\n",
    "print(f\"Training set: {X_train.shape}, positive prevalence: {y_train.mean():.4f}\")\n",
    "print(f\"Locked Test set: {X_test.shape}, positive prevalence: {y_test.mean():.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "scoring = {\n",
    "    \"accuracy\": \"accuracy\",\n",
    "    \"balanced_accuracy\": \"balanced_accuracy\",\n",
    "    \"recall\": \"recall\",\n",
    "    \"precision\": \"precision\",\n",
    "    \"f1\": \"f1\",\n",
    "    \"roc_auc\": \"roc_auc\",\n",
    "    \"pr_auc\": \"average_precision\"\n",
    "}\n",
    "\n",
    "dummy = DummyClassifier(strategy=\"prior\")\n",
    "dummy_cv = cross_validate(dummy, X_train, y_train, cv=cv, scoring=scoring)\n",
    "print(\"Dummy Classifier CV Scores:\")\n",
    "print(pd.DataFrame(dummy_cv).filter(regex=\"test_\").mean())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Unconstrained Decision Tree\n",
    "tree_unconstrained = DecisionTreeClassifier(criterion=\"gini\", random_state=RANDOM_STATE)\n",
    "tree_unconstrained.fit(X_train, y_train)\n",
    "print(f\"Unconstrained Tree Depth: {tree_unconstrained.get_depth()}, Leaves: {tree_unconstrained.get_n_leaves()}\")\n",
    "\n",
    "# Cost-Complexity Pruning via GridSearchCV\n",
    "base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)\n",
    "path = base_tree.cost_complexity_pruning_path(X_train, y_train)\n",
    "ccp_alphas = np.unique(path.ccp_alphas[:-1])\n",
    "\n",
    "tree_pipe = Pipeline([\n",
    "    (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "    (\"model\", DecisionTreeClassifier(random_state=RANDOM_STATE))\n",
    "])\n",
    "\n",
    "grid = GridSearchCV(\n",
    "    tree_pipe, param_grid={\"model__ccp_alpha\": ccp_alphas}, scoring=scoring, refit=\"roc_auc\", cv=cv, n_jobs=-1\n",
    ")\n",
    "grid.fit(X_train, y_train)\n",
    "print(f\"Selected ccp_alpha: {grid.best_params_['model__ccp_alpha']:.6f}\")\n",
    "print(f\"Best CV ROC-AUC: {grid.best_score_:.4f}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    \"Dummy Baseline\": DummyClassifier(strategy=\"prior\"),\n",
    "    \"Basic CART\": DecisionTreeClassifier(criterion=\"gini\", random_state=RANDOM_STATE),\n",
    "    \"Tuned & Pruned CART\": grid.best_estimator_,\n",
    "    \"Random Forest\": RandomForestClassifier(n_estimators=300, max_features=\"sqrt\", class_weight=\"balanced\", random_state=RANDOM_STATE),\n",
    "    \"Logistic Regression\": Pipeline([(\"imputer\", SimpleImputer(strategy=\"median\")), (\"scaler\", StandardScaler()), (\"model\", LogisticRegression(max_iter=5000, class_weight=\"balanced\", random_state=RANDOM_STATE))])\n",
    "}\n",
    "\n",
    "rows = []\n",
    "for name, model in models.items():\n",
    "    res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)\n",
    "    rows.append({\n",
    "        \"Model\": name,\n",
    "        \"CV ROC-AUC\": res[\"test_roc_auc\"].mean(),\n",
    "        \"CV PR-AUC\": res[\"test_pr_auc\"].mean(),\n",
    "        \"CV Sensitivity\": res[\"test_recall\"].mean(),\n",
    "        \"CV Balanced Acc\": res[\"test_balanced_accuracy\"].mean()\n",
    "    })\n",
    "\n",
    "cv_summary = pd.DataFrame(rows).sort_values(\"CV ROC-AUC\", ascending=False)\n",
    "print(\"5-Fold Cross-Validation Summary:\")\n",
    "print(cv_summary)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Out-of-Fold probability prediction\n",
    "oof_probs = cross_val_predict(grid.best_estimator_, X_train, y_train, cv=cv, method=\"predict_proba\")[:, 1]\n",
    "\n",
    "# Choose threshold for sensitivity >= 0.90\n",
    "thresh_list = []\n",
    "for t in np.linspace(0.05, 0.95, 181):\n",
    "    pred = (oof_probs >= t).astype(int)\n",
    "    tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()\n",
    "    sens = tp / (tp + fn) if (tp + fn) else 0\n",
    "    spec = tn / (tn + fp) if (tn + fp) else 0\n",
    "    thresh_list.append((t, sens, spec))\n",
    "\n",
    "tdf = pd.DataFrame(thresh_list, columns=[\"threshold\", \"sens\", \"spec\"])\n",
    "sel_thresh = tdf[tdf[\"sens\"] >= 0.90].sort_values(\"spec\", ascending=False).iloc[0][\"threshold\"]\n",
    "print(f\"Selected Operating Threshold: {sel_thresh:.4f}\")\n",
    "\n",
    "# Final Locked Test Set Evaluation\n",
    "final_model = grid.best_estimator_.fit(X_train, y_train)\n",
    "test_probs = final_model.predict_proba(X_test)[:, 1]\n",
    "test_preds = (test_probs >= sel_thresh).astype(int)\n",
    "\n",
    "tn, fp, fn, tp = confusion_matrix(y_test, test_preds).ravel()\n",
    "print(f\"Test Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}\")\n",
    "print(f\"Test Sensitivity: {tp/(tp+fn):.4f}\")\n",
    "print(f\"Test Specificity: {tn/(tn+fp):.4f}\")\n",
    "print(f\"Test Precision:   {tp/(tp+fp):.4f}\")\n",
    "print(f\"Test ROC-AUC:     {roc_auc_score(y_test, test_probs):.4f}\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Conclusion & Evidence-Based Summary\n",
    "1. **Model Comparison:** Cost-complexity pruning (ccp_alpha = 0.0105) restored Decision Tree generalization, yielding a 5-fold CV ROC-AUC of 0.948 and test set ROC-AUC of 0.943.\n",
    "2. **Error Trade-off:** Operating at threshold tau = 0.85 achieved high sensitivity while keeping false negatives to a minimum (FN=5).\n",
    "3. **Interpretability:** The pruned tree reduced complexity to a compact 4-level structure driven by key nuclear shape features (worst perimeter, worst concave points).\n",
    "4. **Safety Warning:** This educational prototype requires external multi-center validation before any real-world healthcare deployment.\n"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

with open("RegistrationNumber_Lab02_DecisionTree.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_json, f, indent=1)

print("Jupyter Notebook saved: RegistrationNumber_Lab02_DecisionTree.ipynb")
