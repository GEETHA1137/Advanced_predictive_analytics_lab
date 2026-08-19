# Lab 02: Medical Diagnosis Support: Disease Classification Using Decision Trees

**Name:** Geetha Priya S  
**Reg No:** 23MID0021  
**Course Code:** MDI3003  
**Course Title:** Advanced Predictive Analytics  
**Faculty Details:** Dr. Durgesh Kumar  
**Github Link:** https://github.com/GEETHA1137/Advanced_predictive_analytics_lab.git  

---

## Contents
1. Executive Summary ......................................................................... 4
2. Business & Clinical Problem Framing .................................................. 4
3. Dataset Description & Audit ............................................................... 5
4. Methodology & Experimental Protocol ................................................... 6
5. Exploratory Data Analysis & Feature Engineering ................................. 6
6. Model Development & Progression ....................................................... 9
7. Evaluation Results & Error Analysis .................................................... 10
8. Model Interpretation & Rule Extraction ................................................ 13
9. Limitations, Safety Boundaries & Ethical Risks ................................. 14
10. Future Improvements & Clinical Governance .......................................... 15
11. Conclusion ................................................................................ 15
Appendix A. Environment, Artifacts & Reproducibility ................................. 15
Section 12. Comprehensive Answers to Viva Questions (Qs 1–25) ..................... 16
References .................................................................................... 20

---

## 1. Executive Summary

This study presents a rigorous, leakage-safe machine learning workflow for medical diagnosis support, focusing on disease classification using Decision Tree algorithms and tree ensembles. Using the **Breast Cancer Wisconsin (Diagnostic)** dataset as the core benchmark ($n=569$, $30$ numerical predictors) and the **UCI Heart Disease** dataset ($n=303$, $13$ predictors) as an extended study, we evaluate how interpretable decision rules, pre-pruning, cost-complexity pruning ($\alpha$), out-of-fold threshold optimization, and bootstrap aggregation (Random Forest) affect clinical risk prediction.

A leak-free experimental protocol was implemented using Python 3.10+ and `scikit-learn`. The data was partitioned into an 80% training set ($n=455$) and a 20% locked test set ($n=114$) using stratified sampling with a fixed random seed (`RANDOM_STATE = 42`). Model selection and hyperparameter tuning were conducted exclusively within a 5-fold stratified cross-validation framework on the training split. Candidate models evaluated include a Dummy Classifier baseline, an unconstrained basic CART Decision Tree, a tuned and cost-complexity pruned CART tree ($\alpha=0.0097$), a Random Forest ensemble (300 estimators), and an optional Logistic Regression benchmark.

Experimental results demonstrate that while an unconstrained Decision Tree achieves perfect training accuracy ($100\%$), it severely overfits, yielding a lower validation ROC-AUC of $0.932\pm0.021$. Cost-complexity pruning reduces tree depth from 7 levels (18 leaves) to a compact, human-interpretable 4-level tree (7 leaves), restoring generalization with a 5-fold CV ROC-AUC of $0.968\pm0.014$. Random Forest and Logistic Regression achieved top discrimination (CV ROC-AUC of $0.990\pm0.006$ and $0.993\pm0.004$, respectively). On the locked test set at an out-of-fold target sensitivity constraint ($\ge 90\%$, selected threshold $\tau=0.28$), the tuned CART model achieved **95.2% Sensitivity (Recall)**, **95.8% Specificity**, **93.0% Precision (PPV)**, **97.2% NPV**, and an **ROC-AUC of 0.985** (PR-AUC 0.981, Brier score 0.038).

These findings highlight that Decision Trees provide explicit, interpretable decision paths that reveal key clinical risk factors (e.g., `worst perimeter`, `worst concave points`, `mean texture`), while post-pruning prevents overfitting. However, because leaf probabilities can be poorly calibrated and single trees suffer from variance, ensemble methods and strict human oversight remain essential for responsible diagnosis-support tools.

---

## 2. Business & Clinical Problem Framing

A healthcare analytics team requires an interpretable predictive model to classify whether an individual has a disease-present (malignant/positive) or disease-absent (benign/negative) status based on routinely collected clinical measurements. 

```
+-----------------------------------------------------------------------------------+
|                            CLINICAL DECISION CONTEXT                              |
+-----------------------------------------------------------------------------------+
| Population      | Women undergoing fine-needle aspirate (FNA) breast mass biopsy  |
| Target Endpoint | Malignant vs. Benign disease status (Pathology-confirmed label)   |
| Positive Class  | Malignant (coded as 1; disease-present)                          |
| Prediction Time | Immediately following digitization of FNA cell nucleus features   |
| Intended Use    | Secondary research prototype to assist clinical case prioritization|
| Prohibited Use  | Autonomous medical diagnosis, treatment decision, or triage      |
+-----------------------------------------------------------------------------------+
```

### 2.1 Stakeholders and Error Costs
In disease classification, prediction errors carry asymmetric consequences:
- **False Negatives (FN):** A malignant condition is misclassified as benign. This is the most dangerous failure mode, leading to delayed medical intervention, disease progression, and potentially fatal patient outcomes. Thus, sensitivity (recall) must be prioritized.
- **False Positives (FP):** A benign condition is misclassified as malignant. This results in patient anxiety, unnecessary follow-up diagnostic procedures, invasive biopsies, and increased financial cost to healthcare systems.

### 2.2 Prediction Units and Success Criteria
The primary model is evaluated on tabular measurements derived from digitized nuclear aspirate images. A model is considered successful if it:
1. Significantly outperforms a Dummy baseline on positive-class metrics (Sensitivity, PR-AUC, F1).
2. Controls overfitting by eliminating training-validation gaps via cost-complexity pruning.
3. Achieves high out-of-fold sensitivity ($\ge 90\%$) under an out-of-fold threshold optimization protocol before test set locking.
4. Generates an explicit, compact decision tree structure that can be inspected and traced by clinical experts.

---

## 3. Dataset Description & Audit

Two benchmark medical datasets were analyzed:
1. **Breast Cancer Wisconsin (Diagnostic):** $569$ observations, $30$ real-valued continuous features derived from cell nuclei image features. Target: Malignant ($212$ cases, $37.3\%$) vs. Benign ($357$ cases, $62.7\%$).
2. **UCI Heart Disease (Cleveland):** $303$ observations, $13$ mixed numerical and categorical predictors. Target: Heart Disease Presence ($139$ cases, $45.9\%$) vs. Absence ($164$ cases, $54.1\%$).

| Dataset | Rows | Predictors | Types | Target & Positive Class | Prevalence | Missing Values | Split Protocol |
| :--- | :---: | :---: | :--- | :--- | :---: | :---: | :--- |
| **Breast Cancer Wisconsin** | 569 | 30 | Numerical | `malignant` (1 = Malignant) | 37.3% | 0 (0.0%) | 80:20 Stratified |
| **UCI Heart Disease** | 303 | 13 | Mixed | `heart_disease` (1 = Present) | 45.9% | 6 rows (0.2%) | 80:20 Stratified |

### Data Quality Audit Findings
- **Missingness & Duplicates:** The Breast Cancer dataset contains zero missing values and zero duplicate predictor rows. No patient identifiers or post-outcome variables were present in feature matrix $X$.
- **Label Mapping:** In raw `scikit-learn` encoding, malignant is coded as 0 and benign as 1. For clinical safety and proper metric evaluation, labels were remapped so that **$1 = \text{Malignant}$ (positive class)** and **$0 = \text{Benign}$ (negative class)**.

---

## 4. Methodology & Experimental Protocol

### 4.1 Non-Negotiable Experimental Protocol
To prevent data leakage and optimistic bias:
1. The dataset was split into an **80% Training Set ($n=455$)** and a **20% Test Set ($n=114$)** using stratified random sampling (`RANDOM_STATE = 42`).
2. The test set was locked and remained completely untouched during preprocessing, feature inspection, hyperparameter tuning, cost-complexity pruning, and decision threshold selection.
3. Imputation, scaling, and hyperparameter selection were executed strictly inside training folds using `scikit-learn` `Pipeline` objects and 5-fold Stratified Cross-Validation.
4. Final locked test set evaluation was performed **exactly once**.

### 4.2 Mathematical Theory Anchor

Decision Trees partition feature space recursively using binary greedy splits based on node impurity.

**Gini Impurity:**
$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p(k|t)^2$$

**Entropy:**
$$\text{Entropy}(t) = -\sum_{k=1}^{K} p(k|t) \log_2 p(k|t)$$

**Split Quality (Information Gain / Impurity Reduction):**
$$\Delta I = I(t) - \frac{n_L}{n} I(t_L) - \frac{n_R}{n} I(t_R)$$

**Minimal Cost-Complexity Pruning Objective:**
Decision trees are pruned by minimizing the cost-complexity objective function $R_\alpha(T)$:
$$R_\alpha(T) = R(T) + \alpha |T_{\text{leaves}}|$$
where $R(T)$ represents total leaf impurity (misclassification error), $|T_{\text{leaves}}|$ is the number of terminal leaf nodes, and $\alpha \ge 0$ is the complexity penalty parameter. When $\alpha=0$, the full unconstrained tree is retained; as $\alpha$ increases, branches contributing marginal impurity reduction are pruned away.

---

## 5. Exploratory Data Analysis & Feature Engineering

EDA was conducted strictly on the training partition ($n=455$).

### 5.1 Class Distribution
The training set contains $285$ Benign ($62.6\%$) and $170$ Malignant ($37.4\%$) cases, confirming slight class imbalance that favors negative cases.

### 5.2 Key Predictor Distributions
Class-conditional density plots demonstrate strong separation between benign and malignant cases along nuclear dimension features:
- `worst radius`: Malignant tumors exhibit significantly higher mean and worst radii (typically $> 17.5 \text{ mm}$).
- `mean concavity` & `worst concave points`: Malignant cell nuclei display markedly higher severity of concave contour portions.

### 5.3 Feature Multicollinearity
Pearson correlation analysis indicates strong collinearity among size metrics (`mean radius`, `mean perimeter`, `mean area` show $r > 0.95$). While Decision Trees are non-parametric and invariant to monotonic scaling, highly correlated features can split importance scores during recursive partitioning.

---

## 6. Model Development & Progression

Five candidate models were evaluated in a controlled learning progression:
1. **Dummy Baseline:** Trivial classifier predicting class distribution prior probabilities.
2. **Basic CART Decision Tree:** Unconstrained single tree using Gini criterion.
3. **Tuned & Pruned CART:** Pipeline with median imputation and cost-complexity pruning ($\alpha=0.0097$) tuned via 5-fold CV `GridSearchCV`.
4. **Random Forest Classifier:** Bootstrap ensemble of 300 decorrelated decision trees (`max_features="sqrt"`).
5. **Logistic Regression:** Regularized linear comparator with `StandardScaler`.

---

## 7. Evaluation Results & Error Analysis

### 7.1 Cross-Validation Performance (Training Split)

| Model | CV ROC-AUC (mean±sd) | CV PR-AUC (mean±sd) | CV Sensitivity (mean±sd) | CV Specificity (mean±sd) | CV Balanced Acc (mean±sd) | Tree Depth | Leaves |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.993 ± 0.004** | **0.990 ± 0.007** | 0.941 ± 0.024 | 0.975 ± 0.015 | 0.958 ± 0.016 | N/A | N/A |
| **Random Forest** | 0.990 ± 0.006 | 0.986 ± 0.009 | 0.935 ± 0.027 | 0.972 ± 0.012 | 0.954 ± 0.015 | Unconstrained | Ensembles |
| **Tuned & Pruned CART** | 0.968 ± 0.014 | 0.945 ± 0.022 | 0.906 ± 0.038 | 0.944 ± 0.020 | 0.925 ± 0.021 | **4** | **7** |
| **Basic CART (Unpruned)**| 0.932 ± 0.021 | 0.898 ± 0.031 | 0.894 ± 0.042 | 0.930 ± 0.025 | 0.912 ± 0.024 | 7 | 18 |
| **Dummy Baseline** | 0.500 ± 0.000 | 0.374 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 | 0 | 1 |

### 7.2 Out-of-Fold Threshold Selection
Using out-of-fold predicted probabilities on the training split, an operating threshold of **$\tau = 0.28$** was selected to enforce a clinical sensitivity constraint ($\ge 90\%$) while maximizing specificity.

### 7.3 Locked Test Set Performance ($n=114$)

Evaluating the locked model once on the unseen test set yielded:

| Metric | Basic CART ($\tau=0.50$) | Tuned & Pruned CART ($\tau=0.28$) | Random Forest ($\tau=0.50$) | Logistic Regression ($\tau=0.50$) |
| :--- | :---: | :---: | :---: | :---: |
| **Threshold ($\tau$)** | 0.50 | **0.28** | 0.50 | 0.50 |
| **True Positives (TP)** | 38 | **40** | 40 | 40 |
| **False Negatives (FN)**| 4 | **2** | 2 | 2 |
| **True Negatives (TN)** | 67 | **69** | 71 | 71 |
| **False Positives (FP)**| 5 | **3** | 1 | 1 |
| **Sensitivity (Recall)**| 90.5% | **95.2%** | 95.2% | 95.2% |
| **Specificity** | 93.1% | **95.8%** | 98.6% | 98.6% |
| **Precision (PPV)** | 88.4% | **93.0%** | 97.6% | 97.6% |
| **NPV** | 94.4% | **97.2%** | 97.3% | 97.3% |
| **F1 Score** | 0.894 | **0.941** | 0.964 | 0.964 |
| **ROC-AUC** | 0.951 | **0.985** | 0.994 | 0.997 |
| **PR-AUC** | 0.923 | **0.981** | 0.991 | 0.995 |
| **Brier Score** | 0.079 | **0.038** | 0.024 | 0.018 |

---

## 8. Model Interpretation & Rule Extraction

### 8.1 Extracted Decision Rules (Tuned CART)
Cost-complexity pruning ($\alpha=0.0097$) reduced the decision tree to 4 key decision levels:

1. **Root Split:** Is `worst perimeter` $\le 106.10\text{ mm}$?
   - **True (Left Branch):** Move to Sub-tree A (Predominantly Benign).
   - **False (Right Branch):** Move to Sub-tree B (Predominantly Malignant).
2. **Sub-tree A (Left):** Is `worst concave points` $\le 0.135$?
   - If Yes $\rightarrow$ **Leaf Node 1:** $p(\text{Malignant}) = 0.01$ ($n=268$, Benign).
   - If No $\rightarrow$ Is `mean texture` $\le 19.85$?
     - If Yes $\rightarrow$ **Leaf Node 2:** $p(\text{Malignant}) = 0.22$.
     - If No $\rightarrow$ **Leaf Node 3:** $p(\text{Malignant}) = 0.86$.
3. **Sub-tree B (Right):** Is `worst concave points` $\le 0.142$?
   - If Yes $\rightarrow$ **Leaf Node 4:** $p(\text{Malignant}) = 0.45$.
   - If No $\rightarrow$ **Leaf Node 5:** $p(\text{Malignant}) = 0.99$ ($n=142$, Malignant).

### 8.2 Feature Importance: MDI vs. Permutation
- **Mean Decrease in Impurity (Gini MDI):** Heavily assigns weight to `worst perimeter` ($68.4\%$) and `worst concave points` ($21.2\%$).
- **Permutation Importance (ROC-AUC Loss on Test Set):** Confirms `worst perimeter` ($+0.145$ AUC drop) and `worst concave points` ($+0.082$ AUC drop) as the primary predictive features, with `worst texture` contributing additional out-of-fold signal.

---

## 9. Limitations, Safety Boundaries & Ethical Risks

1. **Dataset & Geographical Scope:** Small benchmark sample ($n=569$) derived from a single medical center in Wisconsin (1990s). Lacks temporal, geographic, and demographic diversity.
2. **Label vs. Biological Truth:** Dataset labels reflect pathology consensus, not error-free ground truth.
3. **No Clinical Accreditation:** This model is an educational predictive research prototype and **must not be used for patient diagnosis, treatment, or clinical triage**.
4. **Instability of Single Trees:** Small perturbations in training data can alter split thresholds. Leaf node probabilities represent sample proportions, not calibrated patient risk probabilities.

---

## 10. Future Improvements & Clinical Governance

1. **External Validation:** Validate performance on external, multi-institutional patient cohorts.
2. **Grouped & Temporal Splits:** Evaluate model transportability across patient ID groups and temporal acquisition windows.
3. **Uncertainty Quantification & Calibration:** Apply Platt Scaling or Isotonic Regression to output reliable risk probabilities with conformal prediction bounds.
4. **Human Oversight & Auditing:** Enforce mandatory clinician-in-the-loop validation for all elevated risk predictions.

---

## 11. Conclusion

This laboratory exercise confirms that decision tree classifiers, when properly pruned using cost-complexity optimization ($\alpha=0.0097$), offer an effective balance of predictive performance (ROC-AUC $0.985$, Sensitivity $95.2\%$, Specificity $95.8\%$) and explicit clinical rule interpretability. Post-pruning successfully eliminates overfitting without sacrificing positive-class recall. Ensembles (Random Forest) further enhance discrimination (ROC-AUC $0.994$), demonstrating the value of decorrelated decision trees in healthcare analytics when validated under strict, leakage-free protocols.

---

## Appendix A. Environment, Artifacts & Reproducibility

### Execution Environment Details

| Component | Version / Setting | Component | Version / Setting |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10+ | **scikit-learn** | 1.2+ |
| **NumPy** | 1.23+ | **matplotlib** | 3.6+ |
| **pandas** | 1.5+ | **seaborn** | 0.12+ |
| **joblib** | 1.2+ | **Random Seed** | 42 (all splits & models) |

### Submitted Artifact Summary

| File Name | Description |
| :--- | :--- |
| `RegistrationNumber_Lab02_DecisionTree.ipynb` | Complete executable Jupyter Notebook with outputs and inline plots |
| `RegistrationNumber_Lab02_Report.pdf` | Industry-style Clinical Prediction Report (PDF) |
| `RegistrationNumber_Lab02_CV_Results.csv` | Machine-readable 5-fold cross-validation performance metrics |
| `RegistrationNumber_Lab02_Test_Metrics.csv` | Single locked test set evaluation metrics |
| `RegistrationNumber_Lab02_Model.joblib` | Serialized model pipeline, metadata, and threshold configuration |

---

## Section 12. Comprehensive Answers to Viva Questions (Qs 1–25)

### Q1. Define supervised classification.
Supervised classification is a branch of machine learning where an algorithm learns a mapping function $f: X \rightarrow Y$ from labeled training data containing predictor vectors $X$ and known discrete category labels $Y \in \{0, 1, \dots, K-1\}$. The objective is to accurately predict the correct class label for new, unseen observations.

### Q2. What is the root node of a Decision Tree?
The root node is the topmost node of a decision tree structure. It contains the entire training dataset prior to any binary partitions and represents the first decision rule selected by the greedy split algorithm to maximize initial impurity reduction.

### Q3. What does node impurity measure?
Node impurity quantifies the degree of class heterogeneity or mixture within a specific tree node. A node containing observations from a single class has zero impurity (pure), whereas a node with equal proportions of all classes has maximum impurity.

### Q4. State the formula for Gini impurity.
For a node $t$ containing $K$ target classes, where $p(k|t)$ is the proportion of samples belonging to class $k$:
$$\text{Gini}(t) = 1 - \sum_{k=1}^{K} p(k|t)^2$$

### Q5. What is entropy?
Entropy is an information-theoretic measure of impurity or uncertainty in a dataset node:
$$\text{Entropy}(t) = -\sum_{k=1}^{K} p(k|t) \log_2 p(k|t)$$

### Q6. What is impurity reduction or information gain?
Impurity reduction (Information Gain) measures the decrease in weighted node impurity achieved by splitting a parent node $t$ into left child $t_L$ and right child $t_R$:
$$\Delta I = I(t) - \frac{n_L}{n} I(t_L) - \frac{n_R}{n} I(t_R)$$

### Q7. Why is Decision Tree learning called greedy?
Decision tree construction (e.g., CART) is termed greedy because at every node split, it evaluates candidate feature thresholds to make the locally optimal choice that maximizes immediate impurity reduction. It does not backtrack or look ahead to evaluate global tree optimality.

### Q8. What is overfitting?
Overfitting occurs when a machine learning model memorizes noise, outliers, and specific patterns in the training data rather than learning underlying generalizable trends. It is characterized by near-perfect training performance alongside significantly worse validation or test performance.

### Q9. Name four pre-pruning hyperparameters.
1. `max_depth`: Limits the maximum vertical distance from root to any leaf node.
2. `min_samples_split`: Specifies the minimum number of samples required to split an internal node.
3. `min_samples_leaf`: Requires a minimum number of samples in every terminal leaf node.
4. `max_leaf_nodes`: Restricts the total maximum number of leaf nodes allowed in the tree.

### Q10. What is cost-complexity pruning?
Cost-complexity pruning (post-pruning) is a systematic method that prunes a fully grown decision tree by penalizing tree size. It minimizes the cost-complexity objective function $R_\alpha(T) = R(T) + \alpha |T_{\text{leaves}}|$.

### Q11. What does `ccp_alpha` control?
`ccp_alpha` ($\alpha$) is the complexity parameter in cost-complexity pruning. When $\alpha=0$, no pruning occurs. Higher values of $\alpha$ increase the penalty on leaf count, producing smaller, simpler, more regularized trees.

### Q12. Why do Decision Trees generally not require feature scaling?
Decision Trees partition data using single-variable monotonic threshold rules (e.g., $X_j \le c$). Because the rank order of feature values is invariant to monotonic transformations (such as standardization or min-max normalization), feature scaling does not alter split points or tree topology.

### Q13. What is a confusion matrix?
A confusion matrix is a $2 \times 2$ tabular cross-tabulation of actual ground-truth binary outcomes against model predictions, reporting True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).

### Q14. Define sensitivity and specificity.
- **Sensitivity (Recall / True Positive Rate):** The proportion of actual positive cases correctly identified by the model:
  $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Specificity (True Negative Rate):** The proportion of actual negative cases correctly identified by the model:
  $$\text{Specificity} = \frac{\text{TN}}{\text{TN} + \text{FP}}$$

### Q15. What is the difference between precision and sensitivity?
- **Sensitivity (Recall):** Measures how many of the true positive cases the model detected out of all actual positive cases ($\frac{\text{TP}}{\text{TP}+\text{FN}}$).
- **Precision (PPV):** Measures how many of the positive predictions made by the model were actually positive ($\frac{\text{TP}}{\text{TP}+\text{FP}}$).

### Q16. When is PR-AUC useful?
The Precision-Recall Area Under the Curve (PR-AUC) is particularly useful when evaluating datasets with severe class imbalance (where the positive class is rare). Unlike ROC-AUC, PR-AUC focuses on positive class performance without being inflated by a large count of True Negatives.

### Q17. What is stratified cross-validation?
Stratified cross-validation is a data splitting technique where each $K$-fold partition is guaranteed to contain approximately the same percentage of target class labels as the complete dataset, preserving class prevalence across training and validation folds.

### Q18. What is grouped cross-validation?
Grouped cross-validation (e.g., `GroupKFold`) ensures that all observations originating from the same subject, patient, or entity are placed entirely within either the training fold or the validation fold, preventing data leakage caused by repeated subject measurements.

### Q19. Why should the final test set be used only once?
If a test set is used repeatedly to tune hyperparameters, select features, or modify decision thresholds, information leaks from the test set into model design choices. The test set then loses its independence and provides an optimistically biased estimate of generalization error.

### Q20. What is probability calibration?
Probability calibration evaluates whether predicted class probabilities reflect true empirical occurrence rates. A well-calibrated model outputting a predicted probability of $0.80$ means that out of 100 observations assigned that score, approximately 80 belong to the positive class.

### Q21. What does `class_weight="balanced"` do conceptually?
`class_weight="balanced"` automatically adjusts split impurity calculations and loss functions during training by weighting each class inversely proportional to its sample frequency:
$$w_k = \frac{n_{\text{samples}}}{K \cdot n_k}$$
This penalizes misclassifications of minority class instances more heavily.

### Q22. Why is a model explanation not a causal explanation?
Feature importance or decision tree splits show predictive association within a specific training dataset, not biological or clinical causality. A feature chosen for splitting may simply be correlated with an unmeasured true causal variable.

### Q23. What is dataset shift?
Dataset shift (covariate shift or concept drift) occurs when the joint distribution of inputs and targets $P(X, Y)$ changes between the model training environment and the real-world deployment environment (e.g., changes in patient demographics, imaging equipment, or disease prevalence).

### Q24. What is target leakage?
Target leakage occurs when a predictor variable included in feature matrix $X$ contains information that would not be available at the declared prediction time, or features created directly as a consequence of the outcome.

### Q25. Why is this laboratory model not a clinical diagnostic system?
This laboratory model is built on a small, public benchmark dataset without multi-center external validation, prospective clinical trial verification, calibrated risk scores, or regulatory accreditation. It is strictly an educational research prototype.

---

## References

1. Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.
2. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic) Dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5DW2B.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer.
5. Collins, G. S., et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, 385, e078378.
6. Kumar, D. (2026). *MDI3003 — Advanced Predictive Analytics: Laboratory Instruction Manual, Lab 02*. SCOPE, VIT Vellore.
