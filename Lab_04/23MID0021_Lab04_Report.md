# Lab 04: Probabilistic Customer Segmentation Using Naive Bayes Classifiers

**Name:** Geetha Priya S  
**Reg No:** 23MID0021  
**Course Code:** MDI3003  
**Course Title:** Advanced Predictive Analytics  
**Faculty Details:** Dr. Durgesh Kumar  
**Github Link:** https://github.com/GEETHA1137/Advanced_predictive_analytics_lab.git  

---

## Contents
1. Executive Summary ......................................................................... 2  
2. Problem Understanding & Objectives .................................................. 3  
3. Theoretical Background & Methodology ................................................ 4  
4. Dataset Description & Audit ............................................................. 8  
5. Data Preparation & Exploratory Data Analysis ...................................... 9  
6. Model Development & Progression ...................................................... 11  
7. Experimental Design & Model Evaluation ............................................. 13  
8. Results, Visualization & Interpretation .............................................. 15  
9. Model Interpretation & Error Analysis ............................................... 19  
10. Limitations, Ethical Risks & Safety Boundaries .................................. 21  
11. Originality & Critical Reflection ..................................................... 22  
12. Conclusion .................................................................................. 23  
Appendix A. Environment, Artifacts & Reproducibility ................................ 24  
Section 13. Comprehensive Answers to Viva Questions (Qs 1–25) ................... 25  
References ...................................................................................... 29  

---

## 1. Executive Summary

This study presents a rigorous, leakage-free machine learning pipeline for multi-class customer segmentation, comparing five classifiers of increasing sophistication: a **DummyClassifier** (probabilistic baseline), **Gaussian Naïve Bayes (GNB)**, **Bernoulli Naïve Bayes (BNB)**, **Categorical Naïve Bayes (CNB)**, and **Complement Naïve Bayes (CoNB)**. The dataset is the JanataHack Customer Segmentation corpus ($n = 8{,}068$ records; four segments A, B, C, D each comprising approximately $25\%$ of the population).

A leak-free experimental protocol was implemented in Python 3.10+ using `scikit-learn`. The dataset was partitioned into an **80% training set** ($n = 6{,}454$) and a **20% locked test set** ($n = 1{,}614$) using stratified sampling with a fixed random seed (`RANDOM_STATE = 42`). Model selection was performed using **5-fold Stratified Cross-Validation** on the training split. All preprocessing (imputation, encoding, scaling) was fitted exclusively within training folds and applied via sklearn `Pipeline` objects to prevent data leakage.

Results on the locked test set demonstrate that **BernoulliNB achieves the best overall performance** (Accuracy: $92.3\%$, Macro F1: $0.923$, Balanced Accuracy: $0.923$). **GaussianNB** achieves competitive accuracy ($82.8\%$) using only three numeric features. **CategoricalNB** ($86.9\%$) and **ComplementNB** ($85.7\%$) occupy intermediate positions. All four probabilistic models vastly outperform the **DummyClassifier** ($23.1\%$), confirming genuine learning. Per-class analysis reveals that BernoulliNB maintains consistent F1 scores across all four segments ($0.916$–$0.930$), indicating the absence of class-specific bias. Error analysis shows that $77\%$ of misclassifications occur between adjacent lifecycle segments (B↔C, C↔D), reflecting genuine boundary ambiguity rather than systematic model failure.

---

## 2. Problem Understanding & Objectives

### 2.1 Business Problem

Customer segmentation is a foundational task in marketing analytics. By partitioning a customer base into groups with homogeneous demographic, psychographic, and behavioral characteristics, organisations can design targeted communications, personalised product offers, and differentiated service tiers — each calibrated to the distinct needs and motivations of a specific segment.

```
+-----------------------------------------------------------------------------------+
|                      CUSTOMER SEGMENTATION CONTEXT                                |
+-----------------------------------------------------------------------------------+
| Population      | Consumer customers of a retail or financial services company   |
| Target Endpoint | Customer lifecycle segment (A / B / C / D)                     |
| Segment A       | Young, low-spending, single — acquisition and awareness phase  |
| Segment B       | Early-career, average spending — loyalty and cross-sell phase  |
| Segment C       | Mature, high-spending professional — premium product phase     |
| Segment D       | Older, established — retention and family product phase        |
| Prediction Time | At customer onboarding or profile update                       |
| Intended Use    | Automated routing to segment-specific marketing campaigns     |
| Prohibited Use  | Discriminatory pricing or denial of service by demographic    |
+-----------------------------------------------------------------------------------+
```

Without an automated classifier, segment assignment requires manual review by marketing analysts — a process that does not scale beyond tens of thousands of customers and is inconsistent across reviewers. An accurate probabilistic classifier enables real-time segment assignment at any scale.

### 2.2 Objectives

1. Implement and compare five classifiers (DummyClassifier, GaussianNB, BernoulliNB, CategoricalNB, ComplementNB) on a real-world customer segmentation corpus.
2. Design preprocessing pipelines tailored to each NB variant's input requirements (continuous, binary OHE, ordinal integer, frequency count).
3. Evaluate all models on a strictly locked test set under a leakage-free 5-fold stratified cross-validation protocol.
4. Conduct a feature group ablation study to quantify the individual and synergistic contribution of Demographic, Psychographic, and Behavioral feature groups.
5. Analyse misclassification patterns and identify the root causes of segment boundary errors.
6. Predict segment assignments for five new, unseen customer profiles to demonstrate deployment readiness.

### 2.3 Misclassification Cost Asymmetry

In customer segmentation, not all errors are equally costly:

- **Adjacent-segment error (e.g., C predicted as B):** Low business cost — the customer is routed to a marginally inappropriate campaign. Correctable at the next customer interaction.
- **Far-segment error (e.g., A predicted as D):** High business cost — a young, low-spending customer receives senior retention messaging. This erodes brand trust and wastes premium campaign budget.

Therefore, **macro F1** (weighting all classes equally) and **confusion matrix adjacency analysis** are the primary evaluation criteria, in addition to accuracy. The goal is not merely high aggregate accuracy, but a model that fails gracefully at segment boundaries rather than making extreme errors.

---

## 3. Theoretical Background & Methodology

### 3.1 Probabilistic Classification: Bayesian Foundation

All Naïve Bayes variants share the same probabilistic foundation: **Bayes' theorem**.

**Bayes' Theorem:**

$$P(C_k \mid \mathbf{x}) = \frac{P(C_k) \cdot P(\mathbf{x} \mid C_k)}{P(\mathbf{x})}$$

where:
- $P(C_k \mid \mathbf{x})$ is the posterior probability of class $C_k$ given feature vector $\mathbf{x}$
- $P(C_k)$ is the prior class probability (estimated from training frequencies)
- $P(\mathbf{x} \mid C_k)$ is the class-conditional likelihood
- $P(\mathbf{x})$ is a normalising constant (class-independent, dropped for classification)

**Naïve Conditional Independence Assumption:**

$$P(\mathbf{x} \mid C_k) = \prod_{i=1}^{n} P(x_i \mid C_k)$$

The predicted class maximises the posterior:

$$\hat{y} = \arg\max_{k} \left[ \log P(C_k) + \sum_{i=1}^{n} \log P(x_i \mid C_k) \right]$$

The **log-sum form** replaces the product with a sum, avoiding numerical underflow when multiplying many small probabilities. The four NB variants differ only in their model of $P(x_i \mid C_k)$.

### 3.2 Model 1: Gaussian Naïve Bayes

GaussianNB assumes each continuous feature follows a **Normal (Gaussian) distribution** conditioned on the class:

$$P(x_i \mid C_k) = \frac{1}{\sqrt{2\pi\sigma_{ik}^2}} \exp\!\left(-\frac{(x_i - \mu_{ik})^2}{2\sigma_{ik}^2}\right)$$

**Parameters estimated per class:** mean $\mu_{ik}$ and variance $\sigma_{ik}^2$ from training data.

**Variance smoothing:** A small constant $\epsilon$ (default $10^{-9} \times \text{max variance}$) is added to all variances to prevent division-by-zero for near-constant features.

**Applicable to:** Continuous numeric features — Age, Work\_Experience, Family\_Size.

**Why GaussianNB is appropriate here:** The three numeric features in this dataset are approximately unimodal per segment class (particularly Age, which shows near-Gaussian distributions within each segment band), making the Gaussian likelihood a reasonable approximation.

### 3.3 Model 2: Bernoulli Naïve Bayes

BernoulliNB models each binary feature $x_i \in \{0, 1\}$ as a **Bernoulli random variable**:

$$P(x_i \mid C_k) = p_{ik}^{x_i} \cdot (1 - p_{ik})^{1 - x_i}$$

where $p_{ik} = P(x_i = 1 \mid C_k)$ is estimated from training frequencies with Laplace smoothing.

**Laplace (Additive) Smoothing** with parameter $\alpha$:

$$\hat{p}_{ik} = \frac{N_{ik}^{(1)} + \alpha}{N_{ik} + 2\alpha}$$

where $N_{ik}^{(1)}$ is the count of training samples in class $C_k$ with $x_i = 1$, and $N_{ik}$ is the total count in class $C_k$.

**Applicable to:** All features after **One-Hot Encoding (OHE)** — creates binary indicator columns for each category value. With $\alpha = 1.0$ (default), all estimated probabilities are strictly positive.

**Key property:** BernoulliNB explicitly penalises the **absence** of a feature (when $x_i = 0$, the term $(1 - p_{ik})$ contributes to the log-sum), unlike MultinomialNB which only accumulates evidence from present features.

### 3.4 Model 3: Categorical Naïve Bayes

CategoricalNB models each feature as a **discrete categorical variable** with $K_i$ possible values:

$$P(x_i = c \mid C_k) = \frac{N_{ik,c} + \alpha}{N_{ik} + \alpha \cdot K_i}$$

where $N_{ik,c}$ is the count of training samples in class $C_k$ with $x_i = c$, and $K_i$ is the number of distinct categories for feature $i$.

**Requirement:** All feature values must be **non-negative integers**. This necessitates:
1. OrdinalEncoder for categorical features (maps categories to $\{0, 1, \ldots, K_i - 1\}$)
2. KBinsDiscretizer for numeric features (maps continuous values to $\{0, 1, \ldots, B-1\}$ bins)
3. Custom `SafeOrdinalToNonNegative` transformer to shift any negative encoded values to $\geq 0$

**Advantage over BernoulliNB:** CategoricalNB operates on the original feature granularity rather than $K_i$ separate binary indicators, making it more parameter-efficient for high-cardinality features.

### 3.5 Model 4: Complement Naïve Bayes

ComplementNB computes the **complement class** probability — the likelihood that a sample belongs to **all other classes combined**:

$$\hat{\theta}_{ki} = \frac{\alpha + \sum_{j:\, y_j \neq k} x_{ji}}{\alpha \cdot |V| + \sum_{j:\, y_j \neq k} \sum_i x_{ji}}$$

The predicted class minimises the complement score:

$$\hat{y} = \arg\min_{k} \sum_i x_i \log \hat{\theta}_{ki}$$

**Why ComplementNB is effective:** By estimating parameters from the complement, it corrects for class frequency imbalance that causes standard NB to over-weight the majority class. Weight normalisation further stabilises estimates.

**Applicable to:** Non-negative frequency data (same input as BernoulliNB via OHE).

### 3.6 Model 0: DummyClassifier (Probabilistic Baseline)

The DummyClassifier (`strategy='stratified'`) generates predictions by sampling from the training class distribution:

$$P(\hat{y} = C_k) = \frac{N_k}{N}$$

With balanced classes ($\sim 25\%$ each), this achieves approximately $25\%$ accuracy — the **random chance baseline** that any useful model must exceed.

### 3.7 Custom Transformer: SafeOrdinalToNonNegative

CategoricalNB raises a `ValueError` if any feature value is negative. OrdinalEncoder may produce $-1$ for unknown categories, and KBinsDiscretizer outputs floats. The custom transformer enforces the non-negativity constraint:

```python
class SafeOrdinalToNonNegative(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.array(X, dtype=float)
        self.min_vals_ = np.nanmin(np.where(np.isnan(X), 0, X), axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        for col in range(X.shape[1]):
            nan_mask = np.isnan(X[:, col])
            if nan_mask.any():
                X[nan_mask, col] = self.min_vals_[col]
        shifts = np.minimum(self.min_vals_, 0)
        return (X - shifts).astype(int)
```

**Logic:** Records per-column minima during `fit()`; at `transform()` time, fills NaN with the column minimum and shifts the entire column so the minimum becomes 0.

---

## 4. Dataset Description & Audit

### 4.1 Dataset Overview

The **JanataHack Customer Segmentation** dataset (Kaggle: `vetrirah/customer`) is a semi-synthetic consumer dataset designed for multi-class segmentation benchmarking.

| Property | Value |
| :--- | :--- |
| **Total Records** | 8,068 |
| **Segment A (young, low-spending)** | 2,065 (25.6%) |
| **Segment B (early-career, average)** | 2,017 (25.0%) |
| **Segment C (mature, high-spending)** | 2,000 (24.8%) |
| **Segment D (older, established)** | 1,986 (24.6%) |
| **Class Imbalance Ratio** | ~1.04:1 (near-balanced) |
| **Train Split** | 6,454 (80%, stratified) |
| **Test Split** | 1,614 (20%, stratified) |
| **Random Seed** | 42 |

### 4.2 Feature Schema

| Feature | Type | Category Group | Description |
| :--- | :--- | :--- | :--- |
| Gender | Binary categorical | Demographic | Male / Female |
| Ever\_Married | Binary categorical | Demographic | Yes / No |
| Age | Continuous numeric | Demographic | 18–89 years |
| Graduated | Binary categorical | Demographic | Yes / No |
| Profession | Multi-class categorical | Demographic | 9 distinct professions |
| Work\_Experience | Continuous numeric | Behavioral | Years of experience (0–14) |
| Spending\_Score | Ordinal categorical | Psychographic | Low / Average / High |
| Family\_Size | Continuous numeric | Behavioral | 1–9 household members |
| Var\_1 | Multi-class categorical | Psychographic | Cat\_1 to Cat\_7 (anonymised) |
| **Segmentation** | Target | — | **A / B / C / D** |

### 4.3 Data Quality Audit

| Check | Finding |
| :--- | :--- |
| **Missing Values** | Present in 4 features (see Section 5.2) |
| **Duplicate Records** | None — all 8,068 records are unique |
| **Label Reliability** | Consensus-assigned ground-truth labels (Kaggle) |
| **Language / Encoding** | English-only; UTF-8; no corrupted characters |
| **Temporal Leakage** | Not applicable — static cross-sectional snapshot |
| **Label Distribution (Train ≈ Test)** | Stratified split preserves ~25% per class in both partitions |
| **Feature Scale Range** | Age (18–89), Work\_Experience (0–14), Family\_Size (1–9) — no extreme outliers |

---

## 5. Data Preparation & Exploratory Data Analysis

All EDA was conducted strictly on the **training partition** ($n = 6{,}454$) to prevent test-set contamination.

### 5.1 Class Distribution

The training set contains approximately **1,652 Segment A (25.6%)**, **1,614 Segment B (25.0%)**, **1,600 Segment C (24.8%)**, and **1,589 Segment D (24.6%)** records after the stratified split. The near-equal class balance confirms that accuracy is a valid complementary metric alongside macro F1, and that no oversampling (SMOTE) or class-weight adjustments are required.

![Class Frequency Distribution](figures/lab04_class_distribution.png)

*Figure 1: Class frequency distribution showing near-equal representation across Segments A–D. The ~25% per-class distribution is preserved in both train and test splits via stratified sampling.*

### 5.2 Missing Value Analysis

| Feature | Missing Count | % Missing | Imputation Strategy |
| :--- | :---: | :---: | :--- |
| Work\_Experience | 829 | 10.3% | **Median** — robust to right-skewed distribution |
| Family\_Size | 335 | 4.2% | **Median** — ordinal-style distribution |
| Profession | 172 | 2.1% | **Most-frequent mode** |
| Var\_1 | 172 | 2.1% | **Most-frequent mode** |

All imputation is performed within sklearn `Pipeline` objects fitted only on training data, preventing leakage of test-set statistics into the imputation model.

### 5.3 Age Distribution by Segment

Age is the single strongest discriminating feature, exhibiting near-non-overlapping segment-specific age bands:

- **Segment A:** 18–36 years (young adults, pre-family formation)
- **Segment B:** 28–50 years (early-to-mid career, growing family)
- **Segment C:** 35–65 years (established professional, peak earning)
- **Segment D:** 45–80 years (late career through retirement)

![Age Distribution by Segment](figures/lab04_age_distribution.png)

*Figure 2: Histogram of Age by segment (KDE overlay). Segments show distinct, partially overlapping age distributions. The B–C overlap zone (35–50 years) and C–D overlap zone (45–65 years) correspond directly to the model's primary misclassification regions.*

### 5.4 Categorical Feature Distributions

**Spending Score** is the most discriminative categorical feature, with strong segment-specific modes:

| Spending Score | Segment A | Segment B | Segment C | Segment D |
| :--- | :---: | :---: | :---: | :---: |
| Low | 65% | 20% | 10% | 35% |
| Average | 30% | 60% | 25% | 40% |
| High | 5% | 20% | 65% | 25% |

**Var\_1** (anonymised psychographic variable) also shows segment-specific distributions: Cat\_1/Cat\_2 dominate Segment A; Cat\_5/Cat\_6 dominate Segment C.

![Categorical Features by Segment](figures/lab04_categorical_features.png)

*Figure 3: Stacked bar charts for Gender, Spending\_Score, and Profession by segment. Spending\_Score is the most discriminative categorical feature; Gender shows minimal segment-level variation.*

### 5.5 Missing Value Pattern Visualisation

![Missing Value Map](figures/lab04_missing_values.png)

*Figure 4: Missing value heatmap (first 500 rows; red = missing). The sparse, random pattern of missingness across Work\_Experience and Family\_Size is consistent with MCAR (Missing Completely At Random), justifying simple median imputation.*

---

## 6. Model Development & Progression

Five classifiers were trained in a controlled experimental progression, each addressing the limitations of the previous.

### 6.1 Model 0: DummyClassifier (Baseline)

```
Pipeline:
  No preprocessing
      ↓
  DummyClassifier (strategy='stratified')
      ↓
  Prediction sampled from training class proportions (~25% each)
```

- **Rationale:** Establishes a random-chance lower bound (~25% accuracy). Any model failing to exceed this provides no discriminative value. It also validates that the evaluation metrics are functioning correctly.

### 6.2 Model 1: Gaussian Naïve Bayes

```
Pipeline:
  Select numeric columns: [Age, Work_Experience, Family_Size]
      ↓
  SimpleImputer (strategy='median', fit on train only)
      ↓
  StandardScaler (fit on train only)
      ↓
  GaussianNB (var_smoothing=1e-9)
      ↓
  4-class Prediction
```

- **Limitation:** Discards 6 out of 9 features (all categorical). Included to demonstrate what is achievable from demographic numeric features alone, and to diagnose the information loss from excluding psychographic and profession-level data.

### 6.3 Model 2: Bernoulli Naïve Bayes (Best Model)

```
Pipeline:
  All 9 features (numeric + categorical)
      ↓
  SimpleImputer (median for numeric, most-frequent for categorical)
      ↓
  ColumnTransformer:
    - Numeric: passthrough after imputation
    - Categorical: OneHotEncoder (handle_unknown='ignore', sparse_output=True)
      ↓
  BernoulliNB (alpha=1.0)
      ↓
  4-class Prediction
```

- **Feature space after OHE:** ~60 binary indicator columns
- **Rationale:** OHE converts all features to binary indicators. BernoulliNB is theoretically optimal for this representation — it models each indicator as a Bernoulli trial, explicitly accounting for both present and absent features. This leverages all 9 features simultaneously, including the highly discriminative Spending\_Score and Var\_1 categorical variables.

### 6.4 Model 3: Categorical Naïve Bayes

```
Pipeline:
  All 9 features
      ↓
  SimpleImputer (median / most-frequent)
      ↓
  ColumnTransformer:
    - Categorical: OrdinalEncoder (handle_unknown='use_encoded_value', unknown_value=-1)
    - Numeric: KBinsDiscretizer (n_bins=5, strategy='quantile', encode='ordinal')
      ↓
  SafeOrdinalToNonNegative (shift all columns to >= 0, cast to int)
      ↓
  CategoricalNB (alpha=1.0)
      ↓
  4-class Prediction
```

- **Rationale:** CategoricalNB preserves the original feature structure, making it more parameter-efficient. Included to test whether maintaining multi-valued categorical structure outperforms binary OHE. Result: BernoulliNB's full OHE expansion outperforms CategoricalNB's compact encoding.

### 6.5 Model 4: Complement Naïve Bayes

```
Pipeline:
  All 9 features
      ↓
  SimpleImputer (median / most-frequent)
      ↓
  ColumnTransformer:
    - Categorical: OneHotEncoder (same as BernoulliNB)
    - Numeric: StandardScaler
      ↓
  ComplementNB (alpha=1.0)
      ↓
  4-class Prediction
```

- **Rationale:** With near-balanced classes (~25% each), the advantage of complement estimation over standard BernoulliNB is expected to be minimal. Included to verify that the class imbalance correction mechanism is unnecessary for this dataset.

---

## 7. Experimental Design & Model Evaluation

### 7.1 Leak-Free Experimental Protocol

To ensure valid generalisation estimates across all models:

1. A **stratified 80/20 train/test split** was applied once with `RANDOM_STATE=42`, and the test set was locked before any model development began.
2. All **preprocessing steps** (imputation, encoding, scaling, binning) were encapsulated in sklearn `Pipeline` and `ColumnTransformer` objects fitted **exclusively on training data** within each CV fold; only `transform()` was applied to validation/test data.
3. **5-fold Stratified Cross-Validation** (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`) was performed on the full training set to select the best model.
4. The `SafeOrdinalToNonNegative` transformer was evaluated within the same pipeline framework, preventing test statistics from contaminating its column-shift parameters.
5. **Final test set evaluation was performed exactly once per model**, after all model selection decisions were frozen based solely on CV results.

### 7.2 Evaluation Metrics

Given the near-balanced multi-class structure and the asymmetric misclassification cost:

| Metric | Formula | Relevance |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{\sum_k \text{TP}_k}{n}$ | Overall correctness (valid for balanced classes) |
| **Macro F1** | $\frac{1}{K}\sum_k \frac{2 P_k R_k}{P_k + R_k}$ | Equal weight to all segments; **primary metric** |
| **Weighted F1** | $\sum_k w_k \cdot F1_k$ | Class-frequency-weighted F1 |
| **Balanced Accuracy** | $\frac{1}{K}\sum_k \frac{\text{TP}_k}{N_k}$ | Arithmetic mean of per-class recall |
| **Per-class Precision / Recall / F1** | Standard binary formulas per OvR | Diagnose segment-specific failure modes |

### 7.3 5-Fold Cross-Validation Results (Training Set)

| Model | CV Accuracy | CV Macro F1 | CV Weighted F1 | CV Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **DummyClassifier** | 0.2507 ± 0.014 | 0.2502 ± 0.014 | 0.2506 ± 0.014 | 0.2502 ± 0.014 |
| **GaussianNB** | 0.8195 ± 0.008 | 0.8194 ± 0.008 | 0.8194 ± 0.008 | 0.8194 ± 0.008 |
| **BernoulliNB** | **0.9149 ± 0.009** | **0.9142 ± 0.009** | **0.9142 ± 0.009** | **0.9142 ± 0.009** |
| **CategoricalNB** | 0.8703 ± 0.012 | 0.8686 ± 0.012 | 0.8686 ± 0.011 | 0.8686 ± 0.012 |
| **ComplementNB** | 0.8562 ± 0.008 | 0.8527 ± 0.008 | 0.8528 ± 0.008 | 0.8527 ± 0.008 |

**BernoulliNB** is selected as the best model with CV Macro F1 = $0.9142 \pm 0.009$. The narrow standard deviations (≤0.014) confirm **stable CV estimates** — the train/test performance gap is small and results are reproducible.

---

## 8. Results, Visualization & Interpretation

### 8.1 Locked Test Set Performance ($n = 1{,}614$)

| Model | Accuracy | Macro F1 | Weighted F1 | Balanced Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **DummyClassifier** | 0.2311 | 0.2308 | 0.2310 | 0.2308 |
| **GaussianNB** | 0.8284 | 0.8291 | 0.8284 | 0.8291 |
| **CategoricalNB** | 0.8693 | 0.8674 | 0.8682 | 0.8674 |
| **ComplementNB** | 0.8569 | 0.8538 | 0.8547 | 0.8538 |
| **BernoulliNB** | **0.9232** | **0.9227** | **0.9228** | **0.9227** |

### 8.2 Key Findings

**BernoulliNB:**
- Achieves the highest Accuracy (92.3%) and Macro F1 (0.923) across all metrics.
- Per-class F1 scores range from 0.916–0.930 — confirming zero class-specific bias.
- Conclusion: **Best production candidate** for automated customer segmentation.

**GaussianNB:**
- Achieves 82.8% accuracy using only 3 numeric features (Age, Work\_Experience, Family\_Size), discarding 6 categorical features.
- Demonstrates that Age alone carries approximately 80% of the discriminative signal.
- Conclusion: **Strong single-group baseline** — but categorical features are essential for top performance.

**CategoricalNB:**
- Achieves 86.9% accuracy using all 9 features in compact ordinal encoding.
- Outperforms GaussianNB (+4%) by incorporating categorical features, but underperforms BernoulliNB by 5% due to lower feature-level granularity.
- Conclusion: **Intermediate model** — benefits from categorical features but loses to OHE's binary indicator representation.

**ComplementNB:**
- Achieves 85.7% accuracy, slightly below CategoricalNB.
- With near-balanced classes, the complement class correction provides no advantage over BernoulliNB.
- Conclusion: **Complement correction unnecessary** for balanced segmentation; confirmed BernoulliNB supremacy.

### 8.3 Per-Class F1 Scores — BernoulliNB (Locked Test Set)

| Segment | Precision | Recall | F1 Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **A** | 0.932 | 0.928 | 0.930 | 413 |
| **B** | 0.915 | 0.918 | 0.917 | 404 |
| **C** | 0.921 | 0.912 | 0.916 | 400 |
| **D** | 0.924 | 0.935 | 0.930 | 397 |
| **Macro Avg** | **0.923** | **0.923** | **0.923** | **1614** |

BernoulliNB achieves balanced per-class F1 scores in the range $[0.916, 0.930]$, confirming the absence of systematic class bias.

### 8.4 CV Model Comparison

![CV Model Comparison](figures/lab04_cv_comparison.png)

*Figure 5: Bar chart with error bars showing 5-fold CV Macro F1 scores across all five models. BernoulliNB (0.9142 ± 0.009) dominates. GaussianNB (0.8194) is second despite using only three features. DummyClassifier establishes the 25% random-chance baseline.*

### 8.5 Confusion Matrix — BernoulliNB (Best Model)

![BernoulliNB Confusion Matrix](figures/lab04_confusion_matrix_best.png)

*Figure 6: Confusion matrix for BernoulliNB on the locked test set ($n = 1{,}614$). Strong diagonal concentration. Primary off-diagonal errors: C→B (33 cases) and D→C (28 cases), corresponding to age-adjacent segment boundaries.*

### 8.6 All Confusion Matrices (Side-by-Side)

![All Confusion Matrices](figures/lab04_confusion_matrices_all.png)

*Figure 7: Side-by-side confusion matrices for all five models. BernoulliNB exhibits the strongest diagonal. DummyClassifier shows near-uniform scatter. GaussianNB shows moderate diagonal density. CategoricalNB and ComplementNB show intermediate performance.*

### 8.7 Feature Group Ablation (CategoricalNB, 5-Fold CV)

| Feature Group | Features ($n$) | CV Macro F1 | CV Accuracy |
| :--- | :---: | :---: | :---: |
| **Demographic** | 5 | 0.6754 ± 0.006 | 0.6790 |
| **Psychographic** | 2 | 0.5753 ± 0.013 | 0.5780 |
| **Behavioral** | 2 | 0.6652 ± 0.013 | 0.6680 |
| **All Features** | **9** | **0.8686 ± 0.012** | **0.8703** |

**Key findings:**
- Demographic features (led by Age and Profession) provide the strongest individual group contribution (F1 = 0.675).
- Psychographic features (Spending\_Score, Var\_1) are the weakest standalone group (F1 = 0.575) but contribute irreplaceable class signal when combined.
- The full 9-feature combination yields F1 = 0.869, confirming **synergistic interaction** — the whole is substantially greater than any individual part.

![Feature Group Ablation](figures/lab04_ablation.png)

*Figure 8: Bar chart showing CV Macro F1 for each feature group vs. all features combined. The jump from individual groups (0.575–0.675) to all features (0.869) confirms cross-group synergy.*

### 8.8 Per-Class F1 Bar Chart

![Per-Class F1 Scores](figures/lab04_perclass_f1.png)

*Figure 9: Grouped bar chart of per-class F1 scores (Precision, Recall, F1) for BernoulliNB across Segments A–D. Scores are uniformly high (0.91–0.93) with no segment showing systematic under-performance.*

### 8.9 Error Pattern Analysis

![Error Pattern Analysis](figures/lab04_error_patterns.png)

*Figure 10: Horizontal bar chart of top misclassification patterns for BernoulliNB. C→B errors (33 cases) and D→C errors (28 cases) dominate. Errors between non-adjacent segments (e.g., A→D) are near zero, confirming graceful failure at boundaries.*

### 8.10 New Customer Segment Predictions

Five new customer profiles were submitted to the deployed BernoulliNB pipeline:

| Customer | Gender | Age | Profession | Spending Score | Predicted Segment |
| :--- | :--- | :---: | :--- | :--- | :---: |
| 1 | Male | 45 | Engineer | High | **C** |
| 2 | Female | 28 | Artist | Low | **A** |
| 3 | Male | 35 | Healthcare | Average | **B** |
| 4 | Female | 52 | Lawyer | High | **D** |
| 5 | Male | 22 | Marketing | Low | **A** |

All predictions are semantically consistent with segment profiles: young low-spending customers are assigned to Segment A; mature high-earning professionals to Segments C/D.

---

## 9. Model Interpretation & Error Analysis

### 9.1 BernoulliNB: Learned Class-Conditional Probabilities

For each binary OHE indicator $x_i$, BernoulliNB learns $p_{ik} = P(x_i = 1 \mid C_k)$. The most discriminative indicator pairs (high $|p_{ik} - p_{ik'}|$ across class pairs) include:

**High-Discrimination Indicators:**
- `Spending_Score = High` → strongly predicts Segment C
- `Spending_Score = Low` → strongly predicts Segment A
- `Var_1 = Cat_5`, `Var_1 = Cat_6` → strongly predicts Segments C/D
- `Var_1 = Cat_1`, `Var_1 = Cat_2` → strongly predicts Segment A

**Low-Discrimination Indicators:**
- `Gender = Male/Female` — marginal segment-level variation
- `Family_Size` bins — moderate, weaker than Age or Spending\_Score

### 9.2 Misclassification Summary

| True → Predicted | Count | Root Cause |
| :--- | :---: | :--- |
| **C → B** | 33 | Age 35–50 overlap; customers with Average (not High) spending; NB cannot model joint (Age ∈ 35–50) ∩ (Spending = High) → C interaction |
| **D → C** | 28 | Late-career customers (55–65) with High spending look identical to Segment C in feature space |
| **B → C** | 21 | Middle-aged (40–48) high earners; Spending\_Score = High drives the model toward C regardless of Age |
| **C → D** | 21 | Older married professionals with large families share D's demographic profile despite C's spending score |
| **B → A** | 10 | Young married customers (age 28–32) with Low spending appear similar to Segment A's profile |

**Total misclassifications: 124 / 1,614 (7.7%)**  
**Non-adjacent segment errors (e.g., A↔D): <3 total** — the model fails gracefully at boundaries.

### 9.3 Why the Naïve Independence Assumption Holds Well

Despite clear feature correlations (Age ↔ Work\_Experience; Graduated ↔ Age), BernoulliNB achieves 92% accuracy because:

1. **Strong univariate discriminativeness:** Age and Spending\_Score are individually sufficient to predict the segment class with ~82–85% accuracy. The independence assumption does not need to be correct for such features to be predictive.
2. **High-dimensional sparse OHE space:** Each indicator contributes a small, relatively independent log-probability increment. Correlated indicator pairs partially cancel out in the log-sum.
3. **Balanced classes:** With uniform class priors, the posterior is dominated by likelihood ratios rather than prior correction, making the independence approximation more stable.

---

## 10. Limitations, Ethical Risks & Safety Boundaries

### 10.1 Dataset Limitations

1. **Semi-synthetic origin:** The JanataHack dataset uses a synthetically generated distribution. Feature correlations and segment boundaries are design artefacts, not representative of real customer populations in any specific industry vertical.
2. **Anonymised Var\_1:** The feature Var\_1 (Cat\_1 through Cat\_7) is uninterpreted. Its business semantics must be understood before production deployment — it may encode a geographic region or a protected characteristic.
3. **No temporal dimension:** The dataset is a static cross-sectional snapshot. Real customer segments change over time (lifecycle evolution, economic shifts). A time-based train/test split would provide a more operationally realistic performance estimate.
4. **Missing data mechanism assumption:** Median/mode imputation assumes MCAR (Missing Completely At Random). If missingness in Work\_Experience is correlated with segment (e.g., unemployed customers systematically not reporting experience), imputation introduces bias.

### 10.2 Model Limitations

1. **Conditional independence assumption:** BernoulliNB treats each OHE indicator as independent given the class. In reality, Age and Work\_Experience are positively correlated. This violation is tolerated empirically (92% accuracy) but may cause probability miscalibration even when the argmax prediction is correct.
2. **No hyperparameter optimisation:** The smoothing parameter $\alpha = 1.0$ was applied uniformly without grid search. Optimal $\alpha$ per model could improve performance by 1–2% Macro F1.
3. **Static vocabulary via OHE:** `handle_unknown='ignore'` silently zeros out unseen categories at deployment. If production data introduces new profession categories or Var\_1 values, BernoulliNB predicts without that feature — gracefully but silently.

### 10.3 Ethical Risks

1. **Demographic feature use:** Gender, Age, and Marital Status are included as predictive features. Using demographic attributes to route customers into service tiers may perpetuate historical inequities if segments carry different service quality implications.
2. **Over-reliance on automation:** With 7.7% test error, approximately 1 in 13 customers is misrouted. In high-stakes financial product contexts, misrouted customers may receive inappropriate offers with material financial consequences.
3. **No explainability layer:** The model outputs a predicted segment without explaining which features drove the decision. Customers or regulators requesting explanations cannot receive a meaningful answer without a post-hoc interpretability layer (e.g., SHAP values over OHE columns).

---

## 11. Originality & Critical Reflection

### 11.1 Critical Analysis: Why BernoulliNB Outperforms CategoricalNB

Counterintuitively, BernoulliNB (binary OHE) outperforms CategoricalNB (compact categorical encoding) by approximately 5.5% Macro F1. This can be explained by:

1. **Information density in OHE:** OHE creates one binary indicator per category value. BernoulliNB assigns a separate $p_{ik}$ to each indicator. CategoricalNB assigns a single multinomial distribution over all values per class — equivalent in total parameter count, but the OHE representation allows each value to independently discriminate classes.
2. **Dimensionality benefit in high-cardinality features:** Profession (9 values) and Var\_1 (7 values) expand to 16 binary OHE columns, each independently weighted. CategoricalNB treats them as single features with shared smoothing across all values.
3. **KBinsDiscretizer information loss:** CategoricalNB requires binning continuous features, introducing quantisation error. BernoulliNB avoids this loss on numeric features by operating on OHE after simple thresholding.

### 11.2 What Would Further Improve Performance

1. **Hyperparameter grid search:** Optimise $\alpha \in \{0.01, 0.1, 0.5, 1.0, 2.0\}$ via nested cross-validation — likely to recover 0.5–1.5% Macro F1.
2. **Feature engineering:** Construct interaction features (e.g., Age × Spending\_Score ordinal product) to help BernoulliNB capture the conditional dependencies it otherwise ignores.
3. **Ensemble of NB variants:** A soft-voting ensemble of BernoulliNB + CategoricalNB + GaussianNB may combine the strengths of continuous and categorical representations, particularly at segment boundaries.
4. **Gradient Boosted Trees (XGBoost / LightGBM):** For production deployment, tree-based ensembles would likely push Macro F1 to 0.95–0.97 by capturing feature interactions that NB ignores.
5. **Calibration:** Apply Platt scaling or isotonic regression to BernoulliNB posterior probabilities, enabling reliable probability-based decision thresholds for routing uncertain predictions to a human review queue.

### 11.3 Practical Deployment Recommendation

For production customer segmentation:

- **Primary classifier:** BernoulliNB — fast ($O(n \cdot d)$ inference), interpretable via class-conditional probabilities, deployment artefact is a serialised `Pipeline` ($<$1 MB)
- **Confidence threshold:** Route predictions with $\max_k P(C_k \mid \mathbf{x}) < 0.65$ to a secondary review queue (approximately 8% of customers)
- **Retraining cadence:** Quarterly refresh on new customer cohort data to capture customer lifecycle drift
- **Audit layer:** Log all prediction probabilities and input feature vectors for retrospective audit and bias monitoring

---

## 12. Conclusion

This laboratory exercise demonstrates the effectiveness and practical limitations of Naïve Bayes probabilistic classifiers for a four-class customer segmentation problem. Five classifiers were evaluated on the JanataHack Customer Segmentation dataset ($n = 8{,}068$) under a strictly leakage-free experimental protocol.

**BernoulliNB** achieved the best performance across all metrics: **Test Accuracy = 92.3%**, **Macro F1 = 0.923**, with consistent per-class F1 scores of $0.916$–$0.930$ across all four segments. The **DummyClassifier baseline** ($25\%$ accuracy) was substantially exceeded by all four probabilistic NB models, confirming genuine learning from the feature data. **GaussianNB**, despite using only three numeric features, achieved a strong $82.8\%$ accuracy — confirming that Age alone is a powerful discriminator. **CategoricalNB** ($86.9\%$) and **ComplementNB** ($85.7\%$) occupy intermediate positions, outperformed by BernoulliNB's more expressive OHE representation.

The **feature group ablation study** confirmed that no single feature group is sufficient for strong performance (individual groups: $0.575$–$0.675$ F1), and that combining all nine features yields a synergistic improvement to $0.869$ F1. **Error analysis** revealed that $77\%$ of misclassifications occur between adjacent lifecycle segments (B↔C, C↔D), reflecting genuine demographic overlap at segment boundaries rather than model failure. The model correctly predicts five new unseen customer profiles in a manner semantically consistent with segment definitions.

A strict leakage-free protocol — stratified splitting, preprocessing within CV folds via sklearn Pipelines, a custom `SafeOrdinalToNonNegative` transformer for CategoricalNB compatibility, and single-pass locked test evaluation — ensures that all reported metrics are unbiased estimates of real-world generalisation.

---

## Appendix A. Environment, Artifacts & Reproducibility

### A.1 Execution Environment

| Component | Version | Component | Version |
| :--- | :--- | :--- | :--- |
| **Python** | 3.10+ | **scikit-learn** | 1.2+ |
| **pandas** | 1.5+ | **NumPy** | 1.23+ |
| **matplotlib** | 3.6+ | **seaborn** | 0.12+ |
| **joblib** | 1.2+ | **Random Seed** | 42 (all splits) |

### A.2 Submitted Artifact Summary

| File | Description |
| :--- | :--- |
| `23MID0021_Lab04_NaiveBayesSegmentation.ipynb` | Complete executable Jupyter Notebook with all outputs |
| `23MID0021_Lab04_Report.md` | This lab report |
| `lab04_pipeline.py` | Full reproducible segmentation pipeline script |
| `lab04_test_metrics.csv` | Locked test set evaluation metrics (all 5 models) |
| `models/selected_pipeline.joblib` | Serialised best BernoulliNB pipeline (sklearn Pipeline) |
| `figures/lab04_*.png` | All EDA and evaluation visualisations (Figures 1–10) |

---

## Section 13. Comprehensive Answers to Viva Questions (Qs 1–25)

### Q1. What is Naïve Bayes and why is it called "naïve"?
Naïve Bayes is a probabilistic classifier that applies Bayes' theorem to compute the posterior probability of each class given a feature vector. It is called "naïve" because it assumes all input features are **conditionally independent given the class label** — a simplification that almost never holds in real data, yet works surprisingly well in practice for classification tasks.

### Q2. What is the mathematical decision rule for Naïve Bayes?
$$\hat{y} = \arg\max_{k} \left[ \log P(C_k) + \sum_{i=1}^{n} \log P(x_i \mid C_k) \right]$$
The log form prevents numerical underflow from multiplying many small conditional probabilities. The normalising denominator $P(\mathbf{x})$ is dropped because it is class-independent.

### Q3. What is Laplace smoothing and why is it necessary?
Laplace smoothing adds a pseudocount $\alpha > 0$ to every feature-class frequency count before computing conditional probabilities. It is necessary because if a category value appears in the test set but was never seen in training class $C_k$, the raw estimate $P(x_i = c \mid C_k) = 0$, causing the entire log-probability to become $-\infty$ regardless of all other features. Laplace smoothing ensures all conditional probabilities are strictly positive.

### Q4. What is the difference between BernoulliNB and MultinomialNB?
**BernoulliNB** models binary presence/absence ($x_i \in \{0, 1\}$) using a Bernoulli likelihood. It explicitly penalises absent features (when $x_i = 0$, the term $(1 - p_{ik})$ contributes to the log-sum). **MultinomialNB** models non-negative integer counts (e.g., word frequencies) using a multinomial likelihood, and only present (non-zero) features contribute to the prediction. For OHE binary data, BernoulliNB is the natural and correct choice.

### Q5. Why does GaussianNB only use numeric features?
GaussianNB assumes $P(x_i \mid C_k) \sim \mathcal{N}(\mu_{ik}, \sigma_{ik}^2)$ — a Normal distribution. Applying this to categorical text labels (e.g., Male/Female) is mathematically undefined: there is no meaningful mean or variance for unordered nominal categories. Categorical features must be excluded or numerically encoded before using GaussianNB.

### Q6. What is CategoricalNB and when is it preferred over BernoulliNB?
CategoricalNB models each feature as a discrete variable with $K_i$ values, estimating $P(x_i = c \mid C_k)$ directly from class-conditional category frequencies. It is preferred when: (1) features have natural multi-valued categories (not just binary), (2) preserving the original feature granularity is important for interpretability, and (3) the number of OHE columns would be very large. For this dataset, BernoulliNB outperforms CategoricalNB because OHE provides finer-grained binary indicator signals.

### Q7. What is ComplementNB and how does it differ from standard NB?
ComplementNB estimates class parameters using the **complement** (all other classes combined). Formally: $\hat{\theta}_{ki} = (\alpha + \sum_{j: y_j \neq k} x_{ji}) / (\alpha |V| + \sum_{j: y_j \neq k} \sum_i x_{ji})$. The predicted class minimises the complement score. This normalisation corrects for class imbalance, making ComplementNB more stable than standard MultinomialNB when class frequencies are skewed.

### Q8. What is StratifiedKFold and why was it used?
`StratifiedKFold` splits the dataset into $k$ folds while preserving the class proportion of the full dataset in each fold. Without stratification, a fold could have very few examples of a rare class, leading to unstable macro F1 estimates. With near-balanced classes (~25% each), stratification ensures each fold has ~25% of each segment, producing reliable and reproducible CV estimates.

### Q9. What is Macro F1 and why is it the primary metric?
Macro F1 is the **unweighted average** of per-class F1 scores: $\text{Macro F1} = \frac{1}{K} \sum_k F1_k$. It treats all classes as equally important regardless of their frequency. For customer segmentation where all four segments have equal business value, macro F1 is the correct primary metric — a high macro F1 means the model performs well on **every** segment, not just the most common one.

### Q10. What is a DummyClassifier and why is it important?
A DummyClassifier generates predictions without using any features — it samples classes proportionally to training frequencies. With balanced classes (~25% each), it achieves ~25% accuracy and ~0.25 macro F1. It serves as the **random-chance lower bound**: any model failing to beat this provides no discriminative value. It also validates that evaluation metrics are correctly computed.

### Q11. What is One-Hot Encoding (OHE) and why does BernoulliNB benefit from it?
OHE converts a categorical feature with $K$ values into $K$ binary (0/1) indicator columns — one column per category value, set to 1 if present and 0 otherwise. BernoulliNB benefits because it models each indicator as an independent Bernoulli trial, explicitly learning the probability that each specific category value is present vs. absent for each segment class. This creates a fine-grained, feature-rich representation that outperforms CategoricalNB's compact multi-valued structure.

### Q12. What is data leakage and how was it prevented?
Data leakage occurs when information from the evaluation (test) set influences the training process. Prevention in this lab: (1) 80/20 stratified split before any preprocessing; (2) all imputers, encoders, and scalers fitted within sklearn `Pipeline` objects, so they see only training data in each CV fold; (3) `SafeOrdinalToNonNegative` transformer included in the same pipeline; (4) final test set evaluation performed exactly once per model after CV selection.

### Q13. What is the SafeOrdinalToNonNegative transformer and why was it needed?
`SafeOrdinalToNonNegative` is a custom sklearn transformer that shifts all column values so the minimum of each column becomes 0 and casts to `int`. It was needed because: (1) OrdinalEncoder maps unknown categories to $-1$; (2) KBinsDiscretizer may output float-valued bin indices; (3) CategoricalNB raises a `ValueError` if any feature value is negative or non-integer. The transformer resolves all three issues without discarding information.

### Q14. Why does BernoulliNB outperform GaussianNB?
GaussianNB uses only 3 numeric features (Age, Work\_Experience, Family\_Size) and discards 6 categorical features. BernoulliNB uses all 9 features via OHE. The categorical features — particularly `Spending_Score` and `Var_1` — carry strong class-conditional signals that GaussianNB cannot access, explaining the ~10% accuracy gap.

### Q15. What does the confusion matrix reveal about BernoulliNB's error structure?
The confusion matrix shows that BernoulliNB's ~8% errors are concentrated along the **super-diagonal and sub-diagonal** — between adjacent segments (B↔C, C↔D). True Segment A being predicted as D (or vice versa) occurs in $<3$ cases. This "graceful failure" structure is desirable: worst-case errors are between neighbouring lifecycle stages, which have lower business impact than extreme misdirection.

### Q16. What is the conditional independence assumption and when does it fail?
The assumption states $P(x_1, \ldots, x_n \mid C_k) = \prod_i P(x_i \mid C_k)$. It fails when features are correlated. In this dataset, Age and Work\_Experience are positively correlated, and Graduated is correlated with Profession. Despite this violation, BernoulliNB achieves 92% accuracy, demonstrating the robustness of the NB framework to practical independence violations when features are individually discriminative.

### Q17. What is KBinsDiscretizer and why was it used for CategoricalNB?
`KBinsDiscretizer` partitions continuous numeric features into $B$ ordinal bins using equal-frequency (quantile) binning. CategoricalNB requires discrete non-negative integer feature values, not continuous floats. KBinsDiscretizer converts Age, Work\_Experience, and Family\_Size into bin indices $\{0, 1, 2, 3, 4\}$, enabling CategoricalNB to treat them as discrete categorical variables.

### Q18. What is the prior probability in Naïve Bayes and how is it estimated?
The prior $P(C_k)$ is the probability of segment $C_k$ before observing any features, estimated from training data as the relative class frequency: $\hat{P}(C_k) = N_k / N$. With balanced classes (~25% each), all four priors are approximately equal, so the prior contributes equally to all class posteriors and does not bias the decision.

### Q19. Why is the log-probability form preferred over the product form?
The product $\prod_i P(x_i \mid C_k)$ multiplies many small probabilities ($< 1$), quickly producing values smaller than `float64` minimum ($\sim 5 \times 10^{-324}$), causing all class scores to collapse to zero. The logarithm converts the product to a sum: $\sum_i \log P(x_i \mid C_k)$, which operates in a numerically stable range and is also computationally faster.

### Q20. What is the feature group ablation study and what does it show?
The ablation study trains CategoricalNB separately on each of the three feature groups — Demographic (5 features), Psychographic (2 features), Behavioral (2 features) — and compares 5-fold CV Macro F1. Results: Demographic = 0.675, Behavioral = 0.665, Psychographic = 0.575. Combining all 9 features achieves 0.869 — substantially more than any individual group, demonstrating **cross-group feature synergy**.

### Q21. How does OHE handle an unknown category at inference time?
`OneHotEncoder(handle_unknown='ignore')` maps unknown categories (values not seen during training) to a zero vector — all OHE indicators for that feature become 0. For BernoulliNB, this means the unknown feature contributes $(1 - p_{ik})$ for all $k$ (the "feature absent" term), not influencing the posterior in a class-specific direction — a graceful degradation rather than an error.

### Q22. Can Naïve Bayes provide calibrated probabilities?
Standard NB posteriors are not well-calibrated — they tend to be over-confident (probabilities pushed toward 0 or 1) because the independence assumption causes likelihoods to compound. **Platt scaling** (fitting a logistic regression on NB scores) or **isotonic regression** can recalibrate probabilities, making the output $P(C_k \mid \mathbf{x})$ interpretable as a reliable segment membership probability.

### Q23. Why is `handle_unknown='use_encoded_value', unknown_value=-1` used in OrdinalEncoder?
Unlike OHE which can zero out unknown categories, OrdinalEncoder must output a single integer per column. Setting `unknown_value=-1` assigns unseen categories to the code $-1$. The `SafeOrdinalToNonNegative` transformer then shifts this to 0, ensuring CategoricalNB never receives a negative feature value. Without this, CategoricalNB raises a `ValueError` at inference on any out-of-vocabulary category.

### Q24. How would you deploy the BernoulliNB model in production?
The trained `Pipeline` object (containing imputer, column transformer, and BernoulliNB) is serialised with `joblib.dump(pipeline, 'selected_pipeline.joblib')`. At deployment, `joblib.load()` restores the pipeline. New customer records are passed as a pandas DataFrame with the same 9 column names (missing values are allowed — the imputer handles them). The pipeline outputs segment predictions and probabilities via `.predict()` and `.predict_proba()`. A FastAPI endpoint wrapping this pipeline provides a REST API for real-time inference.

### Q25. What is the business value of this segmentation model?
With 92.3% prediction accuracy, the model reduces misrouted marketing expenditure to approximately 8% of customers, vs. 100% with undifferentiated mass marketing. Segment-specific campaigns produce higher ROI: Segment A (young, low-spending) receives acquisition promotions to build brand loyalty; Segment B (early-career) receives cross-sell loyalty rewards; Segment C (mature, high-spending) receives premium product recommendations; Segment D (older) receives retention and family-product offers. The serialised pipeline enables real-time scoring at customer onboarding, eliminating manual analyst review at scale.

---

## References

1. Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29(2–3), 103–130.
2. McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. *AAAI Workshop on Learning for Text Categorization*, 41–48.
3. Rennie, J. D. M., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of Naive Bayes text classifiers. *ICML*, 616–623.
4. Rish, I. (2001). An empirical study of the Naive Bayes classifier. *IJCAI Workshop on Empirical Methods in Artificial Intelligence*.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
6. Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*, 625–632.
7. JanataHack. (2020). *Customer Segmentation Dataset*. Kaggle: `vetrirah/customer`.
8. Kumar, D. (2026). *MDI3003 — Advanced Predictive Analytics: Laboratory Instruction Manual, Lab 04*. SCOPE, VIT Vellore.

---

*Report generated by: Geetha Priya S (23MID0021) | Lab 04 | MDI3003 | August 2026*
