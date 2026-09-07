# lab04_pipeline.py
# Lab04: Probabilistic Customer Segmentation using Naive Bayes
# Registration: 23MID0021 | Geetha Priya S | Course: MDI3003
# Faculty: Dr. Durgesh Kumar

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import platform

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_validate, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, OrdinalEncoder,
    LabelEncoder, KBinsDiscretizer
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# -------------------------------------------------------------
# Directory setup
# -------------------------------------------------------------
FIG_DIR   = "figures"
MODEL_DIR = "models"
ART_DIR   = "artifacts"
for d in [FIG_DIR, MODEL_DIR, ART_DIR]:
    os.makedirs(d, exist_ok=True)

REG_NO = "23MID0021"
LAB    = "Lab04"

print("=" * 70)
print("LAB 04: PROBABILISTIC CUSTOMER SEGMENTATION (NAIVE BAYES)")
print(f"Registration: {REG_NO}  |  Name: Geetha Priya S")
print("=" * 70)

# -------------------------------------------------------------
# Feature taxonomy (used by all pipelines)
# -------------------------------------------------------------
FEATURE_COLS  = ['Gender', 'Ever_Married', 'Age', 'Graduated',
                 'Profession', 'Work_Experience', 'Spending_Score',
                 'Family_Size', 'Var_1']
TARGET_COL    = 'Segmentation'

NUMERIC_COLS  = ['Age', 'Work_Experience', 'Family_Size']
BINARY_COLS   = ['Gender', 'Ever_Married', 'Graduated']
ORDINAL_COLS  = ['Profession', 'Spending_Score', 'Var_1']
ALL_CAT_COLS  = BINARY_COLS + ORDINAL_COLS

FEATURE_GROUPS = {
    'Demographic':   ['Gender', 'Ever_Married', 'Age', 'Graduated', 'Profession'],
    'Psychographic': ['Spending_Score', 'Var_1'],
    'Behavioral':    ['Work_Experience', 'Family_Size'],
    'All Features':  FEATURE_COLS,
}


# -------------------------------------------------------------
# Custom transformer: SafeOrdinalToNonNegative
# CategoricalNB requires all feature values >= 0 integer codes.
# OrdinalEncoder unknown categories map to -1; KBins outputs 0-based floats.
# This transformer shifts the minimum to 0 and casts to int.
# -------------------------------------------------------------
class SafeOrdinalToNonNegative(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.array(X, dtype=float)
        self.min_vals_ = np.nanmin(np.where(np.isnan(X), 0, X), axis=0)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)
        # Fill NaN with column minimum (safe default)
        for col in range(X.shape[1]):
            nan_mask = np.isnan(X[:, col])
            if nan_mask.any():
                X[nan_mask, col] = self.min_vals_[col]
        # Shift so every column minimum becomes 0
        shifts = np.minimum(self.min_vals_, 0)
        X = X - shifts
        return X.astype(int)


# -------------------------------------------------------------
# Dataset loader / synthetic generator
# -------------------------------------------------------------
def generate_synthetic_dataset(n=8068, seed=42):
    """Synthetic dataset with realistic segment-feature correlations.

    Segment profiles (based on JanataHack actual distributions):
      A - young (18-35), low spending, Var_1 Cat_1/Cat_2, often single
      B - middle-aged (28-50), average spending, Cat_3/Cat_4
      C - mature (35-65), high spending, Cat_5/Cat_6, graduated, professional
      D - older (45-80), low/avg spending, Cat_6/Cat_7, married, larger family
    """
    rng = np.random.default_rng(seed)

    profs = ['Artist','Doctor','Engineer','Entertainment','Executive',
             'Healthcare','Homemaker','Lawyer','Marketing', None]
    vars1 = [f'Cat_{i}' for i in range(1, 8)] + [None]

    SEG_PROFILES = {
        'A': dict(
            gender_p=[0.45, 0.55],
            married_p=[0.20, 0.76, 0.04],
            age_lo=18, age_hi=36,
            grad_p=[0.40, 0.56, 0.04],
            prof_p=[0.18,0.06,0.10,0.18,0.06,0.10,0.14,0.06,0.10,0.02],
            work_lo=0, work_hi=5,
            spending_p=[0.65, 0.30, 0.05],
            fam_lo=1, fam_hi=3,
            var1_p=[0.28,0.28,0.14,0.10,0.08,0.06,0.04,0.02],
        ),
        'B': dict(
            gender_p=[0.55, 0.45],
            married_p=[0.55, 0.41, 0.04],
            age_lo=28, age_hi=51,
            grad_p=[0.55, 0.41, 0.04],
            prof_p=[0.10,0.12,0.14,0.10,0.14,0.12,0.10,0.10,0.06,0.02],
            work_lo=2, work_hi=10,
            spending_p=[0.25, 0.60, 0.15],
            fam_lo=2, fam_hi=5,
            var1_p=[0.10,0.12,0.26,0.26,0.14,0.06,0.04,0.02],
        ),
        'C': dict(
            gender_p=[0.52, 0.48],
            married_p=[0.72, 0.24, 0.04],
            age_lo=35, age_hi=66,
            grad_p=[0.82, 0.14, 0.04],
            prof_p=[0.06,0.16,0.18,0.06,0.16,0.14,0.04,0.12,0.06,0.02],
            work_lo=5, work_hi=15,
            spending_p=[0.10, 0.25, 0.65],
            fam_lo=2, fam_hi=5,
            var1_p=[0.06,0.08,0.10,0.12,0.26,0.26,0.10,0.02],
        ),
        'D': dict(
            gender_p=[0.62, 0.38],
            married_p=[0.80, 0.16, 0.04],
            age_lo=45, age_hi=81,
            grad_p=[0.60, 0.36, 0.04],
            prof_p=[0.10,0.10,0.10,0.08,0.12,0.10,0.14,0.12,0.12,0.02],
            work_lo=8, work_hi=15,
            spending_p=[0.40, 0.45, 0.15],
            fam_lo=3, fam_hi=8,
            var1_p=[0.04,0.06,0.08,0.10,0.12,0.24,0.34,0.02],
        ),
    }

    seg_sizes = {'A': int(n*0.256), 'B': int(n*0.250), 'C': int(n*0.248)}
    seg_sizes['D'] = n - sum(seg_sizes.values())

    rows = []
    for seg, size in seg_sizes.items():
        p = SEG_PROFILES[seg]
        gender   = rng.choice(['Male','Female'], size, p=p['gender_p'])
        married  = rng.choice(['Yes','No',np.nan], size, p=p['married_p'])
        age      = rng.integers(p['age_lo'], p['age_hi'], size).astype(float)
        grad     = rng.choice(['Yes','No',np.nan], size, p=p['grad_p'])
        prof     = rng.choice(profs, size, p=p['prof_p'])
        work_exp = rng.integers(p['work_lo'], p['work_hi'], size).astype(float)
        # ~8% missing for work and family
        work_exp[rng.random(size) < 0.08] = np.nan
        spending = rng.choice(['Low','Average','High'], size, p=p['spending_p'])
        fam_size = rng.integers(p['fam_lo'], p['fam_hi'], size).astype(float)
        fam_size[rng.random(size) < 0.04] = np.nan
        var1     = rng.choice(vars1, size, p=p['var1_p'])
        for i in range(size):
            rows.append({
                'Gender':          gender[i],
                'Ever_Married':    married[i],
                'Age':             age[i],
                'Graduated':       grad[i],
                'Profession':      prof[i],
                'Work_Experience': work_exp[i],
                'Spending_Score':  spending[i],
                'Family_Size':     fam_size[i],
                'Var_1':           var1[i],
                'Segmentation':    seg,
            })

    df = (pd.DataFrame(rows)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True))
    df.insert(0, 'ID', range(1, len(df)+1))
    return df


def load_dataset():
    for path in ['train.csv', 'data/train.csv', 'dataset/train.csv']:
        if os.path.exists(path):
            print(f"Loading dataset from '{path}'...")
            df = pd.read_csv(path)
            if TARGET_COL in df.columns:
                return df
    print("'train.csv' not found — generating synthetic data with identical schema.")
    return generate_synthetic_dataset(n=8068, seed=42)


# -------------------------------------------------------------
# Pipeline factory functions
# -------------------------------------------------------------
def make_dummy_pipeline():
    return Pipeline([
        ('prep', ColumnTransformer(
            [('num', SimpleImputer(strategy='median'), NUMERIC_COLS)],
            remainder='drop'
        )),
        ('clf', DummyClassifier(strategy='stratified', random_state=42)),
    ])


def make_gaussian_pipeline():
    """GaussianNB on numeric features only."""
    return Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('scl', StandardScaler()),
            ]), NUMERIC_COLS),
        ], remainder='drop')),
        ('clf', GaussianNB()),
    ])


def make_bernoulli_pipeline():
    """BernoulliNB: OHE all features, binarize result."""
    return Pipeline([
        ('prep', ColumnTransformer([
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), ALL_CAT_COLS),
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), NUMERIC_COLS),
        ])),
        ('clf', BernoulliNB(alpha=1.0)),
    ])


def make_categorical_pipeline():
    """CategoricalNB: OrdinalEncoder + KBins for numerics + SafeShift."""
    return Pipeline([
        ('prep', ColumnTransformer([
            ('cat', Pipeline([
                ('imp',     SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(
                    handle_unknown='use_encoded_value', unknown_value=-1
                )),
                ('shift',   SafeOrdinalToNonNegative()),
            ]), ALL_CAT_COLS),
            ('num', Pipeline([
                ('imp',   SimpleImputer(strategy='median')),
                ('bins',  KBinsDiscretizer(n_bins=5, encode='ordinal',
                                           strategy='quantile')),
                ('shift', SafeOrdinalToNonNegative()),
            ]), NUMERIC_COLS),
        ])),
        ('clf', CategoricalNB(alpha=1.0)),
    ])


def make_complement_pipeline():
    """ComplementNB: same OHE as BernoulliNB (extension model)."""
    return Pipeline([
        ('prep', ColumnTransformer([
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), ALL_CAT_COLS),
            ('num', Pipeline([
                ('imp', SimpleImputer(strategy='median')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ]), NUMERIC_COLS),
        ])),
        ('clf', ComplementNB(alpha=1.0)),
    ])


# -------------------------------------------------------------
# 1. Load & Audit
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("1. LOADING AND AUDITING DATA")
print("-" * 60)

df_raw = load_dataset()
print(f"Raw shape: {df_raw.shape}")
print(f"\nMissing values:\n{df_raw[FEATURE_COLS + [TARGET_COL]].isnull().sum()}")
print(f"\nClass distribution:\n{df_raw[TARGET_COL].value_counts()}")

# -------------------------------------------------------------
# 2. EDA Visualizations
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("2. EDA VISUALIZATIONS")
print("-" * 60)

df = df_raw[FEATURE_COLS + [TARGET_COL]].copy()

# Figure 1: Class distribution
fig, ax = plt.subplots(figsize=(6, 4))
counts = df[TARGET_COL].value_counts().sort_index()
colors_seg = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
bars = ax.bar(counts.index, counts.values, color=colors_seg,
              edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f'{val}\n({val/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xlabel('Customer Segment', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Target Class Distribution — Segments A–D', fontweight='bold')
ax.set_ylim(0, counts.max() * 1.22)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lab04_class_distribution.png")

# Figure 2: Age distribution by segment
fig, ax = plt.subplots(figsize=(8, 4))
for seg, col in zip(['A', 'B', 'C', 'D'], colors_seg):
    ax.hist(df[df[TARGET_COL] == seg]['Age'].dropna(),
            bins=20, alpha=0.55, label=f'Segment {seg}',
            color=col, edgecolor='none')
ax.set_xlabel('Age', fontweight='bold')
ax.set_ylabel('Count', fontweight='bold')
ax.set_title('Age Distribution by Customer Segment', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_age_by_segment.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lab04_age_by_segment.png")

# Figure 3: Categorical breakdowns
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, feat in zip(axes, ['Gender', 'Spending_Score', 'Profession']):
    ct = pd.crosstab(df[TARGET_COL],
                     df[feat].fillna('Unknown'), normalize='index')
    ct.plot(kind='bar', stacked=True, ax=ax,
            colormap='tab10', edgecolor='white', linewidth=0.5)
    ax.set_title(f'{feat} by Segment', fontweight='bold')
    ax.set_xlabel('Segment')
    ax.set_ylabel('Proportion')
    ax.tick_params(axis='x', rotation=0)
    ax.legend(loc='upper right', fontsize=7)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_categorical_by_segment.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: lab04_categorical_by_segment.png")

# Figure 4: Missing-value heatmap (sample of 1000 rows for clarity)
miss_df = df[FEATURE_COLS].isnull().astype(int).head(500)
if miss_df.sum().sum() > 0:
    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(miss_df.T, cmap='Reds', ax=ax, cbar=True,
                yticklabels=True, xticklabels=False)
    ax.set_title('Missing Value Map — first 500 rows (red = missing)',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/lab04_missing_values.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print("Saved: lab04_missing_values.png")

# -------------------------------------------------------------
# 3. Preprocessing — encode target, train/test split
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("3. PREPROCESSING & TRAIN/TEST SPLIT")
print("-" * 60)

le = LabelEncoder()
y  = le.fit_transform(df[TARGET_COL])
X  = df[FEATURE_COLS].copy()
CLASS_NAMES = list(le.classes_)
print(f"Classes: {CLASS_NAMES}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# Save split manifest
split_manifest = pd.DataFrame({
    'split':   ['train'] * len(y_train) + ['test'] * len(y_test),
    'segment': list(le.inverse_transform(y_train)) +
               list(le.inverse_transform(y_test)),
})
split_manifest.to_csv(f'{ART_DIR}/split_manifest.csv', index=False)

# Feature manifest
with open(f'{ART_DIR}/feature_manifest.json', 'w') as fh:
    json.dump({
        'numeric':            NUMERIC_COLS,
        'binary_categorical': BINARY_COLS,
        'ordinal_categorical':ORDINAL_COLS,
        'feature_groups':     FEATURE_GROUPS,
        'target':             TARGET_COL,
        'classes':            CLASS_NAMES,
    }, fh, indent=2)

# -------------------------------------------------------------
# 4. 5-Fold Cross-Validation
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("4. 5-FOLD STRATIFIED CROSS-VALIDATION")
print("-" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scoring = ['accuracy', 'f1_macro', 'f1_weighted', 'balanced_accuracy']

MODELS = {
    'DummyClassifier': make_dummy_pipeline(),
    'GaussianNB':      make_gaussian_pipeline(),
    'BernoulliNB':     make_bernoulli_pipeline(),
    'CategoricalNB':   make_categorical_pipeline(),
    'ComplementNB':    make_complement_pipeline(),
}

cv_records = []
for name, pipe in MODELS.items():
    print(f"  CV: {name}...", end='', flush=True)
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring=cv_scoring,
                            return_train_score=True, n_jobs=-1)
    rec = {
        'Model':               name,
        'CV_Accuracy_mean':    scores['test_accuracy'].mean(),
        'CV_Accuracy_std':     scores['test_accuracy'].std(),
        'CV_F1_Macro_mean':    scores['test_f1_macro'].mean(),
        'CV_F1_Macro_std':     scores['test_f1_macro'].std(),
        'CV_F1_Weighted_mean': scores['test_f1_weighted'].mean(),
        'CV_F1_Weighted_std':  scores['test_f1_weighted'].std(),
        'CV_BalAcc_mean':      scores['test_balanced_accuracy'].mean(),
        'CV_BalAcc_std':       scores['test_balanced_accuracy'].std(),
        'Train_F1_Macro_mean': scores['train_f1_macro'].mean(),
    }
    cv_records.append(rec)
    print(f"  F1-macro={rec['CV_F1_Macro_mean']:.4f} ± {rec['CV_F1_Macro_std']:.4f}")

cv_df = pd.DataFrame(cv_records)
cv_df.to_csv(f'{REG_NO}_{LAB}_CV_Results.csv', index=False)
print(f"\nSaved: {REG_NO}_{LAB}_CV_Results.csv")
print(cv_df[['Model', 'CV_F1_Macro_mean', 'CV_F1_Macro_std',
             'CV_Accuracy_mean']].to_string(index=False))

# -------------------------------------------------------------
# 5. Feature Group Ablation (CategoricalNB)
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("5. FEATURE GROUP ABLATION STUDY")
print("-" * 60)


def make_group_cat_pipeline(feature_subset):
    cat_sub = [f for f in feature_subset if f in ALL_CAT_COLS]
    num_sub = [f for f in feature_subset if f in NUMERIC_COLS]
    transformers = []
    if cat_sub:
        transformers.append(('cat', Pipeline([
            ('imp',     SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value',
                                       unknown_value=-1)),
            ('shift',   SafeOrdinalToNonNegative()),
        ]), cat_sub))
    if num_sub:
        transformers.append(('num', Pipeline([
            ('imp',   SimpleImputer(strategy='median')),
            ('bins',  KBinsDiscretizer(n_bins=5, encode='ordinal',
                                       strategy='quantile')),
            ('shift', SafeOrdinalToNonNegative()),
        ]), num_sub))
    if not transformers:
        return None
    return Pipeline([('prep', ColumnTransformer(transformers)),
                     ('clf', CategoricalNB(alpha=1.0))])


abl_records = []
for group_name, group_feats in FEATURE_GROUPS.items():
    pipe = make_group_cat_pipeline(group_feats)
    if pipe is None:
        continue
    sc = cross_validate(pipe, X_train[group_feats], y_train, cv=cv,
                        scoring=['f1_macro', 'accuracy'], n_jobs=-1)
    abl_records.append({
        'Feature_Group':    group_name,
        'Features':         ', '.join(group_feats),
        'N_Features':       len(group_feats),
        'CV_F1_Macro_mean': sc['test_f1_macro'].mean(),
        'CV_F1_Macro_std':  sc['test_f1_macro'].std(),
        'CV_Accuracy_mean': sc['test_accuracy'].mean(),
    })
    print(f"  {group_name:15s}: F1={sc['test_f1_macro'].mean():.4f} ± {sc['test_f1_macro'].std():.4f}")

abl_df = pd.DataFrame(abl_records)

# -------------------------------------------------------------
# 6. Select Best Model & Fit on Full Training Set
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("6. MODEL SELECTION & FINAL FITTING")
print("-" * 60)

best_row  = cv_df.loc[cv_df['CV_F1_Macro_mean'].idxmax()]
best_name = best_row['Model']
print(f"Best model (CV Macro F1): {best_name} = {best_row['CV_F1_Macro_mean']:.4f}")

best_pipe = {
    'DummyClassifier': make_dummy_pipeline(),
    'GaussianNB':      make_gaussian_pipeline(),
    'BernoulliNB':     make_bernoulli_pipeline(),
    'CategoricalNB':   make_categorical_pipeline(),
    'ComplementNB':    make_complement_pipeline(),
}[best_name]
best_pipe.fit(X_train, y_train)

joblib.dump(best_pipe, f'{MODEL_DIR}/{REG_NO}_{LAB}_selected_pipeline.joblib')
joblib.dump(best_pipe, f'{MODEL_DIR}/selected_pipeline.joblib')
print(f"Model saved to {MODEL_DIR}/selected_pipeline.joblib")

# -------------------------------------------------------------
# 7. Locked Test Set Evaluation
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("7. LOCKED TEST SET EVALUATION")
print("-" * 60)

test_records = []
fitted_models = {}

for name, pipe in MODELS.items():
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe
    y_pred = pipe.predict(X_test)
    rpt    = classification_report(y_test, y_pred,
                                   target_names=CLASS_NAMES,
                                   output_dict=True)
    rec = {
        'Model':           name,
        'Accuracy':        accuracy_score(y_test, y_pred),
        'F1_Macro':        f1_score(y_test, y_pred, average='macro'),
        'F1_Weighted':     f1_score(y_test, y_pred, average='weighted'),
        'Precision_Macro': precision_score(y_test, y_pred, average='macro',
                                           zero_division=0),
        'Recall_Macro':    recall_score(y_test, y_pred, average='macro',
                                        zero_division=0),
        'BalancedAcc':     balanced_accuracy_score(y_test, y_pred),
    }
    for cls in CLASS_NAMES:
        rec[f'F1_{cls}']        = rpt[cls]['f1-score']
        rec[f'Precision_{cls}'] = rpt[cls]['precision']
        rec[f'Recall_{cls}']    = rpt[cls]['recall']
    test_records.append(rec)
    print(f"  {name:20s} | Acc={rec['Accuracy']:.4f} | F1-Macro={rec['F1_Macro']:.4f}")

test_df = pd.DataFrame(test_records)
test_df.to_csv(f'{REG_NO}_{LAB}_Test_Results.csv', index=False)
print(f"\nSaved: {REG_NO}_{LAB}_Test_Results.csv")

# -------------------------------------------------------------
# 8. Error Analysis
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("8. ERROR ANALYSIS")
print("-" * 60)

y_pred_best   = best_pipe.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred_best)
y_true_labels = le.inverse_transform(y_test)

error_mask    = (y_pred_best != y_test)
X_test_r      = X_test.reset_index(drop=True)
error_df      = X_test_r[error_mask].copy()
error_df['True_Segment']      = y_true_labels[error_mask]
error_df['Predicted_Segment'] = y_pred_labels[error_mask]
error_df['Error_Type']        = (
    'True=' + error_df['True_Segment'] +
    ' -> Pred=' + error_df['Predicted_Segment']
)
error_df.to_csv(f'{REG_NO}_{LAB}_Error_Analysis.csv', index=False)

total_err = error_mask.sum()
print(f"Misclassifications: {total_err}/{len(y_test)} "
      f"({total_err/len(y_test)*100:.1f}%)")
print("\nTop patterns:")
print(error_df['Error_Type'].value_counts().head(8))
print(f"Saved: {REG_NO}_{LAB}_Error_Analysis.csv")

# -------------------------------------------------------------
# 9. New Customer Predictions
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("9. NEW CUSTOMER PREDICTIONS")
print("-" * 60)

new_customers = pd.DataFrame({
    'Gender':          ['Male',   'Female', 'Male',      'Female', 'Male'],
    'Ever_Married':    ['Yes',    'No',     'Yes',       'Yes',    'No'],
    'Age':             [45,       28,       35,          52,       22],
    'Graduated':       ['Yes',    'Yes',    'No',        'Yes',    'No'],
    'Profession':      ['Engineer','Artist','Healthcare','Lawyer','Marketing'],
    'Work_Experience': [10,       3,        7,           20,       1],
    'Spending_Score':  ['High',   'Low',    'Average',   'High',   'Low'],
    'Family_Size':     [4,        1,        3,           5,        2],
    'Var_1':           ['Cat_6',  'Cat_2',  'Cat_4',     'Cat_6',  'Cat_1'],
})

preds  = best_pipe.predict(new_customers)
labels = le.inverse_transform(preds)
new_customers['Predicted_Segment'] = labels

try:
    proba = best_pipe.predict_proba(new_customers)
    for i, cls in enumerate(CLASS_NAMES):
        new_customers[f'P({cls})'] = proba[:, i].round(4)
except Exception:
    pass

new_customers.to_csv(f'{REG_NO}_{LAB}_NewCustomer_Predictions.csv', index=False)
print(new_customers[['Gender', 'Age', 'Profession',
                      'Spending_Score', 'Predicted_Segment']].to_string(index=False))
print(f"Saved: {REG_NO}_{LAB}_NewCustomer_Predictions.csv")

# -------------------------------------------------------------
# 10. Visualizations
# -------------------------------------------------------------
print("\n" + "-" * 60)
print("10. GENERATING VISUALIZATIONS")
print("-" * 60)

# Figure 5: CV model comparison
fig, ax = plt.subplots(figsize=(9, 5))
x      = np.arange(len(cv_df))
colors = ['#BDBDBD', '#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
bars   = ax.bar(x, cv_df['CV_F1_Macro_mean'],
                yerr=cv_df['CV_F1_Macro_std'], capsize=5,
                color=colors, edgecolor='white', linewidth=1.2, width=0.55)
for bar, val in zip(bars, cv_df['CV_F1_Macro_mean']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cv_df['Model'], rotation=15, ha='right')
ax.set_ylabel('5-Fold CV Macro F1', fontweight='bold')
ax.set_title('Cross-Validation Model Comparison (Macro F1 ± 1 std)',
             fontweight='bold')
ax.set_ylim(0, min(1.0, cv_df['CV_F1_Macro_mean'].max() + 0.14))
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_cv_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lab04_cv_comparison.png")

# Figure 6: Confusion matrix of best model
cm_best = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            annot_kws={'size': 13, 'weight': 'bold'})
ax.set_xlabel('Predicted Segment', fontweight='bold')
ax.set_ylabel('True Segment', fontweight='bold')
ax.set_title(f'Confusion Matrix — {best_name} (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_confusion_matrix.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: lab04_confusion_matrix.png")

# Figure 7: All confusion matrices
fig, axes = plt.subplots(1, len(MODELS), figsize=(22, 4))
for ax, (name, pipe) in zip(axes, fitted_models.items()):
    cm_i = confusion_matrix(y_test, pipe.predict(X_test))
    sns.heatmap(cm_i, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={'size': 9})
    ax.set_title(name, fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=8)
    ax.set_ylabel('True', fontsize=8)
plt.suptitle('Confusion Matrices — All Models (Test Set)',
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_all_confusion_matrices.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: lab04_all_confusion_matrices.png")

# Figure 8: Feature group ablation
fig, ax = plt.subplots(figsize=(7, 4))
abl_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
bars = ax.bar(abl_df['Feature_Group'], abl_df['CV_F1_Macro_mean'],
              yerr=abl_df['CV_F1_Macro_std'], capsize=4,
              color=abl_colors[:len(abl_df)], edgecolor='white', linewidth=1.2)
for bar, val in zip(bars, abl_df['CV_F1_Macro_mean']):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{val:.4f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_ylabel('CV Macro F1', fontweight='bold')
ax.set_title('Feature Group Ablation — CategoricalNB', fontweight='bold')
ax.set_ylim(0, abl_df['CV_F1_Macro_mean'].max() + 0.12)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_feature_group_ablation.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: lab04_feature_group_ablation.png")

# Figure 9: Per-class F1 grouped bar
fig, ax = plt.subplots(figsize=(9, 4))
x  = np.arange(len(CLASS_NAMES))
w  = 0.17
m_names  = [m for m in test_df['Model'] if m != 'DummyClassifier']
m_colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
for i, mname in enumerate(m_names):
    row = test_df[test_df['Model'] == mname].iloc[0]
    f1s = [row[f'F1_{c}'] for c in CLASS_NAMES]
    ax.bar(x + i * w, f1s, width=w, label=mname,
           color=m_colors[i], alpha=0.85, edgecolor='white')
ax.set_xticks(x + w * 1.5)
ax.set_xticklabels([f'Segment {c}' for c in CLASS_NAMES])
ax.set_ylabel('F1 Score', fontweight='bold')
ax.set_title('Per-Class F1 — All Models (Test Set)', fontweight='bold')
ax.legend(fontsize=8, loc='lower right')
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_per_class_f1.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: lab04_per_class_f1.png")

# Figure 10: Error pattern bar chart
error_top = error_df['Error_Type'].value_counts().head(8)
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(error_top.index, error_top.values,
        color='#E91E63', edgecolor='white')
ax.set_xlabel('Count', fontweight='bold')
ax.set_title(f'Top Misclassification Patterns — {best_name}', fontweight='bold')
ax.invert_yaxis()
for i, val in enumerate(error_top.values):
    ax.text(val + 0.3, i, str(val), va='center', fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/lab04_error_patterns.png', dpi=150,
            bbox_inches='tight')
plt.close()
print("Saved: lab04_error_patterns.png")

# -------------------------------------------------------------
# 11. Save versions artifact
# -------------------------------------------------------------
import sklearn
with open(f'{ART_DIR}/versions.json', 'w') as fh:
    json.dump({
        'python':        platform.python_version(),
        'numpy':         np.__version__,
        'pandas':        pd.__version__,
        'sklearn':       sklearn.__version__,
        'joblib':        joblib.__version__,
        'random_state':  42,
        'cv_folds':      5,
        'best_model':    best_name,
    }, fh, indent=2)

# -------------------------------------------------------------
# Final summary
# -------------------------------------------------------------
best_test_row = test_df[test_df['Model'] == best_name].iloc[0]
print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print(f"Best model  : {best_name}")
print(f"CV F1-Macro : {best_row['CV_F1_Macro_mean']:.4f} ± {best_row['CV_F1_Macro_std']:.4f}")
print(f"Test Acc    : {best_test_row['Accuracy']:.4f}")
print(f"Test F1-Mac : {best_test_row['F1_Macro']:.4f}")
print("=" * 70)
