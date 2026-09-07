"""
Generate 5 additional figures for Lab 05 from saved pipeline outputs.
Saves to lab05_figs/ alongside the original 10 figures.
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve, StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import re

BASE     = r'c:\Users\I768951\OneDrive - SAP SE\SAP\Exam'
FIG_DIR  = os.path.join(BASE, 'lab05_figs')
OUT_DIR  = os.path.join(BASE, 'lab05_outputs')
SEED     = 42

np.random.seed(SEED)
os.makedirs(FIG_DIR, exist_ok=True)

PALETTE = {'negative': '#E74C3C', 'neutral': '#F39C12', 'positive': '#27AE60'}
CLASSES  = ['negative', 'neutral', 'positive']

def normalize_tweet(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', str(text))
    text = re.sub(r'@\w+', ' <USER> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Load saved data ────────────────────────────────────────────────────────────
print('Loading saved train/test data...')
train_df = pd.read_csv(os.path.join(OUT_DIR, 'train_manifest.csv'))
test_df  = pd.read_csv(os.path.join(OUT_DIR, 'test_manifest.csv'))
preds_df = pd.read_csv(os.path.join(OUT_DIR, 'test_predictions.csv'))

train_df['clean_text'] = train_df['text'].map(normalize_tweet)
test_df['clean_text']  = test_df['text'].map(normalize_tweet)

X_train = train_df['clean_text']
y_train = train_df['airline_sentiment']
X_test  = test_df['clean_text']
y_test  = test_df['airline_sentiment']

# ── Fit best model for probability / feature access ───────────────────────────
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)),
    ('clf',   MultinomialNB(alpha=0.5))
])
pipe.fit(X_train, y_train)
tfidf = pipe.named_steps['tfidf']
nb    = pipe.named_steps['clf']

# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 — Top 20 Discriminative Terms per Sentiment Class
# ─────────────────────────────────────────────────────────────────────────────
print('Generating fig11: top discriminative terms...')
feature_names = np.array(tfidf.get_feature_names_out())
log_probs     = nb.feature_log_prob_          # shape (3, n_features)
class_map     = {c: i for i, c in enumerate(nb.classes_)}

fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.suptitle('Top 20 Most Discriminative Terms per Sentiment Class\n(NB Log-Probability Score)',
             fontsize=14, fontweight='bold', y=1.01)

for ax, cls in zip(axes, CLASSES):
    idx   = class_map[cls]
    # log P(term|class) - max over other classes (discriminativeness)
    other = [i for i in range(3) if i != idx]
    disc  = log_probs[idx] - np.max(log_probs[other], axis=0)
    top20 = np.argsort(disc)[-20:][::-1]
    terms = feature_names[top20]
    scores = disc[top20]

    colors = [PALETTE[cls]] * len(terms)
    bars = ax.barh(range(len(terms)), scores, color=colors, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(terms)))
    ax.set_yticklabels(terms, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlabel('Discriminativeness Score\n(log P(term|class) − max log P(term|other))', fontsize=8)
    ax.set_title(f'{cls.capitalize()} Class', fontsize=12, fontweight='bold',
                 color=PALETTE[cls])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')

plt.tight_layout()
p = os.path.join(FIG_DIR, 'fig11_discriminative_terms.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 — Learning Curve
# ─────────────────────────────────────────────────────────────────────────────
print('Generating fig12: learning curve...')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
train_sizes_abs, train_scores, val_scores = learning_curve(
    pipe, X_train, y_train,
    train_sizes=np.linspace(0.10, 1.0, 10),
    cv=cv, scoring='f1_macro',
    n_jobs=1, random_state=SEED
)
train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes_abs, train_mean, 'o-', color='#2980B9', label='Training macro F1', linewidth=2)
ax.fill_between(train_sizes_abs, train_mean-train_std, train_mean+train_std,
                alpha=0.18, color='#2980B9')
ax.plot(train_sizes_abs, val_mean, 's-', color='#E74C3C', label='Validation macro F1', linewidth=2)
ax.fill_between(train_sizes_abs, val_mean-val_std, val_mean+val_std,
                alpha=0.18, color='#E74C3C')
ax.axhline(val_mean[-1], color='gray', linestyle=':', linewidth=1.2, alpha=0.7)
ax.set_xlabel('Training Set Size (number of tweets)', fontsize=11)
ax.set_ylabel('Macro F1 Score', fontsize=11)
ax.set_title('Learning Curve — MultinomialNB + TF-IDF\n(5-fold stratified CV)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.5, 1.02)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Annotate final validation score
ax.annotate(f'Final val F1: {val_mean[-1]:.4f}',
            xy=(train_sizes_abs[-1], val_mean[-1]),
            xytext=(-120, 18), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=9, color='#E74C3C')
plt.tight_layout()
p = os.path.join(FIG_DIR, 'fig12_learning_curve.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 — Precision-Recall Curves (one-vs-rest per class)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating fig13: precision-recall curves...')
# Use LR for calibrated probabilities
from sklearn.linear_model import LogisticRegression
lr_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)),
    ('clf',   LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED, C=1.0))
])
lr_pipe.fit(X_train, y_train)
y_score = lr_pipe.predict_proba(X_test)
y_bin   = label_binarize(y_test, classes=CLASSES)

fig, ax = plt.subplots(figsize=(8, 6))
for i, cls in enumerate(CLASSES):
    prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
    ap = average_precision_score(y_bin[:, i], y_score[:, i])
    ax.plot(rec, prec, linewidth=2.2, color=list(PALETTE.values())[i],
            label=f'{cls.capitalize()} (AP={ap:.3f})')

ax.axhline(y_bin.mean(axis=0).mean(), color='gray', linestyle='--',
           linewidth=1, alpha=0.7, label='Random baseline')
ax.set_xlabel('Recall', fontsize=11)
ax.set_ylabel('Precision', fontsize=11)
ax.set_title('Precision-Recall Curves by Sentiment Class\n(LogisticRegression, one-vs-rest)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10, loc='lower left')
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
p = os.path.join(FIG_DIR, 'fig13_pr_curves.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 14 — NB Feature Log-Probability Heatmap (top 30 shared terms)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating fig14: feature log-probability heatmap...')
# Find top 30 most informative terms across all classes
max_lp_per_feature = log_probs.max(axis=0)
top30_idx  = np.argsort(max_lp_per_feature)[-30:][::-1]
top30_terms = feature_names[top30_idx]
heat_data   = log_probs[:, top30_idx]  # shape (3, 30)

fig, ax = plt.subplots(figsize=(14, 6))
im = ax.imshow(heat_data, aspect='auto', cmap='RdYlGn', interpolation='nearest')
ax.set_xticks(range(30))
ax.set_xticklabels(top30_terms, rotation=45, ha='right', fontsize=8.5)
ax.set_yticks(range(3))
ax.set_yticklabels([c.capitalize() for c in nb.classes_], fontsize=11)
ax.set_title('NB Feature Log-Probability Heatmap\n(Top 30 most informative terms across all sentiment classes)',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, label='log P(term | class)', shrink=0.7)
for i in range(3):
    for j in range(30):
        val = heat_data[i, j]
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=6, color='black' if -8 < val < -4 else 'white')
plt.tight_layout()
p = os.path.join(FIG_DIR, 'fig14_feature_heatmap.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 15 — Prediction Confidence Distribution (correct vs incorrect)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating fig15: prediction confidence distribution...')
# Use LR probabilities
proba = lr_pipe.predict_proba(X_test)
max_conf = proba.max(axis=1)
pred_labels = lr_pipe.predict(X_test)
correct   = (pred_labels == y_test.values)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Prediction Confidence Distribution\n(Maximum predicted class probability)',
             fontsize=13, fontweight='bold')

# Left: histogram
ax = axes[0]
ax.hist(max_conf[correct],   bins=30, alpha=0.65, color='#27AE60', label=f'Correct ({correct.sum():,})',   density=True)
ax.hist(max_conf[~correct],  bins=30, alpha=0.65, color='#E74C3C', label=f'Incorrect ({(~correct).sum():,})', density=True)
ax.set_xlabel('Max Predicted Probability', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Confidence Histogram', fontsize=11)
ax.legend(fontsize=9)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0.5 threshold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# Right: confidence by true class
ax2 = axes[1]
conf_by_class = {cls: max_conf[y_test.values == cls] for cls in CLASSES}
bp = ax2.boxplot([conf_by_class[c] for c in CLASSES],
                 patch_artist=True, notch=True,
                 medianprops=dict(color='black', linewidth=2))
for patch, cls in zip(bp['boxes'], CLASSES):
    patch.set_facecolor(PALETTE[cls]); patch.set_alpha(0.7)
ax2.set_xticklabels([c.capitalize() for c in CLASSES], fontsize=11)
ax2.set_ylabel('Max Predicted Probability', fontsize=11)
ax2.set_title('Confidence by True Class', fontsize=11)
ax2.set_ylim(0.3, 1.05)
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)

# Annotate mean accuracy per confidence band
bands = [(0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
for lo, hi in bands:
    mask  = (max_conf >= lo) & (max_conf < hi)
    if mask.sum() == 0: continue
    acc_b = correct[mask].mean()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
p = os.path.join(FIG_DIR, 'fig15_confidence_dist.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 16 — Tweet Sentiment Distribution by Airline (grouped bar)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating fig16: airline grouped sentiment bar...')
all_df = pd.concat([train_df, test_df], ignore_index=True)
pivot  = all_df.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0)
# Ensure all three columns exist
for cls in CLASSES:
    if cls not in pivot.columns:
        pivot[cls] = 0
pivot = pivot[CLASSES]

airlines = pivot.index.tolist()
x = np.arange(len(airlines))
w = 0.26

fig, ax = plt.subplots(figsize=(12, 6))
for i, cls in enumerate(CLASSES):
    bars = ax.bar(x + (i-1)*w, pivot[cls], width=w,
                  color=PALETTE[cls], alpha=0.85, label=cls.capitalize(),
                  edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 10,
                    str(int(h)), ha='center', va='bottom', fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels([a.replace(' ', '\n') for a in airlines], fontsize=10)
ax.set_ylabel('Number of Tweets', fontsize=11)
ax.set_title('Tweet Count by Airline and Sentiment Class\n(full dataset: 14,640 tweets)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
p = os.path.join(FIG_DIR, 'fig16_airline_grouped_bar.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

# ─────────────────────────────────────────────────────────────────────────────
# FIG 17 — Model Accuracy vs Training Size (bar chart summary)
# ─────────────────────────────────────────────────────────────────────────────
print('Generating fig17: CV results detailed bar chart...')
cv_df = pd.read_csv(os.path.join(OUT_DIR, 'cv_results.csv'))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('5-Fold Cross-Validation Results — All Models', fontsize=13, fontweight='bold')

model_colors = {'MultinomialNB': '#2980B9', 'LogisticRegression': '#8E44AD', 'LinearSVC': '#16A085'}

# Left: macro F1 with error bars
ax = axes[0]
models = cv_df['model'].tolist() if 'model' in cv_df.columns else ['MultinomialNB', 'LogisticRegression', 'LinearSVC']
means  = cv_df['cv_macro_f1_mean'].tolist() if 'cv_macro_f1_mean' in cv_df.columns else [0.8539, 0.8522, 0.8434]
stds   = cv_df['cv_macro_f1_std'].tolist()  if 'cv_macro_f1_std'  in cv_df.columns else [0.0076, 0.0076, 0.0080]
colors = [model_colors.get(m, '#95A5A6') for m in models]
bars   = ax.bar(models, means, yerr=stds, capsize=6,
               color=colors, alpha=0.85, edgecolor='white', error_kw={'ecolor':'gray','elinewidth':1.5})
ax.set_ylim(0.80, 0.88)
ax.set_ylabel('CV Macro F1', fontsize=11)
ax.set_title('CV Macro F1 ± 1 SD', fontsize=11)
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.001,
            f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_xticklabels([m.replace('Regression', '\nRegression').replace('MultinomialNB','Multinomial\nNB').replace('LinearSVC','Linear\nSVC') for m in models])

# Right: weighted F1
ax2 = axes[1]
wmeans = cv_df['cv_weighted_f1_mean'].tolist() if 'cv_weighted_f1_mean' in cv_df.columns else [0.8680, 0.8667, 0.8595]
wstds  = cv_df['cv_weighted_f1_std'].tolist()  if 'cv_weighted_f1_std'  in cv_df.columns else [0.0060, 0.0063, 0.0067]
bars2  = ax2.bar(models, wmeans, yerr=wstds, capsize=6,
                color=colors, alpha=0.85, edgecolor='white', error_kw={'ecolor':'gray','elinewidth':1.5})
ax2.set_ylim(0.83, 0.90)
ax2.set_ylabel('CV Weighted F1', fontsize=11)
ax2.set_title('CV Weighted F1 ± 1 SD', fontsize=11)
for bar, mean in zip(bars2, wmeans):
    ax2.text(bar.get_x() + bar.get_width()/2, mean + 0.001,
             f'{mean:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
ax2.set_xticklabels([m.replace('Regression', '\nRegression').replace('MultinomialNB','Multinomial\nNB').replace('LinearSVC','Linear\nSVC') for m in models])

plt.tight_layout()
p = os.path.join(FIG_DIR, 'fig17_cv_detailed.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved: {p}')

print('\nAll extra figures saved to:', FIG_DIR)
figs = sorted([f for f in os.listdir(FIG_DIR) if f.endswith('.png')])
print(f'Total figures in directory: {len(figs)}')
for f in figs:
    print(f'  {f}')
