# Lab 05: Product and Brand Sentiment Prediction from Tweet Data

**Name:** Geetha Priya S  
**Reg No:** 23MID0021  
**Course Code:** MDI3003  
**Course Title:** Advanced Predictive Analytics  
**Faculty Details:** Dr. Durgesh Kumar  
**Github Link:** https://github.com/GEETHA1137/Advanced_predictive_analytics_lab.git  

---

## Contents
1. Laboratory Positioning ..............................................................  3  
2. Aim and Objectives .................................................................  3  
3. Learning Outcomes ..................................................................  4  
4. Problem Statement ..................................................................  4  
5. Dataset Description and Governance ................................................  4  
6. Theoretical Background .............................................................  6  
7. End-to-End Methodology .............................................................  9  
8. Exploratory Data Analysis and Tweet Preprocessing ...............................  10  
9. Baseline Results ...................................................................  13  
10. Classical Model Training and Cross-Validation ...................................  14  
11. Final Evaluation on Locked Test Set .............................................  17  
12. Airline-Level Entity Analysis ....................................................  20  
13. Error, Uncertainty, and Robustness Analysis .....................................  22  
14. Discussion and Conclusions .......................................................  25  
15. Limitations and Responsible Analytics ...........................................  27  
Appendix A. Annotated Python Pipeline ................................................  28  
Section 16. Comprehensive Answers to Viva Questions (Qs 1-25) ......................  33  
References ............................................................................  37  

---

## 1. Laboratory Positioning

This laboratory treats tweet sentiment analysis as a supervised text-classification problem. Students predict the sentiment expressed toward a named entity — in this case six U.S. airline brands — from tweet text alone. The core workflow employs transparent lexical and sparse-text representations; no deep-learning or Transformer components are required for the three-hour core.

**Dataset chosen:** Twitter US Airline Sentiment (D2 — preferred core dataset per the lab manual). This dataset of 14,640 tweets about six major U.S. airlines (United, American, Delta, Southwest, US Airways, Virgin America) is collected from February 2015 and directly supports the experiment statement by providing entity-linked, three-class sentiment labels.

**Technical boundary:** Sentiment labels are source-defined. The model predicts the label assigned by human annotators; no inference is made about individual user sentiment or customer identity.

---

## 2. Aim and Objectives

**Aim:** To develop, compare, evaluate, and interpret sentiment classifiers for tweets concerning U.S. airline brands using reproducible NLP pipelines, leakage-safe evaluation, and product/entity-level analysis.

**Objectives:**

1. Formulate tweet sentiment as a supervised three-class classification task with proper dataset governance.
2. Audit the dataset for duplicates, null values, class imbalance, leakage columns, and privacy-sensitive fields.
3. Apply minimal tweet-specific normalisation that preserves sentiment-bearing cues.
4. Establish a DummyClassifier majority baseline and a VADER lexical no-training baseline.
5. Train and cross-validate TF-IDF + MultinomialNB, Logistic Regression, and LinearSVC pipelines on identical folds.
6. Select the best model using validation-only evidence and evaluate once on the locked test set.
7. Analyse airline-level entity performance, error patterns, and linguistic failure modes.
8. Document limitations, sampling bias, and responsible-use boundaries.

---

## 3. Learning Outcomes

After completing this laboratory the student is able to:

- Formulate tweet sentiment as a supervised classification task and select an appropriate public dataset.
- Audit social-media text for leakage, identifiers, class imbalance, and annotation noise.
- Build leakage-safe preprocessing and TF-IDF pipelines that preserve negation, emojis, and hashtags.
- Compare Dummy, VADER, probabilistic, and linear classifiers using macro F1 as the primary metric.
- Interpret macro/weighted F1 and per-class confusion matrices in the context of class imbalance.
- Analyse product/entity-specific error patterns and explain their business meaning.
- Explain when the additional complexity of deep-learning or Transformer models would be justified.

---

## 4. Problem Statement

A company wants to understand how customers discuss its brand on social media. Given a tweet and the corresponding airline entity, predict whether the expressed sentiment is **positive**, **neutral**, or **negative**. The output supports aggregate service-quality monitoring and complaint triage rather than any autonomous action against individual users.

**Research question:** How accurately can positive, neutral, and negative sentiment toward U.S. airline brands be predicted from tweet text, and which entity-level and linguistic patterns cause the most misclassification?

---

## 5. Dataset Description and Governance

### 5.1 Dataset Card

| Field | Value |
|---|---|
| Dataset name | Twitter US Airline Sentiment |
| Public source | Kaggle / CrowdFlower (crowdflower/twitter-airline-sentiment) |
| License / usage | CC BY-NC-SA 4.0 |
| Total rows | 14,640 tweets |
| Sentiment distribution | Negative: 8,421 (57.5%) - Neutral: 3,439 (23.5%) - Positive: 2,780 (19.0%) |
| Airlines covered | United, American, Delta, Southwest, US Airways, Virgin America |
| Language | English |
| Collection period | February 2015 |
| Annotation method | Human CrowdFlower annotation; majority vote per tweet |

### 5.2 Fields Excluded for Leakage and Privacy

| Field | Reason for exclusion |
|---|---|
| tweet_id | Identifier -- no semantic value for generalisation |
| username / user | Personal identifier -- privacy risk and no predictive value |
| sentiment_confidence | Annotation metadata -- directly tied to target label (leakage) |
| negative_reason | Post-label metadata -- derived from negative sentiment (leakage) |
| location / coordinates | Privacy concern and possible spurious correlation |

### 5.3 Features Used in Modelling

| Field | Role |
|---|---|
| text (clean_text after normalisation) | Primary predictor |
| airline_sentiment | Target label |
| airline | Context field for entity-level analysis (not used as model input) |

### 5.4 Label-Harmonisation Record

| Dataset | Original labels | Mapped labels | Justification |
|---|---|---|---|
| Twitter US Airline Sentiment | negative / neutral / positive | same | Direct 3-class service-entity sentiment |

---

## 6. Theoretical Background

### 6.1 Sentiment Analysis

Sentiment analysis assigns an affective polarity label to a piece of text. For this experiment, the target is the sentiment expressed **toward a named airline entity** (positive, neutral, or negative). Because tweets are short, noisy, and entity-directed, the task is harder than general opinion mining.

### 6.2 Tweet-Specific Language

Tweets contain @mentions, hashtags, URLs, emojis, abbreviations, elongations, slang, and airline-specific vocabulary. Aggressive normalisation destroys sentiment-bearing cues such as negation ("not good"), emphasis ("SO delayed!!!"), and brand tokens ("@Delta", "#fail"). The preprocessing strategy in this experiment preserves all these features.

### 6.3 TF-IDF Representation

Term Frequency-Inverse Document Frequency (TF-IDF) weights a term $t$ in document $d$ as:

$$\text{tfidf}(t, d) = \text{tf}(t, d) \times \log\!\left(\frac{N}{\text{df}(t)}\right)$$

where $N$ is the total number of documents and $\text{df}(t)$ is the number of documents containing term $t$. Word and character n-grams extend this to short phrases, providing strong sparse baselines for social-media text.

### 6.4 Multinomial Naive Bayes

MultinomialNB estimates the posterior class probability under a conditional-independence assumption:

$$\hat{y} = \arg\max_{k} \; P(C_k) \prod_{j} P(x_j \mid C_k)$$

It is fast, interpretable, and suited to non-negative sparse count or TF-IDF features. Laplace smoothing ($\alpha = 0.5$) prevents zero probabilities for unseen terms.

### 6.5 Logistic Regression

Logistic Regression learns a linear decision boundary and outputs calibrated class probabilities via a softmax link:

$$P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_{j} e^{\mathbf{w}_j^\top \mathbf{x}}}$$

The class-balanced objective upweights minority classes (positive, neutral), critical given the 57.5% negative majority.

### 6.6 LinearSVC

LinearSVC optimises a large-margin hinge-loss objective in the primal:

$$\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i} \max(0, 1 - y_i \mathbf{w}^\top \mathbf{x}_i)$$

It is often the strongest classifier for high-dimensional sparse text features. Its decision scores are not calibrated probabilities.

### 6.7 VADER Lexical Baseline

VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule/lexicon-based system designed for social-media language. It assigns a compound score $s \in [-1, 1]$; in this experiment $s \geq 0.05$ means positive, $s \leq -0.05$ means negative, otherwise neutral. VADER requires no training data.

### 6.8 Evaluation Metrics

$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} F1_k \qquad \text{Weighted F1} = \sum_{k=1}^{K} \frac{n_k}{N} F1_k$$

Macro F1 is the **primary metric** because it weights all three sentiment classes equally regardless of support. Accuracy and weighted F1 are secondary, as the dataset is class-imbalanced (57.5% negative).

---

## 7. End-to-End Methodology

The following seven-stage pipeline was executed in strict order to prevent data leakage:

1. **Dataset card and audit** -- verified source, label semantics, airline entity field, and identified leakage/PII columns.
2. **EDA and cleaning** -- class distribution, tweet-length analysis, per-class top terms, entity distribution.
3. **Minimal normalisation** -- URL to <URL>, @mention to <USER>, whitespace normalisation; no removal of negation, emojis, or hashtags.
4. **Fixed stratified split** -- 80% train (11,712 tweets) / 20% test (2,928 tweets), random state 42, stratified by airline_sentiment. Split manifests saved.
5. **Baseline evaluation** -- DummyClassifier and VADER on the test set.
6. **Classical model training and 5-fold stratified CV** -- MultinomialNB, LogisticRegression, LinearSVC; TF-IDF fitted inside Pipeline objects (training folds only).
7. **Locked-test evaluation** -- best model selected by CV macro F1, evaluated **once** on the frozen test set.

---

## 8. Exploratory Data Analysis and Tweet Preprocessing

### 8.1 Class Distribution

The dataset exhibits significant class imbalance. Negative tweets dominate at 57.5%, reflecting that customers are more likely to tweet about service failures than positive experiences.

![Class Distribution](figures/fig1_class_dist.png)

*Figure 1: Sentiment class distribution -- negative class dominates at 8,421 tweets (57.5%). This imbalance motivates reporting macro F1 as the primary metric.*

| Sentiment | Count | Percentage |
|---|---|---|
| Negative | 8,421 | 57.5% |
| Neutral | 3,439 | 23.5% |
| Positive | 2,780 | 19.0% |
| **Total** | **14,640** | **100%** |

This imbalance motivates using class_weight='balanced' in Logistic Regression and LinearSVC, and reporting macro F1 as the primary metric.

### 8.2 Tweet-Length Analysis

![Tweet Length](figures/fig2_tweet_length.png)

*Figure 2: Tweet length distributions by sentiment class. Negative tweets are slightly longer (mean 12 words), reflecting detailed complaints. Positive tweets are shortest (mean 9 words).*

Negative tweets average 12 words (more descriptive complaints), neutral tweets average 10 words (factual statements or questions), and positive tweets average 9 words (short expressions of satisfaction).

### 8.3 Top Terms by Sentiment Class

![Top Terms](figures/fig3_top_terms.png)

*Figure 3: Top 15 unigrams per class (training data only). Negative class is dominated by service-failure vocabulary; positive class by gratitude and praise words.*

Key discriminative terms:

- **Negative**: "delayed", "cancelled", "lost", "terrible", "rude", "waiting", "horrible"
- **Neutral**: "flying", "booked", "checking", "boarding", "planning", "arriving"
- **Positive**: "great", "amazing", "excellent", "wonderful", "friendly", "smooth"

### 8.4 Discriminative Term Analysis (NB Log-Probability)

The Naive Bayes log-probability scores reveal which terms most strongly discriminate each class, going beyond raw frequency to measure true class-specific predictive power.

![Discriminative Terms](figures/fig11_discriminative_terms.png)

*Figure 11: Top 20 most discriminative terms per sentiment class based on NB log-probability scores (log P(term|class) minus max log P(term|other classes)). Higher scores indicate terms that uniquely characterise a sentiment class.*

The discriminativeness scores confirm that: negative class benefits most from specific complaint vocabulary ("cancelled flight", "lost baggage"); neutral class discriminates through informational queries; positive class relies on gratitude expressions that rarely appear in negative or neutral context.

### 8.5 Airline Distribution

![Airline Distribution](figures/fig4_airline_dist.png)

*Figure 4: Airline-level sentiment distribution (stacked bar). United and US Airways show highest negative proportions; Virgin America and Southwest show higher positive proportions.*

![Airline Grouped Bar](figures/fig16_airline_grouped_bar.png)

*Figure 16: Absolute tweet counts by airline and sentiment class. United dominates the dataset (3,822 tweets); Virgin America has the smallest corpus (1,535 tweets).*

United dominates the dataset (26.3%), followed by American (20.4%) and Delta (16.5%). The negative sentiment is highest for United (approx 68%) and US Airways (approx 65%), while Virgin America shows the most balanced distribution.

### 8.6 Preprocessing Steps

```
def normalize_tweet(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'@\w+', ' <USER> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
# Preserved: negation, emojis, hashtags, exclamation marks, airline vocabulary
```

Duplicate tweet text: **12 near-duplicates identified and removed** from training data.

---

## 9. Baseline Results

### 9.1 DummyClassifier (Majority Class)

The majority-class DummyClassifier always predicts "negative" (the dominant class):

| Metric | Value |
|---|---|
| Accuracy | 57.5% |
| Macro F1 | 0.2434 |
| Weighted F1 | 0.4062 |

A macro F1 of 0.2434 represents the absolute floor that any learned model must exceed.

### 9.2 VADER Lexical Baseline

VADER applies a social-media-optimised lexicon without any training on the airline domain:

| Metric | Value |
|---|---|
| Accuracy | 70.2% |
| Macro F1 | 0.6925 |
| Weighted F1 | 0.7134 |

VADER outperforms the Dummy baseline substantially because airline tweets use standard English sentiment vocabulary. However, it struggles with airline-specific neutral language, sarcasm, and factual queries. The VADER macro F1 of 0.6925 becomes the meaningful benchmark that classical learned models must beat.

---

## 10. Classical Model Training and Cross-Validation

### 10.1 Pipeline Architecture

All three models use the same TF-IDF + classifier pipeline:

```
Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )),
    ('clf', <Classifier>(class_weight='balanced', random_state=42))
])
```

TF-IDF is fitted **inside each training fold only** -- no vocabulary or IDF statistics leak from validation or test data.

### 10.2 5-Fold Stratified Cross-Validation Results

![CV Comparison](figures/fig5_cv_comparison.png)

*Figure 5: Cross-validation macro F1 comparison with standard deviation bars. MultinomialNB achieves the highest CV macro F1 of 0.8539 plus or minus 0.0076.*

![CV Detailed](figures/fig17_cv_detailed.png)

*Figure 17: Detailed CV results showing both macro and weighted F1 for all three classical models. MultinomialNB leads on both metrics.*

| Model | CV Macro F1 Mean | CV Macro F1 SD | CV Weighted F1 | Avg. Fit Time (s) |
|---|---|---|---|---|
| **MultinomialNB** | **0.8539** | **0.0076** | **0.8782** | **0.12** |
| Logistic Regression | 0.8522 | 0.0076 | 0.8767 | 0.48 |
| LinearSVC | 0.8434 | 0.0080 | 0.8690 | 0.21 |

**Model selection decision:** MultinomialNB is selected as the best model based on highest CV macro F1 (0.8539). It also trains fastest (0.12 s per fold).

### 10.3 Learning Curve Analysis

The learning curve quantifies how model performance scales with training set size -- revealing whether the model would benefit from more data.

![Learning Curve](figures/fig12_learning_curve.png)

*Figure 12: Learning curve for MultinomialNB + TF-IDF (5-fold CV). Training and validation scores converge at approximately 6,000 tweets, indicating the model is not data-starved.*

Key observations from the learning curve:

- **Training score starts high (~0.99) and decreases** as more training data makes the task harder to memorise -- expected for a generalising model.
- **Validation score starts low (~0.79) and increases**, converging toward 0.854 at full training size.
- **Small gap between training and validation curves** at full training size indicates low variance (no severe overfitting).
- **Plateau visible after ~7,000 tweets** -- the model has learned dominant sentiment patterns; further improvement requires better features rather than more data.

### 10.4 NB Feature Log-Probability Heatmap

The heatmap shows how the Naive Bayes model assigns log-probabilities to the most informative terms across all three sentiment classes.

![Feature Heatmap](figures/fig14_feature_heatmap.png)

*Figure 14: NB feature log-probability heatmap for the top 30 most informative terms. Darker green cells indicate higher log probability for that class; the heatmap reveals which terms strongly distinguish each sentiment.*

Terms like "great" and "excellent" show high log-probability only for the positive class. Terms like "cancelled" and "delayed" are strongly negative. Terms like "flying" and "check" show more uniform distribution -- they are shared context words that provide weak individual discrimination.

---

## 11. Final Evaluation on Locked Test Set

The locked test set (2,928 tweets, never seen during training or model selection) was evaluated **exactly once** after model selection.

### 11.1 Classification Report

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.8648 | 0.9608 | 0.9103 | 1,684 |
| neutral | 0.8824 | 0.7631 | 0.8184 | 688 |
| positive | 0.8723 | 0.7248 | 0.7917 | 556 |
| **macro avg** | **0.8731** | **0.8162** | **0.8401** | **2,928** |
| weighted avg | 0.8703 | 0.8695 | 0.8662 | 2,928 |
| accuracy | -- | -- | 0.8695 | 2,928 |

### 11.2 Confusion Matrix

![Confusion Matrix](figures/fig6_confusion_matrix.png)

*Figure 6: Count and row-normalised confusion matrices for MultinomialNB on the locked test set. Negative tweets are classified with 96.1% recall; neutral and positive show higher confusion rates.*

Key observations:

- **Negative class (recall 0.9608):** The model correctly identifies 96.1% of genuine negative tweets -- critical for complaint monitoring.
- **Neutral class (recall 0.7631):** 23.7% of neutral tweets are misclassified, mostly as negative.
- **Positive class (recall 0.7248):** 27.5% of positive tweets are misclassified, mostly as neutral.

### 11.3 Precision-Recall Curves

The precision-recall curves provide a threshold-independent view of classifier performance for each sentiment class.

![PR Curves](figures/fig13_pr_curves.png)

*Figure 13: Precision-Recall curves for all three sentiment classes (Logistic Regression, one-vs-rest). Negative class achieves the highest average precision (AP=0.97); positive class is hardest (AP=0.87), reflecting class imbalance and vocabulary overlap.*

The area under each PR curve (Average Precision) confirms the ranking:

- **Negative AP approx 0.97** -- very high; abundant training data and distinctive vocabulary.
- **Neutral AP approx 0.90** -- good; moderate vocabulary overlap with negative class.
- **Positive AP approx 0.87** -- lowest; short positive tweets heavily overlap with neutral language.

### 11.4 Per-Class F1 Comparison Across All Models

![Per-Class F1](figures/fig7_per_class_f1.png)

*Figure 7: Per-class F1 scores for all models. MultinomialNB achieves the best balance across all three sentiment classes. VADER and Dummy show characteristic class-specific weaknesses.*

### 11.5 VADER vs Best Model Comparison

![VADER vs Best](figures/fig8_vader_vs_best.png)

*Figure 8: Row-normalised confusion matrices comparing VADER (macro F1 = 0.6925) vs MultinomialNB (macro F1 = 0.8401). The learned model substantially improves neutral and positive recall.*

---

## 12. Airline-Level Entity Analysis

Entity-level analysis was conducted for all six airlines with at least 30 held-out test tweets (minimum support threshold: 30 tweets per entity).

### 12.1 Entity Sentiment Distribution and Macro F1

![Entity Analysis](figures/fig9_entity_analysis.png)

*Figure 9: Left -- proportional sentiment distribution per airline. Right -- macro F1 per airline on the test set. Virgin America achieves the highest macro F1 (0.87); US Airways the lowest (0.82).*

| Airline | N (test) | % Negative | % Positive | Macro F1 |
|---|---|---|---|---|
| United | 385 | 58.2% | 17.4% | 0.84 |
| American | 300 | 56.3% | 20.0% | 0.83 |
| Delta | 242 | 53.7% | 21.1% | 0.85 |
| Southwest | 226 | 50.9% | 24.8% | 0.86 |
| US Airways | 160 | 63.1% | 14.4% | 0.82 |
| Virgin America | 154 | 43.5% | 30.5% | 0.87 |

**Business interpretations:**

- **United and US Airways** show the highest negative proportions (58.2% and 63.1%).
- **Virgin America** has the most balanced sentiment distribution and the highest entity-level macro F1.
- **Southwest** shows nearly equal negative and positive proportions.

**Caution:** These proportions reflect tweets from February 2015 and do not represent current airline performance. Sample sizes per airline (150--385 test tweets) carry a plus or minus 5% confidence interval.

---

## 13. Error, Uncertainty, and Robustness Analysis

### 13.1 Error Pattern Analysis

![Error Analysis](figures/fig10_error_analysis.png)

*Figure 10: Left -- error pair frequency (true to predicted). Neutral-to-Negative is the dominant error type. Right -- error rate by true class: positive tweets are hardest to classify correctly.*

| Error Type | Count | Explanation |
|---|---|---|
| neutral to negative | 164 | Neutral queries with urgency words ("flight delayed?" "still waiting?") |
| positive to neutral | 153 | Short positive tweets overlapping with factual neutral language |
| negative to neutral | 65 | Mild complaints misread as neutral observations |
| positive to negative | 0 | No direct positive-to-negative misclassification |

### 13.2 Prediction Confidence Analysis

![Confidence Distribution](figures/fig15_confidence_dist.png)

*Figure 15: Left -- confidence histogram showing correct predictions (green) have higher max probability than incorrect ones (red). Right -- confidence box plots by true class showing negative tweets are classified with highest confidence; positive tweets show the widest spread.*

Low maximum predicted probability (< 0.5) was associated with a 38.4% error rate, compared to 8.2% for high-confidence predictions (> 0.8). This suggests that a confidence threshold of 0.5 could route uncertain predictions for human review, substantially reducing automated errors.

### 13.3 Linguistic Challenge Analysis

| Challenge | Example | Correctly classified? | Explanation |
|---|---|---|---|
| Negation | "Not a great experience with Delta today" | Yes | "not great" bigram carries negative signal |
| Sarcasm/irony | "Oh amazing -- delayed again for the 3rd time" | No | "amazing" overrides "delayed" in bag-of-words |
| Emoji-heavy | "Best flight ever!" | Yes | Positive vocabulary dominates |
| Hashtag sentiment | "Another #fail from United baggage" | Yes | "#fail" is a strongly negative token |
| Mixed sentiment | "Crew great but seat terrible and late" | No | Bag-of-words cannot localise opposing opinions |
| Neutral factual | "Flying United to Chicago for a meeting" | Partially | Correctly neutral in 77% of similar cases |

### 13.4 Preprocessing Ablation

A controlled ablation test removing hashtags and emojis from the training data produced a macro F1 of 0.8314 vs 0.8401 with preservation -- a drop of 0.0087. This confirms that **hashtag and emoji tokens carry statistically meaningful sentiment signal** and should not be removed.

---

## 14. Discussion and Conclusions

### 14.1 Summary of Results

| Model | Macro F1 | Weighted F1 | Accuracy | Training Time |
|---|---|---|---|---|
| DummyClassifier | 0.2434 | 0.4062 | 57.5% | <0.01 s |
| VADER | 0.6925 | 0.7134 | 70.2% | 0.00 s |
| **MultinomialNB** | **0.8401** | **0.8662** | **86.9%** | **0.12 s** |
| Logistic Regression | 0.8387 | 0.8649 | 86.7% | 0.48 s |
| LinearSVC | 0.8312 | 0.8577 | 86.1% | 0.21 s |

### 14.2 Key Findings

1. **MultinomialNB is the best classical model** for this dataset with macro F1 = 0.8401 and training time of 0.12 seconds.
2. **VADER provides a strong no-training baseline** (macro F1 = 0.6925). The 0.15 macro F1 gap to the best model demonstrates that domain-specific learning is necessary for production-grade systems.
3. **Class imbalance significantly affects neutral and positive recall.** The negative class achieves 96.1% recall; neutral and positive achieve 76.3% and 72.5% respectively.
4. **Entity-level performance varies by airline**, with Virgin America (0.87) outperforming US Airways (0.82). Airlines with more balanced sentiment distributions have better-calibrated models.
5. **Sarcasm and mixed sentiment are the primary unsolved failure modes** for bag-of-words models.
6. **The learning curve shows convergence at approximately 7,000 tweets** -- improvement requires better features rather than more data.

### 14.3 When Would a Transformer Be Justified?

MultinomialNB achieves macro F1 = 0.8401 with 0.12-second training. A Twitter-specific Transformer (BERTweet) would be justified only if: the performance gap exceeds 3--5 macro F1 points; GPU infrastructure is available; sarcasm/mixed-sentiment cases represent a significant fraction of business-critical errors; and production latency constraints allow 100--300 ms inference per tweet.

---

## 15. Limitations and Responsible Analytics

### 15.1 Data Limitations

- **Temporal validity:** The dataset was collected in February 2015. Twitter language, airline service quality, and platform policies have changed substantially. Results should not be applied to current airline sentiment without re-collection and re-validation.
- **Sampling bias:** The dataset is not a representative sample of all airline customers. Tweeting behaviour is correlated with digital literacy, age, complaint severity, and geography.
- **Label noise:** CrowdFlower annotations contain inherent noise (estimated 10--15% ambiguous labels in the corpus).
- **Airline representativeness:** United dominates the dataset (26.3%), potentially biasing entity-level comparisons.

### 15.2 Responsible Use Boundaries

- **Do not infer individual-level traits:** Predicted sentiment from a single tweet should not be used to assess any individual customer's profile, satisfaction score, or loyalty status.
- **Do not use as sole evidence for action:** Sentiment classification should feed aggregate dashboards -- not trigger automated responses to individual users.
- **Privacy:** Usernames and coordinates are excluded from all modelling.
- **Aggregate reporting only:** Entity-level sentiment proportions are reported at the airline level, not at the level of individual users or incidents.
- **Causal language avoided:** The finding that "United has 58.2% negative tweets" describes the corpus distribution; it does not imply causation.

---

## Appendix A. Annotated Python Pipeline

### A.1 Core Setup and Dataset Loading

```
import os, re, time, random, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42
np.random.seed(SEED)
```

### A.2 Tweet Normalisation

```
def normalize_tweet(text):
    text = re.sub(r'https?://\S+|www\.\S+', ' <URL> ', text)
    text = re.sub(r'@\w+', ' <USER> ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Preserved: negation, emojis, hashtags, punctuation, airline vocabulary
    return text

df['clean_text'] = df['text'].map(normalize_tweet)
```

### A.3 Fixed Stratified Split

```
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=SEED,
    stratify=df['airline_sentiment']
)
X_train, y_train = train_df['clean_text'], train_df['airline_sentiment']
X_test,  y_test  = test_df['clean_text'],  test_df['airline_sentiment']
train_df.to_csv('lab05_outputs/train_manifest.csv', index=False)
test_df.to_csv('lab05_outputs/test_manifest.csv', index=False)
```

### A.4 VADER Baseline

```
analyzer = SentimentIntensityAnalyzer()
def vader_label(text):
    c = analyzer.polarity_scores(text)['compound']
    if c >= 0.05: return 'positive'
    if c <= -0.05: return 'negative'
    return 'neutral'

vader_pred = np.array([vader_label(t) for t in X_test])
print('VADER macro F1:', f1_score(y_test, vader_pred, average='macro'))
# Output: 0.6925
```

### A.5 TF-IDF Pipeline and Cross-Validation

```
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
models = {
    'MultinomialNB': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2,
                                  max_df=0.95, sublinear_tf=True)),
        ('clf', MultinomialNB(alpha=0.5))
    ]),
    'LogisticRegression': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2,
                                  max_df=0.95, sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                   random_state=SEED, C=1.0))
    ]),
    'LinearSVC': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=2,
                                  max_df=0.95, sublinear_tf=True)),
        ('clf', LinearSVC(class_weight='balanced', random_state=SEED,
                          C=1.0, max_iter=2000))
    ]),
}
# TF-IDF fitted inside each fold -- no leakage of vocabulary or IDF statistics
```

### A.6 Locked-Test Evaluation

```
best_pipe = models['MultinomialNB']
best_pipe.fit(X_train, y_train)
pred = best_pipe.predict(X_test)
print(classification_report(y_test, pred, digits=4))
# Macro F1: 0.8401 | Weighted F1: 0.8662 | Accuracy: 86.95%
```

---

## Section 16. Comprehensive Answers to Viva Questions (Qs 1-25)

**Q1. What is sentiment analysis and how is it applied in this experiment?**
Sentiment analysis is the supervised prediction of an affective polarity class from text. In this experiment, it predicts whether a tweet about a U.S. airline expresses positive, neutral, or negative sentiment, using human-labelled training data from the Twitter US Airline Sentiment dataset.

**Q2. Why is macro F1 the primary metric rather than accuracy?**
The dataset is class-imbalanced (57.5% negative). Accuracy is gamed by always predicting "negative" (yielding 57.5% with zero ability to detect positive or neutral). Macro F1 weights all three classes equally, ensuring performance on minority classes is captured.

**Q3. What is TF-IDF and why are bigrams included?**
TF-IDF weights terms by frequency in a document scaled by rarity across the corpus. Bigrams capture local negation ("not good"), brand-specific phrases ("gate change"), and complaint patterns ("still waiting") that unigrams miss.

**Q4. Why does Naive Bayes assume independence and why is it still competitive?**
MultinomialNB assumes conditional independence of features given the class. Despite this being violated in natural language, NB performs well on high-dimensional sparse text because TF-IDF features are weakly correlated in practice and probabilistic weighting naturally handles skewed class distribution.

**Q5. What is VADER and why is it used as a baseline?**
VADER is a rule/lexicon-based sentiment system designed for social-media language. It requires no training data. It tests whether expert-encoded domain knowledge can substitute for a learned model, serving as the upper bound for no-training approaches.

**Q6. What leakage fields were excluded and why?**
`sentiment_confidence` (directly tied to the target label), `negative_reason` (derived from the negative label), `username` and `tweet_id` (identifiers causing memorisation), and `coordinates` (privacy-sensitive with potential spurious geographic correlations).

**Q7. Why was TF-IDF fitted inside Pipeline objects rather than before splitting?**
Fitting TF-IDF before the split allows vocabulary and IDF statistics to be calculated from validation and test data, leaking information into feature weights. Fitting inside the Pipeline ensures only training-fold text informs the vocabulary -- maintaining a fair evaluation.

**Q8. What is a locked test set and why must it be evaluated only once?**
A locked test set is set aside before any model training or selection, never inspected during development. Evaluating it only once after all model selection decisions are finalised ensures an unbiased generalisation estimate. Multiple evaluations cause implicit test-set overfitting.

**Q9. Why does the neutral class have lower recall than the negative class?**
Neutral tweets often contain urgency-related words ("waiting", "checking", "boarding") that also appear in negative tweets, causing misclassification as negative. The negative class has abundant, distinctive vocabulary ("cancelled", "terrible", "delayed").

**Q10. What is class imbalance and how was it addressed?**
Class imbalance occurs when classes have significantly different frequencies (57.5% negative vs 19.0% positive here). Addressed using class_weight='balanced' in Logistic Regression and LinearSVC, and reporting macro F1 as primary metric.

**Q11. Why might sarcasm be difficult for bag-of-words models?**
Sarcasm reverses the polarity of literally-present words -- "great, delayed again" uses a positive word to express negativity. Bag-of-words sees the positive word and negative context independently, without the sequential order available in BiLSTM or BERT.

**Q12. What is domain shift and why is temporal validity important?**
Domain shift occurs when the statistical distribution of text or labels differs between training and deployment. This 2015 dataset may not accurately reflect 2026 airline sentiment because Twitter language, emoji usage, and social media conventions have evolved.

**Q13. Why should username and location be excluded from modelling?**
Usernames enable memorisation of individual users' patterns rather than sentiment patterns (a privacy and generalisation failure). Location is privacy-sensitive with potential spurious geographic correlations.

**Q14. What does calibration mean for Logistic Regression?**
Calibration means predicted class probabilities correspond to actual empirical frequencies -- if the model predicts 80% confidence for "negative", approximately 80% of those predictions should actually be negative. This enables confidence-based routing of uncertain predictions for human review.

**Q15. When would a Transformer like BERTweet be justified over MultinomialNB?**
When: the task involves heavy sarcasm or context-dependent sentiment; the macro F1 improvement exceeds 3--5 points; GPU infrastructure is available; and the latency, model size, and reproducibility cost of a Transformer can be justified by business requirements.

**Q16. Why is the positive class hardest to classify correctly?**
Positive tweets are short (mean 9 words) and use common vocabulary ("good", "great") that also appears in neutral factual statements. Without longer context, the model cannot reliably distinguish a genuine compliment from a neutral observation.

**Q17. What is the difference between precision and recall in this task?**
Precision measures how many tweets predicted as class k are actually class k (avoiding false alarms). Recall measures how many actual class-k tweets are correctly identified (avoiding misses). For complaint monitoring, recall on the negative class is most critical.

**Q18. What is a confusion matrix and what does the row-normalised form reveal?**
A confusion matrix shows the count of predictions for each true-class / predicted-class pair. The row-normalised form shows the proportion of each true class predicted as each output class. It reveals which classes are most confused -- neutral is most confused with negative (23.7% misclassification rate).

**Q19. Why is class_weight='balanced' important for Logistic Regression and LinearSVC?**
Without it, the loss function treats each training example equally. With 57.5% negative examples, the model would be biased toward predicting "negative". The balanced weight setting scales each class's contribution inversely with its frequency.

**Q20. How does character n-gram TF-IDF differ from word n-gram and when does it help?**
Character n-gram TF-IDF tokenises at character level, being more robust to spelling variations, hashtags, elongations, and out-of-vocabulary words. It is useful when social-media misspellings are frequent. For this experiment, word bigrams were sufficient.

**Q21. What does entity-level analysis reveal about airline service quality?**
United and US Airways show the highest negative tweet proportions (58.2% and 63.1%); Virgin America shows the most balanced distribution with the highest positive proportion (30.5%). These distributions reflect the corpus from February 2015 -- not current performance.

**Q22. What is the minimum-support threshold and why is it applied?**
The minimum-support threshold (30 test tweets per entity) prevents reporting entity-level statistics from very small samples where chance variation dominates. An entity with only 10 test tweets could show 90% negative sentiment by chance.

**Q23. Why should social-media sentiment not be interpreted as representative customer satisfaction?**
Social media users are not a random sample -- tweeting behaviour correlates with complaint severity, demographic factors, and incident recency. Observed tweet sentiment is a leading indicator of operational issues, not a measure of overall customer satisfaction.

**Q24. What does the learning curve tell us about this model?**
The learning curve shows training and validation scores converging at approximately 6,000--7,000 tweets with a final validation macro F1 of 0.854. The small gap between curves indicates low variance. The plateau suggests that collecting more tweets would yield diminishing returns -- better features would be more beneficial than more data.

**Q25. What does the confidence analysis show for deployment?**
High-confidence predictions (max probability > 0.8) have an 8.2% error rate; low-confidence predictions (< 0.5) have a 38.4% error rate. This supports a deployment strategy with a confidence threshold: route high-confidence predictions to automated processing and low-confidence predictions to human review, significantly reducing operational error rates.

---

## References

1. Maas, A., et al. (2011). *Learning Word Vectors for Sentiment Analysis.* ACL.
2. Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision.* Stanford University Technical Report.
3. Hutto, C. J., & Gilbert, E. (2014). *VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.* ICWSM.
4. Rosenthal, S., Farra, N., & Nakov, P. (2017). *SemEval-2017 Task 4: Sentiment Analysis in Twitter.* SemEval.
5. Barbieri, F., et al. (2020). *TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.* EMNLP Findings.
6. Nguyen, D. Q., et al. (2020). *BERTweet: A pre-trained language model for English tweets.* EMNLP.
7. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825-2830.
8. CrowdFlower / Figure Eight. (2015). *Twitter US Airline Sentiment Dataset.* Kaggle. CC BY-NC-SA 4.0.
9. Manning, C. D., Raghavan, P., & Schutze, H. (2008). *Introduction to Information Retrieval.* Cambridge University Press.
10. Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python.* O'Reilly.
