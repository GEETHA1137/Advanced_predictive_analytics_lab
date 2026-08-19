import os
import sys
import re
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Directories
FIG_DIR = "figures"
MODEL_DIR = "models"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Add current dir to path to import local tools
sys.path.append(os.getcwd())
import tools

print("="*75)
print("LAB 03: CLASSIFICATION EXPERIMENTS (NAIVE BAYES, KNN, LSTM)")
print("="*75)

# ----------------------------------------------------------------------
# 1. Load and Preprocess Data
# ----------------------------------------------------------------------
print("\n--- 1. Loading and Preprocessing Dataset ---")
df = tools.load_data()
df = tools.clean_corpus(df)

# Text length and details
print(f"Cleaned dataset shape: {df.shape}")
print("Class counts (0 = Ham, 1 = Spam):")
print(df['label'].value_counts())

# Split data: 80% train, 20% test (stratified)
df_train, df_test = train_test_split(
    df, test_size=0.20, stratify=df['label'], random_state=RANDOM_STATE
)

X_train_text = df_train['text_cleaned'].values
y_train = df_train['label'].values
X_test_text = df_test['text_cleaned'].values
y_test = df_test['label'].values

print(f"Train split size: {len(df_train)}")
print(f"Test split size: {len(df_test)}")

# ----------------------------------------------------------------------
# 2. Feature Extraction (TF-IDF)
# ----------------------------------------------------------------------
print("\n--- 2. Vectorizing Text using TF-IDF ---")
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.98,
    sublinear_tf=True,
    max_features=20000
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

print(f"TF-IDF Feature shape: {X_train_tfidf.shape}")

# ----------------------------------------------------------------------
# 3. Model 1: Naive Bayes
# ----------------------------------------------------------------------
print("\n--- 3. Training & Tuning Naive Bayes (MultinomialNB) ---")
nb_param_grid = {'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]}
nb_clf = MultinomialNB()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

nb_grid = GridSearchCV(nb_clf, nb_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
nb_grid.fit(X_train_tfidf, y_train)

best_nb = nb_grid.best_estimator_
print(f"Best Naive Bayes parameters: {nb_grid.best_params_}")
print(f"Naive Bayes 5-Fold CV Macro F1: {nb_grid.best_score_:.4f}")

# ----------------------------------------------------------------------
# 4. Model 2: KNN
# ----------------------------------------------------------------------
print("\n--- 4. Training & Tuning K-Nearest Neighbors (KNN) ---")
knn_param_grid = {'n_neighbors': [3, 5, 7, 11, 15], 'weights': ['uniform', 'distance']}
knn_clf = KNeighborsClassifier()

knn_grid = GridSearchCV(knn_clf, knn_param_grid, cv=cv, scoring='f1_macro', n_jobs=-1)
knn_grid.fit(X_train_tfidf, y_train)

best_knn = knn_grid.best_estimator_
print(f"Best KNN parameters: {knn_grid.best_params_}")
print(f"KNN 5-Fold CV Macro F1: {knn_grid.best_score_:.4f}")

# ----------------------------------------------------------------------
# 5. Model 3: Pre-trained GloVe Embeddings + PyTorch LSTM
# ----------------------------------------------------------------------
print("\n--- 5. Preparing Deep Learning Model (LSTM) ---")

# Download GloVe file directly from HuggingFace mirror if not exists
glove_path = "glove.6B.50d.txt"
if not os.path.exists(glove_path):
    url = "https://huggingface.co/JeremiahZ/glove/resolve/main/glove.6B.50d.txt"
    print(f"Downloading pre-trained GloVe vectors (50d) from {url}...")
    try:
        def dl_progress(block_num, block_size, total_size):
            read_so_far = block_num * block_size
            if total_size > 0:
                percent = read_so_far * 100 / total_size
                print(f"\rDownload Progress: {percent:.1f}% ({read_so_far/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end="")
            else:
                print(f"\rDownload Progress: {read_so_far/(1024*1024):.1f}MB", end="")
        urllib.request.urlretrieve(url, glove_path, dl_progress)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nFailed to download GloVe from HuggingFace: {e}")
        print("Falling back to random initialization...")

# Custom Tokenizer
class CustomTokenizer:
    def __init__(self, max_vocab=20000, oov_token="<OOV>"):
        self.max_vocab = max_vocab
        self.oov_token = oov_token
        self.word_index = {}
        self.index_word = {}

    def fit_on_texts(self, texts):
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\b\w+\b', str(text).lower())
            word_counts.update(words)
        
        most_common = word_counts.most_common(self.max_vocab - 2)
        self.word_index = {self.oov_token: 1}
        for i, (word, _) in enumerate(most_common):
            self.word_index[word] = i + 2
        self.index_word = {v: k for k, v in self.word_index.items()}

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            words = re.findall(r'\b\w+\b', str(text).lower())
            seq = [self.word_index.get(w, 1) for w in words]
            sequences.append(seq)
        return sequences

def pad_sequences(sequences, maxlen=200):
    padded = []
    for seq in sequences:
        if len(seq) > maxlen:
            padded.append(seq[:maxlen])
        else:
            padded.append(seq + [0] * (maxlen - len(seq)))
    return np.array(padded)

# Fit tokenizer
MAX_VOCAB = 20000
MAX_LEN = 200
tokenizer = CustomTokenizer(max_vocab=MAX_VOCAB)
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN)

# Load pre-trained GloVe vectors
EMBED_DIM = 50
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, EMBED_DIM))

if os.path.exists(glove_path):
    print("Loading GloVe vectors into memory...")
    glove_vectors = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == EMBED_DIM + 1:
                word = parts[0]
                vector = np.asarray(parts[1:], dtype='float32')
                glove_vectors[word] = vector
    
    # Fill matrix
    for word, i in tokenizer.word_index.items():
        if word in glove_vectors:
            embedding_matrix[i] = glove_vectors[word]
        else:
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(EMBED_DIM,))
else:
    print("GloVe file not found, initializing embedding matrix randomly.")
    embedding_matrix = np.random.normal(scale=0.6, size=(vocab_size, EMBED_DIM))

# PyTorch Datasets
# Validation split for deep learning early stopping (20% of training set)
X_tr_dl, X_val_dl, y_tr_dl, y_val_dl = train_test_split(
    X_train_pad, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_STATE
)

train_dataset = TensorDataset(torch.tensor(X_tr_dl, dtype=torch.long), torch.tensor(y_tr_dl, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val_dl, dtype=torch.long), torch.tensor(y_val_dl, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test_pad, dtype=torch.long), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# LSTM Architecture
class PyTorchLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # Keep embedding fine-tuneable
        self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        pooled = torch.mean(outputs, dim=1)
        logits = self.fc(self.dropout(pooled))
        return logits.squeeze(1)

model = PyTorchLSTM(vocab_size, EMBED_DIM, embedding_matrix)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

print(f"Training on device: {device}")

# Training Loop with Early Stopping
epochs = 20
patience = 3
best_val_loss = float('inf')
patience_counter = 0
best_model_path = os.path.join(MODEL_DIR, "best_lstm.pt")

history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

for epoch in range(epochs):
    # Train Phase
    model.train()
    train_loss = 0
    correct_train = 0
    total_train = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_x.size(0)
        probs = torch.sigmoid(predictions)
        preds = (probs >= 0.5).float()
        correct_train += (preds == batch_y).sum().item()
        total_train += batch_x.size(0)
        
    train_loss /= len(train_loader.dataset)
    train_acc = correct_train / total_train
    
    # Validation Phase
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            val_loss += loss.item() * batch_x.size(0)
            probs = torch.sigmoid(predictions)
            preds = (probs >= 0.5).float()
            correct_val += (preds == batch_y).sum().item()
            total_val += batch_x.size(0)
            
    val_loss /= len(val_loader.dataset)
    val_acc = correct_val / total_val
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    # Early Stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

# Load Best Model Weights
model.load_state_dict(torch.load(best_model_path))

# ----------------------------------------------------------------------
# 6. Evaluation and Metrics on Locked Test Set
# ----------------------------------------------------------------------
print("\n--- 6. Evaluating Models on Locked Test Set ---")

# Evaluate Naive Bayes
y_pred_nb = best_nb.predict(X_test_tfidf)
y_prob_nb = best_nb.predict_proba(X_test_tfidf)[:, 1]

# Evaluate KNN
y_pred_knn = best_knn.predict(X_test_tfidf)
y_prob_knn = best_knn.predict_proba(X_test_tfidf)[:, 1]

# Evaluate LSTM
model.eval()
y_pred_lstm = []
y_prob_lstm = []

with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        predictions = model(batch_x)
        probs = torch.sigmoid(predictions)
        preds = (probs >= 0.5).float()
        
        y_prob_lstm.extend(probs.cpu().numpy())
        y_pred_lstm.extend(preds.cpu().numpy())

y_pred_lstm = np.array(y_pred_lstm)
y_prob_lstm = np.array(y_prob_lstm)

# Compute metrics dict
def get_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    }

nb_metrics = get_metrics(y_test, y_pred_nb, y_prob_nb)
knn_metrics = get_metrics(y_test, y_pred_knn, y_prob_knn)
lstm_metrics = get_metrics(y_test, y_pred_lstm, y_prob_lstm)

# Print results
metrics_df = pd.DataFrame({
    'Naive Bayes': nb_metrics,
    'KNN': knn_metrics,
    'LSTM (GloVe)': lstm_metrics
}).T
print(metrics_df.to_string())

# Save metrics CSV
metrics_df.to_csv("lab03_test_metrics.csv")
print("Saved test metrics to lab03_test_metrics.csv")

# ----------------------------------------------------------------------
# 7. Visualization Requirements
# ----------------------------------------------------------------------
print("\n--- 7. Generating Visualizations ---")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#1f4e78'
plt.rcParams['axes.linewidth'] = 1.2

# Figure 4: EDA Class Frequency
print("Generating Class Frequency Plot...")
plt.figure(figsize=(6, 5))
tools.plot_class_frequency(df)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lab03_class_frequency.png"), dpi=300)
plt.close()

# Figure 5: EDA Most Common Words
print("Generating Most Common Words Plot...")
tools.plot_most_common_words(df, 30)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lab03_most_common_words.png"), dpi=300)
plt.close()

# Figure 6: EDA Feature Distributions
print("Generating Feature Distributions Plot...")
tools.get_features(df)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lab03_feature_distributions.png"), dpi=300)
plt.close()

# Figure 1: ROC Curves
from sklearn.metrics import roc_curve
plt.figure(figsize=(8, 6))
for name, y_prob in [('Naive Bayes', y_prob_nb), ('KNN', y_prob_knn), ('LSTM (GloVe)', y_prob_lstm)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold', color='#1f4e78')
plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold', color='#1f4e78')
plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=12, fontweight='bold', color='#1f4e78')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lab03_roc_curves.png"), dpi=300)
plt.close()

# Figure 2: Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cms = [
    ('Naive Bayes', confusion_matrix(y_test, y_pred_nb)),
    ('KNN', confusion_matrix(y_test, y_pred_knn)),
    ('LSTM (GloVe)', confusion_matrix(y_test, y_pred_lstm))
]

for idx, (name, cm) in enumerate(cms):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'},
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[idx].set_title(f"{name} Confusion Matrix", fontsize=12, fontweight='bold', color='#1f4e78')
    axes[idx].set_xlabel('Predicted Label', fontsize=10, fontweight='bold', color='#1f4e78')
    axes[idx].set_ylabel('True Label', fontsize=10, fontweight='bold', color='#1f4e78')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lab03_confusion_matrices.png"), dpi=300)
plt.close()

# Figure 3: LSTM Training Curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
plt.xlabel('Epochs', fontsize=10, color='#1f4e78')
plt.ylabel('Loss', fontsize=10, color='#1f4e78')
plt.title('Loss Curve', fontsize=11, fontweight='bold', color='#1f4e78')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', linewidth=2)
plt.plot(history['val_acc'], label='Val Acc', linewidth=2)
plt.xlabel('Epochs', fontsize=10, color='#1f4e78')
plt.ylabel('Accuracy', fontsize=10, color='#1f4e78')
plt.title('Accuracy Curve', fontsize=11, fontweight='bold', color='#1f4e78')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "lab03_lstm_training.png"), dpi=300)
plt.close()

print("All figures successfully saved to 'figures/' folder!")
print("="*75)
print("Pipeline execution complete!")
print("="*75)
