Lab 03 — Email AI / Text Classification (NLP)
Registration Number: 23MID0021  
Lab: Natural Language Processing — LSTM-based Email Spam/Ham Classifier
---
Overview
This project implements a text classification pipeline to classify emails as spam or ham using:
TF-IDF features
GloVe word embeddings
LSTM deep learning model (PyTorch)
---
Project Structure
```
Lab03_EmailAI/
├── 23MID0021_Lab03_EmailAI.ipynb       # Main notebook
├── Text_classification.ipynb            # Supporting text classification notebook
├── lab03_pipeline.py                    # Full NLP pipeline script
├── create_lab03_notebook.py             # Execute notebook script
├── generate_lab03_docx_report.py        # DOCX report generator
├── generate_lab03_pdf_report.py         # PDF report generator
├── tools.py                             # Utility functions
│
├── lab03_test_metrics.csv               # Model test metrics & comparison
│
├── models/
│   └── best_lstm.pt                     # Saved best LSTM model (PyTorch)
│
└── figures/
    ├── lab03_class_frequency.png
    ├── lab03_most_common_words.png
    ├── lab03_feature_distributions.png
    ├── lab03_roc_curves.png
    ├── lab03_confusion_matrices.png
    └── lab03_lstm_training.png
```
> **Note:** GloVe embeddings file (`glove.6B.50d.txt`) is not included due to its size (~163 MB).  
> Download from: https://nlp.stanford.edu/projects/glove/ and place it in the project root.
---
How to Run
Install dependencies:
```bash
   pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
   ```
Download GloVe embeddings and place `glove.6B.50d.txt` in the project root.
Run the full pipeline:
```bash
   python lab03_pipeline.py
   ```
Or open and run the notebook:
```
   23MID0021_Lab03_EmailAI.ipynb
   ```
---
Key Results
Best LSTM model saved as `models/best_lstm.pt` (PyTorch format)
Model evaluation metrics in `lab03_test_metrics.csv`
Note: Lab03 uses a deep learning LSTM model (PyTorch `.pt`) instead of a `.joblib` sklearn model
