Lab 02 — Decision Tree Classifier
Registration Number: 23MID0021  
Lab: Machine Learning — Decision Tree with Cross Validation & Model Comparison
---
Overview
This project implements a Decision Tree Classifier on three medical datasets:
Breast Cancer
Heart Disease
Early Stage Diabetes
It covers EDA, preprocessing, model training, CCP alpha pruning, cross validation, and comparison with other classifiers.
---
Project Structure
```
Lab02\_DecisionTree/
├── RegistrationNumber\_Lab02\_DecisionTree.ipynb   # Main notebook
├── lab02\_pipeline.py                              # Full ML pipeline script
├── create\_notebook.py                             # Execute notebook script
├── create\_datasets.py                             # Dataset generation script
├── generate\_docx\_report.py                        # DOCX report generator
├── generate\_pdf\_report.py                         # PDF report generator
│
├── Datasets
│   ├── breast\_cancer\_dataset.csv
│   ├── early\_stage\_diabetes\_dataset.csv
│   └── heart\_disease\_dataset.csv
│
├── Cross Validation Results
│   ├── RegistrationNumber\_Lab02\_CV\_Results.csv
│   └── heart\_disease\_cv\_results.csv
│
├── Model Comparison
│   ├── breast\_cancer\_model\_comparison.csv
│   ├── diabetes\_model\_comparison.csv
│   └── heart\_disease\_model\_comparison.csv
│
├── RegistrationNumber\_Lab02\_Test\_Metrics.csv      # Test set metrics
├── RegistrationNumber\_Lab02\_Model.joblib          # Saved trained model
├── RegistrationNumber\_Lab02\_Report.md             # Lab report
│
└── figures/
    ├── fig1\_class\_distribution.png
    ├── fig2\_eda\_distributions.png
    ├── fig3\_feature\_correlations.png
    ├── fig4\_unconstrained\_tree.png
    ├── fig5\_ccp\_alpha\_pruning.png
    ├── fig6\_confusion\_matrix.png
    ├── fig7\_roc\_pr\_curves.png
    ├── fig8\_pruned\_tree\_structure.png
    ├── fig9\_feature\_importance.png
    ├── fig10\_3datasets\_synthesis.png
    └── fig10\_model\_comparison.png
```
---
How to Run
Install dependencies:
```bash
   pip install scikit-learn pandas numpy matplotlib seaborn joblib
   ```
Run the full pipeline:
```bash
   python lab02\_pipeline.py
   ```
Or open and run the notebook:
```
   RegistrationNumber\_Lab02\_DecisionTree.ipynb
   ```
---
Key Results
Best model saved as `RegistrationNumber\_Lab02\_Model.joblib`
Cross validation results in `RegistrationNumber\_Lab02\_CV\_Results.csv`
Model comparison across 3 datasets in `\*\_model\_comparison.csv` files
