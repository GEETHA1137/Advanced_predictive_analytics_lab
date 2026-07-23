# House Price Prediction using Machine Learning

**Student Name:** Geetha Priya S  
**Register Number:** 23MID0021  
**Course:** Advanced Predictive Analytics  

---

## Project Overview

This project was completed as part of the Advanced Predictive Analytics Laboratory course. The objective was to compare different regression algorithms for predicting house prices using three publicly available datasets.

The project follows a complete machine learning workflow, including data preprocessing, exploratory data analysis, model training, model evaluation, cross-validation, and saving the best-performing models. All experiments were implemented in Python using Scikit-learn in Google Colab.

---

## Datasets

### 1. Ames Housing

The Ames Housing dataset contains information about residential properties sold in Ames, Iowa.

**Target Variable:** SalePrice

---

### 2. California Housing

The California Housing dataset contains demographic and geographical information collected from the California census.

**Target Variable:** MedHouseVal

---

### 3. UCI Real Estate Valuation

The UCI Real Estate dataset contains real estate transaction records from Taiwan.

**Target Variable:** House Price of Unit Area

---

## Workflow

The following workflow was followed for each dataset.

- Load the dataset
- Explore the data
- Handle missing values
- Perform Exploratory Data Analysis (EDA)
- Build preprocessing pipeline
- Split into training and testing datasets
- Train multiple regression models
- Compare model performance
- Perform cross-validation
- Save the best model

---

## Regression Models

The following models were implemented.

- Dummy Regressor
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression
- Polynomial Regression

---

## Evaluation Metrics

The following metrics were used for evaluation.

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- 5-Fold Cross Validation

---

## Repository Structure

```
23MID0021_Lab01/

│
├── 23MID0021_Lab01_Ames.ipynb
├── 23MID0021_Lab01_California.ipynb
├── 23MID0021_Lab01_UCI.ipynb
│
├── ames_house_price_pipeline.joblib
├── ames_model_comparison.csv
├── ames_cross_validation.csv
│
├── California_house_price_pipeline.joblib
├── California_model_comparison.csv
├── California_cross_validation_results.csv
│
├── uci_real_estate_model.joblib
├── uci_model_comparison.csv
├── uci_cross_validation.csv
│
├── execute_notebooks.py
│
└── 23MID0021_Lab01_README.md
```

---

## File Description

### Ames Housing

- `23MID0021_Lab01_Ames.ipynb` – Complete notebook for the Ames Housing dataset.
- `ames_house_price_pipeline.joblib` – Saved trained model.
- `ames_model_comparison.csv` – Performance comparison of all regression models.
- `ames_cross_validation.csv` – Cross-validation results.

### California Housing

- `23MID0021_Lab01_California.ipynb` – Complete notebook for the California Housing dataset.
- `California_house_price_pipeline.joblib` – Saved trained Random Forest model.
- `California_model_comparison.csv` – Model comparison results.
- `California_cross_validation_results.csv` – Cross-validation results.

### UCI Real Estate

- `23MID0021_Lab01_UCI.ipynb` – Complete notebook for the UCI Real Estate dataset.
- `uci_real_estate_model.joblib` – Saved Random Forest model.
- `uci_model_comparison.csv` – Model comparison results.
- `uci_cross_validation.csv` – Cross-validation results.

---

## Software Requirements

Python 3.11+

Required Libraries

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

Install them using:

```bash
pip install numpy pandas matplotlib scikit-learn joblib
```

---

## Running the Project

### Using Google Colab

1. Open the required notebook.
2. Upload the dataset.
3. Run all cells sequentially.

### Using Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook and run all cells.

### Execute All Notebooks

```bash
python execute_notebooks.py
```

---

## Output Files

Running the notebooks generates:

- Trained machine learning models (.joblib)
- Model comparison tables (.csv)
- Cross-validation results (.csv)
- Data visualizations
- Performance metrics

---

## Acknowledgements

The datasets used in this project are publicly available and were used only for academic purposes.

---

## Author

**Geetha Priya S**

Register Number: **23MID0021**

Vellore Institute of Technology
