# Pulmonary Disease Classification - Introduction to AI

**Author:** Nouman Ahmed  
**Course:** Introduction to AI  
**Program:** MSc Artificial Intelligence  
**Institution:** Berlin School of Business and Innovation (BSBI)

---

## 📌 Project Overview
This project is submitted as part of the "Introduction to AI" course assessment. The goal of this analysis is to build a Machine Learning pipeline to predict the **Disease Type** (e.g., COPD, Asthma, Pneumonia) based on a patient's demographic, clinical, and spirometry data.

The project covers the end-to-end data science lifecycle: from data cleaning and exploratory analysis to model training, hyperparameter tuning, and feature importance interpretation.

## 📂 Dataset
The dataset contains **12,000+ patient records** with the following key features:
* **Demographics:** Age, Gender, BMI, Smoking Status.
* **Clinical Indicators:** CRP levels, White Blood Cell count (WBC), SpO2, Respiratory Rate.
* **Spirometry Data:** FEV1, FVC, FEV1/FVC Ratio (Lung function tests).
* **Imaging:** CXR Findings (Chest X-Ray results).
* **Target Variable:** `disease_type` (Multi-class classification).

## 🛠️ Methodology

### 1. Data Preprocessing
* **Missing Value Imputation:** * Numerical features (e.g., Age) filled with **Median** to handle outliers.
    * Categorical features (e.g., Family History) filled with **Mode** or marked as 'Unknown'.
* **Encoding:** * `LabelEncoding` for the target variable.
    * `One-Hot Encoding` for categorical predictors (Sex, Smoking Status, etc.).
* **Scaling:** Applied `StandardScaler` to normalize numerical ranges for distance-based algorithms.

### 2. Exploratory Data Analysis (EDA)
* Visualized feature distributions using Histograms and Boxplots.
* Analyzed correlations using Heatmaps to detect multicollinearity.
* Investigated class imbalance in the target variable.

### 3. Model Selection
Six different machine learning algorithms were trained and compared:
1.  **Logistic Regression** (Selected as the Baseline & Best Model)
2.  Support Vector Machine (SVM)
3.  Decision Tree Classifier
4.  Random Forest Classifier
5.  Gradient Boosting Classifier
6.  K-Nearest Neighbors (KNN)

### 4. Hyperparameter Tuning
* **Method:** `RandomizedSearchCV`
* **Model Tuned:** Logistic Regression
* **Parameters Optimized:** Regularization strength (`C`), Penalty type (`l1`, `l2`), and Solver.

## wm Results & Key Findings

* **Best Performing Model:** Logistic Regression.
* **Feature Importance:** High impact features identified include:
    * `cxr_finding_Hyperinflation`
    * `cxr_finding_Infiltrate`
    * `cxr_finding_Mass/Nodule`
    * `crp_mg_L` (C-Reactive Protein)
* **Interpretation:** Chest X-Ray findings and inflammatory markers (CRP) were the strongest predictors for distinguishing between disease types.

## 💻 Dependencies
To run this project locally, ensure you have the following Python libraries installed:

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
shap
