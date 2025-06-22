# 🛍️ Supermarket Sales Analytics Dashboard

This Streamlit app provides end-to-end analysis, clustering, and predictive modeling on the Supermarket Sales dataset. It helps business stakeholders understand customer behaviors, segment customers, and predict churn using machine learning.

## 🔧 Features

### 📄 Dataset Overview
- Data preview
- Summary info
- Missing values report
- Descriptive statistics

### 🔍 Exploratory Data Analysis
- Gender distribution
- Total purchase by gender
- Sales breakdown by product line

### 🧮 Customer Segmentation
- K-Means clustering based on Total, Quantity, and Gross Income
- Silhouette score for cluster quality
- Scatterplot visualizing cluster groups

### 📉 Churn Prediction
- Predicts churn based on `Customer type`
- Random Forest model with hyperparameter tuning
- Displays classification report, confusion matrix, and ROC AUC score

### 📌 Feature Insights
- Highlights most important features using Random Forest feature importances

## 🗂️ Dataset
Ensure the file `supermarket_sales.csv` is placed in the same directory as this app script.

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run sales.py
