# Bank Customer Churn Prediction

This project focuses on predicting whether a customer will churn (leave the bank) using machine learning models based on customer demographics and account activity. It aims to help banks identify at-risk customers and take proactive retention measures.

## Dataset

- **Source**: [Bank Customer Churn Dataset](https://www.kaggle.com/datasets/cat-reloaded-data-science/bank-customer-churn)
- **Description**: Includes customer demographics (e.g., age, gender, geography), account details (e.g., credit score, balance, number of products), activity status, and a binary churn label indicating whether the customer has exited the bank.


## Objectives

- Perform Exploratory Data Analysis (EDA) to understand the dataset.
- Preprocess data for machine learning.
- Train and evaluate various classification models.
- Identify important features influencing churn.

## Tools & Libraries

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn (classification models, preprocessing, evaluation)

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Visualize distributions of numerical and categorical features.
- Analyze correlations and feature relationships.

### 2. Data Preprocessing
- Encode categorical variables (Label/One-Hot Encoding).
- Handle missing values (if any).
- Feature scaling using `StandardScaler`.

### 3. Model Building
Trained and compared the following models:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

### 4. Model Evaluation
Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Visualization tools:
- Confusion Matrix
- ROC Curve

## Results

- **Best Model**: XGBoost
- **Best Accuracy**: ~86%

## Conclusion

This project demonstrates a full machine learning pipeline for churn prediction. Identifying key features allows banks to take strategic actions to reduce churn and improve customer retention.

View the full notebook on [Kaggle](https://www.kaggle.com/code/khaledhellmy/bank-customer-churn-prediction).
