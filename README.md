# 📉 Telco Customer Churn Prediction

This project focuses on analyzing customer behavior and predicting churn in a telecommunications company using machine learning models.

## 🗂️ Project Overview

Customer churn refers to the phenomenon where clients stop doing business with a company. Retaining customers is critical in subscription-based businesses like telecom. This project uses a real-world dataset to:

- Analyze customer demographics and usage patterns through comprehensive visualizations
- Build and evaluate machine learning models to predict customer churn

## 📊 Dataset

The dataset is sourced from the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn), and includes features such as:

- Customer demographics (e.g., gender, SeniorCitizen)
- Service usage (e.g., InternetService, StreamingTV)
- Account information (e.g., tenure, MonthlyCharges, TotalCharges)
- Churn label (`Yes`/`No`)

## 🧪 Workflow

1. **Data Loading & Cleaning**
   - Load data from CSV file
   - Remove customerID column
   - Convert TotalCharges to numeric type
   - Handle missing values

2. **Exploratory Data Analysis (EDA)**
   - Univariate Analysis:
     - Distribution plots for categorical variables using pie charts
     - Histograms and box plots for numerical variables
   - Bivariate Analysis:
     - Correlation heatmap of all features
   - Data type inspection and summary statistics

3. **Data Preprocessing**
   - Label encoding for categorical variables
   - Feature scaling using StandardScaler for numerical features
   - Train-test split (80-20)

4. **Modeling**
   - Support Vector Machine (SVC)
   - Random Forest Classifier
   - Naive Bayes (GaussianNB)

## 📈 Results

| Model            | Precision Score |
|------------------|-----------------|
| SVM              | 78%             | 
| Random Forest    | 77%             | 
| Naive Bayes      | 78%             | 


## 🛠️ Libraries Used

- Data Manipulation: `pandas`, `numpy`
- Visualization: `matplotlib`, `seaborn`
- Machine Learning: `scikit-learn`
  - `StandardScaler` for feature scaling
  - `LabelEncoder` for categorical encoding
  - `train_test_split` for data splitting
  - Various classification models

## 🚀 How to Run

1. Clone the repository
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Run the notebook:
   ```bash
   jupyter notebook telco-churn-prediction.ipynb
   ```

## 📌 Future Improvements

- Implement additional machine learning models
- Perform hyperparameter tuning
- Add model interpretability analysis
