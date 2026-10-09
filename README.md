# Customer Churn Prediction

**Author:** Shoaib ur Rehman | Quaid-i-Azam University (QAU)
**Course:** Introduction to Applied AI - Project 1: ML-Powered Customer Analytics

## Week 1: Exploratory Data Analysis

### Dataset
- Source: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Size: 7,043 customers, 21 features
- Target: Predict customer churn (Yes/No)

### Notebook
- `Week_1_Customer_Churn_EDA.ipynb` - full EDA with 7 visualizations (tenure, monthly charges, total charges, contract type, internet service, payment method, correlation heatmap) and written insights.

### Key Findings
- Overall churn rate is about 26.5% (1,869 of 7,043 customers), so the target is imbalanced.
- Month-to-month contracts churn about 43%, versus about 11% (1-year) and about 3% (2-year).
- Churned customers have much lower tenure (median about 10 months vs about 38 for retained customers).
- Fiber optic customers churn about 42%, versus about 19% for DSL.
- Electronic check users churn the most (about 45%); automatic payment methods churn far less.
- `TotalCharges` had 11 hidden missing values (all customers with tenure 0) and is strongly correlated with `tenure`.

### Setup
Open the Kaggle notebook or run locally:

```
pip install pandas numpy matplotlib seaborn
```

Then download the dataset from Kaggle and update the file path in the notebook.
