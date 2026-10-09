# Customer Churn Prediction

**Author:** Shoaib ur Rehman | Quaid-i-Azam University (QAU)
**Course:** Introduction to Applied AI - Project 1: ML-Powered Customer Analytics
**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7,043 customers, 21 features

## Repository contents
- `week1-eda.ipynb` - Week 1: exploratory data analysis
- `week2-ml-models.ipynb` - Week 2: building, evaluating and interpreting ML models
- `README.md` - this file

## Week 1: Exploratory Data Analysis
- Notebook: `week1-eda.ipynb`
- Overall churn rate is about 26.5%, so the target is imbalanced.
- Month-to-month contracts churn about 43%, versus about 11% (1-year) and about 3% (2-year).
- Churned customers have much lower tenure (median about 10 months vs about 38).
- Fiber optic customers churn about 42% vs about 19% for DSL; electronic check users churn the most (about 45%).

## Week 2: Building ML Models
- Notebook: `week2-ml-models.ipynb`
- Baseline (always "stay"): accuracy 0.735, catches 0 of 374 test churners
- Best model: Logistic Regression, AUC 0.842, recall 0.920 at threshold 0.15
- Top churn drivers (permutation importance): tenure, TotalCharges, Contract_Two year
- Threshold chosen: 0.15, because a missed churner (PKR 6,000) costs six times a wasted retention offer (PKR 1,000), so t* = 0.14; this cuts total cost from PKR 1,082,000 (at 0.5) to PKR 617,000
- Engineered features: n_services, is_new, charge_per_mo, price_jump; effect on AUC: 0.8422 -> 0.8420 (no improvement)
- Biggest lesson: accuracy is misleading on imbalanced churn data, and a cost-based threshold mattered more than the choice of model (Logistic Regression and Random Forest both reach AUC 0.842).

### Model comparison (test set, 1,409 customers, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.807 | 0.658 | 0.567 | 0.609 | 0.842 |
| LR balanced | 0.739 | 0.505 | 0.781 | 0.613 | 0.841 |
| Decision Tree (d=5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| Random Forest | 0.807 | 0.673 | 0.529 | 0.593 | 0.842 |

### Setup
Open the Kaggle notebooks, or run locally:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

Then download the dataset from Kaggle and update the file path in the notebook.
