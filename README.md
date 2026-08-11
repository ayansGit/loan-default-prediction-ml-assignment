# Loan Default Prediction — ML Classification

## a. Problem Statement

The goal of this project is to predict whether a credit card client will default on their payment next month based on their demographic information, credit history, and past payment behaviour. This is a binary classification problem where the target variable is `default` (1 = will default, 0 = will not default).

Accurately identifying potential defaulters helps financial institutions manage credit risk, reduce losses, and make better lending decisions.

---

## b. Dataset Description

- **Dataset:** Default of Credit Card Clients
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
- **Instances:** 30,000
- **Features:** 23 (after dropping ID column)
- **Target Variable:** `default` — whether the client defaulted next month (1 = Yes, 0 = No)
- **Class Distribution:** 23,364 non-defaulters (77.88%) and 6,636 defaulters (22.12%)

### Key Features:
| Feature | Description |
|---------|-------------|
| LIMIT_BAL | Credit limit amount |
| SEX | Gender (1=male, 2=female) |
| EDUCATION | Education level |
| MARRIAGE | Marital status |
| AGE | Age in years |
| PAY_0 to PAY_6 | Repayment status for past 6 months |
| BILL_AMT1–6 | Bill statement amount for past 6 months |
| PAY_AMT1–6 | Amount paid for past 6 months |

---

## c. GitHub Repository Link

> https://github.com/ayansGit/loan-default-prediction-ml-assignment

---

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.7178 | 0.6084 | 0.3738 | 0.4084 | 0.3903 | 0.2076 |
| KNN | 0.7928 | 0.7014 | 0.5487 | 0.3564 | 0.4322 | 0.3233 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Random Forest (Ensemble) | 0.8112 | 0.7515 | 0.6283 | 0.3580 | 0.4561 | 0.3725 |

---

### Observations on Model Performance

| ML Model Name | Observation |
|---|---|
| Logistic Regression | Achieved good accuracy (80.77%) and the highest precision (0.6868), meaning when it predicts a default, it is correct most often. However, its recall is very low (0.2396), indicating it misses many actual defaulters. Suitable when false positives are costly. |
| Decision Tree | Lowest overall performance with accuracy of 71.78% and AUC of 0.6084. It shows the most balanced precision and recall among all models but still performs poorly. Prone to overfitting and does not generalise well on this dataset. |
| KNN | Moderate performance with accuracy 79.28% and AUC 0.7014. Performs reasonably well but is computationally slow on large datasets. Sensitive to the scale of features, which is why scaling was critical here. |
| Naive Bayes | Achieved the highest recall (0.5539) and best AUC (0.7249) among models with lower accuracy. Best at catching actual defaulters, making it valuable in scenarios where missing a defaulter (false negative) is very costly. Trade-off is lower precision. |
| Random Forest (Ensemble) | Best overall model — highest accuracy (0.8112), highest AUC (0.7515), highest MCC (0.3725), and best F1 (0.4561). As an ensemble method, it reduces overfitting by combining multiple decision trees. Most reliable for this dataset. |
| **Overall Winner** | **Random Forest** — consistently outperforms all other models across Accuracy, AUC, F1, and MCC metrics. Its ensemble nature makes it robust to noise and class imbalance in this dataset. |

---

## Project Structure

```
loan-default-prediction-ml-assignement/
├── app.py               # Streamlit web application
├── train_models.py      # Model training script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── test_data.csv        # Test dataset (20% holdout)
├── loan_default.xls     # Original dataset
└── model/
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── metrics_summary.csv
```

## How to Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (generates .pkl files)
python train_models.py

# Run Streamlit app
streamlit run app.py
```
