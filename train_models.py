import pandas as pd
import numpy as np
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

print("All libraries imported successfully!\n")

# Load the dataset
print("Loading datasets...\n")

df = pd.read_excel("loan_default.xls", header=1)

print("Dataset loaded successfully!\n")

# Rows and columns
print("Shape:", df.shape)

# Print the first 5 rows of the dataset
print(df.head())

# Data Preprocessing
print("\nData Preprocessing...\n")

#Drop the 'ID' column as it is not needed for modeling, just a row number
df = df.drop(columns=['ID'])

# Rename the target variable column to 'default' which simpler
df = df.rename(columns={'default payment next month': 'default'})

# Separate features (X) and target variable (y)
X = df.drop(columns=['default'])
y = df['default']

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("Class distribution:\n", y.value_counts())

# Split data: 80% for training, 20% for testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")
print("Scaler saved!")

# Print top 5 rows of scaled data
print("Top 5 rows of scaled training data:\n", X_train[:5])
print("Top 5 rows of scaled testing data:\n", X_test[:5])

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

print("\nData Preprocessing completed successfully!\n")

# ******* Train & Evaluate Models ********

# Define all 5 models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "KNN":                 KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes":         GaussianNB(),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42)
}

# Store results
results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_prob)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    mcc  = matthews_corrcoef(y_test, y_pred)

    # Save results
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC":      round(auc, 4),
        "Precision":round(prec, 4),
        "Recall":   round(rec, 4),
        "F1":       round(f1, 4),
        "MCC":      round(mcc, 4)
    })

    # Save the trained model
    filename = name.lower().replace(" ", "_")
    joblib.dump(model, f"model/{filename}.pkl")
    print(f"{name} done! Saved to model/{filename}.pkl")

print("\nAll models trained!")

# Save results to a CSV file
results_df = pd.DataFrame(results)

#  prints the full table in terminal without row numbers
print("\nModel Comparison:\n")
print(results_df.to_string(index=False))

results_df.to_csv("model/metrics_summary.csv", index=False)
print("\nMetrics saved to model/metrics_summary.csv")

# Save the test data as CSV

test_data = df.loc[y_test.index]
test_data.to_csv("test_data.csv", index=False)

print("Test data saved to test_data.csv")
print("Shape:", test_data.shape)