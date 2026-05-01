# traditional_ml_.py
# Traditional ML
# Project: The Semantic Job Miner
#
# Goal:
# Train and compare traditional ML baseline models:
#   1. Random Forest
#   2. Multinomial Naive Bayes
#   3. Linear SVM
#
# Predict salary classes:
#   low / medium / high

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


# Config
DATA_DIR = "."
MODEL_DIR = "saved_models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42


# Load Data
print("Loading TF-IDF data...")

X_train = load_npz(f"{DATA_DIR}/X_tfidf_train.npz")
X_test = load_npz(f"{DATA_DIR}/X_tfidf_test.npz")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").iloc[:,0].astype(str).values
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").iloc[:,0].astype(str).values

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
print()


# Define Models
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    ),

    "Naive Bayes": MultinomialNB(alpha=0.5),

    "Linear SVM": LinearSVC(
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}


# Train & Evaluate
results = []

for name, model in models.items():

    print("=" * 60)
    print("Training:", name)

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="weighted"
    )

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Train Time: {train_time:.2f} sec")
    print()

    print("Classification Report:")
    print(classification_report(y_test, preds))

    # Save metrics
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Train Time (sec)": train_time
    })

    # Save model
    joblib.dump(model, f"{MODEL_DIR}/{name.replace(' ', '_')}.pkl")

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds, labels=["low", "medium", "high"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["low", "medium", "high"],
        yticklabels=["low", "medium", "high"]
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/{name.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

print("=" * 60)


# Save Comparison Table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)

print("\nFINAL MODEL RANKINGS")
print(results_df)

results_df.to_csv(f"{RESULT_DIR}/traditional_model_results.csv", index=False)


# Bar Chart Comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Traditional Model Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/accuracy_comparison.png")
plt.close()


print("\nSaved all outputs to:")
print("saved_models/")
print("results/")
print("\nDone.")