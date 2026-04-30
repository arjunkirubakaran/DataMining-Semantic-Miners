import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz

INPUT_FILE = "clean_all_jobs.csv"
OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFIDF_MAX_FEATURES = 5000
BERT_MAX_CHARS = 400
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Load cleaned data
print("Loading data...")
df = pd.read_csv(INPUT_FILE)
print(f"  Loaded {len(df)} rows")

# Normalize all salaries to annual so they can be compared on the same scale
print("\nNormalizing salaries to annual...")

multipliers = {
    'yearly':  1,
    'monthly': 12,
    'weekly':  52,
    'hourly':  2080
}

df['annual_salary'] = df['max_salary'] * df['pay_period'].map(multipliers)

before = len(df)
df = df.dropna(subset=['annual_salary'])
print(f"  Dropped {before - len(df)} rows with unrecognized pay period")
print(f"  Annual salary - min: ${df['annual_salary'].min():,.0f} | "
      f"median: ${df['annual_salary'].median():,.0f} | "
      f"max: ${df['annual_salary'].max():,.0f}")

# Bucket salaries into low, medium, high using 33rd and 66th percentiles
# so each class has roughly equal representation
print("\nBucketing salaries...")

p33 = df['annual_salary'].quantile(0.33)
p66 = df['annual_salary'].quantile(0.66)
print(f"  Low  = under ${p33:,.0f}")
print(f"  Mid  = ${p33:,.0f} to ${p66:,.0f}")
print(f"  High = over  ${p66:,.0f}")

def bucket_salary(salary):
    if salary < p33:
        return 'low'
    elif salary < p66:
        return 'medium'
    else:
        return 'high'

df['salary_label'] = df['annual_salary'].apply(bucket_salary)
print("\n  Salary label distribution:")
print(df['salary_label'].value_counts().to_string())

# Build combined text field - title is repeated to give it more weight
print("\nBuilding combined text field...")

df['combined_text'] = (
    df['title'] + ' ' + df['title'] + ' ' + df['description']
)

print(f"  Sample (first 120 chars):")
print(f"  {df['combined_text'].iloc[0][:120]}...")

# TF-IDF vectorization for traditional models
print(f"\nBuilding TF-IDF matrix (max_features={TFIDF_MAX_FEATURES})...")

tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=3,
    stop_words='english'
)

X_tfidf = tfidf.fit_transform(df['combined_text'])
print(f"  TF-IDF matrix shape: {X_tfidf.shape}")

# BERT input - title and truncated description separated by [SEP]
# Description is capped at 400 chars to stay within BERT's 512 token limit
print("\nBuilding BERT input field...")

df['bert_input'] = (
    df['title'] + ' [SEP] ' + df['description'].str[:BERT_MAX_CHARS]
)

print(f"  Sample (first 120 chars):")
print(f"  {df['bert_input'].iloc[0][:120]}...")

# Train/test split - same indices used for both TF-IDF and BERT
# so both models are evaluated on identical data
print("\nSplitting into train/test sets...")

y = df['salary_label']
indices = np.arange(len(df))

idx_train, idx_test = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"  Train size: {len(idx_train)} | Test size: {len(idx_test)}")

X_tfidf_train = X_tfidf[idx_train]
X_tfidf_test  = X_tfidf[idx_test]

df_train = df.iloc[idx_train].reset_index(drop=True)
df_test  = df.iloc[idx_test].reset_index(drop=True)

y_train = y.iloc[idx_train].reset_index(drop=True)
y_test  = y.iloc[idx_test].reset_index(drop=True)

# Save all outputs
print(f"\nSaving outputs to '{OUTPUT_DIR}/'...")

save_npz(f"{OUTPUT_DIR}/X_tfidf_train.npz", X_tfidf_train)
save_npz(f"{OUTPUT_DIR}/X_tfidf_test.npz",  X_tfidf_test)
print("  Saved X_tfidf_train.npz and X_tfidf_test.npz")

with open(f"{OUTPUT_DIR}/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)
print("  Saved tfidf_vectorizer.pkl")

y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv",   index=False)
print("  Saved y_train.csv and y_test.csv")

df_train[['bert_input', 'salary_label']].to_csv(
    f"{OUTPUT_DIR}/bert_train.csv", index=False)
df_test[['bert_input', 'salary_label']].to_csv(
    f"{OUTPUT_DIR}/bert_test.csv",  index=False)
print("  Saved bert_train.csv and bert_test.csv")

df.to_csv(f"{OUTPUT_DIR}/jobs_enriched.csv", index=False)
print("  Saved jobs_enriched.csv")

print("\nFeature engineering complete.")
print(f"  TF-IDF shape : {X_tfidf.shape}")
print(f"  BERT rows    : {len(df)}")
print(f"  Labels       : {df['salary_label'].value_counts().to_dict()}")