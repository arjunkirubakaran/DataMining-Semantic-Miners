import pandas as pd
import re

# -------------------------------
# Step 1: Load CSV
# -------------------------------
df = pd.read_csv("postings.csv")

print("Original shape:", df.shape)
print("Columns:", df.columns)


# -------------------------------
# Step 2: Fix column names (adjust if needed)
# -------------------------------
df = df.rename(columns={
    'job_title': 'title',
    'job_description': 'description',
    'salary_max': 'max_salary'
})

# Handle missing company_name
if 'company_name' not in df.columns:
    if 'company_id' in df.columns:
        df['company_name'] = df['company_id']
    else:
        df['company_name'] = "unknown"


# -------------------------------
# Step 3: Keep required columns
# -------------------------------
df = df[['company_name', 'title', 'description', 'max_salary', 'pay_period']]


# -------------------------------
# Step 4: Drop missing values
# -------------------------------
df = df.dropna(subset=['company_name', 'title', 'description'])


# -------------------------------
# Step 5: Remove duplicates
# -------------------------------
df = df.drop_duplicates(subset=['company_name', 'title', 'description'])


# -------------------------------
# Step 6: Clean text
# -------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['company_name'] = df['company_name'].apply(clean_text)
df['title'] = df['title'].apply(clean_text)
df['description'] = df['description'].apply(clean_text)


# -------------------------------
# Step 7: Clean salary
# -------------------------------
df['max_salary'] = pd.to_numeric(df['max_salary'], errors='coerce')

df = df[
    (df['max_salary'] > 1000) &
    (df['max_salary'] < 1000000)
]


# -------------------------------
# Step 8: Clean pay period
# -------------------------------
df['pay_period'] = df['pay_period'].astype(str).str.lower()

valid_periods = ['yearly', 'monthly', 'weekly', 'hourly']
df = df[df['pay_period'].isin(valid_periods)]


# -------------------------------
# Step 9: Remove bad descriptions
# -------------------------------
df = df[df['description'].str.len() > 100]

df = df[~df['description'].str.contains(
    'click here|apply now|subscribe|sign up',
    case=False,
    na=False
)]

df = df[df['description'].str.split().str.len() < 1000]


# -------------------------------
# ✅ FINAL STEP: Save ALL cleaned rows
# -------------------------------
print("Final cleaned shape:", df.shape)

df.to_csv( "clean_all_jobs.csv", index=False, encoding='utf-8', quoting=1 )

print("✅ Saved all cleaned rows to clean_all_jobs.csv")