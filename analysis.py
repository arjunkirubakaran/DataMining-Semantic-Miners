import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy.sparse import load_npz
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

# For Association Rule Mining (to find the skill gaps)
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("Warning: mlxtend not installed. Skill Gap analysis will not be done.")
    print("Install with: pip install mlxtend")

# Configuration
DATA_DIR = "."
MODEL_DIR = "saved_models"
RESULT_DIR = "results"
ANALYSIS_DIR = f"{RESULT_DIR}/analysis"

os.makedirs(ANALYSIS_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\nAnalysis: Traditional and Modern ML Analysis")

# Load Data
print("\n#1 Loading Data...")

X_train = load_npz(f"{DATA_DIR}/X_tfidf_train.npz")
X_test = load_npz(f"{DATA_DIR}/X_tfidf_test.npz")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").iloc[:, 0].astype(str).values
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").iloc[:, 0].astype(str).values

print(f"Train set: {X_train.shape}")
print(f"Test set:  {X_test.shape}")
print(f"Classes:   {np.unique(y_test)}")

# Load trained models and predictions
print("\n#2 Loading Trained Models...")

models_dict = {}
predictions_dict = {}
probabilities_dict = {}

model_names = ["Random Forest", "Linear SVM", "Naive Bayes"]
model_files = ["Random_Forest", "Linear_SVM", "Naive_Bayes"]

for name, file in zip(model_names, model_files):
    try:
        model = joblib.load(f"{MODEL_DIR}/{file}.pkl")
        models_dict[name] = model
        
        # Get predictions
        preds = model.predict(X_test)
        predictions_dict[name] = preds
        
        # Get probability-like scores for ROC curves
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            probabilities_dict[name] = probs
            print(f"{name} (with probabilities)")
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            probabilities_dict[name] = scores
            print(f"{name} (with decision scores)")
        else:
            print(f"{name}")
    except Exception as e:
        print(f"   ✗ Error loading {name}: {e}")


# Load results from csv files
print("\n#3 Loading Results Summary")

results_df = pd.read_csv(f"{RESULT_DIR}/traditional_model_results.csv")
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))

# ROC Curve
print("\n#4 Generating ROC Curves")

# Binarize labels for ROC curve
classes = np.unique(y_test)
n_classes = len(classes)
y_test_bin = label_binarize(y_test, classes=classes)

fig, ax = plt.subplots(figsize=(10, 8))
model_colors = {'Random Forest': '#1f77b4', 'Linear SVM': '#ff7f0e', 'Naive Bayes': '#2ca02c'}

# Plot random classifier diagonal
ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Random Classifier')

# Compute macro-average ROC for each traditional model
for name, scores in probabilities_dict.items():
    if scores is None:
        continue
    
    # Compute ROC curve and ROC area 
    fpr_list = []
    tpr_list = []
    auc_list = []
    
    for class_idx, class_label in enumerate(classes):
        class_scores = scores[:, class_idx] if np.ndim(scores) > 1 else scores
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], class_scores)
        roc_auc = auc(fpr, tpr)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(roc_auc)
    
    # Use macro-average AUC
    macro_auc = np.mean(auc_list)
    
    # Use average FPR/TPR for visualization 
    # Interpolate to common FPR range
    all_fpr = np.unique(np.concatenate([fpr for fpr in fpr_list]))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fpr_list, tpr_list):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    
    color = model_colors.get(name, 'black')
    ax.plot(all_fpr, mean_tpr, color=color, lw=2.5, label=f'{name} (AUC = {macro_auc:.3f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Traditional Models Comparison', fontsize=14, fontweight='bold')
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/roc_curve_all_models_combined.png", dpi=300, bbox_inches='tight')
plt.close()

print("Combined ROC curves created")

# Confusion Matrix - DistilBERT Only
print("\n#5 Generating Confusion Matrix")

# Load DistilBERT predictions
distilbert_preds = None
try:
    bert_df = pd.read_csv(f"{ANALYSIS_DIR}/../bert_predictions.csv")
    distilbert_preds = bert_df['predicted_label'].values
    distilbert_labels = bert_df['true_label'].values
except Exception as e:
    print(f"Error loading DistilBERT predictions: {e}")
    distilbert_preds = None

# Create single confusion matrix for DistilBERT
if distilbert_preds is not None:
    fig, ax = plt.subplots(figsize=(8, 7))
    
    cm = confusion_matrix(distilbert_labels, distilbert_preds, labels=classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )
    
    ax.set_title('Confusion Matrix - DistilBERT', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_DIR}/confusion_matrix_distilbert.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("DistilBERT confusion matrix created")
else:
    print("DistilBERT predictions not available")

# Model Comparison
print("\n#6 Creating Model Comparisons")

# Calculate per-class metrics
comparison_data = []

if len(predictions_dict) > 0:
    for name, preds in predictions_dict.items():
        for class_label in classes:
            y_binary = (y_test == class_label).astype(int)
            pred_binary = (preds == class_label).astype(int)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_binary, pred_binary, average='binary'
            )
            
            comparison_data.append({
                'Model': name,
                'Class': class_label,
                'Precision': precision,
                'Recall': recall,
                'F1': f1
            })

    if len(comparison_data) > 0:
        comparison_df = pd.DataFrame(comparison_data)
    else:
        print("No comparison data generated for per-class metrics")
else:
    print("No models loaded, skipping detailed comparison")

#
# Skill Gap Analysis: Association Rule Mining using the Apriori Algorithm
print("\n#7 Performing Skill Gap Analysis")

if HAS_MLXTEND:
    try:
        # Load original job descriptions
        clean_data = pd.read_csv(f"{DATA_DIR}/clean_all_jobs.csv")

        if 'salary_label' not in clean_data.columns and {'max_salary', 'pay_period'}.issubset(clean_data.columns):
            multipliers = {
                'yearly': 1,
                'monthly': 12,
                'weekly': 52,
                'hourly': 2080
            }

            clean_data['annual_salary'] = clean_data['max_salary'] * clean_data['pay_period'].map(multipliers)
            clean_data = clean_data.dropna(subset=['annual_salary'])

            p33 = clean_data['annual_salary'].quantile(0.33)
            p66 = clean_data['annual_salary'].quantile(0.66)

            def bucket_salary(salary):
                if salary < p33:
                    return 'low'
                elif salary < p66:
                    return 'medium'
                else:
                    return 'high'

            clean_data['salary_label'] = clean_data['annual_salary'].apply(bucket_salary)

        def extract_skill_transactions(text_series, skill_list):
            transactions = []
            for text in text_series.fillna('').str.lower():
                skills_found = [skill for skill in skill_list if skill in text]
                if skills_found:
                    transactions.append(skills_found)
            return transactions

        def mine_and_export_rules(transactions, output_prefix, title, min_support=0.05, min_confidence=0.3, top_n=10):
            if len(transactions) == 0:
                print(f"No transactions found for {title}")
                return

            te = TransactionEncoder()
            te_ary = te.fit(transactions).transform(transactions)
            df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

            frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
            if len(frequent_itemsets) == 0:
                print(f"No frequent itemsets found for {title}")
                return

            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
            if len(rules) == 0:
                print(f"No strong association rules found for {title}")
                return

            rules = rules.sort_values('lift', ascending=False).head(top_n)

            fig, ax = plt.subplots(figsize=(12, 8))

            rule_labels = []
            for _, row in rules.iterrows():
                antecedent = ', '.join(list(row['antecedents']))
                consequent = ', '.join(list(row['consequents']))
                rule_labels.append(f"{antecedent} → {consequent}")

            y_pos = np.arange(len(rule_labels))
            lifts = rules['lift'].values
            confidences = rules['confidence'].values
            supports = rules['support'].values

            bars = ax.barh(y_pos, lifts, alpha=0.9, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(rule_labels, fontsize=10)
            if output_prefix == "skill_gap_rules":
                ax.set_xlabel('Lift (how many times more often these skills appear together in jobs)', fontsize=12)
            else:
                ax.set_xlabel('Lift', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            for i, bar in enumerate(bars):
                w = bar.get_width()
                conf = confidences[i]
                sup = supports[i]
                ax.text(w + 0.02 * max(lifts), bar.get_y() + bar.get_height()/2,
                        f"conf={conf:.2f}, sup={sup:.2f}", va='center', fontsize=9)

            plt.tight_layout()
            plt.savefig(f"{ANALYSIS_DIR}/{output_prefix}.png", dpi=300, bbox_inches='tight')
            plt.close()

            rules_export = pd.DataFrame({
                'Antecedent': rules['antecedents'].apply(lambda x: ', '.join(list(x))),
                'Consequent': rules['consequents'].apply(lambda x: ', '.join(list(x))),
                'Support': rules['support'],
                'Confidence': rules['confidence'],
                'Lift': rules['lift']
            })
            rules_export.to_csv(f"{ANALYSIS_DIR}/{output_prefix}.csv", index=False)

            print(f"Skill Gap analysis completed")
            print(f"Found {len(rules)} association rules")
        
        if 'description' in clean_data.columns:
            # Extract keywords/skills from job descriptions
            # Define common skills to track
            skills = ['python', 'java', 'sql', 'aws', 'azure', 'docker', 'kubernetes',
                     'javascript', 'react', 'node', 'machine learning', 'deep learning',
                     'data science', 'devops', 'agile', 'scrum', 'git', 'api', 'rest',
                     'microservices', 'cloud', 'distributed', 'nosql', 'mongodb']
            
            # Create transaction data (which skills are in each job description)
            transactions = extract_skill_transactions(clean_data['description'], skills)

            mine_and_export_rules(
                transactions,
                output_prefix="skill_gap_rules",
                title="Top 10 Skill Co-occurrence Rules (ordered by Lift)"
            )

            if 'salary_label' in clean_data.columns:
                skill_salary_transactions = []
                for _, row in clean_data[['description', 'salary_label']].fillna('').iterrows():
                    description = str(row['description']).lower()
                    salary_label = str(row['salary_label']).strip().lower()
                    skills_found = [skill for skill in skills if skill in description]
                    if skills_found and salary_label in {'low', 'medium', 'high'}:
                        skill_salary_transactions.append(skills_found + [f"salary_{salary_label}"])

                if len(skill_salary_transactions) > 0:
                    te = TransactionEncoder()
                    te_ary = te.fit(skill_salary_transactions).transform(skill_salary_transactions)
                    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

                    frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
                    if len(frequent_itemsets) > 0:
                        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)
                        if len(rules) > 0:
                            rules = rules[
                                rules['consequents'].apply(lambda x: any(item.startswith('salary_') for item in x)) &
                                rules['antecedents'].apply(lambda x: not any(item.startswith('salary_') for item in x))
                            ]

                            if len(rules) > 0:
                                rules = rules.sort_values('lift', ascending=False).head(10)

                                fig, ax = plt.subplots(figsize=(12, 8))
                                rule_labels = []
                                for _, row in rules.iterrows():
                                    antecedent = ', '.join(list(row['antecedents']))
                                    consequent = ', '.join(list(row['consequents']))
                                    rule_labels.append(f"{antecedent} → {consequent}")

                                y_pos = np.arange(len(rule_labels))
                                lifts = rules['lift'].values
                                confidences = rules['confidence'].values
                                supports = rules['support'].values

                                bars = ax.barh(y_pos, lifts, alpha=0.9, color='salmon')
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(rule_labels, fontsize=10)
                                ax.set_xlabel('Lift (how many times more often these skills appear with this salary range)', fontsize=12)
                                ax.set_title('Top Skill-to-Salary Association Rules (ordered by Lift)', fontsize=14, fontweight='bold')
                                ax.grid(axis='x', alpha=0.3)

                                for i, bar in enumerate(bars):
                                    w = bar.get_width()
                                    conf = confidences[i]
                                    sup = supports[i]
                                    ax.text(w + 0.02 * max(lifts), bar.get_y() + bar.get_height()/2,
                                            f"conf={conf:.2f}, sup={sup:.2f}", va='center', fontsize=9)

                                plt.tight_layout()
                                plt.savefig(f"{ANALYSIS_DIR}/skill_to_salary_rules.png", dpi=300, bbox_inches='tight')
                                plt.close()

                                rules_export = pd.DataFrame({
                                    'Antecedent': rules['antecedents'].apply(lambda x: ', '.join(list(x))),
                                    'Consequent': rules['consequents'].apply(lambda x: ', '.join(list(x))),
                                    'Support': rules['support'],
                                    'Confidence': rules['confidence'],
                                    'Lift': rules['lift']
                                })
                                rules_export.to_csv(f"{ANALYSIS_DIR}/skill_to_salary_rules.csv", index=False)

                                print("Skill to salary analysis completed")
                                print(f"Found {len(rules)} association rules")
                            else:
                                print("No skill-to-salary rules found after filtering")
                        else:
                            print("No strong skill-to-salary association rules found")
                    else:
                        print("No frequent itemsets found for skill-to-salary analysis")
                else:
                    print("No skill-to-salary transactions found")
            else:
                print("'salary_label' column not found in clean data")

            # Description length analysis by salary range
            if 'salary_label' in clean_data.columns and 'description' in clean_data.columns:
                description_length_data = clean_data[['description', 'salary_label']].dropna().copy()
                description_length_data['description_length'] = description_length_data['description'].astype(str).str.len()

                length_summary = description_length_data.groupby('salary_label')['description_length'].agg(
                    count='count',
                    mean='mean',
                    median='median',
                    std='std'
                ).reset_index()
                length_summary.to_csv(f"{ANALYSIS_DIR}/description_length_by_salary.csv", index=False)

                salary_order = ['low', 'medium', 'high']
                length_summary['salary_label'] = pd.Categorical(length_summary['salary_label'], categories=salary_order, ordered=True)
                length_summary = length_summary.sort_values('salary_label')

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(
                    length_summary['salary_label'].astype(str),
                    length_summary['mean'],
                    color=['#4c78a8', '#f58518', '#54a24b']
                )
                ax.set_title('Average Job Description Length by Salary Range', fontsize=14, fontweight='bold')
                ax.set_xlabel('Salary Range', fontsize=12)
                ax.set_ylabel('Average Description Length (characters)', fontsize=12)
                ax.grid(axis='y', alpha=0.3)

                for bar, count in zip(bars, length_summary['count']):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + max(length_summary['mean']) * 0.01,
                            f"n={int(count)}", ha='center', va='bottom', fontsize=9)

                plt.tight_layout()
                plt.savefig(f"{ANALYSIS_DIR}/description_length_by_salary.png", dpi=300, bbox_inches='tight')
                plt.close()

                print("Description length analysis completed")
                print("Saved description_length_by_salary.csv and description_length_by_salary.png")
        else:
            print("'job_description' column not found in clean data")
    except Exception as e:
        print(f"Error in skill gap analysis: {e}")
else:
    print("mlxtend not installed, skipping skill gap analysis")
    print("Install with: pip install mlxtend")

# Traditional vs Modern ML Comparison
print("\n#8 Comparing Traditional vs Modern ML")
distilbert_accuracy = 0.7007
distilbert_precision = 0.7003
distilbert_recall = 0.7007
distilbert_f1 = 0.7005
distilbert_available = True
if distilbert_available:
    # Create comparison dataframe
    trad_modern_comparison = pd.DataFrame({
        'Model': ['Random Forest', 'Linear SVM', 'Naive Bayes', 'DistilBERT'],
        'Accuracy': [
            results_df.iloc[0]['Accuracy'],
            results_df.iloc[1]['Accuracy'],
            results_df.iloc[2]['Accuracy'],
            distilbert_accuracy
        ],
        'Precision': [
            results_df.iloc[0]['Precision'],
            results_df.iloc[1]['Precision'],
            results_df.iloc[2]['Precision'],
            distilbert_precision
        ],
        'Recall': [
            results_df.iloc[0]['Recall'],
            results_df.iloc[1]['Recall'],
            results_df.iloc[2]['Recall'],
            distilbert_recall
        ],
        'F1 Score': [
            results_df.iloc[0]['F1 Score'],
            results_df.iloc[1]['F1 Score'],
            results_df.iloc[2]['F1 Score'],
            distilbert_f1
        ],
        'Type': ['Traditional', 'Traditional', 'Traditional', 'Modern']
    })
    print("Model Performance Comparison:")
    print(trad_modern_comparison.to_string(index=False))
    
    # Save comparison
    trad_modern_comparison.to_csv(f"{ANALYSIS_DIR}/traditional_vs_modern_comparison.csv", index=False)
    # Also include the combined metrics table (CSV) as an image for reports
    try:
        combined_csv = 'results/combined_metrics_table.csv'
        if os.path.exists(combined_csv):
            df_tab = pd.read_csv(combined_csv)
            # Format values for display
            df_display = df_tab.copy()
            for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

            out_img = f"{ANALYSIS_DIR}/combined_metrics_table.png"
            fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(df_display)))
            ax.axis('off')
            table = ax.table(cellText=df_display.values,
                             colLabels=df_display.columns,
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 1.3)
            plt.title('Model Metrics Comparison', pad=12)
            plt.tight_layout()
            plt.savefig(out_img, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved combined metrics table image to: {out_img}")
        else:
            print(f"Combined CSV not found at: {combined_csv}")
    except Exception as e:
        print(f"Failed to create combined metrics image: {e}")
    
print("Analysis complete!")
