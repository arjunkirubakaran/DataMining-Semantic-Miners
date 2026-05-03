import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize

# For Association Rule Mining (Skill Gap)
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    print("Warning: mlxtend not installed. Skill Gap analysis will be skipped.")
    print("Install with: pip install mlxtend")

# Config
DATA_DIR = "."
MODEL_DIR = "saved_models"
RESULT_DIR = "results"
ANALYSIS_DIR = f"{RESULT_DIR}/analysis"

os.makedirs(ANALYSIS_DIR, exist_ok=True)

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("Analysis: Traditional ML and Modern ML Analysis & Visualization")
print("="*70)

# 1. Load Data
print("\n#1 Loading Data...")

X_train = load_npz(f"{DATA_DIR}/X_tfidf_train.npz")
X_test = load_npz(f"{DATA_DIR}/X_tfidf_test.npz")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").iloc[:, 0].astype(str).values
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").iloc[:, 0].astype(str).values

print(f"   Train set: {X_train.shape}")
print(f"   Test set:  {X_test.shape}")
print(f"   Classes:   {np.unique(y_test)}")

# 2. Load trained models and predictions
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
        
        # Get probabilities (if available)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)
            probabilities_dict[name] = probs
            print(f"    {name} (with probabilities)")
        else:
            print(f"    {name}")
    except Exception as e:
        print(f"   ✗ Error loading {name}: {e}")


# 3. Load results from csv files
print("\n#3 Loading Results Summary...")

results_df = pd.read_csv(f"{RESULT_DIR}/traditional_model_results.csv")
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))


# 4. Detailed Classification Reports
print("\n#4 Generating Detailed Classification Reports")

report_file = open(f"{ANALYSIS_DIR}/classification_reports.txt", "w")

for name, preds in predictions_dict.items():
    report_file.write(f"\n{'='*70}\n")
    report_file.write(f"{name.upper()}\n")
    report_file.write(f"{'='*70}\n\n")
    
    # Overall metrics
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted"
    )
    
    report_file.write(f"Overall Metrics:\n")
    report_file.write(f"  Accuracy:  {acc:.4f}\n")
    report_file.write(f"  Precision: {precision:.4f}\n")
    report_file.write(f"  Recall:    {recall:.4f}\n")
    report_file.write(f"  F1 Score:  {f1:.4f}\n\n")
    
    # Per-class metrics
    report_file.write("Classification Report (Per-Class):\n")
    report_file.write(classification_report(y_test, preds) + "\n")

report_file.close()
print(f"   Saved to: {ANALYSIS_DIR}/classification_reports.txt")

# 5. ROC Curves
print("\n#5 Generating ROC Curves...")

# Binarize labels for ROC curve
classes = np.unique(y_test)
n_classes = len(classes)
y_test_bin = label_binarize(y_test, classes=classes)

for name, probs in probabilities_dict.items():
    if probs is None:
        print(f"    {name} doesn't support probability predictions")
        continue
    
    plt.figure(figsize=(10, 8))
    
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    colors = ['blue', 'red', 'green']
    
    for i, class_label in enumerate(classes):
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        else:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], probs[:, i])
        
        roc_auc = auc(fpr, tpr)
        fpr_dict[class_label] = fpr
        tpr_dict[class_label] = tpr
        roc_auc_dict[class_label] = roc_auc
        
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_label} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = f"{ANALYSIS_DIR}/roc_curve_{name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"     {name}")

# 6. Confusion Matrices
print("\n#6 Generating Confusion Matrices...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, preds) in enumerate(predictions_dict.items()):
    cm = confusion_matrix(y_test, preds, labels=classes)
    
    # Normalize for visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=cm,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        ax=axes[idx],
        cbar_kws={'label': 'Normalized Count'}
    )
    
    axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=11)
    axes[idx].set_xlabel('Predicted Label', fontsize=11)

plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/confusion_matrices_combined.png", dpi=300, bbox_inches='tight')
plt.close()

print("     Confusion matrices saved")

# 7. Model Comparison
print("\n#7 Creating Model Comparison Visualizations...")

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

    comparison_df = pd.DataFrame(comparison_data)

    # Plot 1: Precision/Recall/F1 by Model (only if we have data)
    if len(comparison_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for idx, metric in enumerate(['Precision', 'Recall', 'F1']):
            pivot_data = comparison_df.pivot(index='Model', columns='Class', values=metric)
            
            pivot_data.plot(kind='bar', ax=axes[idx], width=0.8)
            axes[idx].set_title(f'{metric} by Model', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(metric, fontsize=11)
            axes[idx].set_xlabel('Model', fontsize=11)
            axes[idx].legend(title='Class', fontsize=10)
            axes[idx].set_ylim([0, 1.0])
            axes[idx].grid(axis='y', alpha=0.3)
            axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/model_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("     Model metrics comparison")
    else:
        print("      No predictions available for detailed comparison")
else:
    print("    No models loaded, skipping detailed comparison")

# Plot 2: Overall Accuracy/F1 Comparison
fig, ax = plt.subplots(figsize=(10, 6))

results_sorted = results_df.sort_values('Accuracy', ascending=True)
x = np.arange(len(results_sorted))
width = 0.35

ax.barh(x - width/2, results_sorted['Accuracy'], width, label='Accuracy', alpha=0.8)
ax.barh(x + width/2, results_sorted['F1 Score'], width, label='F1 Score', alpha=0.8)

ax.set_xlabel('Score', fontsize=12)
ax.set_ylabel('Model', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_xticklabels([f'{i:.1%}' for i in np.arange(0, 1.1, 0.1)])
ax.set_yticks(x)
ax.set_yticklabels(results_sorted['Model'])
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/overall_performance_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("    Overall performance comparison")


# 8. Feature Importance (Random Forest)
print("\n#8 Analyzing Feature Importance (Random Forest)...")

if 'Random Forest' in models_dict:
    rf_model = models_dict['Random Forest']
    
    if hasattr(rf_model, 'feature_importances_'):
        # Get top features
        feature_importance = rf_model.feature_importances_
        top_indices = np.argsort(feature_importance)[-20:]  # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_indices)), feature_importance[top_indices], color='steelblue')
        plt.yticks(range(len(top_indices)), [f'Feature {i}' for i in top_indices])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Random Forest - Top 20 Most Important Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/feature_importance_random_forest.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"     Top 20 features identified")


# 9. Training Time Comparison
print("\n#9 Creating Training Time Comparison...")

fig, ax = plt.subplots(figsize=(10, 6))

results_sorted = results_df.sort_values('Train Time (sec)', ascending=True)
colors_time = ['green' if x < 1 else 'orange' if x < 5 else 'red' 
               for x in results_sorted['Train Time (sec)']]

ax.barh(results_sorted['Model'], results_sorted['Train Time (sec)'], color=colors_time, alpha=0.8)
ax.set_xlabel('Training Time (seconds)', fontsize=12)
ax.set_title('Model Training Time Comparison', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (model, time) in enumerate(zip(results_sorted['Model'], results_sorted['Train Time (sec)'])):
    ax.text(time + 0.1, i, f'{time:.2f}s', va='center', fontsize=11)

plt.tight_layout()
plt.savefig(f"{ANALYSIS_DIR}/training_time_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("     Training time comparison saved")

# 10. Skill Gap Analysis: Association Rule Mining using the Apriori Algorithm
print("\n#10 Performing Skill Gap Analysis (Association Rules)...")

if HAS_MLXTEND:
    try:
        # Load original job descriptions
        clean_data = pd.read_csv(f"{DATA_DIR}/clean_all_jobs.csv")
        
        if 'job_description' in clean_data.columns:
            # Extract keywords/skills from job descriptions
            # Define common skills to track
            skills = ['python', 'java', 'sql', 'aws', 'azure', 'docker', 'kubernetes',
                     'javascript', 'react', 'node', 'machine learning', 'deep learning',
                     'data science', 'devops', 'agile', 'scrum', 'git', 'api', 'rest',
                     'microservices', 'cloud', 'distributed', 'nosql', 'mongodb']
            
            # Create transaction data (which skills are in each job description)
            transactions = []
            
            for description in clean_data['job_description'].fillna('').str.lower():
                skills_found = [skill for skill in skills if skill in description]
                if skills_found:
                    transactions.append(skills_found)
            
            if len(transactions) > 0:
                # Apply Apriori
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                
                frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
                
                if len(frequent_itemsets) > 0:
                    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
                    
                    if len(rules) > 0:
                        # Sort by lift
                        rules = rules.sort_values('lift', ascending=False).head(10)
                        
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        rule_labels = []
                        for idx, row in rules.iterrows():
                            antecedent = ', '.join(list(row['antecedents']))
                            consequent = ', '.join(list(row['consequents']))
                            label = f"{antecedent} → {consequent}"
                            rule_labels.append(label)
                        
                        y_pos = np.arange(len(rule_labels))
                        ax.barh(y_pos, rules['lift'].values, alpha=0.8, color='skyblue')
                        ax.set_yticks(y_pos)
                        ax.set_yticklabels(rule_labels, fontsize=10)
                        ax.set_xlabel('Lift', fontsize=12)
                        ax.set_title('Top 10 Skill Co-occurrence Rules (Lift)', fontsize=14, fontweight='bold')
                        ax.grid(axis='x', alpha=0.3)
                        
                        plt.tight_layout()
                        plt.savefig(f"{ANALYSIS_DIR}/skill_gap_association_rules.png", dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        # Save rules to CSV
                        rules_export = pd.DataFrame({
                            'Antecedent': rules['antecedents'].apply(lambda x: ', '.join(list(x))),
                            'Consequent': rules['consequents'].apply(lambda x: ', '.join(list(x))),
                            'Support': rules['support'],
                            'Confidence': rules['confidence'],
                            'Lift': rules['lift']
                        })
                        rules_export.to_csv(f"{ANALYSIS_DIR}/skill_gap_rules.csv", index=False)
                        
                        print("     Skill gap analysis completed")
                        print(f"     Found {len(rules)} association rules")
                    else:
                        print("     No strong association rules found")
                else:
                    print("     No frequent itemsets found")
            else:
                print("     No skill transactions found")
        else:
            print("      'job_description' column not found in clean data")
    except Exception as e:
        print(f"     ✗ Error in skill gap analysis: {e}")
else:
    print("   mlxtend not installed, skipping skill gap analysis")
    print("   Install with: pip install mlxtend")

# ============================================================================
# 11. SUMMARY REPORT
# ============================================================================

print("\n[11] Generating Summary Report...")

summary_file = open(f"{ANALYSIS_DIR}/ANALYSIS_SUMMARY.txt", "w")

summary_file.write("="*70 + "\n")
summary_file.write("Traditional ML Models Analysis\n")
summary_file.write("="*70 + "\n\n")


best_model = results_df.iloc[0]
summary_file.write(f"Best Performing Model: {best_model['Model']}\n")
summary_file.write(f"  - Accuracy:  {best_model['Accuracy']:.4f}\n")
summary_file.write(f"  - Precision: {best_model['Precision']:.4f}\n")
summary_file.write(f"  - Recall:    {best_model['Recall']:.4f}\n")
summary_file.write(f"  - F1 Score:  {best_model['F1 Score']:.4f}\n")
summary_file.write(f"  - Training Time: {best_model['Train Time (sec)']:.2f}s\n\n")

summary_file.write("ALL MODELS RANKED BY ACCURACY\n")
summary_file.write("-"*70 + "\n")
for idx, row in results_df.iterrows():
    summary_file.write(f"{idx+1}. {row['Model']:<20} Acc: {row['Accuracy']:.4f}  F1: {row['F1 Score']:.4f}\n")

summary_file.write("\n\nKEY INSIGHTS\n")
summary_file.write("-"*70 + "\n")
summary_file.write("1. Model Comparison:\n")
summary_file.write("   - Random Forest shows best overall accuracy\n")
summary_file.write("   - Linear SVM offers good balance of speed and accuracy\n")
summary_file.write("   - Naive Bayes is fastest but with lower accuracy\n\n")

summary_file.write("2. Class-wise Performance:\n")
for class_label in classes:
    class_data = comparison_df[comparison_df['Class'] == class_label]
    avg_precision = class_data['Precision'].mean()
    avg_recall = class_data['Recall'].mean()
    summary_file.write(f"   - {class_label}: Avg Precision={avg_precision:.4f}, Avg Recall={avg_recall:.4f}\n")

summary_file.write("\n3. Recommendations:\n")
summary_file.write(f"   - Deploy {best_model['Model']} as baseline for production\n")
summary_file.write("   - Consider ensemble methods for further improvement\n")
summary_file.write("   - Evaluate against BERT/RoBERTa for modern comparison\n\n")

summary_file.write("\nGENERATED VISUALIZATIONS\n")
summary_file.write("-"*70 + "\n")
summary_file.write("1. roc_curve_*.png - ROC curves for each model\n")
summary_file.write("2. confusion_matrices_combined.png - All confusion matrices\n")
summary_file.write("3. model_metrics_comparison.png - Precision/Recall/F1 by model\n")
summary_file.write("4. overall_performance_comparison.png - Accuracy vs F1\n")
summary_file.write("5. training_time_comparison.png - Training speed comparison\n")
summary_file.write("6. feature_importance_random_forest.png - Top features\n")
summary_file.write("7. skill_gap_association_rules.png - Skill co-occurrence patterns\n\n")

summary_file.write("="*70 + "\n")

summary_file.close()

print(f"   Saved summary to: {ANALYSIS_DIR}/ANALYSIS_SUMMARY.txt")

