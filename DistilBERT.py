# DistilBERT.py
# Project: The Semantic Job Miner

import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Config
DATA_DIR = "."
MODEL_DIR = "saved_models/distilbert_learning_rate_2e-5"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# 1. Load Data
print("\nLoading BERT-ready data...")
# Read the CSVs and instantly drop any entirely blank rows
train_df = pd.read_csv(f"{DATA_DIR}/bert_train.csv").dropna(how='all')
test_df = pd.read_csv(f"{DATA_DIR}/bert_test.csv").dropna(how='all')

# Clean up the label column strings (lowercase and remove extra spaces)
label_map = {"low": 0, "medium": 1, "high": 2}
train_df['clean_label'] = train_df['salary_label'].astype(str).str.lower().str.strip().map(label_map)
test_df['clean_label'] = test_df['salary_label'].astype(str).str.lower().str.strip().map(label_map)

# Drop any rows where the text is missing or the label map failed
train_df = train_df.dropna(subset=['clean_label', 'bert_input'])
test_df = test_df.dropna(subset=['clean_label', 'bert_input'])

# Extract the final clean lists
train_texts = train_df['bert_input'].astype(str).tolist()
test_texts = test_df['bert_input'].astype(str).tolist()

train_labels = train_df['clean_label'].astype(int).tolist()
test_labels = test_df['clean_label'].astype(int).tolist()

# 2. Tokenize Text
print("Tokenizing text...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Convert text into numerical vectors that PyTorch can read
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

# 3. Create PyTorch Dataset & DataLoader
class JobDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Force the labels to be integers (torch.long)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = JobDataset(train_encodings, train_labels)
test_dataset = JobDataset(test_encodings, test_labels)

# DataLoaders handle feeding the data in batches during training
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 4. Setup PyTorch Model & Optimizer
print("Initializing DistilBERT...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)

# AdamW is the standard optimizer for Transformer models
optim = AdamW(model.parameters(), lr=2e-5)

# 5. The PyTorch Training Loop
epochs = 3
print("=" * 60)
print("Training: DistilBERT (PyTorch implementation)")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch data to GPU/CPU
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass: get predictions and loss
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass: calculate gradients and update weights
        loss.backward()
        optim.step()

        # Print progress every 100 batches
        if batch_idx % 100 == 0 and batch_idx != 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f} sec. Average Loss: {total_loss/len(train_loader):.4f}\n")

# 6. Evaluation Loop
print("Evaluating model...")
model.eval()
all_preds = []
all_labels = []

# Turn off gradients for validation to save memory and speed up
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # Get the highest probability class
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate Metrics
acc = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save predictions to CSV for analysis
try:
    inv_label_map = {0: 'low', 1: 'medium', 2: 'high'}
    pred_labels = [inv_label_map[int(p)] for p in all_preds]
    true_labels = [inv_label_map[int(t)] for t in all_labels]
    pred_df = pd.DataFrame({
        'bert_input': test_texts,
        'true_label': true_labels,
        'predicted_label': pred_labels
    })
    pred_file = f"{RESULT_DIR}/bert_predictions.csv"
    pred_df.to_csv(pred_file, index=False)
    print(f"Saved BERT predictions to: {pred_file}")
except Exception as e:
    print(f"Could not save BERT predictions: {e}")

# 7. Save Model & Outputs
print("\nSaving PyTorch model...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt="d", 
    cmap="Purples", 
    xticklabels=["low", "medium", "high"], 
    yticklabels=["low", "medium", "high"]
)
plt.title("DistilBERT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/DistilBERT_confusion_matrix.png")
plt.close()

print("=" * 60)
print("Saved model to:", MODEL_DIR)
print("Saved confusion matrix to:", RESULT_DIR)
print("\nDone.")