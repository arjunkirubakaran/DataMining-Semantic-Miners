import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = 'results/combined_metrics_table.csv'
OUT_DIR = 'results/analysis'
OUT_FILE = os.path.join(OUT_DIR, 'combined_metrics_table.png')

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df_display = df.copy()
# Round numeric columns for display
for col in ['Accuracy','Precision','Recall','F1 Score']:
    if col in df_display.columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

# Create table image
fig, ax = plt.subplots(figsize=(8, 1 + 0.5 * len(df_display)))
ax.axis('off')

# Build table
table = ax.table(cellText=df_display.values,
                 colLabels=df_display.columns,
                 cellLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.3)

plt.title('Model Metrics Comparison', pad=12)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300, bbox_inches='tight')
print('Saved image to', OUT_FILE)
