import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = '/content/drive/MyDrive/predictive_maintenance/Code'

dataset_path = os.path.abspath(os.path.join(current_dir, '..', 'DataSet', 'ai4i2020.csv'))
print(f"[*] Membaca dataset dari: {dataset_path}")

df = pd.read_csv(dataset_path)
df_cleaned = df.drop(columns=['UDI', 'Product ID'])

le = LabelEncoder()
df_cleaned['Type'] = le.fit_transform(df_cleaned['Type'])
X = df_cleaned.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
y = df_cleaned['Machine failure']
X.columns = [re.sub(r'[\[\]<]', '', col) for col in X.columns]

best_xgb_params = {
    'n_estimators': 105, 'max_depth': 3, 'learning_rate': 0.021935303148955785, 
    'subsample': 0.690798672448651, 'colsample_bytree': 0.783944978570833,
    'random_state': 42, 'eval_metric': 'logloss'
}

scenarios = {
    "1. Baseline (Raw XGB)": ImbPipeline(steps=[
        ('model', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
    ]),
    "2. RF + SMOTE (Standar)": ImbPipeline(steps=[
        ('resampler', SMOTE(random_state=42)),
        ('model', RandomForestClassifier(random_state=42))
    ]),
    "3. LGBM + SMOTE (Literatur 1)": ImbPipeline(steps=[
        ('resampler', SMOTE(random_state=42)),
        ('model', lgb.LGBMClassifier(random_state=42, verbose=-1))
    ]),
    "4. XGB + SMOTE (Literatur 3)": ImbPipeline(steps=[
        ('resampler', SMOTE(random_state=42)),
        ('model', xgb.XGBClassifier(random_state=42, eval_metric='logloss'))
    ]),
    "5. CatBoost + SMOTE-ENN (Literatur 2)": ImbPipeline(steps=[
        ('resampler', SMOTEENN(random_state=42)),
        ('model', CatBoostClassifier(random_state=42, verbose=0))
    ]),
    "6. Usulan Utama (Optuna XGB + SMOTE-ENN)": ImbPipeline(steps=[
        ('resampler', SMOTEENN(random_state=42)),
        ('model', xgb.XGBClassifier(**best_xgb_params))
    ])
}

# Penambahan Precision, Accuracy, dan Balanced Accuracy pada K-Fold
scoring_metrics = {
    'Recall': 'recall', 
    'Precision': 'precision',
    'F1': 'f1', 
    'Accuracy': 'accuracy',
    'Balanced_Acc': 'balanced_accuracy',
    'ROC_AUC': 'roc_auc', 
    'PR_AUC': 'average_precision'
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n[*] Menjalankan 5-Fold Cross Validation dengan Metrik Komprehensif (Estimasi waktu: 3-5 menit)...")
results = []
cv_recall_data = []

for name, pipeline in scenarios.items():
    print(f"    -> Memproses: {name}...")
    cv_scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring_metrics, return_train_score=False)
    
    for score in cv_scores['test_Recall']:
        cv_recall_data.append({'Model': name, 'Recall Score': score})
        
    results.append({
        "Model / Skenario": name,
        "Recall": np.mean(cv_scores['test_Recall']),
        "Precision": np.mean(cv_scores['test_Precision']),
        "F1-Score": np.mean(cv_scores['test_F1']),
        "Accuracy": np.mean(cv_scores['test_Accuracy']),
        "Balanced Acc": np.mean(cv_scores['test_Balanced_Acc']),
        "ROC AUC": np.mean(cv_scores['test_ROC_AUC']),
        "PR AUC": np.mean(cv_scores['test_PR_AUC']),
        "Train Time (s)": np.mean(cv_scores['fit_time']),
        "Inference Time (s)": np.mean(cv_scores['score_time'])
    })

results_df = pd.DataFrame(results)
csv_path = os.path.join(current_dir, 'benchmark_metrics.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n[V] Tabel matriks CSV disimpan di: {csv_path}")

print("\n--- HASIL BENCHMARK LENGKAP (Rata-rata 5 Lipatan) ---")
# Menampilkan kolom-kolom utama ke layar agar mudah dibaca
print(results_df[['Model / Skenario', 'Recall', 'Precision', 'F1-Score', 'Balanced Acc', 'ROC AUC']].to_string(index=False))

# Visualisasi 1: Boxplot Stabilitas Recall (K-Fold)
plt.figure(figsize=(14, 6)) 
sns.boxplot(data=pd.DataFrame(cv_recall_data), x='Recall Score', y='Model', palette='Set2')
plt.title('Stabilitas Metrik Recall (5-Fold Cross Validation)')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'benchmark_cv_boxplot.png'))
plt.show()

# Visualisasi 2: Barplot Latensi Waktu
fig, ax1 = plt.subplots(figsize=(14, 6)) 
ax2 = ax1.twinx()
results_df.plot(x='Model / Skenario', y='Train Time (s)', kind='bar', ax=ax1, color='lightgray', position=1, width=0.3, label='Train Time')
results_df.plot(x='Model / Skenario', y='Inference Time (s)', kind='bar', ax=ax2, color='orange', position=0, width=0.3, label='Inference Time')
ax1.set_ylabel('Training Time (detik)')
ax2.set_ylabel('Inference Time (detik)')
plt.title('Komparasi Waktu Komputasi (Latensi)')
plt.xticks(rotation=15, ha='right')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'benchmark_latency.png'))
plt.show()

print(f"\n[+] Seluruh evaluasi benchmark telah selesai dan disimpan di: {current_dir}")
