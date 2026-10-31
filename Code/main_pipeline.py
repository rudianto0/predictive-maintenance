import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTEENN
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.columns = [re.sub(r'[\[\]<]', '', col) for col in X_train.columns]
X_test.columns = [re.sub(r'[\[\]<]', '', col) for col in X_test.columns]

print("[*] Mengeksekusi SMOTE-ENN dan melatih model...")
smote_enn = SMOTEENN(random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)

best_xgb_params = {
    'n_estimators': 105, 'max_depth': 3, 'learning_rate': 0.021935303148955785, 
    'subsample': 0.690798672448651, 'colsample_bytree': 0.783944978570833,
    'random_state': 42, 'eval_metric': 'logloss'
}
final_model = xgb.XGBClassifier(**best_xgb_params)
final_model.fit(X_train_resampled, y_train_resampled)

print("[*] Membangun visualisasi evaluasi...\n")
y_pred = final_model.predict(X_test)
y_prob = final_model.predict_proba(X_test)[:, 1]

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Rusak (1)']))

# 1. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Rusak'], yticklabels=['Normal', 'Rusak'])
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'confusion_matrix.png'))
plt.show()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'roc_curve.png'))
plt.show()

# 3. Precision-Recall Curve (PRC)
precision, recall, _ = precision_recall_curve(y_test, y_prob)
prc_auc = average_precision_score(y_test, y_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color='blue', lw=2, label=f'PRC curve (AUC = {prc_auc:.3f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'prc_curve.png'))
plt.show()

# 4. Native Feature Importance
plt.figure(figsize=(8, 5))
xgb.plot_importance(final_model, importance_type='weight', title='XGBoost Native Feature Importance', show_values=False)
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'feature_importance.png'))
plt.show()

# 5. TreeSHAP
print("\nMembangun visualisasi TreeSHAP...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("TreeSHAP Summary Plot")
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'shap_summary.png'))
plt.show()

print(f"\n[+] 5 File visualisasi telah disimpan secara permanen di: {current_dir}")
