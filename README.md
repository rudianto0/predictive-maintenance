# Predictive Maintenance

Machine failure prediction on the AI4I 2020 Predictive Maintenance dataset using tree-based classifiers with class-imbalance handling.

## Dataset

- `DataSet/ai4i2020.csv` — [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

## Approach

- Remove identifier columns (`UDI`, `Product ID`) and label-encode `Type`
- Handle class imbalance with **SMOTE-ENN**
- Tune **XGBoost** and benchmark against Random Forest, LightGBM, and CatBoost
- Evaluate with stratified cross-validation and hold-out test set
- Interpret model behavior with **SHAP**

## Structure

```
Code/
├── main_pipeline.py        # Training, evaluation, and SHAP analysis
├── benchmark_pipeline.py   # Cross-validated model benchmark
├── benchmark_metrics.csv   # Benchmark results
└── *.png                   # Evaluation plots (ROC, PRC, confusion matrix, etc.)
DataSet/
└── ai4i2020.csv
```

## Requirements

```
pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost shap matplotlib seaborn
```

## Usage

```bash
python Code/main_pipeline.py
python Code/benchmark_pipeline.py
```
