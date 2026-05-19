"""
train.py  –  IS 108 Student Performance Prediction
Run this script once to train KNN, SVM, and ANN models
and save them to the /models/ directory.

Usage:
    python train.py
"""

import os
import joblib
import numpy  as np
import pandas as pd

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'student-mat.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Categorical columns ────────────────────────────────────────────
CAT_COLS = [
    'school','sex','address','famsize','Pstatus',
    'Mjob','Fjob','reason','guardian',
    'schoolsup','famsup','paid','activities',
    'nursery','higher','internet','romantic'
]

# ── 1. Load dataset ────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep=';')
print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")

# ── 2. Encode categoricals ─────────────────────────────────────────
le = LabelEncoder()
for col in CAT_COLS:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# ── 3. Create binary target ────────────────────────────────────────
df['target'] = (df['G3'] >= 10).astype(int)
feature_cols = [c for c in df.columns if c not in ('G3', 'target')]

X = df[feature_cols].values
y = df['target'].values
print(f"  Features: {len(feature_cols)}  |  Pass: {y.sum()}  |  Fail: {(1-y).sum()}")

# ── 4. Scale ───────────────────────────────────────────────────────
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 5. Train / Test split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── 6. Train models ────────────────────────────────────────────────
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', C=1.0, probability=True, random_state=42),
    'ANN': MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                         max_iter=500, early_stopping=True, random_state=42),
}

print("\nTraining models...")
for name, model in models.items():
    print(f"  Training {name}...", end=' ')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = round(accuracy_score(y_test, y_pred)  * 100, 2)
    prec = round(precision_score(y_test, y_pred, zero_division=0) * 100, 2)
    rec  = round(recall_score(y_test, y_pred, zero_division=0) * 100, 2)
    f1   = round(f1_score(y_test, y_pred, zero_division=0) * 100, 2)
    print(f"Done  |  Acc: {acc}%  Precision: {prec}%  Recall: {rec}%  F1: {f1}%")

# ── 7. Save models & scaler ────────────────────────────────────────
print("\nSaving files to /models/...")
joblib.dump(models['KNN'], os.path.join(MODEL_DIR, 'knn_model.pkl'))
joblib.dump(models['SVM'], os.path.join(MODEL_DIR, 'svm_model.pkl'))
joblib.dump(models['ANN'], os.path.join(MODEL_DIR, 'ann_model.pkl'))
joblib.dump(scaler,        os.path.join(MODEL_DIR, 'scaler.pkl'))
print("  knn_model.pkl  [OK]")
print("  svm_model.pkl  [OK]")
print("  ann_model.pkl  [OK]")
print("  scaler.pkl     [OK]")
print("\nAll done! Restart Flask (python app.py) to load the new models.")
