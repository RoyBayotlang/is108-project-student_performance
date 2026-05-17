"""
app.py  –  IS 108 Student Performance Prediction
Flask backend that:
  1. Loads & preprocesses student-mat.csv
  2. Loads pre-trained .pkl models exported from Google Colab
  3. Exposes REST API endpoints consumed by the frontend

Models are trained in Google Colab (student_performance_colab.ipynb)
and placed in the /models/ directory before running this app.
"""

import os, json
import numpy  as np
import pandas as pd
import joblib

from flask            import Flask, jsonify, render_template, request
from flask_cors       import CORS

from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score, confusion_matrix)

# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'student-mat.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Categorical columns that need label-encoding
CAT_COLS = [
    'school','sex','address','famsize','Pstatus',
    'Mjob','Fjob','reason','guardian',
    'schoolsup','famsup','paid','activities',
    'nursery','higher','internet','romantic'
]

# Fixed label-encoding maps (same order as fit_transform on full data)
# These mirror what scikit-learn LabelEncoder produces alphabetically.
LABEL_MAPS = {
    'school'    : {'GP': 0, 'MS': 1},
    'sex'       : {'F': 0, 'M': 1},
    'address'   : {'R': 0, 'U': 1},
    'famsize'   : {'GT3': 0, 'LE3': 1},
    'Pstatus'   : {'A': 0, 'T': 1},
    'Mjob'      : {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'Fjob'      : {'at_home': 0, 'health': 1, 'other': 2, 'services': 3, 'teacher': 4},
    'reason'    : {'course': 0, 'home': 1, 'other': 2, 'reputation': 3},
    'guardian'  : {'father': 0, 'mother': 1, 'other': 2},
    'schoolsup' : {'no': 0, 'yes': 1},
    'famsup'    : {'no': 0, 'yes': 1},
    'paid'      : {'no': 0, 'yes': 1},
    'activities': {'no': 0, 'yes': 1},
    'nursery'   : {'no': 0, 'yes': 1},
    'higher'    : {'no': 0, 'yes': 1},
    'internet'  : {'no': 0, 'yes': 1},
    'romantic'  : {'no': 0, 'yes': 1},
}

# ── Global state ──────────────────────────────────────────────────────────────
_state = {
    'df'      : None,   # raw dataframe
    'X_train' : None,
    'X_test'  : None,
    'y_train' : None,
    'y_test'  : None,
    'scaler'  : None,
    'models'  : {},     # {'KNN': ..., 'SVM': ..., 'ANN': ...}
    'metrics' : {},
    'feature_cols': [],
}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path=None):
    """Load CSV and store raw dataframe."""
    path = path or DATA_PATH
    df = pd.read_csv(path, sep=';')
    _state['df'] = df
    return df


def preprocess():
    """
    Encode categoricals, create binary target, scale features,
    split 80/20. Returns (X_train, X_test, y_train, y_test).
    """
    df = _state['df'].copy()

    # ── Encode categorical columns ────────────────────────────────────────────
    le = LabelEncoder()
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])

    # ── Binary target: Pass (1) = G3 >= 10, Fail (0) = G3 < 10 ──────────────
    df['target'] = (df['G3'] >= 10).astype(int)

    # ── Feature columns (everything except raw G3 and new target) ─────────────
    feature_cols = [c for c in df.columns if c not in ('G3', 'target')]
    _state['feature_cols'] = feature_cols

    X = df[feature_cols].values
    y = df['target'].values

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    _state['scaler'] = scaler

    # ── Split 80 / 20 ─────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    _state['X_train'] = X_train
    _state['X_test']  = X_test
    _state['y_train'] = y_train
    _state['y_test']  = y_test

    return X_train, X_test, y_train, y_test


def compute_metrics(model, X_test, y_test, name):
    """Return dict of evaluation metrics for a single model."""
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred).tolist()
    return {
        'name'      : name,
        'accuracy'  : round(accuracy_score (y_test, y_pred) * 100, 2),
        'precision' : round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
        'recall'    : round(recall_score   (y_test, y_pred, zero_division=0) * 100, 2),
        'f1'        : round(f1_score       (y_test, y_pred, zero_division=0) * 100, 2),
        'confusion_matrix': cm,
    }



def load_pretrained_models():
    """Load .pkl models saved from Colab (or previous local training)."""
    files = {
        'KNN': 'knn_model.pkl',
        'SVM': 'svm_model.pkl',
        'ANN': 'ann_model.pkl',
    }
    loaded = []
    for name, fname in files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            _state['models'][name] = joblib.load(path)
            loaded.append(name)

    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    if os.path.exists(scaler_path):
        _state['scaler'] = joblib.load(scaler_path)

    return loaded


# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP  – load data + try pre-trained models automatically
# ─────────────────────────────────────────────────────────────────────────────
def startup():
    if os.path.exists(DATA_PATH):
        load_dataset()
        preprocess()
        loaded = load_pretrained_models()
        if loaded and _state['X_test'] is not None:
            for name, model in _state['models'].items():
                _state['metrics'][name] = compute_metrics(
                    model, _state['X_test'], _state['y_test'], name
                )
        print(f"[OK] Dataset loaded  |  Models found: {loaded or 'none (train first)'}")
    else:
        print("[WARN] student-mat.csv not found -- run: python data/generate_data.py")

startup()

# ─────────────────────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


# ── Dataset ──────────────────────────────────────────────────────────────────

@app.route('/api/dataset', methods=['GET'])
def api_dataset():
    """Return first 100 rows of the raw dataset as JSON."""
    if _state['df'] is None:
        return jsonify({'error': 'Dataset not loaded'}), 404
    df = _state['df']
    return jsonify({
        'columns': df.columns.tolist(),
        'rows'   : df.head(100).values.tolist(),
        'total'  : len(df),
    })


@app.route('/api/dataset/info', methods=['GET'])
def api_dataset_info():
    """Return dataset statistics and class distribution."""
    if _state['df'] is None:
        return jsonify({'error': 'Dataset not loaded'}), 404
    df = _state['df']
    pass_count = int((df['G3'] >= 10).sum())
    fail_count = int((df['G3'] < 10).sum())

    # Missing values per column
    missing = {col: int(df[col].isnull().sum()) for col in df.columns}

    return jsonify({
        'rows'       : len(df),
        'columns'    : len(df.columns),
        'pass_count' : pass_count,
        'fail_count' : fail_count,
        'missing'    : missing,
        'dtypes'     : {col: str(df[col].dtype) for col in df.columns},
        'g3_mean'    : round(float(df['G3'].mean()), 2),
        'g3_std'     : round(float(df['G3'].std()),  2),
    })


@app.route('/api/dataset/upload', methods=['POST'])
def api_upload():
    """Accept a CSV upload, replace current dataset."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    path = os.path.join(BASE_DIR, 'data', 'student-mat.csv')
    f.save(path)
    load_dataset(path)
    preprocess()
    _state['models'] = {}
    _state['metrics'] = {}
    return jsonify({'message': 'Dataset uploaded and preprocessed successfully.'})


# ── Preprocessing ─────────────────────────────────────────────────────────────

@app.route('/api/preprocess', methods=['GET'])
def api_preprocess():
    """Return preprocessing summary."""
    if _state['df'] is None:
        return jsonify({'error': 'Dataset not loaded'}), 404
    df    = _state['df']
    total = len(df)
    train_n = len(_state['X_train']) if _state['X_train'] is not None else 0
    test_n  = len(_state['X_test'])  if _state['X_test']  is not None else 0

    return jsonify({
        'total_rows'     : total,
        'train_rows'     : train_n,
        'test_rows'      : test_n,
        'missing_total'  : int(df.isnull().sum().sum()),
        'categorical_cols': CAT_COLS,
        'numeric_cols'   : [c for c in df.columns if c not in CAT_COLS],
        'feature_count'  : len(_state['feature_cols']),
        'target_col'     : 'G3 → binary (Pass/Fail at G3 ≥ 10)',
        'scaler'         : 'StandardScaler',
        'split'          : '80% train / 20% test (stratified)',
        'label_maps'     : LABEL_MAPS,
    })


# ── Model Status ──────────────────────────────────────────────────────────────

@app.route('/api/model-status', methods=['GET'])
def api_model_status():
    """Return which models are currently loaded and their metrics."""
    model_files = {
        'KNN': 'knn_model.pkl',
        'SVM': 'svm_model.pkl',
        'ANN': 'ann_model.pkl',
    }
    status = {}
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        loaded = name in _state['models']
        file_exists = os.path.exists(path)
        size = os.path.getsize(path) if file_exists else 0
        status[name] = {
            'loaded'     : loaded,
            'file_exists': file_exists,
            'file_size'  : size,
            'metrics'    : _state['metrics'].get(name),
        }
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    return jsonify({
        'models'      : status,
        'scaler_ready': os.path.exists(scaler_path),
        'all_loaded'  : len(_state['models']) == 3,
    })


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.route('/api/evaluate', methods=['GET'])
def api_evaluate():
    """Return evaluation metrics for all trained models."""
    if not _state['metrics']:
        return jsonify({'error': 'No models trained yet. Click Train Models first.'}), 404
    return jsonify(_state['metrics'])


# ── Prediction ────────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Accept a JSON body with student feature values,
    return Pass/Fail prediction + probability from each model.
    """
    if not _state['models']:
        return jsonify({'error': 'No models available. Train first.'}), 400
    if _state['scaler'] is None:
        return jsonify({'error': 'Scaler not loaded.'}), 400

    data = request.json  # dict of feature_name -> value

    # Build a single-row dataframe in the same column order used during training
    feature_cols = _state['feature_cols']
    row = {}
    for col in feature_cols:
        val = data.get(col, 0)
        # Apply label encoding for categorical cols
        if col in LABEL_MAPS:
            val = LABEL_MAPS[col].get(str(val), 0)
        row[col] = float(val)

    X_input = np.array([[row[c] for c in feature_cols]])
    X_scaled = _state['scaler'].transform(X_input)

    results = {}
    for name, model in _state['models'].items():
        pred  = int(model.predict(X_scaled)[0])
        label = 'Pass' if pred == 1 else 'Fail'
        try:
            proba = model.predict_proba(X_scaled)[0]
            conf  = round(float(max(proba)) * 100, 1)
            pass_prob = round(float(proba[1]) * 100, 1)
        except Exception:
            conf      = 100.0
            pass_prob = 100.0 if pred == 1 else 0.0
        results[name] = {
            'prediction' : label,
            'label'      : pred,
            'confidence' : conf,
            'pass_prob'  : pass_prob,
        }

    # Majority vote
    votes     = [v['label'] for v in results.values()]
    majority  = 'Pass' if sum(votes) >= 2 else 'Fail'

    return jsonify({'models': results, 'majority': majority})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
