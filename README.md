# IS 108 – Student Performance Prediction
### Business Intelligence Predictive Modeling Application
**Final Project | SY 2025-2026**

A full-stack Business Intelligence web application that predicts whether a student will **Pass or Fail** their final exam using three machine learning algorithms: **KNN**, **SVM**, and **ANN**.

---

## Features
- 📊 **Dashboard** — Dataset overview, class distribution, grade distribution charts
- 📁 **Dataset** — Browse all 395 student records in a searchable table
- ⚙️ **Preprocessing** — Step-by-step data pipeline visualization
- 🤖 **Training** — View Colab-trained model status, parameters, and accuracy
- 📈 **Evaluation** — Compare models with accuracy, F1, precision, recall & confusion matrix
- ⚡ **Predict** — Enter student info and get Pass/Fail prediction from all 3 models

---

## Tech Stack
| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Training | Google Colab + scikit-learn |
| ML Algorithms | KNN, SVM, ANN (MLPClassifier) |
| Data | pandas, numpy |
| Frontend | HTML, Bootstrap 5, JavaScript |
| Charts | Chart.js |

---

## Dataset
[UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) (Cortez & Silva, 2008).
- **395 students**, **33 features** (demographics, grades, study habits, family background)
- **Target**: G3 ≥ 10 → Pass (1), G3 < 10 → Fail (0)

---

## Getting Started

### Requirements
- Python 3.x with pip
- A Google account (for Google Colab)

### 1 · Install Flask dependencies
```bash
pip install -r requirements.txt
```

### 2 · Train models in Google Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `student_performance_colab.ipynb`
3. Run all cells — upload `data/student-mat.csv` when prompted
4. Download the 4 generated `.pkl` files
5. Place them in the `/models/` folder of this project

```
models/
├── knn_model.pkl
├── svm_model.pkl
├── ann_model.pkl
└── scaler.pkl
```

### 3 · Run the Flask app
```bash
python app.py
```

### 4 · Open in browser
```
http://127.0.0.1:5000
```

---

## Model Results
| Model | Accuracy | F1 Score |
|---|---|---|
| **SVM** ⭐ | **89.87%** | **90.48%** |
| ANN | 82.28% | 82.93% |
| KNN | 73.42% | 75.86% |

---

## Project Structure
```
is/
├── app.py                           # Flask backend & API endpoints
├── requirements.txt                 # Python dependencies
├── student_performance_colab.ipynb  # Colab notebook for model training
├── data/
│   └── student-mat.csv              # UCI Student Performance dataset
├── models/                          # Pre-trained .pkl files (from Colab)
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── ann_model.pkl
│   └── scaler.pkl
├── static/
│   ├── css/style.css                # Custom styles (Bootstrap override)
│   └── js/app.js                    # Frontend logic & API calls
└── templates/
    └── index.html                   # Main UI (Bootstrap 5, Chart.js)
```

---

## Team
IS 108 – Intelligence System | Final Project SY 2025-2026
