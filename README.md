# IS 108 – Student Performance Prediction
### Business Intelligence Predictive Modeling Application
**Final Project | SY 2025-2026**

A full-stack Business Intelligence web application that predicts whether a student will **Pass or Fail** their final exam using three machine learning algorithms: **KNN**, **SVM**, and **ANN**.

---

## Features
- 📊 **Dashboard** — Dataset overview, class distribution, grade distribution charts
- 📁 **Dataset** — Browse all 395 student records in a searchable table
- ⚙️ **Preprocessing** — Step-by-step data pipeline visualization
- 🤖 **Training** — Train KNN, SVM, and ANN models with one click
- 📈 **Evaluation** — Compare models with accuracy, F1, precision, recall & confusion matrix
- ⚡ **Predict** — Enter student info and get Pass/Fail prediction from all 3 models

---

## Tech Stack
| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Algorithms | scikit-learn (KNN, SVM, ANN) |
| Data | pandas, numpy |
| Frontend | HTML, Bootstrap 5, JavaScript |
| Charts | Chart.js |

---

## Dataset
Synthetic dataset modeled after the [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance) (Cortez & Silva, 2008).
- **395 students**, **33 features** (demographics, grades, study habits, family background)
- **Target**: G3 ≥ 10 → Pass (1), G3 < 10 → Fail (0)

---

## Getting Started

### Requirements
- [Anaconda](https://www.anaconda.com/) (includes Python, pip)

### Setup
```bash
# 1. Clone the repo
git clone https://github.com/RoyBayotlang/is108-project-student_performance.git
cd is108-project-student_performance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset
python data/generate_data.py

# 4. Run the app
python app.py
```

### Open in browser
```
http://127.0.0.1:5000
```

---

## Model Results (on synthetic dataset)
| Model | Accuracy | F1 Score |
|---|---|---|
| **SVM** ⭐ | **89.87%** | **90.48%** |
| ANN | 82.28% | 82.93% |
| KNN | 73.42% | 75.86% |

---

## Project Structure
```
is/
├── app.py                 # Flask backend & ML pipeline
├── requirements.txt       # Python dependencies
├── data/
│   └── generate_data.py   # Synthetic dataset generator
├── static/
│   ├── css/style.css      # Custom styles (Bootstrap override)
│   └── js/app.js          # Frontend logic & API calls
└── templates/
    └── index.html         # Main UI (Bootstrap 5, Chart.js)
```

---

## Team
IS 108 – Intelligence System | Final Project
"# is108-project-student_performance" 
