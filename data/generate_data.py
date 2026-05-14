"""
generate_data.py
Generates a realistic synthetic Student Performance dataset
matching the UCI Student Performance dataset distributions.
Run this once: python data/generate_data.py
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 395  # Same size as real UCI math dataset

# ── Demographic & Family ──────────────────────────────────────────────────────
school    = np.random.choice(['GP', 'MS'], n, p=[0.91, 0.09])
sex       = np.random.choice(['F', 'M'],   n, p=[0.54, 0.46])
age       = np.random.randint(15, 23, n)
address   = np.random.choice(['U', 'R'],       n, p=[0.73, 0.27])
famsize   = np.random.choice(['GT3', 'LE3'],   n, p=[0.74, 0.26])
Pstatus   = np.random.choice(['T', 'A'],       n, p=[0.90, 0.10])
Medu      = np.random.choice([0,1,2,3,4],      n, p=[0.02,0.09,0.26,0.28,0.35])
Fedu      = np.random.choice([0,1,2,3,4],      n, p=[0.02,0.17,0.27,0.29,0.25])
Mjob      = np.random.choice(['teacher','health','services','at_home','other'], n,
                              p=[0.11,0.10,0.23,0.20,0.36])
Fjob      = np.random.choice(['teacher','health','services','at_home','other'], n,
                              p=[0.08,0.04,0.28,0.04,0.56])
reason    = np.random.choice(['home','reputation','course','other'], n,
                              p=[0.28,0.26,0.31,0.15])
guardian  = np.random.choice(['mother','father','other'], n, p=[0.55,0.38,0.07])

# ── School-related ────────────────────────────────────────────────────────────
traveltime = np.random.choice([1,2,3,4], n, p=[0.47,0.36,0.12,0.05])
studytime  = np.random.choice([1,2,3,4], n, p=[0.24,0.49,0.20,0.07])
failures   = np.random.choice([0,1,2,3], n, p=[0.67,0.22,0.07,0.04])
schoolsup  = np.random.choice(['yes','no'], n, p=[0.18,0.82])
famsup     = np.random.choice(['yes','no'], n, p=[0.62,0.38])
paid       = np.random.choice(['yes','no'], n, p=[0.46,0.54])
activities = np.random.choice(['yes','no'], n, p=[0.52,0.48])
nursery    = np.random.choice(['yes','no'], n, p=[0.81,0.19])
higher     = np.random.choice(['yes','no'], n, p=[0.92,0.08])
internet   = np.random.choice(['yes','no'], n, p=[0.76,0.24])
romantic   = np.random.choice(['yes','no'], n, p=[0.34,0.66])

# ── Social ────────────────────────────────────────────────────────────────────
famrel   = np.random.choice([1,2,3,4,5], n, p=[0.02,0.06,0.19,0.47,0.26])
freetime = np.random.choice([1,2,3,4,5], n, p=[0.04,0.20,0.36,0.27,0.13])
goout    = np.random.choice([1,2,3,4,5], n, p=[0.08,0.22,0.32,0.24,0.14])
Dalc     = np.random.choice([1,2,3,4,5], n, p=[0.54,0.25,0.12,0.06,0.03])
Walc     = np.random.choice([1,2,3,4,5], n, p=[0.29,0.22,0.24,0.15,0.10])
health   = np.random.choice([1,2,3,4,5], n, p=[0.08,0.11,0.22,0.27,0.32])
absences = np.clip(np.round(np.random.exponential(5, n)).astype(int), 0, 75)

# ── Grades (with realistic correlations) ─────────────────────────────────────
G1 = np.clip(np.round(np.random.normal(10.9, 3.3, n)), 0, 20).astype(int)
G2 = np.clip(np.round(G1 + np.random.normal(0, 1.5, n)), 0, 20).astype(int)

# G3 is influenced by G2, studytime, failures, absences
G3_raw = (G2
          + np.random.normal(0, 1.0, n)
          - failures * 2.5
          + (studytime - 2) * 0.8
          - absences * 0.05)
G3 = np.clip(np.round(G3_raw), 0, 20).astype(int)

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    'school': school, 'sex': sex, 'age': age, 'address': address,
    'famsize': famsize, 'Pstatus': Pstatus, 'Medu': Medu, 'Fedu': Fedu,
    'Mjob': Mjob, 'Fjob': Fjob, 'reason': reason, 'guardian': guardian,
    'traveltime': traveltime, 'studytime': studytime, 'failures': failures,
    'schoolsup': schoolsup, 'famsup': famsup, 'paid': paid,
    'activities': activities, 'nursery': nursery, 'higher': higher,
    'internet': internet, 'romantic': romantic,
    'famrel': famrel, 'freetime': freetime, 'goout': goout,
    'Dalc': Dalc, 'Walc': Walc, 'health': health, 'absences': absences,
    'G1': G1, 'G2': G2, 'G3': G3
})

out_path = os.path.join(os.path.dirname(__file__), 'student-mat.csv')
df.to_csv(out_path, sep=';', index=False)
print(f"[OK] Dataset generated: {out_path}  ({len(df)} rows)")
pass_rate = (df['G3'] >= 10).mean() * 100
print(f"     Pass rate: {pass_rate:.1f}%  |  Fail rate: {100 - pass_rate:.1f}%")
