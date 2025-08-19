# 🧮 MAP – Charting Student Math Misunderstandings

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%20%7C%20Pandas%20%7C%20NumPy-orange.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue.svg)

Baseline solution for the [Kaggle: MAP – Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings) competition.  
The task is to **predict student misconceptions** from text (question stem, student answer, and explanation).  
Evaluation metric: **MAP@3**.

---

## 📂 Project Structure
map-misconceptions/ ├─ README.md ├─ requirements.txt ├─ .gitignore ├─ src/ │ └─ train_baseline.py ├─ submissions/ │ └─ submission.csv ├─ models/ ├─ notebooks/ │ └─ make_cv_chart.py ├─ assets/ │ └─ cv_scores.png └─ data/ └─ raw/

---

## Script

import os, numpy as np, pandas as pd, joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

KAGGLE_DIR = "/kaggle/input/map-charting-student-math-misunderstandings"
if os.path.exists(KAGGLE_DIR):
    TRAIN = os.path.join(KAGGLE_DIR, "train.csv")
    TEST = os.path.join(KAGGLE_DIR, "test.csv")
else:
    TRAIN = "data/raw/train.csv"
    TEST = "data/raw/test.csv"

OUT_SUB = "submissions/submission.csv"
os.makedirs("submissions", exist_ok=True)
os.makedirs("models", exist_ok=True)

def combine_text(df):
    cols = [c for c in ["QuestionText", "MC_Answer", "StudentExplanation"] if c in df.columns]
    txt = df[cols].astype(str).agg(" ".join, axis=1)
    return txt.str.replace(r"\s+", " ", regex=True).str.strip()

def mapk(y_true_int, probas, k=3):
    idx = np.argsort(-probas, axis=1)[:, :k]
    score = 0.0
    for i, true_lab in enumerate(y_true_int):
        hits = (idx[i] == true_lab)
        if hits.any():
            precisions = [(hits[: j+1].sum()) / (j+1) for j in range(k) if hits[j]]
            score += np.mean(precisions)
    return score / len(y_true_int)

train = pd.read_csv(TRAIN)
test = pd.read_csv(TEST)

for c in ["QuestionText","MC_Answer","StudentExplanation","Category","Misconception"]:
    if c in train.columns: train[c] = train[c].fillna("")
    if c in test.columns: test[c] = test[c].fillna("")

train["Misconception"] = train["Misconception"].replace("", "NA")
train["target"] = (train["Category"].astype(str).str.strip() 
                   + ":" + train["Misconception"].astype(str).str.strip())

X = combine_text(train)
X_test = combine_text(test)

le = LabelEncoder()
y = le.fit_transform(train["target"].values)
classes_ = le.classes_

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200_000)),
    ("lr", LogisticRegression(
        solver="saga", max_iter=4000, C=4.0, class_weight="balanced", multi_class="multinomial"
    ))
])

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []
for tr_idx, va_idx in skf.split(X, y):
    pipe.fit(X.iloc[tr_idx], y[tr_idx])
    va_proba = pipe.predict_proba(X.iloc[va_idx])
    scores.append(mapk(y[va_idx], va_proba, k=3))
print(f"CV MAP@3: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

pipe.fit(X, y)
test_proba = pipe.predict_proba(X_test)
topk = np.argsort(-test_proba, axis=1)[:, :3]
pred_strings = [" ".join(classes_[row_idx]) for row_idx in topk]

sub = pd.DataFrame({
    "row_id": test["row_id"],
    "Category:Misconception": pred_strings
})
sub.to_csv(OUT_SUB, index=False)
print(f"Saved -> {OUT_SUB}")

joblib.dump(pipe, "models/tfidf_logreg_pipeline.joblib")
joblib.dump(le, "models/label_encoder.joblib")
