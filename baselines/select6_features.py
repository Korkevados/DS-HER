"""
HW3 extra — reduce the 561 UCI engineered features to (at most) 6 real features.

We rank the 561 features by mutual information with the activity label (computed on
the TRAIN set only, no test leakage), keep the top 6 *named* features, and retrain
the three classical baselines (Logistic Regression, Random Forest, SVM-RBF) on just
those 6. Results are compared against the full-561 baseline.

Run:  .venv/bin/python baselines/select6_features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

SEED = 42
K = 6
DATA = (Path(__file__).resolve().parent.parent / "Data"
        / "human+activity+recognition+using+smartphones")
OUT = Path(__file__).resolve().parent / "feature6_outputs"
OUT.mkdir(exist_ok=True)
CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]


def feature_names():
    names = pd.read_csv(DATA / "features.txt", sep=r"\s+", header=None)[1].tolist()
    # disambiguate the 84 duplicate names (UCI quirk)
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__dup{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def load(split):
    X = pd.read_csv(DATA / split / f"X_{split}.txt", sep=r"\s+", header=None).to_numpy(np.float64)
    y = pd.read_csv(DATA / split / f"y_{split}.txt", header=None).to_numpy().ravel()
    return X, y


def models():
    return {
        "LogisticRegression": Pipeline([("sc", StandardScaler()),
                                        ("clf", LogisticRegression(max_iter=2000, C=2.0, random_state=SEED))]),
        "RandomForest": Pipeline([("sc", StandardScaler()),
                                  ("clf", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1))]),
        "SVM_RBF": Pipeline([("sc", StandardScaler()),
                             ("clf", SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED))]),
    }


def evaluate_set(Xtr, ytr, Xte, yte):
    rows = {}
    for name, pipe in models().items():
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        rows[name] = {"accuracy": accuracy_score(yte, p),
                      "macro_f1": f1_score(yte, p, average="macro"),
                      "y_pred": p}
    return rows


def main():
    names = feature_names()
    X_tr, y_tr = load("train")
    X_te, y_te = load("test")
    y_tr_names = np.array([CLASS_NAMES[i - 1] for i in y_tr])
    y_te_names = np.array([CLASS_NAMES[i - 1] for i in y_te])
    print(f"Loaded train {X_tr.shape}  test {X_te.shape}  ({len(names)} feature names)")

    # ---- mutual information ranking (train only) ----
    Xs = StandardScaler().fit_transform(X_tr)
    mi = mutual_info_classif(Xs, y_tr, random_state=SEED)
    order = np.argsort(mi)[::-1]
    top = order[:K]
    print(f"\nTop {K} features by mutual information:")
    for r, idx in enumerate(top, 1):
        print(f"  {r}. {names[idx]:45s}  MI={mi[idx]:.4f}  (col {idx})")

    sel_names = [names[i] for i in top]

    # ---- train classical models on the 6 features vs all 561 ----
    res6 = evaluate_set(X_tr[:, top], y_tr_names, X_te[:, top], y_te_names)
    res561 = evaluate_set(X_tr, y_tr_names, X_te, y_te_names)

    print("\n" + "=" * 64)
    print(f"{'model':22s} {'acc(6)':>8s} {'F1(6)':>8s} {'acc(561)':>10s} {'F1(561)':>9s}")
    print("=" * 64)
    summary = []
    for m in res6:
        print(f"{m:22s} {res6[m]['accuracy']:8.4f} {res6[m]['macro_f1']:8.4f} "
              f"{res561[m]['accuracy']:10.4f} {res561[m]['macro_f1']:9.4f}")
        summary.append({"model": m,
                        "acc_6feat": res6[m]["accuracy"], "f1_6feat": res6[m]["macro_f1"],
                        "acc_561feat": res561[m]["accuracy"], "f1_561feat": res561[m]["macro_f1"]})

    best = max(res6, key=lambda m: res6[m]["macro_f1"])
    print(f"\nBest 6-feature model: {best}")
    print(classification_report(y_te_names, res6[best]["y_pred"], labels=CLASS_NAMES, digits=3))

    # ---- figures ----
    plt.figure(figsize=(10, 5))
    show = order[:15]
    cols = ["#d62728" if i in set(top) else "#7f9bbf" for i in show]
    plt.barh([names[i] for i in show][::-1], [mi[i] for i in show][::-1], color=cols[::-1])
    plt.xlabel("Mutual information with activity label")
    plt.title(f"Top-15 features by MI (red = selected top {K})")
    plt.tight_layout(); plt.savefig(OUT / "mi_top_features.png", dpi=150); plt.close()

    cm = confusion_matrix(y_te_names, res6[best]["y_pred"], labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(f"{best} on {K} features — test confusion matrix")
    plt.tight_layout(); plt.savefig(OUT / "confusion_6features.png", dpi=150); plt.close()

    json.dump({"selected_features": sel_names,
               "selected_columns": [int(i) for i in top],
               "mutual_information": {names[i]: float(mi[i]) for i in top},
               "comparison": summary, "best_6feature_model": best},
              open(OUT / "feature6_results.json", "w"), indent=2)
    pd.DataFrame(summary).to_csv(OUT / "feature6_comparison.csv", index=False)
    print("\nArtifacts written to:", OUT)


if __name__ == "__main__":
    main()
