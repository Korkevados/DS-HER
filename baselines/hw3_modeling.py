"""
HW3 - CRISP-DM Modeling phase for the HAR project.

Trains three baseline classifiers (Logistic Regression, Random Forest, SVM/RBF)
on the UCI HAR engineered-feature dataset and reports honest, subject-disjoint
generalization metrics for the HW3 document (sections 4.3 / 4.4).

Design decisions (see HW1/HW2):
  - Use the provided subject-disjoint train/test split (never split by window).
  - Tune hyper-parameters with subject-grouped 5-fold CV on the TRAIN set only,
    so the test set stays untouched until the final evaluation (no leakage).
  - Primary metric: macro-F1 (classes are mildly imbalanced, ratio ~1.43);
    also report accuracy, per-class precision/recall, and a confusion matrix
    with the fixed class order used throughout the project.

Run:
    .venv/bin/python code/hw3_modeling.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42

# Fixed class order used across the whole project (HW2 section 3.5).
CLASS_ORDER = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

DATA_ROOT = (
    Path(__file__).resolve().parent.parent
    / "Data"
    / "human+activity+recognition+using+smartphones"
)


def load_split(split):
    """Load X, y (int 1-6) and subject ids for a UCI HAR split ('train'/'test')."""
    base = DATA_ROOT / split
    X = pd.read_csv(base / f"X_{split}.txt", sep=r"\s+", header=None).to_numpy(
        dtype=np.float32
    )
    y = pd.read_csv(base / f"y_{split}.txt", header=None).to_numpy().ravel()
    subj = pd.read_csv(base / f"subject_{split}.txt", header=None).to_numpy().ravel()
    return X, y, subj


def label_names(y):
    """Map integer labels 1-6 to activity names via activity_labels.txt."""
    labels = pd.read_csv(
        DATA_ROOT / "activity_labels.txt", sep=r"\s+", header=None,
        index_col=0,
    )[1].to_dict()
    return np.array([labels[v] for v in y])


def main():
    X_train, y_train, subj_train = load_split("train")
    X_test, y_test, subj_test = load_split("test")

    y_train_names = label_names(y_train)
    y_test_names = label_names(y_test)

    # --- Sanity checks (HW2 facts) -------------------------------------------
    train_subjects = set(np.unique(subj_train))
    test_subjects = set(np.unique(subj_test))
    assert train_subjects.isdisjoint(test_subjects), "train/test subjects overlap!"
    print("=" * 70)
    print("DATA SUMMARY")
    print("=" * 70)
    print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(
        f"train subjects: {len(train_subjects)}  "
        f"test subjects: {len(test_subjects)}  (disjoint: True)"
    )
    counts = pd.Series(y_test_names).value_counts().reindex(CLASS_ORDER)
    print("test per-class counts:")
    print(counts.to_string())

    # --- Model definitions + small hyper-parameter grids ---------------------
    # Each model lives in a Pipeline with StandardScaler so SVM/LogReg see
    # comparable feature scales. RF is scale-invariant but the scaler is a
    # no-op harm-wise, so we keep the pipeline uniform.
    models = {
        "LogisticRegression": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "RandomForest": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        RandomForestClassifier(
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 20],
            },
        ),
        "SVM_RBF": (
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
                ]
            ),
            {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.001]},
        ),
    }

    cv = GroupKFold(n_splits=5)
    results = {}

    for name, (pipe, grid) in models.items():
        print("\n" + "=" * 70)
        print(f"MODEL: {name}")
        print("=" * 70)
        search = GridSearchCV(
            pipe,
            grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train_names, groups=subj_train)

        print(f"grid searched: {grid}")
        print(f"best CV macro-F1: {search.best_score_:.4f}")
        print(f"best params: {search.best_params_}")

        # Final, single evaluation on the held-out test set.
        y_pred = search.predict(X_test)
        acc = accuracy_score(y_test_names, y_pred)
        f1_macro = f1_score(y_test_names, y_pred, average="macro")
        f1_weighted = f1_score(y_test_names, y_pred, average="weighted")
        print(f"\nTEST accuracy:     {acc:.4f}")
        print(f"TEST macro-F1:     {f1_macro:.4f}")
        print(f"TEST weighted-F1:  {f1_weighted:.4f}")
        print("\nper-class report (test):")
        print(
            classification_report(
                y_test_names, y_pred, labels=CLASS_ORDER, digits=3
            )
        )
        cm = confusion_matrix(y_test_names, y_pred, labels=CLASS_ORDER)
        print("confusion matrix (rows=true, cols=pred, order=CLASS_ORDER):")
        cm_df = pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER)
        print(cm_df.to_string())

        results[name] = {
            "cv_f1": search.best_score_,
            "best_params": search.best_params_,
            "test_acc": acc,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
        }

    # --- Ranking summary -----------------------------------------------------
    print("\n" + "=" * 70)
    print("RANKING (by test macro-F1)")
    print("=" * 70)
    summary = pd.DataFrame(results).T.sort_values("test_f1_macro", ascending=False)
    summary = summary[
        ["test_acc", "test_f1_macro", "test_f1_weighted", "cv_f1"]
    ].astype(float).round(4)
    print(summary.to_string())


if __name__ == "__main__":
    main()
