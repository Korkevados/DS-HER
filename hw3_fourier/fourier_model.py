"""
HW3 - Modeling phase (CRISP-DM section 4)
Modeling technique: Fourier-transform (frequency-domain) features.

Our team owns the Fourier-transform track. Instead of using the 561 engineered
UCI features (other team's track), we compute our OWN frequency-domain feature
set directly from the raw 9-channel inertial windows via the FFT, and train
classical classifiers on those Fourier features only.

Pipeline:
  raw inertial signals  -> 11 channels (9 raw + 2 magnitudes)
  each 128-sample window -> rFFT -> per-channel spectral features
  feature matrix         -> StandardScaler -> {LogReg, RandomForest, SVM-RBF}
  assessment             -> subject-aware GroupKFold CV on train,
                            final hold-out evaluation on the UCI test set.

Run:  conda activate study && python fourier_model.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
FS = 50.0  # UCI sampling rate (Hz)
N = 128  # samples per window (2.56 s)
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "UCI HAR Dataset" / "UCI HAR Dataset"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# 9 raw inertial channels shipped by UCI (fixed order)
RAW_CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

ACTIVITY_NAMES = {
    1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
    4: "SITTING", 5: "STANDING", 6: "LAYING",
}
LABEL_ORDER = [1, 2, 3, 4, 5, 6]  # for a consistent confusion-matrix order

# Frequency bands (Hz) used for band-energy features
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 25.0)]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_signals(split):
    """Return raw inertial array of shape (n_windows, 128, 9) for a split."""
    sig_dir = DATA / split / "Inertial Signals"
    chans = []
    for ch in RAW_CHANNELS:
        arr = np.loadtxt(sig_dir / f"{ch}_{split}.txt")  # (n_windows, 128)
        chans.append(arr)
    return np.stack(chans, axis=-1)  # (n_windows, 128, 9)


def load_labels(split):
    y = np.loadtxt(DATA / split / f"y_{split}.txt").astype(int)
    subj = np.loadtxt(DATA / split / f"subject_{split}.txt").astype(int)
    return y, subj


def add_magnitude_channels(sig):
    """Append orientation-invariant magnitude channels for acc and gyro.

    body_acc magnitude  = sqrt(x^2 + y^2 + z^2)  over body_acc_{x,y,z}
    body_gyro magnitude = sqrt(x^2 + y^2 + z^2)  over body_gyro_{x,y,z}
    These are robust to the phone-orientation shift documented in HW2 (2.4).
    """
    body_acc = sig[:, :, 0:3]
    body_gyro = sig[:, :, 3:6]
    acc_mag = np.sqrt((body_acc ** 2).sum(axis=-1, keepdims=True))
    gyro_mag = np.sqrt((body_gyro ** 2).sum(axis=-1, keepdims=True))
    return np.concatenate([sig, acc_mag, gyro_mag], axis=-1)  # (n, 128, 11)


ALL_CHANNELS = RAW_CHANNELS + ["body_acc_mag", "body_gyro_mag"]


# --------------------------------------------------------------------------- #
# Fourier feature extraction
# --------------------------------------------------------------------------- #
FREQS = np.fft.rfftfreq(N, d=1.0 / FS)  # (65,) -> 0 .. 25 Hz


def channel_fft_features(x):
    """Compute frequency-domain features for one (n_windows, 128) channel.

    Returns a dict {feature_name: (n_windows,) array}.
    Uses the one-sided power spectrum P = |rFFT|^2.
    """
    # Remove per-window DC for the *shape* descriptors so a large gravity
    # offset does not dominate centroid / entropy / moments. Energy-type
    # features keep the DC information via the low-band energy fraction.
    spec = np.fft.rfft(x, axis=1)
    power = (np.abs(spec) ** 2)  # (n, 65)

    # AC power: drop the DC bin for shape descriptors
    ac_power = power[:, 1:]  # (n, 64)
    ac_freqs = FREQS[1:]  # (n,)
    p_sum = ac_power.sum(axis=1) + 1e-12
    p_norm = ac_power / p_sum[:, None]  # probability distribution over freq

    feats = {}

    # Total spectral energy (per sample), log-scaled for dynamic range
    total_power = power.sum(axis=1) / N
    feats["log_energy"] = np.log1p(total_power)

    # Dominant (peak) AC frequency and its relative magnitude
    dom_idx = ac_power.argmax(axis=1)
    feats["dom_freq"] = ac_freqs[dom_idx]
    feats["dom_ratio"] = ac_power[np.arange(len(x)), dom_idx] / p_sum

    # Spectral centroid (mean frequency) and bandwidth (spread)
    centroid = (ac_freqs[None, :] * p_norm).sum(axis=1)
    feats["centroid"] = centroid
    var = ((ac_freqs[None, :] - centroid[:, None]) ** 2 * p_norm).sum(axis=1)
    feats["bandwidth"] = np.sqrt(var)

    # Spectral skewness and kurtosis (shape of the spectrum)
    std = feats["bandwidth"] + 1e-12
    z = (ac_freqs[None, :] - centroid[:, None]) / std[:, None]
    feats["skew"] = (z ** 3 * p_norm).sum(axis=1)
    feats["kurtosis"] = (z ** 4 * p_norm).sum(axis=1)

    # Spectral entropy (flat spectrum -> high; single peak -> low)
    feats["entropy"] = -(p_norm * np.log(p_norm + 1e-12)).sum(axis=1)

    # Spectral roll-off: frequency below which 85% of energy lies
    cums = np.cumsum(ac_power, axis=1) / p_sum[:, None]
    rolloff_idx = (cums < 0.85).sum(axis=1)
    rolloff_idx = np.clip(rolloff_idx, 0, len(ac_freqs) - 1)
    feats["rolloff85"] = ac_freqs[rolloff_idx]

    # Fraction of energy in each frequency band (incl. DC band -> static cue)
    for lo, hi in BANDS:
        mask = (FREQS >= lo) & (FREQS < hi)
        band_energy = power[:, mask].sum(axis=1)
        feats[f"band_{lo:g}_{hi:g}"] = band_energy / (power.sum(axis=1) + 1e-12)

    return feats


def build_feature_matrix(sig):
    """sig: (n_windows, 128, n_channels) -> (DataFrame features, names)."""
    cols = {}
    for c, name in enumerate(ALL_CHANNELS):
        chan_feats = channel_fft_features(sig[:, :, c])
        for fname, vals in chan_feats.items():
            cols[f"{name}__{fname}"] = vals
    df = pd.DataFrame(cols)
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print(">> Loading raw inertial signals ...")
    sig_train = add_magnitude_channels(load_signals("train"))
    sig_test = add_magnitude_channels(load_signals("test"))
    y_train, subj_train = load_labels("train")
    y_test, subj_test = load_labels("test")
    print(f"   train signals {sig_train.shape}, test signals {sig_test.shape}")

    print(">> Extracting Fourier features ...")
    X_train = build_feature_matrix(sig_train)
    X_test = build_feature_matrix(sig_test)
    print(f"   feature matrix: {X_train.shape[1]} Fourier features per window")
    X_train.to_csv(OUT / "fourier_features_train.csv", index=False)
    X_test.to_csv(OUT / "fourier_features_test.csv", index=False)

    # ---- 4.4 part 1: subject-aware cross-validation on the training set ---- #
    models = {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]),
        "rbf_svm": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=10.0, gamma="scale")),
        ]),
        "random_forest": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=300, random_state=42,
                                           n_jobs=-1)),
        ]),
    }

    print(">> Subject-aware 5-fold GroupKFold CV on the training set ...")
    gkf = GroupKFold(n_splits=5)
    cv_results = {}
    for name, pipe in models.items():
        scores = cross_val_score(pipe, X_train, y_train, groups=subj_train,
                                 cv=gkf, scoring="f1_macro", n_jobs=-1)
        cv_results[name] = (scores.mean(), scores.std())
        print(f"   {name:14s} macro-F1 = {scores.mean():.4f} "
              f"(+/- {scores.std():.4f})")

    best_name = max(cv_results, key=lambda k: cv_results[k][0])
    print(f"   -> best by CV macro-F1: {best_name}")

    # ---- 4.3: fit each model on full train, evaluate on hold-out test ----- #
    print(">> Fitting on full training set and evaluating on UCI test set ...")
    test_results = {}
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1m = f1_score(y_test, pred, average="macro")
        test_results[name] = {"accuracy": acc, "macro_f1": f1m}
        print(f"   {name:14s} test acc = {acc:.4f}  macro-F1 = {f1m:.4f}")

    # ---- 4.4: detailed assessment of the best model ----------------------- #
    best = fitted[best_name]
    pred = best.predict(X_test)
    target_names = [ACTIVITY_NAMES[i] for i in LABEL_ORDER]
    report = classification_report(y_test, pred, labels=LABEL_ORDER,
                                   target_names=target_names, digits=4)
    print("\n=== Classification report (best model: "
          f"{best_name}) ===\n{report}")

    cm = confusion_matrix(y_test, pred, labels=LABEL_ORDER)

    # ---- Save metrics -------------------------------------------------- #
    summary = {
        "n_features": int(X_train.shape[1]),
        "channels": ALL_CHANNELS,
        "cv_macro_f1": {k: {"mean": v[0], "std": v[1]}
                        for k, v in cv_results.items()},
        "test": test_results,
        "best_model": best_name,
        "confusion_matrix_labels": target_names,
        "confusion_matrix": cm.tolist(),
    }
    with open(OUT / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(OUT / "classification_report.txt", "w") as f:
        f.write(f"Best model: {best_name}\n\n{report}\n")

    # ---- Plot 1: confusion matrix -------------------------------------- #
    plot_confusion(cm, target_names, best_name)

    # ---- Plot 2: example spectra per activity -------------------------- #
    plot_example_spectra(sig_train, y_train)

    # ---- Plot 3: RF feature importances (interpretability) ------------- #
    if "random_forest" in fitted:
        plot_feature_importance(fitted["random_forest"], X_train.columns)

    # ---- Save the best trained pipeline for reuse ---------------------- #
    import joblib
    model_path = OUT / "fourier_model_best.joblib"
    joblib.dump({"pipeline": best, "model_name": best_name,
                 "feature_names": list(X_train.columns),
                 "channels": ALL_CHANNELS,
                 "label_order": LABEL_ORDER, "activity_names": ACTIVITY_NAMES},
                model_path)
    print(f">> Saved best pipeline ({best_name}) -> {model_path}")

    print(f"\n>> Done. Outputs written to {OUT}")


def plot_confusion(cm, names, model_name):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title(f"Fourier model ({model_name}) — normalised confusion matrix")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_example_spectra(sig, y):
    """Mean body-acc-magnitude spectrum per activity (the Fourier signature)."""
    mag = sig[:, :, ALL_CHANNELS.index("body_acc_mag")]
    spec = np.abs(np.fft.rfft(mag - mag.mean(axis=1, keepdims=True), axis=1))
    fig, ax = plt.subplots(figsize=(8, 5))
    for lab in LABEL_ORDER:
        m = spec[y == lab].mean(axis=0)
        ax.plot(FREQS, m, label=ACTIVITY_NAMES[lab])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mean |FFT| of body-acc magnitude")
    ax.set_title("Average Fourier spectrum per activity (training set)")
    ax.set_xlim(0, 15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "spectra_per_activity.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(rf_pipe, feat_names, top=20):
    imp = rf_pipe.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:top]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(order)), imp[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel("Random-forest importance")
    ax.set_title(f"Top {top} Fourier features")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
