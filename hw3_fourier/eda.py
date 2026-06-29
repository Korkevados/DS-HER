"""Exploratory data analysis on the UCI HAR raw inertial signals.
Produces plots + a stats summary under hw3_fourier/outputs/eda/.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 50.0
N = 128
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "UCI HAR Dataset" / "UCI HAR Dataset"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

SIGNALS = ["total_acc_x", "total_acc_y", "total_acc_z",
           "body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z"]
NAMES = {1: "WALKING", 2: "WALK_UP", 3: "WALK_DOWN", 4: "SITTING", 5: "STANDING", 6: "LAYING"}
ORDER = [1, 2, 3, 4, 5, 6]
COLORS = plt.cm.tab10(np.linspace(0, 1, 6))


def load(split):
    sig = np.stack([np.loadtxt(DATA / split / "Inertial Signals" / f"{s}_{split}.txt")
                    for s in SIGNALS], axis=-1).astype(np.float32)  # (N,128,9)
    y = np.loadtxt(DATA / split / f"y_{split}.txt").astype(int)
    subj = np.loadtxt(DATA / split / f"subject_{split}.txt").astype(int)
    return sig, y, subj


def main():
    sig_tr, y_tr, subj_tr = load("train")
    sig_te, y_te, subj_te = load("test")
    stats = {}

    # total-acc magnitude and body-acc magnitude per window
    tacc = sig_tr[:, :, 0:3]
    bacc = sig_tr[:, :, 3:6]
    bgyro = sig_tr[:, :, 6:9]
    tacc_mag = np.linalg.norm(tacc, axis=-1)     # (N,128)
    bacc_mag = np.linalg.norm(bacc, axis=-1)
    bgyro_mag = np.linalg.norm(bgyro, axis=-1)

    # ---------- 1. class distribution train vs test ----------
    tr_counts = [int((y_tr == k).sum()) for k in ORDER]
    te_counts = [int((y_te == k).sum()) for k in ORDER]
    stats["class_counts_train"] = dict(zip([NAMES[k] for k in ORDER], tr_counts))
    stats["class_counts_test"] = dict(zip([NAMES[k] for k in ORDER], te_counts))
    stats["class_balance_ratio_train"] = round(max(tr_counts) / min(tr_counts), 3)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(6); w = 0.4
    ax.bar(x - w/2, tr_counts, w, label=f"train (n={len(y_tr)})", color="#4C72B0")
    ax.bar(x + w/2, te_counts, w, label=f"test (n={len(y_te)})", color="#DD8452")
    for i, (a, b) in enumerate(zip(tr_counts, te_counts)):
        ax.text(i - w/2, a + 8, str(a), ha="center", fontsize=8)
        ax.text(i + w/2, b + 8, str(b), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NAMES[k] for k in ORDER], rotation=20)
    ax.set_ylabel("number of windows"); ax.set_title("UCI HAR — class distribution (train vs test)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "01_class_distribution.png", dpi=150); plt.close(fig)

    # ---------- 2. per-subject window counts (train) ----------
    subs = sorted(np.unique(subj_tr))
    counts = [int((subj_tr == s).sum()) for s in subs]
    stats["subjects_train"] = len(subs)
    stats["subjects_test"] = int(len(np.unique(subj_te)))
    stats["per_subject_windows_train"] = {"min": min(counts), "max": max(counts),
                                          "mean": round(float(np.mean(counts)), 1)}
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([str(s) for s in subs], counts, color="#55A868")
    ax.set_xlabel("subject id (train)"); ax.set_ylabel("windows")
    ax.set_title("Windows per training subject"); fig.tight_layout()
    fig.savefig(OUT / "02_windows_per_subject.png", dpi=150); plt.close(fig)

    # ---------- 3. example total-acc 3-axis window per activity ----------
    t = np.arange(N) / FS
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), ORDER):
        idx = np.where(y_tr == k)[0][0]
        for c, lab, col in zip(range(3), ["x", "y", "z"], ["r", "g", "b"]):
            ax.plot(t, sig_tr[idx, :, c], col, lw=0.9, label=lab)
        ax.set_title(NAMES[k]); ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.supxlabel("time (s)"); fig.supylabel("total acceleration (g)")
    fig.suptitle("Example 2.56 s windows — total acceleration per axis")
    fig.tight_layout(); fig.savefig(OUT / "03_example_windows.png", dpi=150); plt.close(fig)

    # ---------- 4. mean FFT spectrum per activity (body-acc magnitude) ----------
    spec = np.abs(np.fft.rfft(bacc_mag - bacc_mag.mean(axis=1, keepdims=True), axis=1))
    freqs = np.fft.rfftfreq(N, 1 / FS)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, col in zip(ORDER, COLORS):
        ax.plot(freqs, spec[y_tr == k].mean(0), label=NAMES[k], color=col)
    ax.set_xlim(0, 15); ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean |FFT| of body-acc magnitude")
    ax.set_title("Average Fourier spectrum per activity"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "04_mean_spectrum.png", dpi=150); plt.close(fig)

    # ---------- 5. distribution of within-window std (dynamic vs static) ----------
    win_std = bacc_mag.std(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, col in zip(ORDER, COLORS):
        ax.hist(win_std[y_tr == k], bins=40, alpha=0.5, label=NAMES[k], color=col, density=True)
    ax.set_xlabel("within-window std of body-acc magnitude (g)")
    ax.set_ylabel("density"); ax.set_title("Motion energy separates dynamic vs static")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "05_motion_std_hist.png", dpi=150); plt.close(fig)

    # ---------- 6. per-axis gravity means (orientation cue: sitting vs standing) ----------
    means = {NAMES[k]: tacc[y_tr == k].mean(axis=(0, 1)).round(3).tolist() for k in ORDER}
    stats["total_acc_axis_means_g"] = means
    fig, ax = plt.subplots(figsize=(9, 5))
    xa = np.arange(6); w = 0.25
    for j, (lab, col) in enumerate(zip(["x", "y", "z"], ["#c44", "#4a4", "#44c"])):
        vals = [means[NAMES[k]][j] for k in ORDER]
        ax.bar(xa + (j - 1) * w, vals, w, label=f"axis {lab}", color=col)
    ax.set_xticks(xa); ax.set_xticklabels([NAMES[k] for k in ORDER], rotation=20)
    ax.set_ylabel("mean total acceleration (g)")
    ax.set_title("Per-axis gravity direction by activity (orientation signature)")
    ax.axhline(0, color="k", lw=0.6); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "06_axis_gravity_means.png", dpi=150); plt.close(fig)

    # ---------- summary stats ----------
    stats["sampling_rate_hz"] = FS
    stats["window_len_samples"] = N
    stats["window_seconds"] = N / FS
    stats["channels"] = SIGNALS
    stats["raw_value_range"] = {
        "total_acc_min": round(float(tacc.min()), 3), "total_acc_max": round(float(tacc.max()), 3),
        "body_gyro_min": round(float(bgyro.min()), 3), "body_gyro_max": round(float(bgyro.max()), 3),
    }
    stats["motion_std_by_class_g"] = {NAMES[k]: round(float(win_std[y_tr == k].mean()), 4) for k in ORDER}
    (OUT / "data_stats.json").write_text(json.dumps(stats, indent=2))

    print(json.dumps(stats, indent=2))
    print("\nPlots written to", OUT)


if __name__ == "__main__":
    main()
