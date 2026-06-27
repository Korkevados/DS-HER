"""
har_plots — readable plotting helpers for the HW4 HAR notebooks.

Every function takes already-loaded arrays and returns a matplotlib Figure, so
the same code renders live in the notebook and exports PNGs for the slides.
Styling follows a minimal academic palette (navy / blue, white background).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from har_utils import ACTIVITIES, ACTIVITY_SHORT, CHANNELS, FS, WINDOW

# --- academic palette --------------------------------------------------------
NAVY, BLUE, INK, MUTED = "#1F4E79", "#2E75B6", "#2D2D2D", "#777777"
# 6 colour-blind-friendly hues: 3 "dynamic" (warm) + 3 "static" (cool)
ACT_COLORS = ["#D55E00", "#E69F00", "#F0C200", "#0072B2", "#56B4E9", "#009E73"]


def set_style():
    """Apply a clean, presentation-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#888888", "axes.labelcolor": INK,
        "axes.titlecolor": NAVY, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
        "figure.dpi": 110,
    })


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def fig_class_distribution(y):
    """Bar chart: number of windows per activity."""
    counts = [int((y == a).sum()) for a in range(6)]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(ACTIVITY_SHORT, counts, color=ACT_COLORS, edgecolor="white")
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("windows"); ax.set_title("Class distribution is balanced (ratio 1.4:1)")
    ax.margins(y=0.12); fig.tight_layout()
    return fig


def fig_windows_per_subject(subjects, y):
    """Stacked bar: windows per subject, coloured by activity (shows coverage)."""
    subs = sorted(set(subjects.tolist()))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bottom = np.zeros(len(subs))
    for a in range(6):
        vals = [int(((subjects == s) & (y == a)).sum()) for s in subs]
        ax.bar(range(len(subs)), vals, bottom=bottom, color=ACT_COLORS[a],
               label=ACTIVITY_SHORT[a], edgecolor="white", linewidth=.3)
        bottom += vals
    ax.set_xticks(range(len(subs))); ax.set_xticklabels(subs, fontsize=8)
    ax.set_xlabel("subject id"); ax.set_ylabel("windows")
    ax.set_title("Each subject contributes all six activities")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    return fig


def fig_raw_window(X, y, activity, title=None):
    """Plot one raw 2.56 s window (9 channels in 3 sensor groups)."""
    idx = int(np.where(y == activity)[0][0])
    w = X[idx]                       # (128, 9)
    t = np.arange(WINDOW) / FS
    groups = [("total acc (g)", [0, 1, 2]), ("body acc (g)", [3, 4, 5]),
              ("gyroscope (rad/s)", [6, 7, 8])]
    axis_c = ["#d62728", "#2ca02c", "#1f77b4"]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 5.2), sharex=True)
    for ax, (name, chans) in zip(axes, groups):
        for k, ch in enumerate(chans):
            ax.plot(t, w[:, ch], color=axis_c[k], lw=1.3,
                    label=CHANNELS[ch].split("_")[-1])
        ax.set_ylabel(name, fontsize=10); ax.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(title or f"Raw window — {ACTIVITIES[activity]}")
    fig.tight_layout()
    return fig


def fig_dynamic_vs_static(X, y):
    """Boxplot of within-window body-acceleration magnitude std, per activity."""
    body_mag = np.sqrt((X[:, :, 3:6] ** 2).sum(axis=2))   # (N,128)
    stds = body_mag.std(axis=1)
    data = [stds[y == a] for a in range(6)]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    bp = ax.boxplot(data, patch_artist=True, labels=ACTIVITY_SHORT, showfliers=False)
    for patch, c in zip(bp["boxes"], ACT_COLORS):
        patch.set_facecolor(c); patch.set_alpha(.85)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_ylabel("body-accel magnitude std (g)")
    ax.set_title("Motion energy separates dynamic from static activities")
    ax.axvline(3.5, color=MUTED, ls="--", lw=1)
    ax.text(2, ax.get_ylim()[1]*.92, "dynamic", ha="center", color=NAVY, fontsize=11, weight="bold")
    ax.text(5, ax.get_ylim()[1]*.92, "static", ha="center", color=BLUE, fontsize=11, weight="bold")
    fig.tight_layout()
    return fig


def fig_gravity_orientation(X, y):
    """Scatter of mean total-acceleration on X vs Z — separates static postures by tilt."""
    mx, mz = X[:, :, 0].mean(1), X[:, :, 2].mean(1)
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for a in range(6):
        m = y == a
        ax.scatter(mx[m], mz[m], s=8, alpha=.45, color=ACT_COLORS[a], label=ACTIVITY_SHORT[a])
    ax.set_xlabel("mean total-acc X (g)"); ax.set_ylabel("mean total-acc Z (g)")
    ax.set_title("Gravity direction separates sitting / standing / laying")
    ax.legend(fontsize=8, markerscale=2, loc="best"); fig.tight_layout()
    return fig


def fig_fft_spectra(X, y, channel=3):
    """Mean amplitude spectrum per activity for one channel (gait peak annotated)."""
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for a in range(6):
        rows = X[y == a, :, channel]
        rows = rows - rows.mean(1, keepdims=True)
        amp = (np.abs(np.fft.rfft(rows, axis=1)) * 2 / WINDOW).mean(0)
        ax.plot(freqs, amp, color=ACT_COLORS[a], lw=1.8, label=ACTIVITY_SHORT[a])
    ax.axvline(1.6, color=MUTED, ls="--", lw=1)
    ax.annotate("≈1.6 Hz gait peak", xy=(1.6, ax.get_ylim()[1]*.7),
                xytext=(3.2, ax.get_ylim()[1]*.78), color=NAVY, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlim(0, 10); ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("amplitude")
    ax.set_title(f"Walking shows a sharp gait peak; static postures are near-DC ({CHANNELS[channel]})")
    ax.legend(fontsize=8, ncol=2); fig.tight_layout()
    return fig


def fig_dominant_freq(X, y, channel=3):
    """Boxplot of the dominant (non-DC) frequency per window, by activity."""
    freqs = np.fft.rfftfreq(WINDOW, d=1.0 / FS)
    rows = X[:, :, channel] - X[:, :, channel].mean(1, keepdims=True)
    amp = np.abs(np.fft.rfft(rows, axis=1))
    amp[:, 0] = 0
    dom = freqs[amp.argmax(1)]
    data = [dom[y == a] for a in range(6)]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    bp = ax.boxplot(data, patch_artist=True, labels=ACTIVITY_SHORT, showfliers=False)
    for patch, c in zip(bp["boxes"], ACT_COLORS):
        patch.set_facecolor(c); patch.set_alpha(.85)
    for med in bp["medians"]:
        med.set_color(INK)
    ax.set_ylabel("dominant frequency (Hz)")
    ax.set_title("Walking variants cluster near gait cadence; static near 0 Hz")
    fig.tight_layout()
    return fig


def fig_channel_correlation(X):
    """Correlation heatmap across the 9 channels (all timesteps pooled)."""
    flat = X.reshape(-1, 9)
    c = np.corrcoef(flat.T)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [ch.replace("_", " ") for ch in CHANNELS]
    ax.set_xticks(range(9)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(9)); ax.set_yticklabels(short, fontsize=8)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, f"{c[i,j]:.1f}", ha="center", va="center",
                    color="white" if abs(c[i, j]) > .55 else INK, fontsize=7)
    ax.set_title("Sensor-channel correlation"); fig.colorbar(im, shrink=.8)
    fig.tight_layout()
    return fig


def fig_pca_2d(X, y, n=2500, seed=42):
    """2-D PCA of standardized flattened windows (subsampled)."""
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(X), size=min(n, len(X)), replace=False)
    flat = X[sel].reshape(len(sel), -1)
    flat = (flat - flat.mean(0)) / (flat.std(0) + 1e-8)
    pcs = PCA(n_components=2, random_state=seed).fit_transform(flat)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for a in range(6):
        m = y[sel] == a
        ax.scatter(pcs[m, 0], pcs[m, 1], s=9, alpha=.5, color=ACT_COLORS[a],
                   label=ACTIVITY_SHORT[a])
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.set_title("Activities form clear clusters in 2-D PCA")
    ax.legend(fontsize=8, markerscale=2); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def fig_model_comparison(final_comparison, deployed_f1=0.9477, eng_logreg_f1=0.9557):
    """Head-to-head macro-F1 of the raw-signal models (sensor-ablation rows excluded).

    Fourier+Transformer is highlighted; an honest dashed reference marks the best
    engineered-feature classical baseline, and the deployed (augmented) score is
    annotated on the winning bar.
    """
    rows = [r for r in final_comparison if "Sensor ablation" not in r["model"]]
    rows = sorted(rows, key=lambda r: r["macro_f1"])
    names = [r["model"].split(" - ")[0] for r in rows]
    vals = [r["macro_f1"] for r in rows]
    colors = [NAVY if "Fourier+Transformer" in r["model"] else
              (BLUE if "Transformer" in r["model"] else "#A9C4E0") for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(names, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + .006, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    # honest context: best engineered-feature classical baseline
    ax.axvline(eng_logreg_f1, color=MUTED, ls="--", lw=1.3)
    ax.text(eng_logreg_f1 + 0.006, len(rows) / 2 - 0.5,
            f"561-feat LogReg = {eng_logreg_f1:.3f}", rotation=90,
            va="center", ha="left", fontsize=8, color=MUTED)
    # deployed (augmented) score on the winning bar
    fi = next(i for i, r in enumerate(rows) if "Fourier+Transformer" in r["model"])
    ax.annotate(f"deployed (+aug): {deployed_f1:.3f}", xy=(vals[fi], fi),
                xytext=(0.50, fi - 1.4), fontsize=9, weight="bold", color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("test macro-F1")
    ax.set_title("Fourier+Transformer is the best raw-signal model", pad=12)
    fig.tight_layout()
    return fig


def fig_confusion(y_true, y_pred, title="Confusion matrix — Fourier+Transformer (test)"):
    """Confusion-matrix heatmap (counts) with the sitting/standing pair annotated."""
    cm = np.zeros((6, 6), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6)); ax.set_xticklabels(ACTIVITY_SHORT, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(6)); ax.set_yticklabels(ACTIVITY_SHORT, fontsize=9)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * .5 else INK)
    # box the sitting<->standing confusion (indices 3,4)
    ax.add_patch(plt.Rectangle((2.5, 2.5), 2, 2, fill=False, edgecolor="#D55E00", lw=2))
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title)
    fig.colorbar(im, shrink=.8); fig.tight_layout()
    return fig


def fig_architecture_ablation(ablation):
    """Bar of validation macro-F1 per architecture-ablation config (A5 highlighted)."""
    rows = sorted(ablation, key=lambda r: r["val_macro_f1"])
    names = [r["name"].split("_", 1)[1].replace("_", " ") for r in rows]
    vals = [r["val_macro_f1"] for r in rows]
    colors = [NAVY if "patch8" in r["name"] else BLUE for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.barh(names, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + .002, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.85, 0.95); ax.set_xlabel("validation macro-F1")
    ax.set_title("Positional encoding + patching drive the gains")
    fig.tight_layout()
    return fig


def fig_sensor_ablation(sensor_ablation):
    """Bar of test macro-F1 per sensor subset."""
    label = {"all_sensors": "all 9 channels",
             "acceleration_only_total_plus_body": "acceleration (6)",
             "total_acc_only": "total acc (3)", "body_acc_only": "body acc (3)",
             "gyroscope_only": "gyroscope (3)"}
    rows = sorted(sensor_ablation, key=lambda r: r["test_macro_f1"])
    names = [label.get(r["sensor_group"], r["sensor_group"]) for r in rows]
    vals = [r["test_macro_f1"] for r in rows]
    colors = [NAVY if r["sensor_group"] == "all_sensors" else
              ("#C0504D" if r["sensor_group"] == "gyroscope_only" else BLUE) for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.barh(names, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.4, 0.95); ax.set_xlabel("test macro-F1")
    ax.set_title("Acceleration carries the signal; gyroscope alone is near-chance")
    fig.tight_layout()
    return fig


def _box(ax, x, y, w, h, text, fc, ec, tc, fs=10, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, weight=weight)


def fig_docker_architecture():
    """What the deployment Docker image contains and where it runs."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 7)
    # outer container = the Docker image
    ax.add_patch(FancyBboxPatch((0.3, 1.35), 9.4, 5.25, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#F5F9FD", ec=NAVY, lw=2.2))
    ax.text(5.0, 6.28, "Docker image  —  python:3.11-slim, CPU-only torch (no CUDA → small)",
            ha="center", va="center", color=NAVY, fontsize=11.5, weight="bold")
    # four component boxes (2 x 2)
    _box(ax, 0.8, 4.45, 4.0, 1.05, "app.py\nFlask API + live dashboard", C := "#EBF3FA", BLUE, NAVY, 10)
    _box(ax, 5.2, 4.45, 4.0, 1.05, "fourier_transformer.py\nmodel + load / predict", NAVY, NAVY, "white", 10)
    _box(ax, 0.8, 3.15, 4.0, 1.05, "live_core.py\nwindowing + preprocessing", "#EBF3FA", BLUE, NAVY, 10)
    _box(ax, 5.2, 3.15, 4.0, 1.05, "artifacts/\ncheckpoint + meta + samples", "#EBF3FA", BLUE, NAVY, 10)
    # dependency bar
    _box(ax, 0.8, 1.75, 8.4, 0.95, "Python deps:  torch-cpu · numpy · scipy · flask · gunicorn",
         "#E8F0F8", "#A9C4E0", INK, 10, "normal")
    # exposed endpoint below the image
    ax.add_patch(FancyArrowPatch((5.0, 1.35), (5.0, 1.12), arrowstyle="-|>",
                                 mutation_scale=16, color=MUTED, lw=1.8))
    _box(ax, 2.9, 0.5, 4.2, 0.56, "serves HTTP on :5050  (dashboard + /data)",
         "#E8F3EC", "#4F9D69", "#2E6B45", 10)
    ax.text(5.0, 0.2, "pushed to Docker Hub  →  pulled and run on a RunPod cloud pod",
            ha="center", va="center", color=MUTED, fontsize=9.5)
    fig.tight_layout()
    return fig


def fig_inference_flow():
    """How a single live prediction flows from the phone to the dashboard."""
    fig, ax = plt.subplots(figsize=(10, 3.5)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5)
    stages = ["Phone\naccel + gyro\n~50 Hz", "Window\n2.56 s\n128 × 9",
              "z-score\nsaved\ntrain stats", "Fourier+\nTransformer",
              "softmax\n+ EMA\nsmoothing", "Live\ndashboard"]
    w, h, gap, x0, y = 1.42, 1.15, 0.18, 0.12, 1.45
    cx = []
    for i, s in enumerate(stages):
        x = x0 + i * (w + gap)
        cx.append(x + w / 2)
        hot = i == 3
        _box(ax, x, y, w, h, s, NAVY if hot else "#EBF3FA", NAVY if hot else BLUE,
             "white" if hot else NAVY, 9.5)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (x + w + gap, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=1.5))
    ax.text((cx[0] + cx[1]) / 2, y + h + 0.18, "HTTP POST /data", ha="center",
            color=NAVY, fontsize=8.5, weight="bold")
    ax.text((cx[4] + cx[5]) / 2, y + h + 0.18, "GET /state", ha="center",
            color=NAVY, fontsize=8.5, weight="bold")
    ax.text(5.0, 0.55, "≈ tens of milliseconds per window on CPU  →  several predictions per second",
            ha="center", color=MUTED, fontsize=9.5)
    fig.tight_layout()
    return fig


def fig_deployment_pipeline():
    """Simple box-and-arrow diagram of the real-time inference pipeline."""
    steps = ["Phone\nsensors\n(50 Hz)", "2.56 s\nwindow\n(128×9)",
             "z-score\n(train stats)", "Fourier+\nTransformer", "Activity\nlabel"]
    fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off")
    n = len(steps); w, h, gap = 1.5, 1.2, 0.55
    for i, s in enumerate(steps):
        x = i * (w + gap)
        fc = NAVY if i == 3 else "#EBF3FA"
        tc = "white" if i == 3 else NAVY
        ax.add_patch(FancyBboxPatch((x, 0), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec=BLUE, lw=1.5))
        ax.text(x + w / 2, h / 2, s, ha="center", va="center", color=tc, fontsize=10, weight="bold")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w, h / 2), (x + w + gap, h / 2),
                                         arrowstyle="-|>", mutation_scale=16, color=MUTED, lw=1.5))
    ax.set_xlim(-0.2, n * (w + gap)); ax.set_ylim(-0.3, h + 0.3)
    fig.tight_layout()
    return fig
