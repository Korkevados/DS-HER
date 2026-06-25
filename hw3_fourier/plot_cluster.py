"""Render plots from the cluster ablation summary.json."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "cluster"
S = json.loads((OUT / "summary.json").read_text())
CLASSES = S["class_names"]

# ---- 1. ablation comparison (test macro-F1 with std + overfit gap) ----
configs = list(S["results"].keys())
f1 = [S["results"][c]["test_f1_mean"] for c in configs]
f1e = [S["results"][c]["test_f1_std"] for c in configs]
gap = [S["results"][c]["overfit_gap_mean"] for c in configs]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
a1.bar(configs, f1, yerr=f1e, capsize=6, color=["#999", "#2a9d8f", "#e76f51"])
a1.set_ylim(0.88, 0.96); a1.set_ylabel("test macro-F1 (3-seed mean)")
a1.set_title("Robustness ablation — test macro-F1")
for i, (v, e) in enumerate(zip(f1, f1e)):
    a1.text(i, v + e + 0.002, f"{v:.3f}", ha="center", fontsize=9)
a2.bar(configs, gap, color=["#999", "#2a9d8f", "#e76f51"])
a2.set_ylabel("train F1 − test F1 (overfit gap)")
a2.set_title("Overfitting gap (lower = better generalization)")
for i, v in enumerate(gap):
    a2.text(i, v + 0.001, f"{v:+.3f}", ha="center", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "ablation_comparison.png", dpi=150); plt.close(fig)

# ---- 2. confusion matrix of best config ----
cm = np.array(S["confusion_matrix_best"])
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(6)); ax.set_yticks(range(6))
ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"Best model (aug) — confusion matrix\nacc {S['results']['aug']['test_acc_mean']:.3f}")
for i in range(6):
    for j in range(6):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout(); fig.savefig(OUT / "confusion_matrix_best.png", dpi=150); plt.close(fig)

# ---- 3. per-subject macro-F1 (best config) ----
ps = S["per_subject"]["subject_macro_f1"]
subs = sorted(ps, key=lambda k: int(k))
vals = [ps[s] for s in subs]
mean = S["per_subject"]["subject_f1_mean"]
fig, ax = plt.subplots(figsize=(11, 4.5))
colors = ["#e76f51" if v < 0.90 else "#2a9d8f" for v in vals]
ax.bar(subs, vals, color=colors)
ax.axhline(mean, color="k", ls="--", lw=1, label=f"mean {mean:.3f}")
ax.set_ylim(0.6, 1.02); ax.set_xlabel("test subject id"); ax.set_ylabel("macro-F1")
ax.set_title("Per-subject generalization (best=aug) — red = hardest subjects")
ax.legend(); fig.tight_layout()
fig.savefig(OUT / "per_subject_f1.png", dpi=150); plt.close(fig)

print("Wrote ablation_comparison.png, confusion_matrix_best.png, per_subject_f1.png")
print("worst subjects:", S["per_subject"]["worst_subjects"])
