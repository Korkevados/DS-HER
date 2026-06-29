# HW3 — Fourier-Transform Track: Process Report

**Project:** Real-Time Human Activity Recognition from Smartphone Sensors
**Scope:** Our team owns the **Fourier-transform modeling technique**. The other
team owns the 561-engineered-feature baselines and the plain transformer. This
report documents everything we did on the Fourier track, end to end.

All results are reproducible. Code: `fourier_model.py`, `eda.py`,
`cluster/har_experiment.py`. Figures are in `outputs/`, `outputs/eda/`,
`outputs/cluster/`. Environments: local `conda activate study`; cluster
`conda activate subpop` (GPU).

---

## 1. The data we worked on

We use the **UCI HAR Dataset**: smartphone accelerometer + gyroscope at 50 Hz,
worn on the waist by 30 subjects performing 6 activities. The signal is already
windowed into **128-sample (2.56 s) windows with 50% overlap**, and ships with
**9 raw inertial channels** per window (`body_acc {x,y,z}`, `body_gyro {x,y,z}`,
`total_acc {x,y,z}`).

| Property | Value |
|---|---|
| Windows | 7,352 train (21 subjects) + 2,947 test (9 subjects) |
| Split | **subject-disjoint** (no subject in both — honest generalization) |
| Window | 128 samples @ 50 Hz = 2.56 s, 9 channels |
| Classes | WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING |
| Class balance | ratio 1.43 (LAYING 1407 ↔ WALK_DOWN 986) — balanced |
| Per-subject windows | min 281, max 409, mean 350 |
| Raw range | total_acc ∈ [−1.64, 2.20] g; body_gyro ∈ [−5.97, 5.75] rad/s |

### Exploratory data analysis (see `outputs/eda/`)

- **`01_class_distribution.png`** — train and test preserve the same class
  proportions; no class is starved.
- **`02_windows_per_subject.png`** — windows are evenly spread across subjects
  (281–409 each); no single subject dominates.
- **`03_example_windows.png`** — one window per activity. Walking variants
  oscillate; sitting/standing/laying are flat. The dynamic-vs-static split is
  visually obvious from the raw signal.
- **`04_mean_spectrum.png`** — average FFT spectrum per class: walking shows a
  gait peak near **1.6 Hz** with harmonics to ~6 Hz; static activities are
  near-zero. This is the empirical justification for Fourier features.
- **`05_motion_std_hist.png`** — within-window std of body-acc magnitude almost
  perfectly separates dynamic from static:

  | | WALKING | WALK_UP | WALK_DOWN | SITTING | STANDING | LAYING |
  |---|---|---|---|---|---|---|
  | motion std (g) | 0.125 | 0.142 | **0.212** | 0.013 | 0.011 | 0.016 |

- **`06_axis_gravity_means.png`** — per-axis gravity means. **SITTING and
  STANDING have nearly identical gravity on x and differ only subtly on y/z**,
  which is exactly why they get confused; LAYING flips gravity onto y/z and is
  always separable.

  | | x | y | z |
  |---|---|---|---|
  | SITTING | 0.95 | +0.14 | +0.15 |
  | STANDING | 1.00 | −0.16 | −0.03 |
  | LAYING | 0.07 | 0.65 | 0.56 |

---

## 2. Fourier-transform classical model (`fourier_model.py`)

Instead of UCI's 561 ready-made features, we compute **our own frequency-domain
features** from the raw windows via the FFT, then classify.

- **Channels:** 9 raw + 2 orientation-invariant magnitude channels = 11.
- **Per channel (15 features):** log spectral energy, dominant AC frequency and
  its relative magnitude, spectral centroid, bandwidth, skewness, kurtosis,
  entropy, 85% roll-off, and energy fractions in 6 bands (0–1, 1–2, 2–3, 3–5,
  5–10, 10–25 Hz). → **165 Fourier features per window.**
- **Classifiers:** Logistic Regression, RBF-SVM, Random Forest (all with
  `StandardScaler`). Model selection by **subject-aware GroupKFold** on train.

**Results (UCI test set):**

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Logistic Regression | 0.925 | 0.926 |
| RBF SVM | 0.928 | 0.929 |
| Random Forest | **0.936** | **0.936** |

Confusion matrix (`outputs/confusion_matrix.png`): **zero** dynamic↔static
confusion; LAYING near-perfect; the only real error is **SITTING↔STANDING** —
precisely the orientation-only difference seen in the EDA. A *linear* model on
Fourier features reaching 92.5% confirms the features are highly informative.

---

## 3. Fourier + Transformer fusion (deep learning)

We then built a dual-branch deep model that uses self-attention over **both**
representations:

- **Time branch:** transformer encoder over the 128 time tokens.
- **Fourier branch:** transformer encoder over the 65 FFT bins (the FFT is
  computed as a fixed **DFT matmul**, so it runs on GPU/TPU without `torch.fft`).
- **Fusion:** the two CLS embeddings are concatenated → classifier. ~148k params.

This was prepared as a single self-contained Colab cell and as a standalone
cluster script. On a GPU it trains in well under a minute.

---

## 4. The core problem: cross-subject generalization

The models learn the *training* subjects well but degrade on *new* people:

- **Overfitting gap:** baseline train macro-F1 ≈ 0.99 vs test ≈ 0.93.
- **The error is concentrated, not uniform:** dynamic↔static is never confused
  and LAYING is essentially perfect; almost all error is SITTING↔STANDING plus
  some within-walking confusion.
- **A few hard subjects dominate the error** (subjects 9, 10).

This motivated a robustness study, run on the BGU GPU cluster.

---

## 5. Robustness study on the GPU cluster (`cluster/`)

We ran a **seed-averaged ablation (3 seeds × 3 configs × 25 epochs)** on one GPU
(`subpop` env, GTX 1080 Ti), training the Fourier+Transformer with on-the-fly
augmentation applied to the raw 128×9 windows in training only:

- **jitter** (σ=0.05), **scaling** (σ=0.10), **small rotation** (±20° — small on
  purpose: full random rotation would erase the gravity cue that separates
  sitting from standing).

**Configurations:** `baseline` (no aug), `aug` (jitter+scale+rotate),
`aug_reg` (aug + dropout 0.30 + weight decay 5e-4 + label smoothing + early stop).

**Results** (`outputs/cluster/ablation_comparison.png`):

| Config | Test macro-F1 | Test acc | Overfit gap |
|---|---|---|---|
| baseline | 0.9287 ± 0.0046 | 0.931 | +0.063 |
| **aug** | **0.9459 ± 0.0013** | **0.947** | **+0.029** |
| aug + heavy reg | 0.9122 ± 0.0178 | 0.913 | +0.046 |

**Findings:**
1. **Augmentation alone is the winner** — +1.7 macro-F1 points over baseline, the
   **overfit gap more than halved** (0.063 → 0.029), and the **lowest seed
   variance** (±0.001).
2. **Stacking heavy regularization on top hurt** at this epoch budget — it
   underfits (one seed fell to 0.888). Honest takeaway: augmentation is the
   high-value lever; piling on dropout + label smoothing + early stopping was
   counterproductive here.
3. **Per-subject** (`outputs/cluster/per_subject_f1.png`): mean 0.941 ± 0.072;
   hardest subjects **10 (0.78), 9 (0.87), 12 (0.90)** — confirming the
   "difficult subjects 9–10" diagnosis. Augmentation lifted subject 10 markedly.

Best model per-class (config `aug`, `outputs/cluster/best_classification_report.txt`):
LAYING 0.997 F1, walking variants 0.94–0.96, and SITTING/STANDING the lowest
(0.91 / 0.93) — the same residual pattern, now softened.

---

## 6. Summary & next steps

- The Fourier track meets the project success criteria: a clean, reproducible
  pipeline that separates the main activities, evaluated honestly on a
  subject-disjoint split.
- Frequency features give a perfect dynamic-vs-static split; the only stubborn
  error (SITTING↔STANDING) is an orientation problem, not a frequency one.
- **Augmentation is the most effective robustness tool** for cross-subject
  generalization; heavy regularization on top is not needed at this scale.

**Recommended next steps:** add a small tilt/orientation feature to attack
SITTING↔STANDING directly; run true Leave-One-Subject-Out CV to quantify the hard
subjects systematically; and (for the live Phyphox demo) raise the rotation
strength to trade some UCI accuracy for cross-orientation transfer.

---

### File index

```
REPORT.md                         this document
fourier_model.py                  classical Fourier-feature model (script)
eda.py                            data EDA (script)
plot_cluster.py                   renders ablation plots from summary.json
cluster/har_experiment.py         Fourier+Transformer robustness ablation (GPU)
cluster/run_har.slurm             SLURM job script
outputs/
  confusion_matrix.png            classical Fourier model confusion matrix
  spectra_per_activity.png        per-activity FFT spectra
  feature_importance.png          top Fourier features (random forest)
  metrics.json                    classical model metrics
  classification_report.txt       classical model per-class report
  eda/                            6 data EDA plots + data_stats.json
  cluster/                        ablation_comparison.png, confusion_matrix_best.png,
                                  per_subject_f1.png, summary.json,
                                  best_classification_report.txt, ablation_job.log
```
