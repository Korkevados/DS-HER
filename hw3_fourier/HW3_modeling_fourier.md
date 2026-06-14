# HW3 — Modeling (Fourier-Transform Track)

**Project:** Real-Time Human Activity Recognition from Smartphone Sensors
**Modeling technique covered by our team:** Fourier-transform (frequency-domain) features
**Scope note:** Our group is responsible only for the Fourier-transform modeling
technique. The other modeling tracks (the 561 engineered-feature baselines and the
transformer on raw windows) are handled by the other team. The text below fills
sections 4.1–4.4 of the template for the Fourier track only.

All numbers below come from `fourier_model.py` (reproducible with
`conda activate study && python fourier_model.py`). Figures and metrics are
saved under `hw3_fourier/outputs/`.

---

## 4.1 Select Modeling Technique

### Output — Modeling Technique

We model human activity recognition with a **Fourier-transform feature pipeline
feeding classical supervised classifiers**. The motivation was fixed in HW1
(frequency-domain features for repeated motion) and empirically justified in HW2
§2.3 (Figure 3): walking shows a dominant gait peak near 1.6 Hz with harmonics up
to ~6 Hz, while static activities concentrate almost all their energy below 0.5 Hz.
The frequency domain therefore gives a clean dynamic-vs-static separation that
time-domain means cannot.

Concretely, the technique has two parts:

1. **Fourier feature extraction.** Unlike the other team's track, we do **not** use
   the 561 pre-engineered UCI features. We start from the 9 raw inertial channels
   (`body_acc_{x,y,z}`, `body_gyro_{x,y,z}`, `total_acc_{x,y,z}`), add 2
   orientation-invariant magnitude channels (`body_acc_mag`, `body_gyro_mag`), and
   compute our own spectral feature set per channel from the real FFT (`np.fft.rfft`)
   of each 128-sample (2.56 s) window at 50 Hz. The 128-point real FFT yields 65
   frequency bins spanning 0–25 Hz. Per channel we extract 15 features:
   log spectral energy, dominant AC frequency and its relative magnitude, spectral
   centroid (mean frequency), bandwidth, spectral skewness and kurtosis, spectral
   entropy, 85% roll-off frequency, and the energy fraction in 6 frequency bands
   (0–1, 1–2, 2–3, 3–5, 5–10, 10–25 Hz). With 11 channels this gives a
   **165-dimensional Fourier feature vector** per window.

2. **Classifier.** On top of these features we train three standard scikit-learn
   classifiers and compare them: multinomial **Logistic Regression**, an **RBF SVM**,
   and a **Random Forest**. All are wrapped in a pipeline with `StandardScaler` so
   the Fourier features are on a comparable scale. These models are chosen for being
   simple, fast on a laptop, and explainable — matching the HW1 success criterion of
   keeping the system understandable for a student project.

### Output — Modeling Assumptions

- **Stationarity within a window.** Each 2.56 s window is treated as approximately
  stationary so that a single FFT spectrum summarises it. This is reasonable at a
  human-activity timescale (gait cadence ~1–2 Hz).
- **50 Hz sampling, 128-sample windows.** The FFT bin layout (0–25 Hz, 65 bins)
  assumes the UCI sampling rate. This matches our HW2 data-preparation contract, so
  the same feature code will apply to the resampled Phyphox windows in the live demo.
- **Discriminative energy lives below 25 Hz.** The Nyquist limit (25 Hz) is far above
  the gait band; no relevant motion energy is lost (HW2 §3.3).
- **Magnitude channels mitigate orientation.** The two Euclidean-norm channels are
  rotation-invariant, addressing the per-axis gravity shift documented in HW2 §2.4 —
  important for transfer to the Phyphox demo even though training/testing here is on
  UCI only.
- **Frequency features alone cannot separate two static postures.** Sitting and
  standing are both near-DC, so we expect them to be the model's hardest pair — an
  assumption the assessment confirms.

---

## 4.2 Generate Test Design

### Output — Test Design

We use a supervised multi-class classification design with **error-rate-style
metrics** (accuracy, per-class precision/recall/F1, macro-F1) plus a **confusion
matrix**, exactly as committed to in HW1's success criteria.

**Data split.** We keep the official UCI split: **7,352 training windows (21
subjects)** and **2,947 test windows (9 subjects)**. The split is **subject-disjoint**
(verified clean in HW2 §2.4), which is the correct setup for honest generalisation —
no subject appears in both train and test, so we measure transfer to *new people*,
not memorisation of individuals.

**Model selection.** Because the final test set must stay untouched during tuning,
we select the model on the training set only, using **subject-aware 5-fold
GroupKFold cross-validation** (folds split by subject ID, scored on macro-F1). Macro-F1
is used so the smaller dynamic classes are weighted equally with the larger static
ones. The model with the best CV macro-F1 is reported as the primary model; the
others are reported for comparison.

**Final evaluation.** Each candidate is then re-fit on the full training set and
evaluated **once** on the held-out UCI test set. We report accuracy and macro-F1 for
all three models and a full per-class report + confusion matrix for the primary model.

**Success threshold.** From HW1, the model must at least cleanly separate the main
activities (walking, sitting, standing, lying). We treat **≥ 90% test accuracy with
near-perfect dynamic-vs-static separation** as a pass for this track.

---

## 4.3 Build Model

### Output — Parameter Settings

| Component | Parameter | Value | Rationale |
|---|---|---|---|
| FFT | window length | 128 samples (2.56 s) | UCI window; matches HW2 prep |
| FFT | sampling rate | 50 Hz | UCI rate; 0–25 Hz usable band |
| Features | channels | 11 (9 raw + 2 magnitudes) | adds orientation-invariant cues |
| Features | per-channel features | 15 → **165 total** | energy + spectral shape + bands |
| Scaler | `StandardScaler` | default | put 165 features on equal scale |
| LogisticRegression | `C`, `max_iter` | 1.0, 2000 | simple linear baseline, converges |
| SVC | `kernel`, `C`, `gamma` | rbf, 10.0, scale | non-linear boundary, modest reg. |
| RandomForest | `n_estimators` | 300, `random_state=42` | stable, gives feature importances |
| CV | `GroupKFold` | 5 folds, group = subject | subject-aware model selection |

Shape descriptors (centroid, bandwidth, skew, kurtosis, entropy) are computed on the
**AC spectrum** (DC bin removed) so a large gravity offset does not dominate them;
the static/dynamic energy split is preserved separately through the band-energy
features, which include the DC band.

### Output — Models

Three trained pipelines are produced (`StandardScaler` + classifier each):
`logreg`, `rbf_svm`, `random_forest`. The extracted Fourier feature matrices are
saved to `outputs/fourier_features_{train,test}.csv` so modeling can be re-run
without recomputing FFTs.

### Output — Model Description

**Subject-aware CV (train, macro-F1):**

| Model | CV macro-F1 |
|---|---|
| Logistic Regression | **0.918 ± 0.023** |
| RBF SVM | 0.915 ± 0.029 |
| Random Forest | 0.906 ± 0.042 |

Logistic Regression is the primary model by CV macro-F1 — notable because a *linear*
model on Fourier features already separates the classes well, confirming the features
are highly informative and the model stays fully explainable. The RBF SVM is within
noise; the Random Forest has the highest CV variance.

**Interpretation.** The Random-Forest feature importances
(`outputs/feature_importance.png`) and the per-activity spectra
(`outputs/spectra_per_activity.png`) agree with the physics: the most important
features are spectral energy / low-band energy fractions on the body-acceleration and
magnitude channels — i.e. the dynamic-vs-static cue — followed by the dominant-frequency
and band-energy features that separate the three walking variants by gait intensity.

**Expected shortcoming.** As assumed in 4.1, sitting and standing share a near-DC
spectrum, so the frequency domain has little to separate them. This is the model's
main expected error mode and is examined next.

---

## 4.4 Assess Model

### Output — Model Assessment

**Held-out UCI test set (2,947 windows, 9 unseen subjects):**

| Model | Test accuracy | Test macro-F1 |
|---|---|---|
| Logistic Regression (primary) | 0.9253 | 0.9259 |
| RBF SVM | 0.9281 | 0.9292 |
| **Random Forest** | **0.9362** | **0.9359** |

All three models clear the 90% bar. The Random Forest generalises best on the
test set (93.6% accuracy) despite the linear model winning CV — a small, honest
CV-vs-test ranking flip we report as-is rather than overfitting to it.

**Per-class results (primary model, Logistic Regression):**

| Activity | Precision | Recall | F1 |
|---|---|---|---|
| WALKING | 0.964 | 0.962 | 0.963 |
| WALKING_UPSTAIRS | 0.950 | 0.958 | 0.954 |
| WALKING_DOWNSTAIRS | 0.971 | 0.964 | 0.968 |
| SITTING | 0.888 | 0.758 | 0.818 |
| STANDING | 0.805 | 0.914 | 0.856 |
| LAYING | 0.998 | 0.998 | 0.998 |

**Confusion matrix** (`outputs/confusion_matrix.png`) — the headline result:

- **Zero** confusion between the dynamic group (the three walking types) and the
  static group (sitting/standing/lying). The Fourier features give a *perfect*
  block separation, directly meeting the HW1 requirement to separate the main
  activity types.
- **LAYING is essentially perfect** (536/537), because lying gives a distinct gravity
  distribution and the flattest spectrum.
- **All meaningful error is within-group:** the three walking variants occasionally
  swap with each other, and — as predicted — **SITTING ↔ STANDING is the dominant
  error** (118 sitting windows predicted as standing). Both are near-DC, so frequency
  content alone cannot reliably tell two static postures apart; this needs the
  orientation/tilt information that lives in the per-axis gravity means rather than
  in the spectrum.

**Ranking.** By test accuracy: Random Forest (0.936) > RBF SVM (0.928) > Logistic
Regression (0.925). All are close; the linear model is the most interpretable and
nearly matches the others, so it remains our recommended Fourier-track model unless
the extra ~1% from the Random Forest is needed.

### Output — Revised Parameter Settings

Based on this assessment, the next Build-Model iterations should target the one real
weakness (SITTING ↔ STANDING) and the small headroom between models:

1. **Add a few tilt/orientation features** (e.g. per-axis gravity means from
   `total_acc`, or the spectral DC term per axis) to the Fourier set. The frequency
   spectrum cannot separate the two static postures; a small amount of low-frequency
   orientation information should recover most of the sitting/standing errors while
   keeping the track "Fourier-based at heart".
2. **Light hyper-parameter tuning** of the Random Forest (`max_depth`,
   `min_samples_leaf`) and the SVM (`C`, `gamma`) under the same GroupKFold, since the
   Random Forest has the highest CV variance and the most test-set headroom.
3. **Prune redundant band features** using the Random-Forest importances to keep the
   model compact for the near-real-time inference goal (1.28 s stride between windows).

The current Fourier track already satisfies the HW1 success criteria for this
modeling technique; the revisions above are refinements for the next iteration.
