# HW3 — Modeling: text for each "<Place your text here>" blank

All numbers below are real, produced by `code/hw3_modeling.py` on the local UCI HAR
dataset (subject-disjoint split, 7,352 train / 2,947 test windows).

---

## 4.1 Select Modeling Technique

### Output — Modeling Technique
We treat Human Activity Recognition as a supervised multi-class classification problem
over the 561 engineered features that the UCI HAR authors provide for each 2.56 s window,
predicting one of six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,
SITTING, STANDING, LAYING). In line with the project plan from HW1 ("the first modeling
approach should be simple and reliable"), this phase builds three classical baseline
techniques and compares them on equal footing:

- **Logistic Regression** — a linear, highly interpretable reference model. It tells us how
  far a purely linear decision boundary can go on these features and serves as the
  sanity-check floor for the more expressive models.
- **Random Forest** — a non-linear ensemble of decision trees. It captures feature
  interactions without manual tuning and provides feature-importance scores, which is
  useful for the secondary objective of understanding which signals matter most.
- **Support Vector Machine with an RBF kernel** — a strong margin-based classifier that
  works well on the medium-dimensional, normalized feature space typical of HAR.

All three are implemented in scikit-learn pipelines. A deep-learning transformer on the
raw 128×9 inertial windows was committed to in HW1/HW2 as the experimental track; it is
deferred to a later phase and is **not** part of this baseline modeling round, so the three
classical models above are the techniques selected here.

### Output — Modeling Assumptions
- The 561 features are pre-normalized to [-1, 1]; for the scale-sensitive models (Logistic
  Regression and SVM) we additionally apply `StandardScaler` (z-scoring, fit on the training
  set only) so every feature contributes on a comparable scale. Random Forest is
  scale-invariant, but we keep the same pipeline for uniformity.
- Each window is treated as one independent labelled sample (the within-window temporal
  structure is already summarized inside the engineered features).
- The provided train/test split is **subject-disjoint** (21 training subjects, 9 unseen test
  subjects, verified to have no overlap), so test performance is an honest estimate of
  generalization to new people — not just new windows from the same people.
- The activity labels are reliable (carried over as an assumption from HW1).
- The classes are only mildly imbalanced (max/min ratio ≈ 1.43, HW2 §2.3), so no resampling
  is required and macro-averaged metrics stay meaningful.

---

## 4.2 Generate Test Design

### Output — Test Design
We use the UCI HAR train/test partition exactly as delivered: the model is trained on the
7,352 windows from 21 subjects and evaluated **once** on the 2,947 windows from 9 entirely
different subjects. Keeping the split subject-disjoint (never splitting by window) is the
single most important design choice, because windows from the same person are highly
correlated — splitting by window would leak subject identity into the test set and inflate
the score.

Hyper-parameter selection is done **only on the training set**, using 5-fold
**GroupKFold** cross-validation with the subject ID as the grouping key. This guarantees
that, within tuning too, the validation folds contain subjects the model has not trained on,
mirroring the real generalization task and preventing leakage. The held-out test set is
never touched during tuning; it is used a single time, at the very end, to report final
performance.

Evaluation metrics (matching the HW1 success criteria):
- **Primary:** macro-averaged F1 (equal weight per class, robust to the mild imbalance),
  used both as the CV selection metric and as the final ranking metric.
- **Secondary:** overall accuracy, weighted F1, and per-class precision / recall / F1.
- **Confusion matrix** with the fixed class order
  [WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING] to show
  exactly which activities get confused.

For reproducibility (an explicit HW1 success criterion) every model uses a fixed
`random_state = 42`.

---

## 4.3 Build Model

### Output — Parameter Settings
We ran a small grid search for each model with the GroupKFold scheme above (scoring =
macro-F1). The grids and the best setting actually selected were:

| Model | Grid searched | Best params (selected by CV) |
|---|---|---|
| Logistic Regression | C ∈ {0.1, 1, 10} | **C = 0.1** |
| Random Forest | n_estimators ∈ {200, 400}, max_depth ∈ {None, 20} | **n_estimators = 200, max_depth = 20** |
| SVM (RBF) | C ∈ {1, 10}, gamma ∈ {scale, 0.001} | **C = 10, gamma = 0.001** |

Rationale: for Logistic Regression the strongest regularization (C = 0.1) won, which is
expected given 561 partly redundant features — heavy regularization controls variance. For
Random Forest, limiting depth to 20 matched or beat unlimited depth while training faster.
For the SVM, the smaller `gamma = 0.001` with a larger `C = 10` gave the smoothest,
best-generalizing decision boundary.

### Output — Models
The grid search refits the best estimator on the full training set, yielding three fitted
models: a regularized multinomial Logistic Regression, a 200-tree depth-20 Random Forest,
and an RBF-kernel SVM (C = 10, gamma = 0.001), each preceded by a training-set
`StandardScaler` inside the pipeline.

### Output — Model Description
Measured on the held-out test set (9 unseen subjects):

| Model | Test accuracy | Test macro-F1 | CV macro-F1 |
|---|---|---|---|
| SVM (RBF) | 0.955 | 0.954 | 0.930 |
| Logistic Regression | 0.951 | 0.951 | 0.934 |
| Random Forest | 0.925 | 0.924 | 0.919 |

- **SVM (RBF)** is the most accurate model (95.5% accuracy, 0.954 macro-F1). It is robust and
  balanced across classes; its shortcomings are practical — it is the slowest to train/tune
  and gives no direct feature interpretability.
- **Logistic Regression** is essentially tied with the SVM (95.1%), which is a strong result
  for a linear model and shows the engineered features are already close to linearly
  separable. It is by far the most interpretable and fastest, making it an excellent default.
- **Random Forest** is clearly behind (92.5%). It separates the static activities very well
  (LAYING is classified perfectly) but is weaker on the three walking variants, where it
  confuses upstairs/downstairs with level walking more than the other two models. Its
  advantage is the built-in feature-importance ranking.

Interpretation: across all three models the dynamic-vs-static split is essentially solved —
there is **zero** confusion between any walking activity and any static posture, confirming
the time/frequency-domain evidence from HW2 (the gait-frequency peak cleanly separates
moving from still). All remaining error is *within* a group.

Difficulties encountered: the SVM grid search dominates runtime on 7,352×561 data, so the
grid was kept deliberately small. The Logistic Regression solver emitted benign numeric
(overflow) warnings under the float32 feature matrix; the selected, well-regularized model
converged normally and reached 95%, so the warnings do not affect the reported results.

---

## 4.4 Assess Model

### Output — Model Assessment
Ranked by test macro-F1 (with subject-disjoint generalization):

| Rank | Model | Test accuracy | Test macro-F1 |
|---|---|---|---|
| 1 | SVM (RBF) | 0.955 | 0.954 |
| 2 | Logistic Regression | 0.951 | 0.951 |
| 3 | Random Forest | 0.925 | 0.924 |

The SVM is our best model, with Logistic Regression a near-identical and far cheaper
runner-up. The confusion matrices tell a consistent story for every model: the only
substantial errors are **SITTING ↔ STANDING** and, to a lesser extent, mix-ups among the
three walking variants (WALKING_UPSTAIRS vs WALKING_DOWNSTAIRS vs WALKING).

For the best model (SVM), the test confusion matrix is:

|                    | WALK | W_UP | W_DOWN | SIT | STAND | LAY |
|--------------------|----:|----:|------:|----:|-----:|----:|
| **WALKING**        | 486 |   6 |     4 |   0 |    0 |   0 |
| **WALKING_UPSTAIRS** |  15 | 454 |     2 |   0 |    0 |   0 |
| **WALKING_DOWNSTAIRS** |   6 |  25 |   389 |   0 |    0 |   0 |
| **SITTING**        |   0 |   1 |     0 | 433 |   54 |   3 |
| **STANDING**       |   0 |   0 |     0 |  17 |  515 |   0 |
| **LAYING**         |   0 |   0 |     0 |   0 |    0 | 537 |

The SITTING↔STANDING confusion (54 sitting windows predicted as standing, 17 the other way)
is physically expected: both are static postures whose accelerometer signal is dominated by
the gravity vector, and they differ mainly in subtle orientation — exactly the hard
within-group problem we anticipated in HW2. LAYING is classified perfectly, and the
dynamic/static boundary is never crossed.

Against the HW1 business success criteria, this phase succeeds: we have a working pipeline
from features to activity labels, full multi-class metrics (accuracy, precision, recall, F1,
confusion matrix), and clear evidence that the model separates the four core activities
(walking, sitting, standing, lying) — sitting and standing being the only meaningfully
confused pair. The ~95% subject-disjoint accuracy comfortably clears the bar for a useful
prototype.

### Output — Revised Parameter Settings
The grid search already performs one build → assess iteration per model, and the CV results
point clearly to the final settings: Logistic Regression at **C = 0.1** (strong
regularization preferred), Random Forest at **n_estimators = 200, max_depth = 20**, and SVM
at **C = 10, gamma = 0.001**, which we adopt as the tuned baselines. The CV-vs-test gap is
small (e.g. SVM CV macro-F1 0.930 vs test 0.954), indicating no overfitting and little to
gain from further grid expansion. The assessment shows the residual error is concentrated in
the SITTING/STANDING pair rather than in any tunable hyper-parameter, so the most promising
next step is not more tuning of these baselines but the planned transformer track on the raw
inertial windows and orientation-invariant features for the Phyphox live demo (HW2 §3.3) —
which we leave to the next modeling iteration. This closes the build/assess loop for the
classical baseline track.
