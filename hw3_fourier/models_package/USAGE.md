# HW3 Models — Human Activity Recognition (Fourier track)

Two trained models for classifying a 2.56 s smartphone sensor window into one of
six activities. Both are bundled with the code needed to load them and a runnable
demo.

```
HW3_Models/
├── USAGE.md                       this file
├── models/
│   ├── fourier_transformer_best.pt    deep model weights (PyTorch state_dict)
│   ├── model_meta.json                deep model metadata (kwargs, normalization, classes)
│   └── fourier_model_best.joblib      classical pipeline (scaler + logistic regression)
├── code/
│   ├── fourier_transformer.py     loader + inference for the deep model
│   ├── fourier_features.py        165-feature extractor + inference for the classical model
│   └── predict.py                 runnable demo using both models
└── sample/
    ├── sample_windows.npy         12 real UCI test windows (2 per class), shape (12,128,9)
    └── sample_labels.npy          their true labels (1..6)
```

**Run the demo:** `cd code && python predict.py`
Requirements: `python>=3.9`, `torch`, `numpy`, `scikit-learn`, `pandas`, `joblib`.

---

## The two models

| | Deep — Fourier+Transformer | Classical — Fourier features + LogReg |
|---|---|---|
| File | `fourier_transformer_best.pt` | `fourier_model_best.joblib` |
| Input | raw windows `(N,128,9)` | raw windows `(N,128,9)` |
| How it works | self-attention over time tokens **and** FFT tokens, fused | 165 hand-built FFT features → logistic regression |
| Test accuracy (UCI) | **0.948** | 0.925 |
| Gives probabilities | yes (softmax) | labels only (`predict_proba` available on the pipeline) |
| Parameters / size | ~148k / 697 KB | ~22 KB |
| Best for | highest accuracy, live demo | fast, fully interpretable baseline |

Both reach the same conclusion on most windows; the only routine disagreement is
**SITTING vs STANDING** (two static postures that differ only by device tilt).

---

## Input format (identical for both models)

A batch of windows: a float array of shape **`(N, 128, 9)`**.

- **128** = time steps (2.56 s at **50 Hz**).
- **9** = sensor channels, in this exact order:

  | idx | channel | sensor | unit |
  |---|---|---|---|
  | 0–2 | total_acc x/y/z | accelerometer (incl. gravity) | g (1 g = 9.80665 m/s²) |
  | 3–5 | body_acc x/y/z | accelerometer (gravity removed) | g |
  | 6–8 | body_gyro x/y/z | gyroscope | rad/s |

> If you record with Phyphox (m/s²), divide accelerometer values by 9.80665 to get
> g, resample to 50 Hz, and window into 128 samples (64-sample stride for 50%
> overlap). `body_acc = total_acc − gravity` (low-pass the gravity component).

**Preprocessing is handled inside the code** — pass RAW windows:
- Deep model: `preprocess()` z-scores each channel with the **saved training
  mean/std** (stored in `model_meta.json`). Never refit normalization on new data.
- Classical model: `extract_features()` computes the 165 Fourier features
  (it also reorders channels and adds the 2 magnitude channels internally).

---

## Output format

- **labels** — int per window. Deep model: `0..5`. Classical model: `1..6`.
- **names** — the activity string per window.
- **probs** (deep model only) — `(N, 6)` softmax probabilities; `probs.max(1)` is
  the confidence. Low confidence flags the hard SITTING/STANDING cases.

Class mapping:

| deep idx | classical label | activity |
|---|---|---|
| 0 | 1 | WALKING |
| 1 | 2 | WALKING_UPSTAIRS |
| 2 | 3 | WALKING_DOWNSTAIRS |
| 3 | 4 | SITTING |
| 4 | 5 | STANDING |
| 5 | 6 | LAYING |

---

## Code examples

### Deep model (Fourier + Transformer)
```python
import numpy as np
from fourier_transformer import load_model, predict

model, meta = load_model("../models/fourier_transformer_best.pt",
                         "../models/model_meta.json", device="cpu")

raw = np.load("../sample/sample_windows.npy")        # (N,128,9) RAW, channel order above
labels, names, probs = predict(model, raw, meta)     # preprocessing done inside
print(names[0], probs[0].max())                      # e.g. 'WALKING' 1.00
```

### Classical model (Fourier features + Logistic Regression)
```python
import joblib, numpy as np
from fourier_features import predict_classical

bundle = joblib.load("../models/fourier_model_best.joblib")
raw = np.load("../sample/sample_windows.npy")        # (N,128,9) RAW
labels, names = predict_classical(bundle, raw)       # feature extraction done inside
print(names[0])                                       # e.g. 'WALKING'
```

### Single live window
```python
window = np.asarray(my_window, dtype=np.float32)[None]   # shape (1,128,9)
_, names, probs = predict(model, window, meta)
print(names[0], float(probs[0].max()))
```

---

## How they were trained (for reference)

- **Data:** UCI HAR Dataset, 50 Hz, waist-worn, subject-disjoint train/test split.
- **Deep model:** trained on the cluster GPU with on-the-fly augmentation
  (jitter + scaling + small ±20° rotation) — the configuration that gave the best
  cross-subject generalization (test macro-F1 0.946, overfit gap halved vs no aug).
- **Classical model:** logistic regression selected by subject-aware GroupKFold
  over the 165 Fourier features.

## Limitations
- Trained on waist-mounted data: a different phone position (hand/pocket) shifts
  the gravity axis and degrades accuracy. Use the magnitude channels / more
  rotation augmentation for cross-orientation robustness.
- SITTING vs STANDING is the main residual error (orientation, not frequency).
- Expects exactly 128 samples at 50 Hz; resample other rates first.
