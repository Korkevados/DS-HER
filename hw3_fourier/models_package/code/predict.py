"""
Runnable demo: load BOTH trained models and classify activity windows.

It uses the bundled real sample (`sample/sample_windows.npy`, 12 windows from the
UCI test set) so it runs with no extra data. Swap in your own (N,128,9) array to
predict on new data.

Run:  python predict.py
Input  to the models: raw inertial windows, shape (N, 128, 9), channel order =
       [total_acc_x, total_acc_y, total_acc_z,
        body_acc_x,  body_acc_y,  body_acc_z,
        body_gyro_x, body_gyro_y, body_gyro_z]
       units: acceleration in g, gyroscope in rad/s, sampled at 50 Hz (2.56 s window).
Output from the models: one activity label per window (+ class probabilities for
       the deep model).
"""
import json
from pathlib import Path

import numpy as np
import joblib

from fourier_transformer import load_model, predict
from fourier_features import predict_classical

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / "models"
SAMPLE = HERE.parent / "sample"

ACT = {1: "WALKING", 2: "WALKING_UPSTAIRS", 3: "WALKING_DOWNSTAIRS",
       4: "SITTING", 5: "STANDING", 6: "LAYING"}


def main():
    # --- load the bundled sample (real UCI test windows) ---
    raw = np.load(SAMPLE / "sample_windows.npy")          # (N,128,9) raw
    y_true = np.load(SAMPLE / "sample_labels.npy")        # (N,) labels 1..6
    print(f"Loaded sample: {raw.shape}  (true labels: {[ACT[i] for i in y_true]})\n")

    # ===== Model 1: Fourier + Transformer (deep) =====
    model, meta = load_model(MODELS / "fourier_transformer_best.pt",
                             MODELS / "model_meta.json")
    labels, names, probs = predict(model, raw, meta)
    print(f"[Deep] Fourier+Transformer  (test acc {meta['test_acc']:.3f})")
    for i, (nm, p) in enumerate(zip(names, probs)):
        print(f"  win {i:2d}: pred={nm:18s} conf={p.max():.2f}  true={ACT[y_true[i]]}")
    acc1 = (labels == (y_true - 1)).mean()
    print(f"  sample accuracy: {acc1:.3f}\n")

    # ===== Model 2: Fourier features + Logistic Regression (classical) =====
    bundle = joblib.load(MODELS / "fourier_model_best.joblib")
    pred2, names2 = predict_classical(bundle, raw)
    print(f"[Classical] Fourier features + {bundle['model_name']}")
    for i, nm in enumerate(names2):
        print(f"  win {i:2d}: pred={nm:18s}  true={ACT[y_true[i]]}")
    acc2 = (np.array(pred2) == y_true).mean()
    print(f"  sample accuracy: {acc2:.3f}")


if __name__ == "__main__":
    main()
