"""
Live preprocessing + class grouping for the real-time HAR demo (self-contained).

Turns the phone's irregular accelerometer + gyroscope stream into the RAW
(128, 9) window the model expects, in the meta channel order:
    [total_acc x/y/z, body_acc x/y/z, body_gyro x/y/z]   (acc in g, gyro rad/s)

The model's own preprocess() then z-scores with the saved training statistics —
we deliberately do NOT normalize here.

For the live demo we report three coarse states — walking / sitting / lying —
by summing the model's six-class probabilities into three groups. Merging the
near-identical static postures and the walking variants makes the real-time
prediction far steadier when streaming from a hand-held phone.
"""
import numpy as np
from scipy.signal import butter, filtfilt

FS = 50.0            # target rate (Hz)
N = 128              # samples per window
WINDOW_SEC = N / FS  # 2.56 s

# The model's native six outputs, in index order 0..5.
MODEL_LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
                "SITTING", "STANDING", "LAYING"]

# Three coarse states shown in the demo, and which model indices feed each.
DEMO_LABELS = ["WALKING", "SITTING", "LYING"]
DEMO_EMOJI = {"WALKING": "🚶", "SITTING": "🪑", "LYING": "🛌"}
GROUPS = {
    "WALKING": [0, 1, 2],   # WALKING + UPSTAIRS + DOWNSTAIRS
    "SITTING": [3, 4],      # SITTING + STANDING (both upright & static)
    "LYING":   [5],         # LAYING
}

# 0.3 Hz high-pass to derive body_acc (gravity removed) from total_acc, matching
# how UCI built body_acc. Same filter used live so the input matches training.
_B_HP, _A_HP = butter(3, 0.3 / (FS / 2.0), btype="highpass")


def collapse_probs(p6):
    """Sum the 6 model probabilities into the 3 demo states (order = DEMO_LABELS)."""
    p6 = np.asarray(p6, dtype=float).ravel()
    return np.array([p6[GROUPS[name]].sum() for name in DEMO_LABELS])


def autoscale_to_g(acc):
    """Sensor Logger may report m/s^2 or g. Convert to g if it looks like m/s^2."""
    acc = np.asarray(acc, dtype=float)
    mag = np.median(np.linalg.norm(acc, axis=1))
    return acc / 9.80665 if mag > 4.0 else acc


def resample_window(times, values, t_end, fs=FS, n=N):
    """Linearly resample irregular samples onto n uniform points ending at t_end."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    grid = np.linspace(t_end - n / fs, t_end, n)
    out = np.empty((n, values.shape[1]))
    for k in range(values.shape[1]):
        out[:, k] = np.interp(grid, times, values[:, k])
    return out


def build_window(acc_g, gyro):
    """acc_g: (128,3) total acceleration in g; gyro: (128,3) rad/s.
    Returns RAW (1,128,9) in the model's channel order."""
    body_acc = filtfilt(_B_HP, _A_HP, acc_g, axis=0)          # gravity removed
    window = np.concatenate([acc_g, body_acc, gyro], axis=1)  # (128,9)
    return window.astype(np.float32)[None]                    # (1,128,9)
