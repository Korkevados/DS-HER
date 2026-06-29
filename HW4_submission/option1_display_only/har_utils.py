"""
har_utils — small, readable helpers shared by the HW4 notebooks.

Keeps the "plumbing" (locating the UCI HAR data on disk, loading the raw
128x9 windows in the model's channel order, and a couple of signal helpers)
out of the notebook so the notebook cells stay focused on EDA and modelling.

The UCI HAR Dataset is loaded directly from the local copy that ships with the
project — no download required.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Constants — match the trained model exactly (see artifacts/model_meta.json)
# --------------------------------------------------------------------------- #
FS = 50                       # sampling rate (Hz)
WINDOW = 128                  # samples per window  (128 / 50 Hz = 2.56 s)
WINDOW_SEC = WINDOW / FS

# Channel order the FourierTransformer was trained on.
CHANNELS = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x",  "body_acc_y",  "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]

# Class index (0..5) -> activity name, as used by the model.
ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
              "SITTING", "STANDING", "LAYING"]

# Short, colour-blind-friendly labels for plots.
ACTIVITY_SHORT = ["Walking", "Up-stairs", "Down-stairs", "Sitting", "Standing", "Laying"]

SEED = 42                     # one seed everywhere for reproducibility


# --------------------------------------------------------------------------- #
# Locating the data
# --------------------------------------------------------------------------- #
def find_uci_dir(start: Path | None = None) -> Path:
    """Walk up from `start` (default: this file) until the UCI HAR folder is found."""
    here = (start or Path(__file__).resolve()).resolve()
    needle = Path("Data") / "human+activity+recognition+using+smartphones"
    for parent in [here, *here.parents]:
        cand = parent / needle
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not locate 'Data/human+activity+recognition+using+smartphones'. "
        "Run the notebook from inside the project tree."
    )


# --------------------------------------------------------------------------- #
# Loading windows
# --------------------------------------------------------------------------- #
def load_raw_windows(split: str = "train", uci_dir: Path | None = None):
    """Load raw inertial windows in the model's channel order.

    Returns
    -------
    X : float32 array (N, 128, 9)   raw signal windows
    y : int array    (N,)           class index 0..5
    subjects : int array (N,)       subject id per window
    """
    uci = Path(uci_dir) if uci_dir else find_uci_dir()
    folder = uci / split
    sig = folder / "Inertial Signals"

    channels = [np.loadtxt(sig / f"{c}_{split}.txt") for c in CHANNELS]  # each (N,128)
    X = np.stack(channels, axis=-1).astype(np.float32)                   # (N,128,9)

    y = np.loadtxt(folder / f"y_{split}.txt").astype(int) - 1            # 1..6 -> 0..5
    subjects = np.loadtxt(folder / f"subject_{split}.txt").astype(int)
    return X, y, subjects


def load_engineered_features(split: str = "train", uci_dir: Path | None = None):
    """Load the 561 pre-engineered UCI features and their names (for EDA only)."""
    uci = Path(uci_dir) if uci_dir else find_uci_dir()
    X = np.loadtxt(uci / split / f"X_{split}.txt")
    names = [ln.split(maxsplit=1)[1] for ln in
             (uci / "features.txt").read_text().strip().splitlines()]
    y = np.loadtxt(uci / split / f"y_{split}.txt").astype(int) - 1
    return X, names, y


# --------------------------------------------------------------------------- #
# Signal helpers
# --------------------------------------------------------------------------- #
def amplitude_spectrum(window_1d: np.ndarray, fs: int = FS):
    """Single-sided amplitude spectrum of one 1-D window. Returns (freqs, amps)."""
    w = np.asarray(window_1d, dtype=float)
    w = w - w.mean()                                  # drop DC (gravity offset)
    n = len(w)
    amp = np.abs(np.fft.rfft(w)) * 2.0 / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, amp


def mean_spectrum_per_activity(X: np.ndarray, y: np.ndarray, channel: int):
    """Average amplitude spectrum of one channel, grouped by activity.

    Returns (freqs, {activity_index: mean_amplitude_vector}).
    """
    freqs = np.fft.rfftfreq(X.shape[1], d=1.0 / FS)
    out = {}
    for a in range(len(ACTIVITIES)):
        rows = X[y == a, :, channel]
        rows = rows - rows.mean(axis=1, keepdims=True)
        amps = np.abs(np.fft.rfft(rows, axis=1)) * 2.0 / X.shape[1]
        out[a] = amps.mean(axis=0)
    return freqs, out


def load_meta(artifacts_dir: Path | str = "artifacts") -> dict:
    """Read the saved model_meta.json (class map, channel order, norm stats)."""
    return json.loads((Path(artifacts_dir) / "model_meta.json").read_text())
