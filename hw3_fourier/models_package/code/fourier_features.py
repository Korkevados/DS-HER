"""
Feature extraction for the classical Fourier model.

The saved pipeline (`fourier_model_best.joblib`) expects a 165-dim Fourier
feature vector per window. This module turns RAW windows (N, 128, 9) into
that feature matrix, then predicts.

Quick use:
    import joblib
    from fourier_features import extract_features, predict_classical
    bundle = joblib.load("../models/fourier_model_best.joblib")
    labels, names = predict_classical(bundle, raw_windows)   # raw: (N,128,9)
"""
import numpy as np
import pandas as pd

FS = 50.0
N = 128
# channel order expected for the RAW input (same as the deep model)
RAW_CHANNELS = ["total_acc_x", "total_acc_y", "total_acc_z",
                "body_acc_x", "body_acc_y", "body_acc_z",
                "body_gyro_x", "body_gyro_y", "body_gyro_z"]
# the classical model was trained with body_acc/body_gyro first; we reorder below.
TRAIN_CHANNELS = ["body_acc_x", "body_acc_y", "body_acc_z",
                  "body_gyro_x", "body_gyro_y", "body_gyro_z",
                  "total_acc_x", "total_acc_y", "total_acc_z"]
ALL_CHANNELS = TRAIN_CHANNELS + ["body_acc_mag", "body_gyro_mag"]
BANDS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 10.0), (10.0, 25.0)]
FREQS = np.fft.rfftfreq(N, d=1.0 / FS)


def _channel_fft_features(x):
    spec = np.fft.rfft(x, axis=1)
    power = np.abs(spec) ** 2
    ac_power = power[:, 1:]
    ac_freqs = FREQS[1:]
    p_sum = ac_power.sum(axis=1) + 1e-12
    p_norm = ac_power / p_sum[:, None]
    f = {}
    f["log_energy"] = np.log1p(power.sum(axis=1) / N)
    dom = ac_power.argmax(axis=1)
    f["dom_freq"] = ac_freqs[dom]
    f["dom_ratio"] = ac_power[np.arange(len(x)), dom] / p_sum
    centroid = (ac_freqs[None, :] * p_norm).sum(axis=1)
    f["centroid"] = centroid
    var = ((ac_freqs[None, :] - centroid[:, None]) ** 2 * p_norm).sum(axis=1)
    f["bandwidth"] = np.sqrt(var)
    std = f["bandwidth"] + 1e-12
    z = (ac_freqs[None, :] - centroid[:, None]) / std[:, None]
    f["skew"] = (z ** 3 * p_norm).sum(axis=1)
    f["kurtosis"] = (z ** 4 * p_norm).sum(axis=1)
    f["entropy"] = -(p_norm * np.log(p_norm + 1e-12)).sum(axis=1)
    cums = np.cumsum(ac_power, axis=1) / p_sum[:, None]
    roll = np.clip((cums < 0.85).sum(axis=1), 0, len(ac_freqs) - 1)
    f["rolloff85"] = ac_freqs[roll]
    for lo, hi in BANDS:
        mask = (FREQS >= lo) & (FREQS < hi)
        f[f"band_{lo:g}_{hi:g}"] = power[:, mask].sum(axis=1) / (power.sum(axis=1) + 1e-12)
    return f


def extract_features(raw):
    """raw: (N,128,9) in RAW_CHANNELS order -> DataFrame of 165 Fourier features."""
    raw = np.asarray(raw, dtype=np.float32)
    assert raw.ndim == 3 and raw.shape[1:] == (128, 9), f"expected (N,128,9), got {raw.shape}"
    # reorder RAW_CHANNELS -> TRAIN_CHANNELS, then append magnitude channels
    order = [RAW_CHANNELS.index(c) for c in TRAIN_CHANNELS]
    sig = raw[:, :, order]
    bacc = sig[:, :, 0:3]; bgyro = sig[:, :, 3:6]
    acc_mag = np.sqrt((bacc ** 2).sum(-1, keepdims=True))
    gyro_mag = np.sqrt((bgyro ** 2).sum(-1, keepdims=True))
    sig = np.concatenate([sig, acc_mag, gyro_mag], axis=-1)  # (N,128,11)
    cols = {}
    for c, name in enumerate(ALL_CHANNELS):
        for fname, vals in _channel_fft_features(sig[:, :, c]).items():
            cols[f"{name}__{fname}"] = vals
    return pd.DataFrame(cols)


def predict_classical(bundle, raw):
    """bundle: dict from joblib.load(). raw: (N,128,9). Returns (labels, names)."""
    X = extract_features(raw)
    X = X[bundle["feature_names"]]            # ensure exact column order
    pred = bundle["pipeline"].predict(X)      # labels in 1..6
    names = [bundle["activity_names"][int(p)] for p in pred]
    return pred, names
