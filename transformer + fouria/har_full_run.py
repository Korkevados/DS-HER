"""
HAR — Transformer + Fourier: full modeling run (SLURM-ready).

Self-contained conversion of HAR_Transformer_Deep_Colab_v2.ipynb into a single
script that runs the complete cycle headless and writes all results to --out:

  Baselines
    1. Logistic Regression on the 561 engineered features
    2. Random Forest on handcrafted time + FFT features
  Deep models (raw 128x9 inertial windows, subject-wise validation split)
    3. Main Transformer  (timestep tokens, sinusoidal PE, CLS pooling)
    4. Transformer ablation study (7 configs, validation-only model selection)
    5. Patch Transformer (patch_size=8)
    6. CNN-Transformer hybrid (Conv1D frontend + Transformer)
    7. Sensor ablation (which channel groups matter)
    8. Fourier+Transformer (dual-branch: time-domain + FFT-magnitude self-attention)
  Analysis
    confusion matrices, per-subject accuracy, calibration/ECE, final comparison

Outputs (under --out, default ./outputs):
    metrics.json                 all numbers in one file
    final_comparison.csv         model leaderboard by test macro-F1
    per_subject_main.csv         per-subject accuracy for the main transformer
    *.png                        training curves, confusion matrices, bar charts

The Fourier+Transformer definition is a faithful reconstruction of the model the
notebook described but no longer contained (see README).

Usage:
    python har_full_run.py                 # full run, auto-detect data + device
    python har_full_run.py --fast          # quick smoke run (few epochs)
    python har_full_run.py --data /path/to/"UCI HAR Dataset" --out ./outputs
"""

import argparse
import json
import math
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: never tries to open a window
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
SIGNAL_NAMES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]
CHANNEL_GROUPS = {
    "all_sensors": list(range(9)),
    "acceleration_only_total_plus_body": [0, 1, 2, 3, 4, 5],
    "gyroscope_only": [6, 7, 8],
    "total_acc_only": [0, 1, 2],
    "body_acc_only": [3, 4, 5],
}
CLASS_NAMES = [
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
    "SITTING", "STANDING", "LAYING",
]

# globals filled in main()
DEVICE: torch.device = torch.device("cpu")
OUT: Path = Path("outputs")
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def has_full_uci_structure(base: Path) -> bool:
    required = [
        base / "train" / "X_train.txt",
        base / "test" / "X_test.txt",
        base / "train" / "y_train.txt",
        base / "test" / "y_test.txt",
        base / "train" / "subject_train.txt",
        base / "test" / "subject_test.txt",
        base / "train" / "Inertial Signals" / "body_acc_x_train.txt",
        base / "test" / "Inertial Signals" / "body_acc_x_test.txt",
    ]
    return all(p.exists() for p in required)


def resolve_dataset(arg_data: Optional[str]) -> Path:
    here = Path(__file__).resolve().parent
    candidates = []
    if arg_data:
        candidates.append(Path(arg_data))
    if os.environ.get("HAR_DATA"):
        candidates.append(Path(os.environ["HAR_DATA"]))
    candidates += [
        here / "data" / "UCI HAR Dataset",
        here / "data",
        here.parent / "Data" / "human+activity+recognition+using+smartphones",
        here / "UCI HAR Dataset",
        Path("/mnt/data/UCI HAR Dataset"),
    ]
    for c in candidates:
        if c and has_full_uci_structure(c):
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        "Could not find the UCI HAR Dataset. Pass --data /path/to/'UCI HAR Dataset' "
        "or set HAR_DATA. Looked in:\n  " + tried
    )


def _load(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def load_engineered(split: str, base: Path):
    X = _load(base / split / f"X_{split}.txt")
    y = _load(base / split / f"y_{split}.txt").astype(int) - 1
    subj = _load(base / split / f"subject_{split}.txt").astype(int)
    return X, y, subj


def load_raw(split: str, base: Path) -> np.ndarray:
    sigs = [_load(base / split / "Inertial Signals" / f"{s}_{split}.txt") for s in SIGNAL_NAMES]
    return np.stack(sigs, axis=-1).astype(np.float32)  # samples x 128 x 9


# ----------------------------------------------------------------------------
# Handcrafted FFT features (baseline 2)
# ----------------------------------------------------------------------------
def spectral_entropy(power: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = power / (np.sum(power, axis=1, keepdims=True) + eps)
    return -np.sum(p * np.log(p + eps), axis=1)


def make_signal_features(X_raw: np.ndarray, fs: float = 50.0) -> pd.DataFrame:
    n, t, c = X_raw.shape
    feats: Dict[str, np.ndarray] = {}
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    for ch in range(c):
        sig = X_raw[:, :, ch]
        name = SIGNAL_NAMES[ch]
        feats[f"{name}_mean"] = sig.mean(axis=1)
        feats[f"{name}_std"] = sig.std(axis=1)
        feats[f"{name}_min"] = sig.min(axis=1)
        feats[f"{name}_max"] = sig.max(axis=1)
        feats[f"{name}_energy"] = np.mean(sig ** 2, axis=1)
        feats[f"{name}_iqr"] = np.percentile(sig, 75, axis=1) - np.percentile(sig, 25, axis=1)
        power = np.abs(np.fft.rfft(sig, axis=1)) ** 2
        dom = np.argmax(power[:, 1:], axis=1)
        feats[f"{name}_fft_dominant_freq"] = freqs[1:][dom]
        feats[f"{name}_fft_total_power"] = power.sum(axis=1)
        feats[f"{name}_fft_entropy"] = spectral_entropy(power)
        feats[f"{name}_fft_mean_freq"] = (power * freqs[None, :]).sum(axis=1) / (power.sum(axis=1) + 1e-12)
    return pd.DataFrame(feats)


# ----------------------------------------------------------------------------
# Model components
# ----------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attn: bool = False):
        attn_out, attn_w = self.self_attn(x, x, x, need_weights=return_attn, average_attn_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x, attn_w


class SensorTransformer(nn.Module):
    def __init__(self, input_channels=9, seq_len=128, num_classes=6, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.15,
                 pe_type="sinusoidal", pooling="mean", patch_size=1):
        super().__init__()
        assert pooling in ["mean", "cls"] and pe_type in ["none", "sinusoidal", "learnable"]
        assert seq_len % patch_size == 0
        self.patch_size = patch_size
        self.pooling = pooling
        self.num_tokens = seq_len // patch_size
        self.input_proj = nn.Linear(input_channels * patch_size, d_model)
        max_len = self.num_tokens + (1 if pooling == "cls" else 0)
        if pe_type == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        elif pe_type == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)
        else:
            self.pos_encoder = nn.Identity()
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def patchify(self, x):
        if self.patch_size == 1:
            return x
        b, t, c = x.shape
        return x.reshape(b, self.num_tokens, self.patch_size * c)

    def forward(self, x, return_attn: bool = False):
        x = self.input_proj(self.patchify(x))
        if self.pooling == "cls":
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = self.dropout(self.pos_encoder(x))
        attns = []
        for blk in self.blocks:
            x, a = blk(x, return_attn=return_attn)
            if return_attn:
                attns.append(a.detach())
        pooled = x[:, 0] if self.pooling == "cls" else x.mean(dim=1)
        logits = self.classifier(pooled)
        return (logits, attns) if return_attn else logits


class CNNTransformerClassifier(nn.Module):
    def __init__(self, input_channels=9, seq_len=128, num_classes=6, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.15,
                 pe_type="sinusoidal", pooling="cls", conv_kernel_size=5):
        super().__init__()
        self.pooling = pooling
        hidden = max(16, d_model // 2)
        pad = conv_kernel_size // 2
        self.conv_frontend = nn.Sequential(
            nn.Conv1d(input_channels, hidden, conv_kernel_size, padding=pad),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, d_model, conv_kernel_size, padding=pad),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout),
        )
        max_len = seq_len + (1 if pooling == "cls" else 0)
        if pe_type == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        elif pe_type == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)
        else:
            self.pos_encoder = nn.Identity()
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x, return_attn: bool = False):
        x = self.conv_frontend(x.transpose(1, 2)).transpose(1, 2)
        if self.pooling == "cls":
            x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = self.dropout(self.pos_encoder(x))
        attns = []
        for blk in self.blocks:
            x, a = blk(x, return_attn=return_attn)
            if return_attn:
                attns.append(a.detach())
        pooled = x[:, 0] if self.pooling == "cls" else x.mean(dim=1)
        logits = self.classifier(pooled)
        return (logits, attns) if return_attn else logits


class FourierTransformer(nn.Module):
    """Dual-branch Transformer: time-domain tokens + FFT log-magnitude spectrum,
    each encoded by its own Transformer, CLS reps concatenated then classified.
    (Reconstruction of the model the notebook described; see README.)"""

    def __init__(self, in_ch=9, seq_len=128, num_classes=6, d_model=64, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.15):
        super().__init__()
        self.freq_len = seq_len // 2 + 1
        # time branch
        self.time_proj = nn.Linear(in_ch, d_model)
        self.time_cls = nn.Parameter(torch.zeros(1, 1, d_model)); nn.init.trunc_normal_(self.time_cls, std=0.02)
        self.time_pos = SinusoidalPositionalEncoding(d_model, seq_len + 1)
        self.time_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        # frequency branch
        self.freq_proj = nn.Linear(in_ch, d_model)
        self.freq_cls = nn.Parameter(torch.zeros(1, 1, d_model)); nn.init.trunc_normal_(self.freq_cls, std=0.02)
        self.freq_pos = SinusoidalPositionalEncoding(d_model, self.freq_len + 1)
        self.freq_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, num_classes))

    def _encode(self, tokens, cls, pos, blocks):
        x = torch.cat([cls.expand(tokens.size(0), -1, -1), tokens], dim=1)
        x = self.dropout(pos(x))
        for blk in blocks:
            x, _ = blk(x)
        return x[:, 0]

    def forward(self, x):
        t = self._encode(self.time_proj(x), self.time_cls, self.time_pos, self.time_blocks)
        mag = torch.log1p(torch.abs(torch.fft.rfft(x, dim=1)))  # B x F x C
        f = self._encode(self.freq_proj(mag), self.freq_cls, self.freq_pos, self.freq_blocks)
        return self.classifier(torch.cat([t, f], dim=-1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Torch dataset + training utilities
# ----------------------------------------------------------------------------
class HARRawDataset(Dataset):
    def __init__(self, X, y, subjects=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subjects = None if subjects is None else torch.tensor(subjects, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.subjects is None:
            return self.X[i], self.y[i]
        return self.X[i], self.y[i], self.subjects[i]


def _unpack(batch):
    x, y = batch[0], batch[1]
    return x.to(DEVICE), y.to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total, preds, tgts = 0.0, [], []
    for batch in loader:
        x, y = _unpack(batch)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * x.size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        tgts.extend(y.detach().cpu().numpy())
    return total / len(loader.dataset), accuracy_score(tgts, preds), f1_score(tgts, preds, average="macro")


@torch.no_grad()
def evaluate(model, loader, criterion=None):
    model.eval()
    total, preds, tgts, probs = 0.0, [], [], []
    for batch in loader:
        x, y = _unpack(batch)
        logits = model(x)
        if criterion is not None:
            total += criterion(logits, y).item() * x.size(0)
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        tgts.extend(y.cpu().numpy())
    return {
        "loss": total / len(loader.dataset) if criterion is not None else float("nan"),
        "accuracy": accuracy_score(tgts, preds),
        "macro_f1": f1_score(tgts, preds, average="macro"),
        "y_true": np.array(tgts), "y_pred": np.array(preds), "probs": np.vstack(probs),
    }


def fit_model(model, train_loader, val_loader, epochs, verbose=True):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    history, best_state, best_f1 = [], None, -1.0
    for ep in range(1, epochs + 1):
        tl, ta, tf = train_one_epoch(model, train_loader, opt, crit)
        vm = evaluate(model, val_loader, crit)
        sched.step()
        history.append({"epoch": ep, "train_loss": tl, "train_acc": ta, "train_macro_f1": tf,
                        "val_loss": vm["loss"], "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"]})
        if vm["macro_f1"] > best_f1:
            best_f1 = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {ep:02d}/{epochs} | train f1 {tf:.4f} acc {ta:.4f} | val f1 {vm['macro_f1']:.4f} acc {vm['accuracy']:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# ----------------------------------------------------------------------------
# Plot helpers (save only)
# ----------------------------------------------------------------------------
def save_history(history: pd.DataFrame, title: str, fname: str):
    plt.figure(figsize=(10, 4))
    plt.plot(history["epoch"], history["train_macro_f1"], label="Train macro-F1")
    plt.plot(history["epoch"], history["val_macro_f1"], label="Validation macro-F1")
    plt.title(title); plt.xlabel("Epoch"); plt.ylabel("Macro-F1")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / fname); plt.close()


def save_confusion(y_true, y_pred, title: str, fname: str):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(title); plt.tight_layout(); plt.savefig(OUT / fname); plt.close()
    return cm


def save_barh(labels, values, xlabel, title, fname):
    order = np.argsort(values)
    plt.figure(figsize=(12, max(4, len(labels) * 0.5)))
    plt.barh([labels[i] for i in order], [values[i] for i in order])
    plt.xlabel(xlabel); plt.title(title); plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / fname); plt.close()


def expected_calibration_error(y_true, y_pred, probs, n_bins=10) -> float:
    conf = probs.max(axis=1)
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if m.sum():
            ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    global DEVICE, OUT, NUM_WORKERS

    ap = argparse.ArgumentParser(description="HAR Transformer+Fourier full run")
    ap.add_argument("--data", default=None, help="Path to 'UCI HAR Dataset' folder")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "outputs"))
    ap.add_argument("--fast", action="store_true", help="quick smoke run (few epochs)")
    ap.add_argument("--device", default=None, choices=[None, "cuda", "cpu"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    DEVICE = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = args.num_workers
    OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)

    # epoch budget
    if args.fast:
        E = dict(main=2, ablation=1, patch=2, cnn=2, sensor=1, fourier=2)
    else:
        E = dict(main=20, ablation=10, patch=20, cnn=20, sensor=8, fourier=20)

    print("=" * 70)
    print("Device:", DEVICE, "| fast:", args.fast, "| out:", OUT)
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    base = resolve_dataset(args.data)
    print("Dataset:", base)

    X_tr_feat, y_train, subj_train = load_engineered("train", base)
    X_te_feat, y_test, subj_test = load_engineered("test", base)
    X_tr_raw = load_raw("train", base)
    X_te_raw = load_raw("test", base)
    print("Engineered:", X_tr_feat.shape, X_te_feat.shape, "| Raw:", X_tr_raw.shape, X_te_raw.shape)

    metrics: Dict[str, object] = {"config": {"device": str(DEVICE), "fast": args.fast,
                                             "epochs": E, "seed": seed, "dataset": str(base)}}
    leaderboard: List[Dict] = []

    # ---- Baseline 1: Logistic Regression on engineered features ----
    print("\n[1/8] Logistic Regression (561 engineered features)")
    sc = StandardScaler()
    lr = LogisticRegression(max_iter=1000, n_jobs=-1, C=2.0)
    lr.fit(sc.fit_transform(X_tr_feat), y_train)
    p = lr.predict(sc.transform(X_te_feat))
    leaderboard.append({"model": "Logistic Regression - 561 engineered features",
                        "accuracy": accuracy_score(y_test, p), "macro_f1": f1_score(y_test, p, average="macro")})
    print("   acc {:.4f}  macroF1 {:.4f}".format(leaderboard[-1]["accuracy"], leaderboard[-1]["macro_f1"]))

    # ---- Baseline 2: Random Forest on handcrafted FFT features ----
    print("\n[2/8] Random Forest (handcrafted time+FFT features)")
    X_tr_hand = make_signal_features(X_tr_raw)
    X_te_hand = make_signal_features(X_te_raw)
    rf = RandomForestClassifier(n_estimators=300 if args.fast else 600, min_samples_leaf=2,
                                random_state=seed, n_jobs=-1)
    rf.fit(X_tr_hand, y_train)
    p = rf.predict(X_te_hand)
    leaderboard.append({"model": "Random Forest - handcrafted time + FFT features",
                        "accuracy": accuracy_score(y_test, p), "macro_f1": f1_score(y_test, p, average="macro")})
    print("   acc {:.4f}  macroF1 {:.4f}".format(leaderboard[-1]["accuracy"], leaderboard[-1]["macro_f1"]))

    # ---- Standardize raw channels (train stats) + subject-wise val split ----
    mean = X_tr_raw.reshape(-1, 9).mean(0)
    std = X_tr_raw.reshape(-1, 9).std(0) + 1e-8
    X_tr_raw_s = (X_tr_raw - mean) / std
    X_te_raw_s = (X_te_raw - mean) / std

    full_ds = HARRawDataset(X_tr_raw_s, y_train, subj_train)
    test_ds = HARRawDataset(X_te_raw_s, y_test)
    test_ds_subj = HARRawDataset(X_te_raw_s, y_test, subj_test)

    uniq = np.array(sorted(np.unique(subj_train)))
    rng = np.random.default_rng(seed)
    n_val = max(3, int(np.ceil(0.20 * len(uniq))))
    val_subj = np.sort(rng.choice(uniq, size=n_val, replace=False))
    tr_idx = np.where(~np.isin(subj_train, val_subj))[0]
    va_idx = np.where(np.isin(subj_train, val_subj))[0]
    print(f"\nSubject-wise split: {len(tr_idx)} train / {len(va_idx)} val windows (val subjects {val_subj.tolist()})")

    def loaders(Xtr, Xte):
        f = HARRawDataset(Xtr, y_train, subj_train)
        tr = DataLoader(Subset(f, tr_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        va = DataLoader(Subset(f, va_idx.tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        te = DataLoader(HARRawDataset(Xte, y_test), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        return tr, va, te

    train_loader = DataLoader(Subset(full_ds, tr_idx.tolist()), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(full_ds, va_idx.tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    crit = nn.CrossEntropyLoss()

    # ---- 3. Main Transformer ----
    print("\n[3/8] Main Transformer")
    main_t = SensorTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                               dropout=0.15, pe_type="sinusoidal", pooling="cls", patch_size=1)
    main_t, hist = fit_model(main_t, train_loader, val_loader, E["main"])
    save_history(hist, "Main Transformer training curve", "main_transformer_training_curve.png")
    mm = evaluate(main_t, test_loader, crit)
    save_confusion(mm["y_true"], mm["y_pred"], "Main Transformer - Confusion Matrix", "main_transformer_confusion_matrix.png")
    leaderboard.append({"model": "Main Transformer - timestep tokens, CLS",
                        "accuracy": mm["accuracy"], "macro_f1": mm["macro_f1"]})
    metrics["main_transformer_report"] = classification_report(mm["y_true"], mm["y_pred"], target_names=CLASS_NAMES, output_dict=True)
    metrics["main_transformer_ece"] = expected_calibration_error(mm["y_true"], mm["y_pred"], mm["probs"])
    print("   test acc {:.4f}  macroF1 {:.4f}  ECE {:.4f}".format(mm["accuracy"], mm["macro_f1"], metrics["main_transformer_ece"]))

    # per-subject analysis for the main transformer
    msubj = evaluate(main_t, DataLoader(test_ds_subj, batch_size=BATCH_SIZE), crit)
    rows = []
    for s in sorted(np.unique(subj_test)):
        m = subj_test == s
        rows.append({"subject": int(s), "n_windows": int(m.sum()),
                     "accuracy": accuracy_score(msubj["y_true"][m], msubj["y_pred"][m]),
                     "macro_f1": f1_score(msubj["y_true"][m], msubj["y_pred"][m], average="macro")})
    pd.DataFrame(rows).sort_values("accuracy").to_csv(OUT / "per_subject_main.csv", index=False)

    # ---- 4. Ablation study (validation-only) ----
    print("\n[4/8] Transformer ablation study (validation-only)")
    ablation_cfgs = [
        {"name": "A1_no_pos_mean", "pe_type": "none", "pooling": "mean", "patch_size": 1, "num_layers": 2, "nhead": 4},
        {"name": "A2_sinusoidal_mean", "pe_type": "sinusoidal", "pooling": "mean", "patch_size": 1, "num_layers": 2, "nhead": 4},
        {"name": "A3_learnable_mean", "pe_type": "learnable", "pooling": "mean", "patch_size": 1, "num_layers": 2, "nhead": 4},
        {"name": "A4_sinusoidal_cls", "pe_type": "sinusoidal", "pooling": "cls", "patch_size": 1, "num_layers": 2, "nhead": 4},
        {"name": "A5_patch8_sinusoidal_cls", "pe_type": "sinusoidal", "pooling": "cls", "patch_size": 8, "num_layers": 2, "nhead": 4},
        {"name": "A6_deeper_4layers_cls", "pe_type": "sinusoidal", "pooling": "cls", "patch_size": 1, "num_layers": 4, "nhead": 4},
        {"name": "A7_more_heads_8_cls", "pe_type": "sinusoidal", "pooling": "cls", "patch_size": 1, "num_layers": 2, "nhead": 8},
    ]
    ablation = []
    for cfg in ablation_cfgs:
        m = SensorTransformer(d_model=64, nhead=cfg["nhead"], num_layers=cfg["num_layers"],
                              dim_feedforward=128, dropout=0.15, pe_type=cfg["pe_type"],
                              pooling=cfg["pooling"], patch_size=cfg["patch_size"])
        m, _ = fit_model(m, train_loader, val_loader, E["ablation"], verbose=False)
        vm = evaluate(m, val_loader, crit)
        ablation.append({**cfg, "params": count_parameters(m), "val_accuracy": vm["accuracy"], "val_macro_f1": vm["macro_f1"]})
        print(f"   {cfg['name']:28} val macroF1 {vm['macro_f1']:.4f}")
    metrics["ablation"] = ablation
    save_barh([a["name"] for a in ablation], [a["val_macro_f1"] for a in ablation],
              "Validation Macro-F1", "Transformer Ablation (validation-only)", "ablation_validation.png")

    # ---- 5. Patch Transformer ----
    print("\n[5/8] Patch Transformer (patch_size=8)")
    patch = SensorTransformer(d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                              dropout=0.15, pe_type="sinusoidal", pooling="cls", patch_size=8)
    patch, hist = fit_model(patch, train_loader, val_loader, E["patch"])
    save_history(hist, "Patch Transformer training curve", "patch_transformer_training_curve.png")
    pm = evaluate(patch, test_loader, crit)
    leaderboard.append({"model": "Patch Transformer - patch size 8, CLS", "accuracy": pm["accuracy"], "macro_f1": pm["macro_f1"]})
    print("   test acc {:.4f}  macroF1 {:.4f}".format(pm["accuracy"], pm["macro_f1"]))

    # ---- 6. CNN-Transformer hybrid ----
    print("\n[6/8] CNN-Transformer hybrid")
    cnn = CNNTransformerClassifier(d_model=64, nhead=4, num_layers=2, dim_feedforward=128,
                                   dropout=0.15, pe_type="sinusoidal", pooling="cls", conv_kernel_size=5)
    cnn, hist = fit_model(cnn, train_loader, val_loader, E["cnn"])
    save_history(hist, "CNN-Transformer training curve", "cnn_transformer_training_curve.png")
    cm = evaluate(cnn, test_loader, crit)
    leaderboard.append({"model": "CNN-Transformer Hybrid - Conv1D + CLS Transformer", "accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"]})
    print("   test acc {:.4f}  macroF1 {:.4f}".format(cm["accuracy"], cm["macro_f1"]))

    # ---- 7. Sensor ablation ----
    print("\n[7/8] Sensor ablation (channel groups)")
    sensor = []
    for gname, idx in CHANNEL_GROUPS.items():
        tr, va, te = loaders(X_tr_raw_s[:, :, idx], X_te_raw_s[:, :, idx])
        m = SensorTransformer(input_channels=len(idx), d_model=64, nhead=4, num_layers=2,
                              dim_feedforward=128, dropout=0.15, pe_type="sinusoidal", pooling="cls", patch_size=1)
        m, _ = fit_model(m, tr, va, E["sensor"], verbose=False)
        tm = evaluate(m, te, crit)
        sensor.append({"sensor_group": gname, "n_channels": len(idx),
                       "test_accuracy": tm["accuracy"], "test_macro_f1": tm["macro_f1"]})
        leaderboard.append({"model": f"Sensor ablation - {gname}", "accuracy": tm["accuracy"], "macro_f1": tm["macro_f1"]})
        print(f"   {gname:36} test macroF1 {tm['macro_f1']:.4f}")
    metrics["sensor_ablation"] = sensor
    save_barh([s["sensor_group"] for s in sensor], [s["test_macro_f1"] for s in sensor],
              "Test Macro-F1", "Sensor ablation (test)", "sensor_ablation_test.png")

    # ---- 8. Fourier+Transformer (dual-branch) ----
    print("\n[8/8] Fourier+Transformer (dual-branch time + FFT)")
    fusion = FourierTransformer(in_ch=9, seq_len=128, num_classes=6, d_model=64, nhead=4,
                                num_layers=2, dim_ff=128, dropout=0.15)
    print("   parameters:", count_parameters(fusion))
    fusion, hist = fit_model(fusion, train_loader, val_loader, E["fourier"])
    save_history(hist, "Fourier+Transformer training curve", "fourier_transformer_training_curve.png")
    fm = evaluate(fusion, test_loader, crit)
    save_confusion(fm["y_true"], fm["y_pred"], "Fourier+Transformer - Confusion Matrix", "fourier_transformer_confusion_matrix.png")
    leaderboard.append({"model": "Fourier+Transformer - Time+FFT Fusion", "accuracy": fm["accuracy"], "macro_f1": fm["macro_f1"]})
    metrics["fourier_transformer_report"] = classification_report(fm["y_true"], fm["y_pred"], target_names=CLASS_NAMES, output_dict=True)
    print("   test acc {:.4f}  macroF1 {:.4f}".format(fm["accuracy"], fm["macro_f1"]))

    # ---- Final comparison ----
    final = pd.DataFrame(leaderboard).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    final.to_csv(OUT / "final_comparison.csv", index=False)
    save_barh(final["model"].tolist(), final["macro_f1"].tolist(),
              "Test Macro-F1", "Final Model Comparison", "final_model_comparison.png")
    metrics["final_comparison"] = leaderboard

    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON (by test macro-F1)")
    print(final.to_string(index=False))
    print("\nAll artifacts written to:", OUT)


if __name__ == "__main__":
    main()
