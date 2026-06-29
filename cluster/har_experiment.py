"""
HW3 robustness study on the GPU cluster.

Model: Fourier + Transformer fusion (dual-branch self-attention over TIME tokens
and over FOURIER/FFT tokens; FFT done as a fixed DFT matmul so it is device-agnostic).

Goal: attack the cross-subject generalization problem. We run a seed-averaged
ABLATION over three configurations and report mean +/- std test accuracy/macro-F1,
plus a per-test-subject breakdown for the best config (reveals "difficult subjects").

Configurations
  baseline   : no augmentation, dropout 0.15, wd 1e-4, no label smoothing
  aug        : + jitter + scaling + small rotation (train only)
  aug_reg    : aug + dropout 0.30 + wd 5e-4 + label smoothing 0.05 + early stopping

Usage
  python har_experiment.py --data_dir <UCI HAR Dataset root> --out_dir results \
         --epochs 25 --seeds 0 1 2

The data root is the folder that directly contains train/ and test/.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SIGNAL_NAMES = [
    "total_acc_x", "total_acc_y", "total_acc_z",
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]
CLASS_NAMES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]
TRIAXIAL_BLOCKS = [(0, 3), (3, 6), (6, 9)]  # total_acc / body_acc / body_gyro


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(root: Path, split: str):
    sig_dir = root / split / "Inertial Signals"
    chans = [np.loadtxt(sig_dir / f"{s}_{split}.txt") for s in SIGNAL_NAMES]
    X = np.stack(chans, axis=-1).astype(np.float32)          # (N, 128, 9)
    y = np.loadtxt(root / split / f"y_{split}.txt").astype(int) - 1
    subj = np.loadtxt(root / split / f"subject_{split}.txt").astype(int)
    return X, y, subj


def load_data(root: Path):
    Xtr, ytr, str_ = load_split(root, "train")
    Xte, yte, ste = load_split(root, "test")
    # per-channel z-score using TRAIN statistics only
    mean = Xtr.reshape(-1, 9).mean(0)
    std = Xtr.reshape(-1, 9).std(0) + 1e-8
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    return Xtr, ytr, str_, Xte, yte, ste, mean, std


# --------------------------------------------------------------------------- #
# Augmentation (train only, on B x 128 x 9)
# --------------------------------------------------------------------------- #
def small_rotations(B, sigma_deg, device):
    s = math.radians(sigma_deg)
    a = torch.randn(B, device=device) * s
    b = torch.randn(B, device=device) * s
    c = torch.randn(B, device=device) * s
    z = torch.zeros(B, device=device); o = torch.ones(B, device=device)
    ca, sa, cb, sb, cc, sc = a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin()
    Rx = torch.stack([o, z, z, z, ca, -sa, z, sa, ca], 1).view(B, 3, 3)
    Ry = torch.stack([cb, z, sb, z, o, z, -sb, z, cb], 1).view(B, 3, 3)
    Rz = torch.stack([cc, -sc, z, sc, cc, z, z, z, o], 1).view(B, 3, 3)
    return Rz @ Ry @ Rx


def augment(x, cfg):
    out = x
    if cfg["scale"]:
        out = out * (1.0 + cfg["scale_sigma"] * torch.randn(out.size(0), 1, 1, device=out.device))
    if cfg["rotate"]:
        R = small_rotations(out.size(0), cfg["rot_sigma_deg"], out.device)
        rot = out.clone()
        for lo, hi in TRIAXIAL_BLOCKS:
            rot[:, :, lo:hi] = torch.einsum("btj,bij->bti", out[:, :, lo:hi], R)
        out = rot
    if cfg["jitter"]:
        out = out + cfg["jitter_sigma"] * torch.randn_like(out)
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SinusoidalPE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        dt = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * dt)
        pe[:, 1::2] = torch.cos(pos * dt)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class LearnablePE(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EncoderBlock(nn.Module):
    def __init__(self, d, h, ff, dp):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, dropout=dp, batch_first=True)
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dp), nn.Linear(ff, d))
        self.drop = nn.Dropout(dp)

    def forward(self, x):
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.n1(x + self.drop(a))
        x = self.n2(x + self.drop(self.ffn(x)))
        return x


class FourierTransformer(nn.Module):
    def __init__(self, in_ch=9, seq_len=128, num_classes=6,
                 d_model=64, nhead=4, num_layers=2, dim_ff=128, dropout=0.15):
        super().__init__()
        n_freq = seq_len // 2 + 1
        k = torch.arange(n_freq).unsqueeze(1).float()
        n = torch.arange(seq_len).unsqueeze(0).float()
        ang = 2.0 * math.pi * k * n / seq_len
        self.register_buffer("dft_cos", torch.cos(ang))
        self.register_buffer("dft_sin", torch.sin(ang))

        self.time_proj = nn.Linear(in_ch, d_model)
        self.time_cls = nn.Parameter(torch.zeros(1, 1, d_model)); nn.init.trunc_normal_(self.time_cls, std=0.02)
        self.time_pe = SinusoidalPE(d_model, max_len=seq_len + 1)
        self.time_blocks = nn.ModuleList([EncoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])

        self.freq_proj = nn.Linear(in_ch, d_model)
        self.freq_cls = nn.Parameter(torch.zeros(1, 1, d_model)); nn.init.trunc_normal_(self.freq_cls, std=0.02)
        self.freq_pe = LearnablePE(d_model, max_len=n_freq + 1)
        self.freq_blocks = nn.ModuleList([EncoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])

        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d_model), nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def fft_mag(self, x):
        real = torch.einsum("ft,btc->bfc", self.dft_cos, x)
        imag = torch.einsum("ft,btc->bfc", self.dft_sin, x)
        return torch.log1p(torch.sqrt(real * real + imag * imag + 1e-8))

    def _encode(self, tok, cls, pe, blocks):
        z = torch.cat([cls.expand(tok.size(0), -1, -1), tok], dim=1)
        z = self.drop(pe(z))
        for blk in blocks:
            z = blk(z)
        return z[:, 0]

    def forward(self, x):
        t = self._encode(self.time_proj(x), self.time_cls, self.time_pe, self.time_blocks)
        f = self._encode(self.freq_proj(self.fft_mag(x)), self.freq_cls, self.freq_pe, self.freq_blocks)
        return self.head(torch.cat([t, f], dim=-1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Train / evaluate one run
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, aug_cfg=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    preds, tgts, tot = [], [], 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        if train and aug_cfg is not None:
            xb = augment(xb, aug_cfg)
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        tot += loss.item() * xb.size(0)
        preds.extend(logits.argmax(1).detach().cpu().numpy())
        tgts.extend(yb.detach().cpu().numpy())
    return tot / len(loader.dataset), accuracy_score(tgts, preds), f1_score(tgts, preds, average="macro")


def train_one(cfg, data, device, args):
    Xtr, ytr, _, Xte, yte, ste, _mean, _std = data
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    full = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long))
    val_n = int(0.15 * len(full)); tr_n = len(full) - val_n
    tr_ds, va_ds = random_split(full, [tr_n, val_n], generator=torch.Generator().manual_seed(cfg["seed"]))
    te_ds = TensorDataset(torch.tensor(Xte), torch.tensor(yte, dtype=torch.long))
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va = DataLoader(va_ds, batch_size=args.batch_size)
    te = DataLoader(te_ds, batch_size=args.batch_size)

    model = FourierTransformer(dropout=cfg["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    aug_cfg = cfg["aug"] if cfg["aug"]["enabled"] else None
    best_f1, best_state, since = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        run_epoch(model, tr, criterion, device, opt, aug_cfg)
        _, _, vf1 = run_epoch(model, va, criterion, device)
        sched.step()
        if vf1 > best_f1:
            best_f1, since = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
        if cfg["early_stop"] and since >= args.patience:
            break
    model.load_state_dict(best_state)

    # final test
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _ in te:
            preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
    preds = np.array(preds)
    acc = accuracy_score(yte, preds)
    mf1 = f1_score(yte, preds, average="macro")
    # train-set fit (overfitting gauge), no aug
    tr_eval = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr, dtype=torch.long)),
                         batch_size=args.batch_size)
    tr_preds = []
    with torch.no_grad():
        for xb, _ in tr_eval:
            tr_preds.extend(model(xb.to(device)).argmax(1).cpu().numpy())
    train_f1 = f1_score(ytr, np.array(tr_preds), average="macro")
    return {"acc": acc, "macro_f1": mf1, "train_f1": train_f1,
            "best_val_f1": best_f1, "preds": preds,
            "state": best_state, "dropout": cfg["dropout"]}


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #
def make_config(name, seed):
    aug_on = {"enabled": True, "jitter": True, "jitter_sigma": 0.05,
              "scale": True, "scale_sigma": 0.10,
              "rotate": True, "rot_sigma_deg": 20.0}
    aug_off = {"enabled": False, "jitter": False, "jitter_sigma": 0.0,
               "scale": False, "scale_sigma": 0.0, "rotate": False, "rot_sigma_deg": 0.0}
    base = {"seed": seed, "dropout": 0.15, "weight_decay": 1e-4,
            "label_smoothing": 0.0, "early_stop": False, "aug": aug_off}
    if name == "baseline":
        return base
    if name == "aug":
        c = dict(base); c["aug"] = aug_on; return c
    if name == "aug_reg":
        return {"seed": seed, "dropout": 0.30, "weight_decay": 5e-4,
                "label_smoothing": 0.05, "early_stop": True, "aug": aug_on}
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--configs", nargs="+", default=["baseline", "aug", "aug_reg"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "| CUDA:", torch.cuda.is_available(),
          "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a", flush=True)

    data = load_data(Path(args.data_dir))
    Xtr, ytr, str_, Xte, yte, ste, norm_mean, norm_std = data
    print(f"train {Xtr.shape} test {Xte.shape} | params={count_params(FourierTransformer())}", flush=True)

    results = {}
    best_overall = {"macro_f1": -1}
    t0 = time.time()
    for name in args.configs:
        accs, f1s, trf1s = [], [], []
        for seed in args.seeds:
            cfg = make_config(name, seed)
            r = train_one(cfg, data, device, args)
            accs.append(r["acc"]); f1s.append(r["macro_f1"]); trf1s.append(r["train_f1"])
            print(f"[{name} seed={seed}] test_acc={r['acc']:.4f} test_f1={r['macro_f1']:.4f} "
                  f"train_f1={r['train_f1']:.4f} (gap={r['train_f1']-r['macro_f1']:+.4f})", flush=True)
            if r["macro_f1"] > best_overall["macro_f1"]:
                best_overall = {"macro_f1": r["macro_f1"], "name": name, "seed": seed,
                                "preds": r["preds"], "state": r["state"], "dropout": r["dropout"]}
        results[name] = {
            "test_acc_mean": float(np.mean(accs)), "test_acc_std": float(np.std(accs)),
            "test_f1_mean": float(np.mean(f1s)), "test_f1_std": float(np.std(f1s)),
            "train_f1_mean": float(np.mean(trf1s)),
            "overfit_gap_mean": float(np.mean(trf1s) - np.mean(f1s)),
            "seeds": args.seeds,
        }
        print(f"==> {name}: test_f1 {results[name]['test_f1_mean']:.4f} "
              f"+/- {results[name]['test_f1_std']:.4f} | overfit gap "
              f"{results[name]['overfit_gap_mean']:+.4f}", flush=True)

    # per-subject breakdown for best config (item #15)
    preds = best_overall["preds"]
    subj_f1 = {}
    for s in np.unique(ste):
        m = ste == s
        subj_f1[int(s)] = float(f1_score(yte[m], preds[m], average="macro"))
    sf = np.array(list(subj_f1.values()))
    per_subject = {"best_config": best_overall["name"], "best_seed": best_overall["seed"],
                   "subject_macro_f1": subj_f1,
                   "subject_f1_mean": float(sf.mean()), "subject_f1_std": float(sf.std()),
                   "worst_subjects": sorted(subj_f1.items(), key=lambda kv: kv[1])[:3]}
    cm = confusion_matrix(yte, preds).tolist()
    report = classification_report(yte, preds, target_names=CLASS_NAMES, digits=4)

    summary = {"results": results, "per_subject": per_subject,
               "confusion_matrix_best": cm, "class_names": CLASS_NAMES,
               "runtime_sec": round(time.time() - t0, 1), "epochs": args.epochs}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "best_classification_report.txt").write_text(
        f"Best config: {best_overall['name']} (seed {best_overall['seed']})\n\n{report}\n")

    # ---- save the best trained model + everything needed for inference ----
    ckpt_path = out / "fourier_transformer_best.pt"
    torch.save(best_overall["state"], ckpt_path)
    model_meta = {
        "architecture": "FourierTransformer (dual-branch time+FFT)",
        "best_config": best_overall["name"], "best_seed": best_overall["seed"],
        "test_acc": float(accuracy_score(yte, best_overall["preds"])),
        "test_macro_f1": float(best_overall["macro_f1"]),
        "model_kwargs": {"in_ch": 9, "seq_len": 128, "num_classes": 6,
                         "d_model": 64, "nhead": 4, "num_layers": 2,
                         "dim_ff": 128, "dropout": best_overall["dropout"]},
        "channel_order": SIGNAL_NAMES,
        "input_norm": {"type": "per-channel z-score (train stats)",
                       "mean": norm_mean.tolist(), "std": norm_std.tolist()},
        "class_index_to_name": {i: n for i, n in enumerate(CLASS_NAMES)},
        "checkpoint_file": ckpt_path.name,
        "how_to_load": ("model = FourierTransformer(**model_kwargs); "
                        "model.load_state_dict(torch.load(checkpoint_file, map_location='cpu')); "
                        "model.eval(). Input: (B,128,9) in SIGNAL_NAMES order, "
                        "z-scored with input_norm mean/std."),
    }
    (out / "model_meta.json").write_text(json.dumps(model_meta, indent=2))
    print(f"Saved model -> {ckpt_path} ({ckpt_path.stat().st_size/1024:.0f} KB) + model_meta.json", flush=True)

    print("\n================ ABLATION SUMMARY ================", flush=True)
    for name in args.configs:
        r = results[name]
        print(f"{name:10s} | test_f1 {r['test_f1_mean']:.4f} +/- {r['test_f1_std']:.4f} "
              f"| acc {r['test_acc_mean']:.4f} | overfit_gap {r['overfit_gap_mean']:+.4f}", flush=True)
    print(f"\nPer-subject (best={best_overall['name']}): "
          f"mean {per_subject['subject_f1_mean']:.4f} +/- {per_subject['subject_f1_std']:.4f} | "
          f"worst {per_subject['worst_subjects']}", flush=True)
    print(f"\nWrote {out/'summary.json'} and {out/'best_classification_report.txt'} "
          f"(runtime {summary['runtime_sec']}s)", flush=True)


if __name__ == "__main__":
    main()
