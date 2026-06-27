"""
Standalone loader for the Fourier+Transformer deep model.

This module contains ONLY what you need to load the trained checkpoint
(`fourier_transformer_best.pt`) and run inference. No training code.

Quick use:
    from fourier_transformer import load_model, predict
    model, meta = load_model("../models/fourier_transformer_best.pt",
                             "../models/model_meta.json")
    labels, names, probs = predict(model, raw_windows, meta)   # raw_windows: (N,128,9)
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Architecture (must match the trained checkpoint exactly)
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
    """Dual-branch transformer: self-attention over TIME tokens and over FFT tokens."""
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


# --------------------------------------------------------------------------- #
# Load / preprocess / predict
# --------------------------------------------------------------------------- #
def load_model(ckpt_path, meta_path, device="cpu"):
    """Return (model in eval mode, meta dict)."""
    meta = json.loads(Path(meta_path).read_text())
    model = FourierTransformer(**meta["model_kwargs"])
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, meta


def preprocess(raw, meta):
    """Z-score raw windows with the SAVED training statistics.

    raw: float array (N, 128, 9) in meta['channel_order'].
    returns float32 array of the same shape, normalized.
    """
    raw = np.asarray(raw, dtype=np.float32)
    assert raw.ndim == 3 and raw.shape[1:] == (128, 9), \
        f"expected (N,128,9), got {raw.shape}"
    mean = np.array(meta["input_norm"]["mean"], dtype=np.float32)
    std = np.array(meta["input_norm"]["std"], dtype=np.float32)
    return (raw - mean) / std


@torch.no_grad()
def predict(model, raw, meta, device="cpu", batch_size=256):
    """raw: (N,128,9) RAW windows. Returns (labels, names, probs).

    labels: int array (N,)         class indices 0..5
    names:  list[str] (N,)         human-readable activity names
    probs:  float array (N, 6)     softmax probabilities
    """
    x = preprocess(raw, meta)
    idx2name = {int(k): v for k, v in meta["class_index_to_name"].items()}
    out = []
    for i in range(0, len(x), batch_size):
        xb = torch.tensor(x[i:i + batch_size], device=device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(out, axis=0)
    labels = probs.argmax(1)
    names = [idx2name[int(l)] for l in labels]
    return labels, names, probs
