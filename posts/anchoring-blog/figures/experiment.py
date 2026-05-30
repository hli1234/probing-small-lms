"""
Periodic anchoring experiment.

Trains a small transformer to predict the next value of a noisy periodic
signal, then probes its attention pattern to look for "anchoring" behavior
at integer multiples of the period.
"""
import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

PERIOD = 24
SEQ_LEN = 96         # 4 days instead of a week — still 3 full periods
N_TRAIN = 8_000
N_VAL = 1_000
NOISE_STD = 0.3
TREND_STD = 0.002
AMP_RANGE = (0.8, 1.2)
PHASE_RANGE = (0.0, 2 * math.pi)

def make_sequence(rng, seq_len=SEQ_LEN, period=PERIOD):
    amp = rng.uniform(*AMP_RANGE)
    phase = rng.uniform(*PHASE_RANGE)
    trend_slope = rng.normal(0, TREND_STD)
    intercept = rng.normal(0, 0.5)
    t = np.arange(seq_len + 1)
    clean = intercept + trend_slope * t + amp * np.sin(2 * math.pi * t / period + phase)
    noise = rng.normal(0, NOISE_STD, size=seq_len + 1)
    series = clean + noise
    return series[:-1].astype(np.float32), series[-1].astype(np.float32)

class PeriodicDataset(Dataset):
    def __init__(self, n, seed):
        self.rng = np.random.default_rng(seed)
        self.xs = np.zeros((n, SEQ_LEN), dtype=np.float32)
        self.ys = np.zeros(n, dtype=np.float32)
        for i in range(n):
            self.xs[i], self.ys[i] = make_sequence(self.rng)
    def __len__(self):
        return len(self.ys)
    def __getitem__(self, i):
        return self.xs[i], self.ys[i]

class TinyTransformer(nn.Module):
    def __init__(self, d_model=32, n_layers=2, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.value_proj = nn.Linear(1, d_model)
        self.pos_emb = nn.Embedding(seq_len + 1, d_model)
        self.query_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(d_model, num_heads=1, batch_first=True),
                "ln1": nn.LayerNorm(d_model),
                "ln2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.GELU(),
                    nn.Linear(4 * d_model, d_model),
                ),
            })
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, return_attn=False):
        B, T = x.shape
        v = self.value_proj(x.unsqueeze(-1))
        q = self.query_token.expand(B, 1, -1)
        h = torch.cat([v, q], dim=1)
        pos = torch.arange(T + 1, device=x.device).unsqueeze(0).expand(B, -1)
        h = h + self.pos_emb(pos)
        attns = []
        for layer in self.layers:
            normed = layer["ln1"](h)
            out, attn_weights = layer["attn"](normed, normed, normed,
                                              need_weights=True,
                                              average_attn_weights=True)
            h = h + out
            h = h + layer["ff"](layer["ln2"](h))
            attns.append(attn_weights)
        z = h[:, -1, :]
        pred = self.head(z).squeeze(-1)
        if return_attn:
            return pred, attns
        return pred

def train(model, train_ds, val_ds, epochs=10, batch_size=128, lr=3e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    history = []
    for ep in range(epochs):
        model.train()
        total = 0.0
        n = 0
        for x, y in train_loader:
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = total / n
        model.eval()
        with torch.no_grad():
            total = 0.0
            n = 0
            for x, y in val_loader:
                pred = model(x)
                total += F.mse_loss(pred, y, reduction="sum").item()
                n += x.size(0)
            val_loss = total / n
        history.append({"epoch": ep, "train_mse": train_loss, "val_mse": val_loss})
        print(f"epoch {ep:2d}  train_mse={train_loss:.4f}  val_mse={val_loss:.4f}")
    return history

if __name__ == "__main__":
    print("Building datasets...")
    train_ds = PeriodicDataset(N_TRAIN, seed=1)
    val_ds = PeriodicDataset(N_VAL, seed=2)
    y_train_mean = train_ds.ys.mean()
    baseline_mse = ((val_ds.ys - y_train_mean) ** 2).mean()
    last_value_mse = ((val_ds.ys - val_ds.xs[:, -1]) ** 2).mean()
    print(f"baseline mse (predict mean): {baseline_mse:.4f}")
    print(f"baseline mse (predict last value): {last_value_mse:.4f}")
    model = TinyTransformer()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params}")
    history = train(model, train_ds, val_ds, epochs=6)
    torch.save({"model": model.state_dict(), "history": history,
                "baseline_mse": float(baseline_mse),
                "last_value_mse": float(last_value_mse)},
               "/home/claude/model.pt")
    with open("/home/claude/history.json", "w") as f:
        json.dump({"history": history,
                   "baseline_mse": float(baseline_mse),
                   "last_value_mse": float(last_value_mse)}, f, indent=2)
    print("saved.")
