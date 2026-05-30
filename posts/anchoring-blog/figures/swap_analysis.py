"""
Refined swap experiment: target the positions where the model actually
puts its attention spikes (lag 23, 45, 69), not the nominal period multiples.

Logic of the content-matched control:
- Position at lag 23 has high attention (~0.063).
- Position at lag 22 has near-zero attention (~0.005).
- These two positions carry near-identical information about the periodic
  signal (autocorrelation between adjacent points is ~0.97 for a sine
  with period 24).
- If we swap the values at these two positions, the values change but
  the positions do not. Under "the model uses informative content", attention
  should follow the values -- the attention should now spike at lag 22.
  Under "the model uses fixed slots", attention stays at lag 23.
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import TinyTransformer, PeriodicDataset, SEQ_LEN, PERIOD, N_VAL

torch.manual_seed(0)
np.random.seed(0)

ckpt = torch.load("/home/claude/model.pt", weights_only=False)
model = TinyTransformer()
model.load_state_dict(ckpt["model"])
model.eval()

eval_ds = PeriodicDataset(N_VAL, seed=42)
xs = torch.from_numpy(eval_ds.xs)
ys = torch.from_numpy(eval_ds.ys)

def query_attention(model, xs):
    with torch.no_grad():
        _, attns = model(xs, return_attn=True)
    layer_attns = torch.stack(attns, dim=0)
    q_attn = layer_attns[:, :, -1, :SEQ_LEN]
    return q_attn.mean(dim=0)

def get_mean_attn(model, xs, batch_size=256):
    all_attn = []
    for i in range(0, len(xs), batch_size):
        a = query_attention(model, xs[i:i+batch_size])
        all_attn.append(a.numpy())
    all_attn = np.concatenate(all_attn, axis=0)
    return all_attn.mean(axis=0)

def get_predictions(model, xs, batch_size=256):
    with torch.no_grad():
        preds = []
        for i in range(0, len(xs), batch_size):
            preds.append(model(xs[i:i+batch_size]).numpy())
    return np.concatenate(preds)

# Lag L -> position SEQ_LEN - L
LAG_ANCHOR = 23  # the position the model anchors to
LAG_NEIGHBOR = 22  # equally informative neighbor

POS_ANCHOR = SEQ_LEN - LAG_ANCHOR    # 73
POS_NEIGHBOR = SEQ_LEN - LAG_NEIGHBOR  # 74

# Baseline
mean_attn_base = get_mean_attn(model, xs)

# Swap intervention: swap values at the anchor position (lag 23) and its
# neighbor (lag 22). Same content, different position.
xs_swap = xs.clone()
tmp = xs_swap[:, POS_ANCHOR].clone()
xs_swap[:, POS_ANCHOR] = xs_swap[:, POS_NEIGHBOR]
xs_swap[:, POS_NEIGHBOR] = tmp
mean_attn_swap = get_mean_attn(model, xs_swap)

# Also: corruption intervention. Replace the anchor value with random noise
# (same distribution as the signal). Does attention move away? Under "fixed
# slot" we expect attention to stay; under "informative content" we expect
# attention to drop or move.
np.random.seed(123)
xs_corrupt = xs.clone()
xs_corrupt[:, POS_ANCHOR] = torch.from_numpy(
    np.random.randn(len(xs)).astype(np.float32) * float(xs[:, POS_ANCHOR].std())
)
mean_attn_corrupt = get_mean_attn(model, xs_corrupt)

# Predictions under each
preds_base = get_predictions(model, xs)
preds_swap = get_predictions(model, xs_swap)
preds_corrupt = get_predictions(model, xs_corrupt)

mse_base = float(((preds_base - ys.numpy()) ** 2).mean())
mse_swap = float(((preds_swap - ys.numpy()) ** 2).mean())
mse_corrupt = float(((preds_corrupt - ys.numpy()) ** 2).mean())

# Print
print(f"Baseline attention:")
print(f"  lag {LAG_ANCHOR} (anchor): {mean_attn_base[POS_ANCHOR]:.4f}")
print(f"  lag {LAG_NEIGHBOR} (neighbor): {mean_attn_base[POS_NEIGHBOR]:.4f}")
print(f"  ratio: {mean_attn_base[POS_ANCHOR] / mean_attn_base[POS_NEIGHBOR]:.2f}x")

print(f"\nAfter swap (values at lag 22 and lag 23 swapped):")
print(f"  attn at lag 23 (still anchor position): {mean_attn_swap[POS_ANCHOR]:.4f}")
print(f"  attn at lag 22 (now holds old anchor value): {mean_attn_swap[POS_NEIGHBOR]:.4f}")
print(f"  Δattn at anchor position: {mean_attn_swap[POS_ANCHOR] - mean_attn_base[POS_ANCHOR]:+.4f}")
print(f"  → attention {'STAYS at the position' if abs(mean_attn_swap[POS_ANCHOR] - mean_attn_base[POS_ANCHOR]) < 0.005 else 'FOLLOWS the value'}")

print(f"\nAfter corruption (anchor value replaced with noise):")
print(f"  attn at lag 23 (anchor): {mean_attn_corrupt[POS_ANCHOR]:.4f}")
print(f"  baseline at same position: {mean_attn_base[POS_ANCHOR]:.4f}")

print(f"\nMSE:")
print(f"  baseline: {mse_base:.4f}")
print(f"  swap:     {mse_swap:.4f}   Δ={mse_swap-mse_base:+.4f}")
print(f"  corrupt:  {mse_corrupt:.4f}   Δ={mse_corrupt-mse_base:+.4f}")

# Save results
results = {
    "baseline_attention": mean_attn_base.tolist(),
    "swap_attention": mean_attn_swap.tolist(),
    "corrupt_attention": mean_attn_corrupt.tolist(),
    "lag_anchor": LAG_ANCHOR,
    "lag_neighbor": LAG_NEIGHBOR,
    "attn_anchor_base": float(mean_attn_base[POS_ANCHOR]),
    "attn_anchor_swap": float(mean_attn_swap[POS_ANCHOR]),
    "attn_anchor_corrupt": float(mean_attn_corrupt[POS_ANCHOR]),
    "attn_neighbor_base": float(mean_attn_base[POS_NEIGHBOR]),
    "attn_neighbor_swap": float(mean_attn_swap[POS_NEIGHBOR]),
    "mse_base": mse_base,
    "mse_swap": mse_swap,
    "mse_corrupt": mse_corrupt,
}
with open("/home/claude/swap_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Plot
lags = np.arange(1, SEQ_LEN + 1)
attn_base_by_lag = mean_attn_base[::-1]  # position 0 = lag 96, position 95 = lag 1
attn_swap_by_lag = mean_attn_swap[::-1]
attn_corrupt_by_lag = mean_attn_corrupt[::-1]

fig, ax = plt.subplots(figsize=(8, 4))
mask = (lags >= 15) & (lags <= 32)
ax.plot(lags[mask], attn_base_by_lag[mask], "o-", color="#2b6cb0",
        label="baseline", markersize=5, linewidth=1.4)
ax.plot(lags[mask], attn_swap_by_lag[mask], "s-", color="#dd6b20",
        label="after swap of values at lag 22 & 23", markersize=5, linewidth=1.4)
ax.plot(lags[mask], attn_corrupt_by_lag[mask], "^-", color="#805ad5",
        label="after corrupting value at lag 23", markersize=5, linewidth=1.4)
ax.axvline(23, color="#e53e3e", alpha=0.3, linestyle="--", label="anchor (lag 23)")
ax.set_xlabel("Lag from end of input window")
ax.set_ylabel("Average attention weight")
ax.set_title("Content interventions at the anchor position")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/fig5_content_swap.png", dpi=140)
plt.close(fig)

print("\nsaved.")
