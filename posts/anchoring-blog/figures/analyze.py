"""
Run the anchoring analyses on the trained model.
"""
import json
import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment import (TinyTransformer, PeriodicDataset, make_sequence,
                        SEQ_LEN, PERIOD, N_VAL)

torch.manual_seed(0)
np.random.seed(0)

ckpt = torch.load("/home/claude/model.pt", weights_only=False)
model = TinyTransformer()
model.load_state_dict(ckpt["model"])
model.eval()

# Use a fresh eval set so we know the model hasn't seen these.
eval_ds = PeriodicDataset(N_VAL, seed=42)
xs = torch.from_numpy(eval_ds.xs)
ys = torch.from_numpy(eval_ds.ys)

# ----- baseline: average attention from the query token -------------------
# The query token is at position SEQ_LEN. We want its attention OVER the
# input value tokens (positions 0..SEQ_LEN-1).
def query_attention(model, xs):
    """Returns attention from the query token to each input position,
    averaged across layers. Shape: (B, SEQ_LEN)."""
    with torch.no_grad():
        _, attns = model(xs, return_attn=True)
    # attns is a list of (B, T+1, T+1). Take the last row (query) and the
    # first SEQ_LEN columns (input positions).
    layer_attns = torch.stack(attns, dim=0)  # (L, B, T+1, T+1)
    q_attn = layer_attns[:, :, -1, :SEQ_LEN]  # (L, B, SEQ_LEN)
    return q_attn.mean(dim=0)  # (B, SEQ_LEN)

# Convert position to "lag" (how many steps before the prediction target).
# Position 0 is the oldest, position SEQ_LEN-1 is the most recent (lag 1).
# So lag = SEQ_LEN - position.
positions = np.arange(SEQ_LEN)
lags = SEQ_LEN - positions  # lag for each position

batch_size = 256
all_attn = []
for i in range(0, len(xs), batch_size):
    a = query_attention(model, xs[i:i+batch_size])
    all_attn.append(a.numpy())
all_attn = np.concatenate(all_attn, axis=0)  # (N, SEQ_LEN)
print(f"baseline attention shape: {all_attn.shape}")

mean_attn = all_attn.mean(axis=0)  # average across samples, for each position
# Plot vs lag
order = np.argsort(lags)  # sort by lag ascending (1, 2, 3, ...)
lags_sorted = lags[order]
mean_attn_sorted = mean_attn[order]

# ----- INTERVENTION 1: anchor swap ----------------------------------------
# Swap value at lag 24 (position SEQ_LEN-24) with value at lag 25.
# Does attention stay at the lag-24 position?
def swap_intervention(xs, lag_a, lag_b):
    xs2 = xs.clone()
    pos_a = SEQ_LEN - lag_a
    pos_b = SEQ_LEN - lag_b
    tmp = xs2[:, pos_a].clone()
    xs2[:, pos_a] = xs2[:, pos_b]
    xs2[:, pos_b] = tmp
    return xs2

xs_swapped = swap_intervention(xs, lag_a=24, lag_b=25)
all_attn_swapped = []
for i in range(0, len(xs_swapped), batch_size):
    a = query_attention(model, xs_swapped[i:i+batch_size])
    all_attn_swapped.append(a.numpy())
all_attn_swapped = np.concatenate(all_attn_swapped, axis=0)
mean_attn_swapped = all_attn_swapped.mean(axis=0)

# ----- INTERVENTION 2: phase shift ----------------------------------------
# Re-generate sequences with the same statistics but phase-shifted by 3
# steps. Then look at where attention peaks. If the spike "follows the
# signal" it moves to lag 21 or 27 (depending on direction). If it stays
# at lag 24, the model is committed to the slot.
#
# Equivalent (and cleaner): build new sequences directly with a controlled
# offset so we know exactly where the "real" period-aligned points are.
def make_shifted_sequence(rng, offset):
    """Same as make_sequence but pretend the period starts 'offset' steps
    later. We do this by sampling normally, but the model's notion of
    'lag 24 from the end' will now land between the natural peaks."""
    from experiment import AMP_RANGE, PHASE_RANGE, TREND_STD, NOISE_STD
    amp = rng.uniform(*AMP_RANGE)
    # We want the structure such that the "natural" period-aligned points
    # from the current request would be at lag 24+offset, not lag 24.
    # Achieve this by picking phase such that t=SEQ_LEN (the target) is
    # at phase 0, but then shifting the input window's sample times.
    phase = rng.uniform(*PHASE_RANGE)
    trend_slope = rng.normal(0, TREND_STD)
    intercept = rng.normal(0, 0.5)
    # Shift time: instead of t in [0, SEQ_LEN], use t in [offset, SEQ_LEN+offset]
    t = np.arange(SEQ_LEN + 1) + offset
    clean = intercept + trend_slope * t + amp * np.sin(2 * math.pi * t / PERIOD + phase)
    noise = rng.normal(0, NOISE_STD, size=SEQ_LEN + 1)
    series = clean + noise
    return series[:-1].astype(np.float32), series[-1].astype(np.float32)

# Actually, the shift above doesn't move anything observable to the model
# because the model only sees relative positions. The right intervention
# is different: at inference time, we feed the model a sequence whose
# period is shifted *relative to the position grid*. The model's position
# embedding is fixed (lag 24 = position SEQ_LEN-24), so if the true
# period-aligned values are now at lag 24+3 instead, we can see whether
# attention moves.
#
# Concrete way to do this: build a sequence where the noisy sine peaks
# at lag 21, 45, 69 instead of 24, 48, 72 -- i.e., the *effective period
# offset from the prediction time* is 21 instead of 24. We do this by
# generating a sequence where time-to-prediction at position p is (24 - 3) - p
# ... easier: just generate the sequence and remove the last 3 timesteps
# before prediction, so what was at lag 24 is now at lag 21.
#
# Cleanest: regenerate sequences with the *target's time index* shifted by
# +3 relative to the input window. So input values come from t=0..SEQ_LEN-1
# and target is at t=SEQ_LEN+3 instead of t=SEQ_LEN. This means the real
# "one period ago" point is at position SEQ_LEN-21 (lag 21), not SEQ_LEN-24.

def make_offset_target_sequence(rng, target_offset):
    from experiment import AMP_RANGE, PHASE_RANGE, TREND_STD, NOISE_STD
    amp = rng.uniform(*AMP_RANGE)
    phase = rng.uniform(*PHASE_RANGE)
    trend_slope = rng.normal(0, TREND_STD)
    intercept = rng.normal(0, 0.5)
    # Generate enough points to include the offset target.
    n_extra = max(target_offset, 0) + 1
    t = np.arange(SEQ_LEN + n_extra)
    clean = intercept + trend_slope * t + amp * np.sin(2 * math.pi * t / PERIOD + phase)
    noise = rng.normal(0, NOISE_STD, size=SEQ_LEN + n_extra)
    series = clean + noise
    x = series[:SEQ_LEN]
    y = series[SEQ_LEN + target_offset - 1]  # target_offset=1 -> next; =4 -> 3 steps later
    return x.astype(np.float32), float(y)

# target_offset=1 reproduces the original setup. target_offset=4 means we
# predict 3 steps further out -- so the "real" period-aligned points from
# the new target are at lag 24+3=27, 48+3=51, etc. relative to the END of
# the input window.
def build_offset_dataset(n, seed, target_offset):
    rng = np.random.default_rng(seed)
    xs_np = np.zeros((n, SEQ_LEN), dtype=np.float32)
    ys_np = np.zeros(n, dtype=np.float32)
    for i in range(n):
        xs_np[i], ys_np[i] = make_offset_target_sequence(rng, target_offset)
    return torch.from_numpy(xs_np), torch.from_numpy(ys_np)

# Note: this intervention tests A vs B in a slightly different way.
# Under A: the model uses the periodic signal, so when the real period
#   lag becomes 27 instead of 24, attention should move to lag 27.
# Under B: the model has compiled attention to fire at fixed lag-24, so
#   it stays put and prediction degrades.
xs_off, ys_off = build_offset_dataset(N_VAL, seed=43, target_offset=4)
all_attn_off = []
for i in range(0, len(xs_off), batch_size):
    a = query_attention(model, xs_off[i:i+batch_size])
    all_attn_off.append(a.numpy())
all_attn_off = np.concatenate(all_attn_off, axis=0)
mean_attn_off = all_attn_off.mean(axis=0)

# Prediction performance check
with torch.no_grad():
    preds_base = []
    preds_off = []
    for i in range(0, len(xs), batch_size):
        preds_base.append(model(xs[i:i+batch_size]).numpy())
        preds_off.append(model(xs_off[i:i+batch_size]).numpy())
    preds_base = np.concatenate(preds_base)
    preds_off = np.concatenate(preds_off)
mse_base = float(((preds_base - ys.numpy()) ** 2).mean())
mse_off = float(((preds_off - ys_off.numpy()) ** 2).mean())

# ----- INTERVENTION 3: content-matched control ----------------------------
# Compare attention at neighbor lags. Lag 24 and lag 25 carry essentially
# the same information (the sine is smooth, autocorrelation is high). If
# the model attends to 24 much more than 25, that's anchoring.
def attn_at_lag(mean_attn_vec, lag):
    return mean_attn_vec[SEQ_LEN - lag]

# ----- SAVE -----
results = {
    "lags": lags_sorted.tolist(),
    "baseline_attention": mean_attn_sorted.tolist(),
    "swapped_attention": mean_attn_swapped[order].tolist(),
    "offset_attention": mean_attn_off[order].tolist(),
    "mse_baseline_eval": mse_base,
    "mse_offset_target": mse_off,
    "attn_at_lag_24_baseline": float(attn_at_lag(mean_attn, 24)),
    "attn_at_lag_25_baseline": float(attn_at_lag(mean_attn, 25)),
    "attn_at_lag_23_baseline": float(attn_at_lag(mean_attn, 23)),
    "attn_at_lag_48_baseline": float(attn_at_lag(mean_attn, 48)),
    "attn_at_lag_47_baseline": float(attn_at_lag(mean_attn, 47)),
    "attn_at_lag_49_baseline": float(attn_at_lag(mean_attn, 49)),
    "attn_at_lag_24_swapped": float(attn_at_lag(mean_attn_swapped, 24)),
    "attn_at_lag_25_swapped": float(attn_at_lag(mean_attn_swapped, 25)),
}
with open("/home/claude/results.json", "w") as f:
    json.dump(results, f, indent=2)

# ----- print summary ------------------------------------------------------
print(f"\nAttention at key lags (baseline):")
for L in [1, 12, 23, 24, 25, 47, 48, 49, 71, 72, 73]:
    print(f"  lag {L:3d}: {attn_at_lag(mean_attn, L):.4f}")

print(f"\nAttention at lag 24 vs lag 25 (baseline):"
      f" {attn_at_lag(mean_attn, 24):.4f} vs {attn_at_lag(mean_attn, 25):.4f}"
      f" -- ratio {attn_at_lag(mean_attn, 24)/attn_at_lag(mean_attn, 25):.2f}x")

print(f"\nAfter anchor swap (24<->25):")
print(f"  attention at position-24: {attn_at_lag(mean_attn_swapped, 24):.4f}"
      f"  (baseline {attn_at_lag(mean_attn, 24):.4f})")
print(f"  attention at position-25: {attn_at_lag(mean_attn_swapped, 25):.4f}"
      f"  (baseline {attn_at_lag(mean_attn, 25):.4f})")

print(f"\nOffset target experiment (predict t+4 instead of t+1):")
print(f"  baseline MSE: {mse_base:.4f}, offset MSE: {mse_off:.4f}")
print(f"  attn at lag 24 (off): {attn_at_lag(mean_attn_off, 24):.4f}"
      f"  baseline: {attn_at_lag(mean_attn, 24):.4f}")
print(f"  attn at lag 27 (off): {attn_at_lag(mean_attn_off, 27):.4f}"
      f"  baseline: {attn_at_lag(mean_attn, 27):.4f}")

# ----- PLOTS --------------------------------------------------------------
plt.style.use("default")

# Plot 1: baseline attention vs lag
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lags_sorted, mean_attn_sorted, linewidth=1.2, color="#2b6cb0")
for k in range(1, SEQ_LEN // PERIOD + 1):
    ax.axvline(k * PERIOD, color="#e53e3e", alpha=0.25, linestyle="--", linewidth=0.8)
ax.set_xlabel("Lag (steps before prediction target)")
ax.set_ylabel("Average attention weight")
ax.set_title("Attention from query token vs lag — periodic anchoring at multiples of 24")
ax.set_xlim(0, SEQ_LEN)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/fig1_baseline_attention.png", dpi=140)
plt.close(fig)

# Plot 2: zoomed view around lag 24
fig, ax = plt.subplots(figsize=(8, 4))
mask = (lags_sorted >= 15) & (lags_sorted <= 35)
ax.plot(lags_sorted[mask], mean_attn_sorted[mask], "o-",
        linewidth=1.5, markersize=5, color="#2b6cb0", label="attention")
ax.axvline(24, color="#e53e3e", alpha=0.5, linestyle="--", label="lag = period (24)")
ax.set_xlabel("Lag (steps before prediction target)")
ax.set_ylabel("Average attention weight")
ax.set_title("Zoomed view: attention spike at lag 24 vs immediate neighbors")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/fig2_zoomed.png", dpi=140)
plt.close(fig)

# Plot 3: swap intervention - delta in attention
delta_swap = (all_attn_swapped.mean(axis=0) - all_attn.mean(axis=0))[order]
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(lags_sorted, delta_swap, color="#805ad5", width=1.0)
ax.axvline(24, color="#e53e3e", alpha=0.4, linestyle="--", label="lag 24 (anchor)")
ax.axvline(25, color="#dd6b20", alpha=0.4, linestyle="--", label="lag 25 (swap partner)")
ax.set_xlabel("Lag")
ax.set_ylabel("Δ attention (swapped − baseline)")
ax.set_title("Effect of swapping the values at lag 24 and lag 25 on attention")
ax.set_xlim(15, 35)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/fig3_swap.png", dpi=140)
plt.close(fig)

# Plot 4: offset target - did the spike follow?
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(lags_sorted, mean_attn_sorted, label="baseline (predict t+1)",
        color="#2b6cb0", linewidth=1.2)
ax.plot(lags_sorted, mean_attn_off[order], label="offset target (predict t+4)",
        color="#dd6b20", linewidth=1.2)
ax.axvline(24, color="#2b6cb0", alpha=0.3, linestyle="--",
           label="lag 24 (period from t+1)")
ax.axvline(27, color="#dd6b20", alpha=0.3, linestyle="--",
           label="lag 27 (period from t+4)")
ax.set_xlabel("Lag from end of input window")
ax.set_ylabel("Average attention weight")
ax.set_title("Does the attention spike follow the periodic signal or stay put?")
ax.set_xlim(15, 80)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("/home/claude/fig4_offset.png", dpi=140)
plt.close(fig)

print("\nplots saved.")
