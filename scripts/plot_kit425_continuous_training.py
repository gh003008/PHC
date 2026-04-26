"""Plot KIT_425 continuous-walking 3-slot training curves."""
import re
from pathlib import Path
import matplotlib.pyplot as plt

LOGS = {
    "S1 TERM_OFF (cycle=F, term off)":     "logs/kit425_cont_S1_3954.out",
    "S2 CYCLE_BASE (cycle=T, no v_cmd)":   "logs/kit425_cont_S2_3955.out",
    "S3 VCMD_HOP (cycle=T, v_cmd resamp)": "logs/kit425_cont_S3_3956.out",
}
PATTERN = re.compile(r"Ep:\s*(\d+)\s+rwd:\s*([-\d.]+).*eps_len:\s*([-\d.]+)")

def load(path):
    eps, rwd, ln = [], [], []
    for line in Path(path).read_text().splitlines():
        m = PATTERN.search(line)
        if m:
            eps.append(int(m.group(1))); rwd.append(float(m.group(2))); ln.append(float(m.group(3)))
    return eps, rwd, ln

def smooth(y, k=200):
    if len(y) < k: return y
    import numpy as np
    y = np.asarray(y, dtype=float)
    out = np.convolve(y, np.ones(k)/k, mode="valid")
    pad = [out[0]] * (len(y) - len(out))
    return list(pad) + list(out)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for label, path in LOGS.items():
    eps, rwd, ln = load(path)
    axes[0].plot(eps, smooth(rwd), label=label, linewidth=1.8)
    axes[1].plot(eps, smooth(ln),  label=label, linewidth=1.8)

axes[0].set_ylabel("reward (smoothed, window=200)")
axes[0].set_title("KIT_425 Continuous-Walking 3-slot training (S1 cycle=F vs S2/S3 cycle=T)")
axes[0].axvline(10000, color="gray", linestyle="--", alpha=0.6, label="curriculum switch")
axes[0].legend(loc="lower right", fontsize=9); axes[0].grid(alpha=0.3)

axes[1].set_ylabel("eps_len (frames, smoothed)")
axes[1].set_xlabel("epoch")
axes[1].axvline(10000, color="gray", linestyle="--", alpha=0.6)
axes[1].axhline(137, color="C0", linestyle=":", alpha=0.5, label="cycle=F avg ceiling ~137")
axes[1].axhline(3000, color="C1", linestyle=":", alpha=0.5, label="cycle=T cap 3000")
axes[1].legend(loc="upper left", fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
out = "logs/kit425_continuous_training_curves.png"
plt.savefig(out, dpi=120)
print(f"saved {out}")

# Also plot S1 alone with proper y-scale
fig2, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
eps, rwd, ln = load(LOGS["S1 TERM_OFF (cycle=F, term off)"])
ax[0].plot(eps, smooth(rwd), color="C0", linewidth=1.8); ax[0].set_ylabel("reward (smoothed)")
ax[0].set_title("KIT_425 Continuous S1 TERM_OFF (zoomed for cycle=F scale)")
ax[0].axvline(10000, color="gray", linestyle="--", alpha=0.6); ax[0].grid(alpha=0.3)
ax[1].plot(eps, smooth(ln), color="C0", linewidth=1.8); ax[1].set_ylabel("eps_len (frames, smoothed)")
ax[1].axhline(137, color="gray", linestyle=":", alpha=0.6, label="cycle=F avg ceiling 137")
ax[1].axvline(10000, color="gray", linestyle="--", alpha=0.6); ax[1].set_xlabel("epoch"); ax[1].grid(alpha=0.3); ax[1].legend()
plt.tight_layout(); plt.savefig("logs/kit425_continuous_S1_only.png", dpi=120)
print("saved logs/kit425_continuous_S1_only.png")
