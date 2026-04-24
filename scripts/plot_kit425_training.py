"""Plot KIT_425 phase-fix 3-slot training curves (epoch vs rwd, epoch vs eps_len)."""
import re
from pathlib import Path
import matplotlib.pyplot as plt

LOGS = {
    "MAIN (cycle=F, phase=T)":   "logs/kit425_main_3942.out",
    "NOPHASE (cycle=F, phase=F)": "logs/kit425_nophase_3943.out",
    "LEGACY (cycle=T, phase=T)":  "logs/kit425_legacy_3944.out",
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
axes[0].set_title("KIT_425 Phase-Fix 3-slot training")
axes[0].axvline(10000, color="gray", linestyle="--", alpha=0.6, label="curriculum switch")
axes[0].legend(loc="lower right", fontsize=9); axes[0].grid(alpha=0.3)

axes[1].set_ylabel("eps_len (frames, smoothed)")
axes[1].set_xlabel("epoch")
axes[1].axvline(10000, color="gray", linestyle="--", alpha=0.6)
axes[1].axhline(137, color="C0", linestyle=":", alpha=0.5, label="cycle=F avg ceiling ~137")
axes[1].axhline(300, color="C2", linestyle=":", alpha=0.5, label="cycle=T cap 300")
axes[1].legend(loc="upper left", fontsize=9); axes[1].grid(alpha=0.3)

plt.tight_layout()
out = "logs/kit425_training_curves.png"
plt.savefig(out, dpi=120)
print(f"saved {out}")
