# Matplotlib Real-time Monitor Debugging Record (260319)

Created: 2026-03-19
Translated from Korean original: `debugging/260319_matplotlib_monitor_디버깅.md`

## Problem

Running `run_visualization.py`: IsaacGym viewer window opens but matplotlib real-time monitor window **does not appear at all**. Plots not saved in headless mode either.

---

## Root Cause: matplotlib Backend Collision

`run_visualization.py` imports `from run_validation import ...`, and `run_validation.py` has:

```python
import matplotlib
matplotlib.use("Agg")     # ← Forces non-display backend
import matplotlib.pyplot as plt
```

Once this import executes, the matplotlib backend is locked to `Agg`. Subsequent `matplotlib.use("TkAgg")` calls in `run_visualization.py` are **ignored** because plt is already initialized.

## Failed Approaches

| Attempt | Result |
|---------|--------|
| Add `matplotlib.use("TkAgg")` at top of run_visualization.py | plt already initialized, ignored |
| `plt.switch_backend("TkAgg")` | ImportError or ignored |
| IsaacGym import order adjustment | Unrelated |

## Final Fixes

### 1. `run_validation.py` — Conditional backend setting
```python
# Only set Agg if already in non-display mode
import matplotlib
if matplotlib.get_backend().lower() in ("agg", ""):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

### 2. `run_visualization.py` — Import order + `force=True`
```python
# Import isaacgym + run_validation FIRST, then override matplotlib
from run_validation import create_sim_and_envs, make_human_body
matplotlib.use("Agg" if _is_headless else "TkAgg", force=True)  # force required
import matplotlib.pyplot as plt
```

### 3. Rolling time window for x-axis
```python
MONITOR_WINDOW_SECS = 10.0
ax.set_xlim(t_lo, t_now + 0.5)  # Sliding window instead of full history
```

## Key Takeaways: IsaacGym + Matplotlib Coexistence

1. Analysis modules' `matplotlib.use()` can silently contaminate the backend
2. Import order: `isaacgym` → other modules → `matplotlib.use(..., force=True)` → `import matplotlib.pyplot`
3. GPU pipeline (`use_gpu_pipeline=True`) required in viewer mode
4. `fig.canvas.draw()` + `fig.canvas.flush_events()` more stable than `plt.pause()`
