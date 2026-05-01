"""Evaluate trained RES_AMP_VCMD residual policy.

Two automated sub-tests per spec §9 (manual qualitative video is Task 23):

  1. Static v_cmd sweep — 9 levels in [0.76, 1.23] m/s, 5s each, no ramp,
     measure mean v_actual and steady-state error per level.
  2. Dynamic v_cmd ramp — scripted schedule 0.85 → 1.15 → 0.85 over 60 s
     with 0.5 m/s² ramp, measure settling time per transition.

Pass criteria (spec §9):
  - sweep: |mean(v_actual) - v_cmd| < 0.10 m/s for every level
  - sweep: zero falls (eps_len_at_5s ≥ 5 × FPS)
  - ramp:  settling time < 2.5 s per transition (where settling = |v_act - v_cmd| < 0.05)

Usage:
  conda activate phc
  python scripts/eval_res_amp_vcmd.py \\
    --ckpt output/HumanoidIm/ResAMPVCmd/Humanoid_*.pth \\
    --out  02_research_dev/260501_res_amp_vcmd_eval_results.json

We boot the env+model in --test mode using PHC's run.py argparse stack
(same path as scripts/launch_res_amp_smoke.sh), then monkey-patch:
  - HumanoidImResAMPVCmd._ramp_v_cmd → no-op (we drive v_cmd manually)
  - Humanoid.pre_physics_step → record v_actual + advance schedule
We exit cleanly when the schedule completes and dump the JSON.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PHC_PKG = os.path.join(_ROOT, "phc")
for _p in (_PHC_PKG, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)

import isaacgym  # noqa: F401  (required before torch)
import numpy as np
import torch


# ----------------------------------------------------------------------
# Schedule
# ----------------------------------------------------------------------

V_CMD_MIN = 0.76
V_CMD_MAX = 1.23

# Sweep: 9 levels (linear), 5s each. Includes both endpoints.
SWEEP_LEVELS = np.linspace(V_CMD_MIN, V_CMD_MAX, 9).tolist()
SWEEP_DURATION_S = 5.0
SETTLE_TOL = 0.05  # m/s — for ramp settling time

# Ramp script: list of (target_v_cmd, hold_s) tuples.
# Three transitions: 0.85 → 1.15 → 0.85 → 1.0 with 6 s holds.
RAMP_SCRIPT = [
    (0.85, 8.0),
    (1.15, 8.0),
    (0.85, 8.0),
    (1.00, 6.0),
]

FPS = 30.0   # control_freq matches env (control_freq_inv=2 @ 60Hz physics → 30Hz control)


# ----------------------------------------------------------------------
# Mutable state used by monkey-patches and main loop
# ----------------------------------------------------------------------

_STATE = {
    "phase": "warmup",      # warmup → sweep → ramp → done
    "phase_t": 0.0,         # seconds in current phase
    "sweep_idx": 0,
    "ramp_idx": 0,
    "ramp_v_target_prev": RAMP_SCRIPT[0][0],
    "samples": {            # per-phase records
        "sweep": [],        # list per level: {"v_cmd": float, "v_act": [..], "fall_step": int|-1}
        "ramp": [],         # list of (t, v_cmd_target, v_actual)
    },
    "warmup_steps": int(2.0 * FPS),  # 2s warmup before sweep
    "current_v_cmd": 1.0,
    "fell": False,
    "done_logged": False,
}


# ----------------------------------------------------------------------
# Monkey-patches
# ----------------------------------------------------------------------

def _install_ramp_suppression():
    """Replace the task's stochastic ramp with a no-op so we drive v_cmd manually."""
    from phc.env.tasks.humanoid_im_res_amp_vcmd import HumanoidImResAMPVCmd
    HumanoidImResAMPVCmd._ramp_v_cmd = lambda self: None
    # Also disable per-reset re-sampling: when the policy falls and resets, we
    # want it to immediately respect the schedule's current v_cmd, not a fresh
    # uniform draw. Override _resample_v_cmd_target if it exists.
    if hasattr(HumanoidImResAMPVCmd, "_resample_v_cmd_target"):
        HumanoidImResAMPVCmd._resample_v_cmd_target = lambda self, env_ids: None


def _drive_schedule(task) -> bool:
    """Update task._v_cmd_target / _v_cmd_ramped per the eval schedule.
    Returns True if we should keep stepping, False if schedule is complete."""
    dt = 1.0 / FPS
    _STATE["phase_t"] += dt

    if _STATE["phase"] == "warmup":
        v = SWEEP_LEVELS[0]
        if _STATE["phase_t"] >= 2.0:
            _STATE["phase"] = "sweep"
            _STATE["phase_t"] = 0.0
            _STATE["samples"]["sweep"].append(
                {"v_cmd": v, "v_act": [], "fall_step": -1}
            )
        _STATE["current_v_cmd"] = v

    elif _STATE["phase"] == "sweep":
        v = SWEEP_LEVELS[_STATE["sweep_idx"]]
        # Record v_actual sample
        root_vel_xy = task._humanoid_root_states[0, 7:9]
        v_act = float(torch.linalg.norm(root_vel_xy).item())
        _STATE["samples"]["sweep"][-1]["v_act"].append(v_act)
        # Detect fall
        pelvis_z = float(task._humanoid_root_states[0, 2].item())
        if pelvis_z < 0.4 and _STATE["samples"]["sweep"][-1]["fall_step"] == -1:
            _STATE["samples"]["sweep"][-1]["fall_step"] = len(
                _STATE["samples"]["sweep"][-1]["v_act"]
            )
        # Advance level when duration elapsed
        if _STATE["phase_t"] >= SWEEP_DURATION_S:
            _STATE["sweep_idx"] += 1
            _STATE["phase_t"] = 0.0
            if _STATE["sweep_idx"] >= len(SWEEP_LEVELS):
                _STATE["phase"] = "ramp"
                _STATE["sweep_idx"] = 0
                _STATE["ramp_idx"] = 0
            else:
                v = SWEEP_LEVELS[_STATE["sweep_idx"]]
                _STATE["samples"]["sweep"].append(
                    {"v_cmd": v, "v_act": [], "fall_step": -1}
                )
        _STATE["current_v_cmd"] = v

    elif _STATE["phase"] == "ramp":
        target, hold_s = RAMP_SCRIPT[_STATE["ramp_idx"]]
        # Linear ramp at 0.5 m/s² toward target
        cur = _STATE["current_v_cmd"]
        max_step = 0.5 * dt
        cur += np.clip(target - cur, -max_step, max_step)
        _STATE["current_v_cmd"] = float(cur)
        # Record (t_global, v_cmd, v_actual)
        root_vel_xy = task._humanoid_root_states[0, 7:9]
        v_act = float(torch.linalg.norm(root_vel_xy).item())
        _STATE["samples"]["ramp"].append(
            (_STATE["phase_t"] + sum(s[1] for s in RAMP_SCRIPT[: _STATE["ramp_idx"]]),
             float(cur),
             float(target),
             v_act)
        )
        if _STATE["phase_t"] >= hold_s:
            _STATE["ramp_idx"] += 1
            _STATE["phase_t"] = 0.0
            if _STATE["ramp_idx"] >= len(RAMP_SCRIPT):
                _STATE["phase"] = "done"

    if _STATE["phase"] == "done":
        return False

    # Force task v_cmd state
    v = _STATE["current_v_cmd"]
    task._v_cmd_target.fill_(v)
    task._v_cmd_ramped.fill_(v)
    return True


def _install_pre_physics_hook(out_path: str):
    """Patch pre_physics_step to drive the eval schedule and dump JSON on done."""
    from phc.env.tasks.humanoid import Humanoid
    orig = Humanoid.pre_physics_step

    def patched(self, actions):
        keep_going = _drive_schedule(self)
        if not keep_going and not _STATE["done_logged"]:
            _STATE["done_logged"] = True
            _summarize_and_dump(out_path)
            print(f"[eval] schedule complete, results saved to {out_path}", flush=True)
            os._exit(0)
        return orig(self, actions)

    Humanoid.pre_physics_step = patched


# ----------------------------------------------------------------------
# Summary + JSON dump
# ----------------------------------------------------------------------

def _summarize_and_dump(out_path: str):
    sweep_results = []
    for entry in _STATE["samples"]["sweep"]:
        v_acts = np.asarray(entry["v_act"], dtype=float)
        if v_acts.size == 0:
            continue
        sweep_results.append({
            "v_cmd": entry["v_cmd"],
            "v_actual_mean": float(v_acts.mean()),
            "v_actual_std": float(v_acts.std()),
            "abs_err": float(abs(v_acts.mean() - entry["v_cmd"])),
            "fell": entry["fall_step"] >= 0,
            "fall_step": entry["fall_step"],
            "n_samples": int(v_acts.size),
        })

    ramp_results = {
        "trace": _STATE["samples"]["ramp"],   # [(t, v_cmd, v_target, v_act), ...]
        "settling_times": _compute_settling_times(_STATE["samples"]["ramp"]),
    }

    summary = {
        "sweep": sweep_results,
        "ramp": ramp_results,
        "pass": {
            "sweep_all_within_0.10": all(s["abs_err"] < 0.10 for s in sweep_results),
            "no_falls": not any(s["fell"] for s in sweep_results),
            "ramp_all_settle_under_2.5s": all(
                t is not None and t < 2.5 for t in ramp_results["settling_times"]
            ),
        },
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)


def _compute_settling_times(trace):
    """For each transition in RAMP_SCRIPT, find settling time = first t after
    transition start where |v_act - v_target| stays < SETTLE_TOL for 0.5 s."""
    if not trace:
        return [None] * (len(RAMP_SCRIPT) - 1)
    arr = np.asarray(trace, dtype=float)  # cols: t, v_cmd, v_target, v_act
    transitions = np.cumsum([s[1] for s in RAMP_SCRIPT[:-1]]).tolist()
    transitions = [0.0] + transitions  # transition START times
    settle_times = []
    for i, t_start in enumerate(transitions):
        target = RAMP_SCRIPT[i][0]
        t_end = t_start + RAMP_SCRIPT[i][1]
        mask = (arr[:, 0] >= t_start) & (arr[:, 0] < t_end)
        sub = arr[mask]
        if sub.size == 0:
            settle_times.append(None)
            continue
        # First time |v_act - target| < tol AND stays < tol for 0.5 s
        within = np.abs(sub[:, 3] - target) < SETTLE_TOL
        win = int(0.5 * FPS)
        settled_at = None
        for j in range(len(within) - win):
            if within[j : j + win].all():
                settled_at = float(sub[j, 0] - t_start)
                break
        settle_times.append(settled_at)
    return settle_times


# ----------------------------------------------------------------------
# Main: boot run.py with --test --epoch -1
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt",
        default="",
        help="Path to trained Humanoid.pth. If empty, picks newest in output/HumanoidIm/ResAMPVCmd/.",
    )
    ap.add_argument(
        "--out",
        default="02_research_dev/260501_res_amp_vcmd_eval_results.json",
    )
    ap.add_argument(
        "--headless",
        action="store_true",
        help="Run without viewer (default: viewer enabled for visual sanity check).",
    )
    args = ap.parse_args()

    # Resolve ckpt path. PHC's argparse path saves to output/<exp_name>_<epoch>.pth
    # (and a 'best so far' at output/<exp_name>.pth). Hydra path uses
    # output/HumanoidIm/<exp_name>/Humanoid_*.pth. Try both.
    ckpt = args.ckpt
    if not ckpt:
        for pattern in (
            "output/ResAMPVCmd_*.pth",                       # argparse path (run.py)
            "output/HumanoidIm/ResAMPVCmd/Humanoid_*.pth",   # hydra path
        ):
            candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
            if candidates:
                ckpt = candidates[-1]
                print(f"[eval] auto-picked latest checkpoint: {ckpt}")
                break
        if not ckpt:
            ckpt = "output/ResAMPVCmd.pth"
    if not os.path.exists(ckpt):
        sys.exit(f"[eval] ERROR: checkpoint not found: {ckpt}")

    _install_ramp_suppression()
    _install_pre_physics_hook(args.out)

    # Boot via run.py with --test --epoch <N from ckpt path or -1>
    sys.argv = [
        "phc/run.py",
        "--task", "HumanoidImResAMPVCmd",
        "--cfg_env", "phc/data/cfg/env/env_im_res_amp_vcmd.yaml",
        "--cfg_train", "phc/data/cfg/learning/im_res_amp_vcmd.yaml",
        "--num_envs", "1",
        "--test",
        "--epoch", "-1",
        "--checkpoint", ckpt,
    ]
    if args.headless:
        sys.argv.append("--headless")
        sys.argv.append("--no_virtual_display")

    print("=" * 70)
    print(f"[eval] ckpt={ckpt}")
    print(f"[eval] sweep: {len(SWEEP_LEVELS)} levels × {SWEEP_DURATION_S}s")
    print(f"[eval] ramp: {len(RAMP_SCRIPT)} segments, total "
          f"{sum(s[1] for s in RAMP_SCRIPT):.1f}s")
    print("=" * 70)

    import runpy
    try:
        runpy.run_path("phc/run.py", run_name="__main__")
    except SystemExit:
        pass

    # If we reach here without _summarize_and_dump being called (e.g. user
    # killed early), still try to dump partial.
    if not _STATE["done_logged"]:
        _summarize_and_dump(args.out)
        print(f"[eval] partial results saved to {args.out}")


if __name__ == "__main__":
    main()
