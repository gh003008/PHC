# V3 Inference Launch (frozen phc_3 + 3 clips + v_cmd + retime)

**Date**: 2026-05-02
**Purpose**: V3 architecture inference verification — zero-training, frozen phc_3 base + multi-clip + v_cmd-driven selection + per-step retime.

## What this demonstrates

The trained component-free version of the V3 design from
`01_research_docs/260502_personalized_diverse_motion_roadmap.md`:

- ✅ 3 motion clips (slow / medium / fast walking)
- ✅ v_cmd input (keyboard or Tk panel)
- ✅ v_cmd-driven clip auto-selection (cycle boundary + hysteresis 0.02)
- ✅ Per-step retime within selected clip (`v_cmd / v_natural`)
- ✅ **Frozen phc_3 walks** — no trained residual head

## Launch

```bash
conda activate phc
DISPLAY=:1 python scripts/phc_walk_demo.py
```

Optional 30 s recording to `videos/phc_walk_demo.mp4`:
```bash
DISPLAY=:1 python scripts/phc_walk_demo.py --record_seconds 30
```

## Controls

- **↑ / ↓**: v_cmd ±0.02 (clamped to [0.76, 1.23])
- **1..9**: v_cmd snap (linear interp between V_CMD_MIN and V_CMD_MAX)
- **R**: episode reset
- **P**: pause / resume
- **Q**: quit

Tk panel also supports numeric input + slider + Apply/Reset/Pause buttons.

## Verified session (2026-05-02, 1860 step ≈ 62 s)

- 3 clip 모두 활성화 (clip 0/1/2 전부 사용됨)
- v_cmd midpoint cross 시 자동 clip switch (0.78 → 1.23 → 0.79 양방향 chain switch)
- Retime ratio 범위 0.847 ~ 1.169
- Falling 0회

## Files in this snapshot

- `phc_walk_demo.py` — main demo script
- `phc_walk_demo_panel.py` — Tk panel
- `env_im_pnn.yaml` — env config (phc_3 base)
- `im_pnn_big.yaml` — learning config (phc_3 PNN architecture)

Frozen ckpt loaded from: `output/HumanoidIm/phc_3/Humanoid.pth`
Motion file: `sample_data/amass_walking_3clips_seamless_60s_v9.pkl`
