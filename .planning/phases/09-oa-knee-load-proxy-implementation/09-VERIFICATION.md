# Phase 9 Verification: OA Knee Load Proxy Implementation

**Status**: PASS
**Date**: 2026-04-29

## Commands

```bash
conda run --no-capture-output -n phc python -m unittest tests.test_pain_knee_proxy -v
```

Result: passed, 5 tests OK.

```bash
conda run --no-capture-output -n phc python -m py_compile \
  phc/env/util/pain_baseline.py \
  phc/env/tasks/humanoid_im_pain.py \
  phc/learning/im_amp_players.py
```

Result: passed.

```bash
conda run --no-capture-output -n phc python - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("phc/data/cfg/env/env_im_pain_v1.yaml").read_text())
assert cfg["pain"]["knee_mechanism"]["proxy_mode"] == "oa_contact_v15"
assert cfg["pain"]["knee_mechanism"]["w_legacy_torque_load"] == 0.0
print("env_im_pain_v1.yaml OK")
PY
```

Result: passed.

```bash
git diff --check
```

Result: passed.

## Headless Probe

```bash
PHC_PAIN_PROBE_JSON=analysis/plots/phc_pain_oa_load_probe/pretrained_oa_probe.json \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn env=env_im_pain_v1 exp_name=phc_shape_pnn_iccv \
  epoch=-1 test=True headless=True env.num_envs=32 \
  env.motion_file=sample_data/amass_isaac_walking_forward_subset23.pkl \
  robot.has_shape_obs_disc=True learning.params.config.player.games_num=3
```

Result: passed. The short probe produced finite values for the required OA proxy fields:

| Metric | Value |
|---|---:|
| `pain_v1_right_knee_compression` | 0.18638019636273384 |
| `pain_v1_right_knee_loaded_flex` | 0.01365843357052654 |
| `pain_v1_right_knee_loading_rate` | 0.0023503638803958893 |
| `pain_v1_right_knee_kam` | 0.05676730442792177 |
| `pain_v1_right_knee_kfm` | 0.058967757504433393 |
| `pain_v1_right_knee_contact_load` | 0.13343286886811256 |
| `pain_v1_right_knee_moment_load` | 0.057427442632615566 |
| `pain_v1_right_knee_torque_load` | 0.28062527626752853 |
| `pain_v1_right_knee_load` | 0.10303070209920406 |
| `pain_v1_right_knee_state` | 0.0011279801838099957 |

## Caveats

- `pytest` is unavailable in the `phc` environment, so the equivalent `unittest` command is the passing local test gate.
- The probe validates wiring only. It is intentionally short and should not be used as gait-quality evidence.
