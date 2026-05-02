# PHC Overlay Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show PHC pretrained and PHC-Pain latest checkpoint in one IsaacGym viewer with different colors on the same walking motion.

**Architecture:** Add an opt-in player overlay mode controlled by `PHC_COMPARE_SECONDARY_CHECKPOINT`. When enabled, env 0 is driven by the secondary checkpoint and env 1 by the primary checkpoint loaded through the normal `exp_name` path.

**Tech Stack:** PHC, IsaacGym, rl_games player, joblib AMASS motion pickle.

---

### Task 1: Opt-In Overlay Player

**Files:**
- Modify: `phc/learning/im_amp_players.py`

- [ ] Add secondary-checkpoint loading after player construction.
- [ ] Override `get_action` only when `PHC_COMPARE_SECONDARY_CHECKPOINT` is set.
- [ ] Color env 0 blue and env 1 red so pretrained/latest can be distinguished.
- [ ] Refuse overlay mode unless `env.num_envs >= 2`, because one env cannot show two policies.

### Task 2: Single-Motion Overlay Input

**Files:**
- Generate: `sample_data/phc_pain_overlay_single_walk.pkl`

- [ ] Load `sample_data/amass_isaac_walking_forward_subset23.pkl`.
- [ ] Keep only `0-KIT_11_WalkingStraightForwards05_poses`.
- [ ] Save the generated one-motion pickle for the viewer command.

### Task 3: Viewer Launch

**Command:**
```bash
PHC_COMPARE_SECONDARY_CHECKPOINT=output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn \
  exp_name=phc_shape_pnn_iccv_pain_v15_oa_sanity_idx3 \
  epoch=-1 \
  test=True \
  env=env_im_pain_v1 \
  env.num_prim=4 \
  robot=smpl_humanoid_shape \
  robot.freeze_hand=True \
  robot.box_body=False \
  robot.has_shape_obs_disc=True \
  env.motion_file=sample_data/phc_pain_overlay_single_walk.pkl \
  env.num_envs=2 \
  env.env_spacing=0.0 \
  headless=False \
  learning.params.config.player.games_num=100000 \
  +learning.params.config.player.render_sleep=0.033
```

Expected: IsaacGym opens with env 0 as blue PHC pretrained and env 1 as red PHC-Pain latest.
