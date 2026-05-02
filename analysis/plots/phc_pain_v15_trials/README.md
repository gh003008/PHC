# PHC-Pain v1.5 Trial Summary

This folder summarizes the training trials used to move from the PHC pretrained walking controller to the current moment-biased OA knee pain controller.

## Files

- `trial_manifest.csv`: one row per trial, with the qualitative purpose and decision.
- `trial_metric_summary.csv`: TensorBoard scalar summaries parsed from `server1-jinsu:/home/jinsu/PHC_v15_sanity/output/HumanoidIm`.
- `training_reward_design.md`: detailed per-trial training setup, reward parameters, pain formulation, and interpretation.

## Trial Chronology

1. `phc_shape_pnn_iccv`: original PHC pretrained baseline.
2. `phc_shape_pnn_iccv_pain_v15_oa_sanity_idx3`: v1.5 OA load proxy sanity/long run. It reduced the logged OA knee proxy but did not visibly change gait much.
3. `phc_shape_pnn_iccv_pain_v15_oa_aggressive_idx3`: aggressive pain pressure. It reduced right knee load too well by learning right-leg non-contact/hopping.
4. `phc_shape_pnn_iccv_pain_v15_oa_middle_idx3`: moderate pain pressure. It was stable but visually close to pretrained and still allowed start/end foot-lift behavior.
5. `phc_shape_pnn_iccv_pain_v15_oa_walkgate_idx3`: walking-gated pain plus right stance contact floor. It prevented the foot-lift cheat but increased right knee loading by over-enforcing contact.
6. `phc_shape_pnn_iccv_pain_v15_oa_momentbias_idx3`: softer contact floor plus stronger moment/KAM/KFM pain. This is the current best candidate because it maintains contact while reducing right knee moment proxies and showing a visible overlaid difference from PHC pretrained.

## Video Capture

The player now supports reproducible local IsaacGym recording through environment variables:

```bash
PHC_VIEW_CAMERA=sagittal_right \
PHC_VIEW_ENV=0 \
PHC_AUTO_RECORD_FRAMES=240 \
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain_v15_oa_momentbias_idx3 epoch=-1 test=True \
  env=env_im_pain_v1 env.num_prim=4 \
  robot=smpl_humanoid_shape robot.has_shape_variation=False robot.freeze_hand=True robot.box_body=False robot.has_shape_obs_disc=True \
  env.motion_file=sample_data/phc_pain_overlay_single_walk.pkl \
  env.num_envs=1 env.env_spacing=0.0 headless=False \
  learning.params.config.player.games_num=100000 \
  +learning.params.config.player.render_sleep=0.033
```

For live visual comparison, use `PHC_COMPARE_SECONDARY_CHECKPOINT=output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth`, `env.num_envs=2`, and `PHC_VIEW_ENV=1`. In comparison mode, env0 is blue/pretrained and env1 is red/current.

The accepted right-leg sagittal view uses:

```bash
PHC_VIEW_CAMERA=sagittal_right
PHC_VIEW_ENV=1
PHC_VIEW_HEIGHT=0.45
PHC_VIEW_TARGET_Z=0.55
PHC_VIEW_DISTANCE=3.2
```

`sagittal_right` is computed from the current humanoid's `R_Hip - L_Hip` body-side vector, not from a fixed world axis. Use `PHC_VIEW_CAMERA_OFFSET=x,y,z` for a custom fixed offset.

To regenerate all overlaid comparison videos:

```bash
bash scripts/record_phc_pain_v15_overlay_videos.sh
```

The script opens IsaacGym with both actors in the same scene (`env0=blue PHC pretrained`, `env1=red trial checkpoint`, `env.env_spacing=0.0`) and records the actual X11 viewer with `ffmpeg x11grab`. No alpha-blending or post-hoc model compositing is used. The videos are written under `analysis/plots/phc_pain_v15_trials/videos/`.

Generated overlaid comparison videos:

- `videos/01_v15_oa_sanity_vs_pretrained_sagittal_right.mp4`
- `videos/02_aggressive_vs_pretrained_sagittal_right.mp4`
- `videos/03_middle_vs_pretrained_sagittal_right.mp4`
- `videos/04_walkgate_vs_pretrained_sagittal_right.mp4`
- `videos/05_momentbias_vs_pretrained_sagittal_right.mp4`
