# PHC-Pain-v1.0 Training And Ablation Commands

These commands define the Phase 4 condition matrix. They are intentionally
bounded smoke/training probes; scale `env.num_envs`, `horizon_length`, and
`max_epochs` only after the condition wiring is verified.

Common base:

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn \
  exp_name=phc_shape_pnn_iccv_pain_v1 \
  epoch=-1 \
  test=False \
  env=env_im_pain_v1 \
  env.num_prim=4 \
  robot=smpl_humanoid_shape \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=128 \
  learning.params.config.horizon_length=32 \
  headless=True \
  learning.params.config.max_epochs=50 \
  learning.params.config.print_stats=True
```

## Main: pain_obs_on + pain_reward_on

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain_v1 epoch=-1 test=False \
  env=env_im_pain_v1 env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=128 learning.params.config.horizon_length=32 headless=True \
  env.pain.obs.enabled=True env.pain.mode=reward_only \
  env.pain.active_knee_side=right \
  learning.params.config.max_epochs=50 \
  learning.params.config.print_stats=True
```

## Ablation: pain_obs_off + pain_reward_on

Tests whether the policy needs to observe the pain map to learn the avoidance
behavior.

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain_v1_no_obs epoch=-1 test=False \
  env=env_im_pain_v1 env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=128 learning.params.config.horizon_length=32 headless=True \
  env.pain.obs.enabled=False env.pain.append_to_obs=False env.pain.mode=reward_only \
  env.pain.active_knee_side=right \
  learning.params.config.max_epochs=50 \
  learning.params.config.print_stats=True
```

## Ablation: pain_obs_on + pain_reward_off

Verifies that adding the observation alone does not drive the mechanism.

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain_v1_no_reward epoch=-1 test=False \
  env=env_im_pain_v1 env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=128 learning.params.config.horizon_length=32 headless=True \
  env.pain.obs.enabled=True env.pain.mode=log_only \
  env.pain.active_knee_side=right \
  learning.params.config.max_epochs=50 \
  learning.params.config.print_stats=True
```

## Side specificity: left-knee impairment

Run the same main condition with the active side flipped:

```bash
conda run --no-capture-output -n phc python phc/run_hydra.py \
  learning=im_pnn exp_name=phc_shape_pnn_iccv_pain_v1_left epoch=-1 test=False \
  env=env_im_pain_v1 env.num_prim=4 \
  robot=smpl_humanoid_shape robot.freeze_hand=True robot.box_body=False \
  env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
  env.num_envs=128 learning.params.config.horizon_length=32 headless=True \
  env.pain.obs.enabled=True env.pain.mode=reward_only \
  env.pain.active_knee_side=left \
  env.pain.knee_mechanism.left_sensitivity=1.0 \
  env.pain.knee_mechanism.right_sensitivity=0.0 \
  learning.params.config.max_epochs=50 \
  learning.params.config.print_stats=True
```

## Interpretation

- Main condition should be compared against both ablations.
- Left/right runs should produce comparable logs for side-specificity checks.
- Any guard-enabled run is debug-only and cannot support the v1.0 mechanism claim.
