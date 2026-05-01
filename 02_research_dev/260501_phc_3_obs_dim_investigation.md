# phc_3 Obs Dim 불일치 원인 조사 (260501)

작성일: 2026-05-01
계기: RES_AMP_VCMD Task 16 smoke 부팅 단계에서 발생한 첫 forward pass crash:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x454 and 934x2048)
```

## 1. 한 줄 요약

**우리 yaml 의 `trackBodies: ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]` 한 줄이 원인.** 이 줄을 제거하면 phc_3 학습 시 사용된 SMPL 24 body 전체가 track 대상이 되어 obs dim 이 934 로 매칭됨.

## 2. 문제

phc_3 PNN 첫 layer 의 weight shape 는 `[2048, 934]` (Task 12 에서 확인). 즉 phc_3 는 934-D obs 를 입력으로 기대.

`HumanoidImResAMPVCmd` 는 v_cmd 1 dim 를 추가했고 frozen base 는 `obs[:, :-1]` 로 v_cmd 를 떼고 phc_3 에 공급함. 따라서 frozen base 입력은 (env obs - 1) dim. 우리 env 는 455-D obs 를 만들었고 → frozen base 입력 454-D → 934 와 미스매치 → CUDA matmul fail.

Gap = 934 - 454 = **480 dims** missing.

## 3. v3 데모는 왜 정상 동작하는가

`scripts/phc_walk_demo.py` 가 phc_3 을 부팅할 때 사용하는 설정:
```python
EXP_NAME = "phc_3"
LEARNING = "im_pnn_big"
ENV_CFG_NAME = "env_im_pnn"
```

즉 `phc/data/cfg/env/env_im_pnn.yaml` + `phc/data/cfg/learning/im_pnn_big.yaml` 조합. 그리고 v3 데모는 `HumanoidIm` task 를 그대로 사용 (override 없음). 이게 phc_3 의 학습 시점 환경과 정확히 일치함.

증거: `output/HumanoidIm/phc_3/.hydra/overrides.yaml` 에 phc_3 가 마지막으로 부팅될 때 사용된 정확한 cli args 가 보존되어 있음:
```yaml
- learning=im_pnn_big
- env=env_im_pnn
- robot=smpl_humanoid
- exp_name=phc_3
- env.num_envs=3
- env.motion_file=sample_data/amass_walking_3clips_seamless_60s_v9.pkl
```

`env=env_im_pnn` 이므로 phc_3 는 `env_im_pnn.yaml` 의 obs 설정으로 학습/추론됨.

## 4. obs 차원 산출

PHC `humanoid_im.py:39-45` 의 핵심 분기:
```python
self._fut_tracks = cfg["env"].get("fut_tracks", False)  # default False
if self._fut_tracks:
    self._num_traj_samples = cfg["env"]["numTrajSamples"]
else:
    self._num_traj_samples = 1   # ← 여기!
```

→ **`fut_tracks: False` 면 `_num_traj_samples = 1`** (yaml 의 `numTrajSamples: 3` 은 무시됨). phc_3 와 우리 env 둘 다 `fut_tracks: False`.

그리고 `humanoid_im.py:64`:
```python
self._track_bodies = cfg["env"].get("trackBodies", self._full_track_bodies)
```

→ yaml 에 `trackBodies` 필드가 **없으면** `_full_track_bodies` (= SMPL 24 bodies 전체) 가 기본값.

### obs_v=6 의 task_obs 산출 (`humanoid_im.py:504-505`):
```python
elif self.obs_v == 6:
    obs_size = len(self._track_bodies) * self._num_traj_samples * 24
```

### self_obs 산출 (`humanoid.py:665`, SMPL 의 경우):
```python
self._num_self_obs = 1 + len(self._body_names) * (3 + 6 + 3 + 3) - 3
                   = 1 + 24*15 - 3 = 358
```

`has_shape_obs=False`, `has_limb_weight_obs=False` (phc_3 의 robot config 에서 확인) → 추가 보정 없음.

### 합산

| | trackBodies 필드 | num_track | num_traj | task_obs | self_obs | total |
|---|---|---|---|---|---|---|
| **phc_3** (env_im_pnn) | 없음 → 24 default | 24 | 1 (fut_tracks=False) | 24×1×24 = **576** | 358 | **934** ✓ |
| **우리 env_im_res_amp_vcmd** | `[R_Ankle, L_Ankle, R_Wrist, L_Wrist]` | 4 | 1 | 4×1×24 = **96** | 358 | **454** ← crash 의 64×454 |

454 + 1 (v_cmd) = 455 = env obs.
455 - 1 (frozen base 가 v_cmd 떼냄) = 454 = mat1.
934 ≠ 454 → matmul mismatch.

## 5. 수정 후속 옵션

### Path A: trackBodies 필드 제거 (권장)

`phc/data/cfg/env/env_im_res_amp_vcmd.yaml` 의 `trackBodies: ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]` 한 줄 삭제.

```diff
-  trackBodies: ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]
```

효과:
- `_track_bodies` 가 `_full_track_bodies` = 24 SMPL bodies 로 fallback
- task_obs = 24 × 1 × 24 = 576 → total = 358 + 576 + 1 (v_cmd) = **935**
- frozen base 입력 = 935 - 1 = **934** ✓ phc_3 과 매칭

장점: 
- 한 줄 삭제로 끝
- phc_3 의 학습 시점 obs 와 정확히 동일 → phc_3 baseline 의 동작이 정상 (학습-추론 분포 일치)
- residual + AMP 학습 의도와 일치 (phc_3 가 24 body 모두 본 채로 동작하고 residual 만 v_cmd 보정)

단점: 
- 우리는 보행 추적용으로 4 발/손목만 보면 충분하다고 생각했었지만, 실제로는 phc_3 가 학습된 24 body obs 를 그대로 줘야 함. 이건 phc_3 와의 호환성 비용.

### Path B: obs projection layer 추가 (비권장)

frozen base 입력으로 480 차원 가짜 데이터를 채워넣거나, 작은 MLP 로 454 → 934 projection. residual 이 자유도 확보하지만 phc_3 가 분포 외 obs 를 받게 됨 → 학습된 baseline 의 가치 훼손. 학습 instability 위험.

### Path C: phc_3 폐기, AMP-only from scratch (escape hatch)

VIC4_VCMD S10-S19B 시리즈의 길. α + AMP 의 핵심 (frozen baseline 활용) 을 포기. 우리 GPU 자원 한계 고려 시 비현실적.

## 6. 권장 다음 단계

1. **Path A 적용** — `phc/data/cfg/env/env_im_res_amp_vcmd.yaml` 에서 `trackBodies` 한 줄 삭제, commit:
   ```bash
   git commit -m "res_amp_vcmd: remove trackBodies override (match phc_3 24-body obs)"
   ```

2. **추가 yaml 정리** (optional but recommended): phc_3 의 `.hydra/config.yaml` 과 우리 yaml 의 obs 무관 기타 차이도 정렬:
   - `zero_out_far: True` → `False` (phc_3 와 일치)
   - `recoveryEpisodeProb: 0.5`, `fallInitProb: 0.3` 제거 (VIC getup 용, 우리 task 와 무관)
   - `has_pnn: False` → `True` (phc_3 와 일치, 향후 일관성)
   - `power_reward: False` → `True` (phc_3 와 일치, 우리는 reward override 하니 효과 없음)
   
   이건 obs dim 과는 무관하지만 학습 분포를 phc_3 와 더 가깝게 만듦.

3. **Task 16 smoke 재시도**:
   ```bash
   bash scripts/launch_res_amp_smoke.sh
   ```
   다음 iteration 에서 다른 issue 가 있으면 같은 방식으로 fix → re-run.

4. **Task 17 (사용자 6h smoke run)** 으로 진행.

## 7. 학습된 교훈 (반영점)

- **trackBodies 는 obs 차원에 직접적 영향**. PHC task 를 base 로 새 task class 만들 때 yaml 의 obs-affecting 필드 (`obs_v`, `fut_tracks`, `numTrajSamples`, `trackBodies`, `numAMPObsSteps`, `self_obs_v`, `has_shape_obs`, `has_limb_weight_obs`, `enable_hist_obs`) 는 baseline 과 일치시키는 게 안전.

- **`.hydra/overrides.yaml` 와 `.hydra/config.yaml` 은 ground truth**. 이번처럼 baseline 의 정확한 학습 설정이 의심될 때 가장 먼저 봐야 할 파일. 다음에 비슷한 통합 작업 시 첫 번째로 확인.

- **fut_tracks=False → num_traj_samples=1 강제**. yaml 에 `numTrajSamples: 3` 이 있어도 fut_tracks 분기에서 1 로 강제되니 obs size 산출 시 주의.

## 8. 참고

- 관련 commit: `dd3f782` (Task 16 smoke launch + 자동 fix 5 종)
- 관련 spec: `docs/superpowers/specs/2026-05-01-phc-residual-amp-vcmd-design.md`
- 관련 plan: `docs/superpowers/plans/2026-05-01-phc-residual-amp-vcmd.md`
- ground truth config: `output/HumanoidIm/phc_3/.hydra/overrides.yaml` + `config.yaml`
