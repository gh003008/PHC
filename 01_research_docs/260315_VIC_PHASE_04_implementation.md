# VIC_PHASE_04 구현 상세 (260315)

## 1. 실험 목적

VIC_PHASE_03에서 bio CCF reward의 효과를 확인했으나 세 가지 한계가 있었다:
1. **Ankle에만 적용**: Knee는 bio reward 대상이 아니어서 항상 compliant(0.65x)로 유지 — 인간은 stance에서 knee가 stiff해야 함
2. **Bio reward weight 부족**: w=0.05가 너무 작아 stance/swing 간 CCF 변화폭이 10~20% 수준
3. **에너지 페널티 과도**: power_coefficient=0.000005로 인해 종종걸음(shuffling gait) 발생. 큰 토크가 필요한 시원한 한 걸음보다 작은 토크의 종종걸음이 에너지 관점에서 유리

VIC_PHASE_04는 이 세 가지를 동시에 조정하여 인간에 더 가까운 phase-dependent 임피던스를 학습시킨다.

## 2. 핵심 컨셉: 인간 보행 임피던스와의 비교

### 인간 보행의 관절별 임피던스 패턴 (생체역학 문헌 기반)

| 관절 | Stance Phase | Swing Phase | 기능적 역할 |
|---|---|---|---|
| Ankle | 매우 높음 (2~5x) | 낮음 | Stance: 체중 지지 + push-off, Swing: dorsiflexion 유지 |
| Knee | 높음 (loading response, push-off) | 낮음 | Stance: weight acceptance, Swing: 수동적 flexion |
| Hip | 중간~높음 | 중간 | 골반 안정화, flexor/extensor 교대 |
| Upper body | 낮음 | 낮음 | 수동적 arm swing |

### VIC_PHASE_03 결과 vs 인간 (gait cycle phase 기준 재평가)

| 관절 | 인간 Stance/Swing 차이 | PHASE_03 결과 | 평가 |
|---|---|---|---|
| Ankle | 극적 (2~5x → 0.5~1x) | 전체적으로 높음 (1.75~1.80x), **phase 변화 없음** | 상수 stiff, swing compliance 부재 |
| Knee | 뚜렷 (높음 → 낮음) | 항상 compliant (0.65x), **phase 변화 없음** | Stance stiffness 완전 부재 |
| Hip | 중간 정도 antiphase | L~0.98x, R~1.16x, **phase 변화 없음** | L/R 수준 차이만 존재, antiphase 없음 |

주의: 이전 분석에서 "phase-dependent 변화" 보고는 clip phase vs gait cycle phase 혼동에 의한 착시였음 (후술 §6.2 참조).

### VIC_PHASE_04에서 기대하는 개선

1. **Knee stance stiffness 출현**: bio CCF reward를 knee에 확대하여 stance에서 knee CCF>0 유도
2. **Phase-dependent 변화폭 증가**: bio reward weight 3배 강화 (0.05→0.15)로 stance/swing 간 CCF 차이 증폭
3. **종종걸음 → 자연스러운 보폭**: 에너지 페널티 완화로 큰 토크 사용 허용

### 비교 대상

- **VIC_PHASE_03**: bio CCF reward ankle only, w=0.05, power=0.000005 (직전 실험, 변경 효과 분리 필요)
- **인간 보행 문헌**: stance/swing에서 2~5배 임피던스 변화가 목표 기준
- **VIC_PHASE(01)**: bio CCF reward 없는 baseline (phase obs만 있을 때의 기본 CCF 분포)

## 3. 변경 사항 요약 (VIC_PHASE_03 대비)

| 항목 | VIC_PHASE_03 | VIC_PHASE_04 | 변경 이유 |
|---|---|---|---|
| Bio CCF reward 대상 | Ankle만 (G2, G5) | **Ankle + Knee (G1, G2, G4, G5)** | Knee stance stiffness 유도 |
| Bio CCF reward weight | 0.05 | **0.15** (3x) | Phase-dependent 변화폭 강화 |
| Power coefficient | 0.000005 | **0.000002** (0.4x) | 종종걸음 억제, 자연스러운 보폭 유도 |
| Motion file | amass_isaac_walking_forward_long.pkl | **amass_isaac_walking_forward_single.pkl** | VIC_PHASE(01)과 동일 짧은 모션으로 복귀 |

## 4. MDP 구성

### 4-1. State (Observation)

VIC_PHASE_03과 동일. 변경 없음.

| 구성 요소 | Dims | 설명 |
|---|---|---|
| Body positions/rotations | ~166 | dof_pos, vel, root state 등 (self_obs) |
| Target tracking (task_obs) | 288 | 4 bodies x 3 samples x 24 (obs_v=6) |
| Phase obs (sin/cos) | 2 | Gait cycle phase 인코딩 |
| **총 obs dims** | **456** | |

### 4-2. Action

VIC_PHASE_03과 동일. 변경 없음.

| 구성 요소 | Dims | 설명 |
|---|---|---|
| PD target (joint angles) | 69 | 각 DOF의 목표 관절 각도 |
| CCF (Compliance Control Factor) | 8 | 8 그룹별 임피던스 배율 (2^ccf) |
| **총 action dims** | **77** | |

CCF 그룹 매핑:
- G0: L_Hip (3 DOFs), G1: L_Knee (3), G2: L_Ankle+Toe (6)
- G3: R_Hip (3), G4: R_Knee (3), G5: R_Ankle+Toe (6)
- G6: Upper-L (30), G7: Upper-R (15)

### 4-3. Reward

| Term | 수식/설명 | 가중치 | 변경 |
|---|---|---|---|
| Point goal | distance 기반 위치 보상 | 1.0 | - |
| Imitation | pos/rot/vel/ang_vel 매칭 (k_pos=200, k_rot=10, k_vel=1.0, k_ang_vel=0.1) | 0.5 | - |
| AMP discriminator | 적대적 모션 자연스러움 판별 | curriculum | - |
| Power penalty | -power_coeff * sum(abs(torque * dof_vel)) | 1.0 | **power_coeff: 0.000005→0.000002** |
| Bio CCF | stance: clamp(ccf, min=0), swing: clamp(-ccf, min=0) | **0.15** | **w: 0.05→0.15, 대상: ankle+knee** |

Reward curriculum: epoch < 10000 → task:0.7/disc:0.3, epoch >= 10000 → task:0.3/disc:0.7

Bio CCF reward 상세:
```
contact_threshold = 10N
is_stance = (foot_force > threshold)

# Ankle (G2=L_Ankle+Toe, G5=R_Ankle+Toe)
ankle_rwd = is_stance * clamp(ankle_ccf, min=0) + (1-is_stance) * clamp(-ankle_ccf, min=0)

# Knee (G1=L_Knee, G4=R_Knee)  ← 신규
knee_rwd = is_stance * clamp(knee_ccf, min=0) + (1-is_stance) * clamp(-knee_ccf, min=0)

bio_reward = (l_ankle + r_ankle + l_knee + r_knee) * 0.25 * w
```

### 4-4. Termination

| 조건 | 값 | 설명 |
|---|---|---|
| enableEarlyTermination | True | 넘어지면 종료 |
| terminationHeight | 0.15m | 루트 높이 기준 |
| terminationDistance | 0.25m | reference 대비 위치 오차 |
| episode_length | 300 steps (10s) | 최대 에피소드 길이 |
| 모션 종료 | ~131 steps (4.4s) | cycle_motion=False → 모션 끝나면 종료 |

## 5. 학습 설정

| 항목 | 값 |
|---|---|
| 실험명 | VIC_PHASE_04 |
| 기반 | VIC_PHASE_03 + knee bio CCF + weight 3x + power 0.4x |
| motion_file | amass_isaac_walking_forward_single.pkl (4.4s, 4걸음) |
| cycle_motion | False |
| vic_curriculum_stage | 2 |
| vic_ccf_num_groups | 8 |
| vic_ccf_sigma_init | -1.0 (std=0.37) |
| vic_phase_obs | True (+2 dims) |
| vic_bio_ccf_reward_w | 0.15 (3x 강화) |
| power_coefficient | 0.000002 (0.4x 완화) |
| max_epochs | 20000 |
| num_envs | 512 |

## 6. 코드 수정 내용

### 6-1. _compute_bio_ccf_reward() 수정 (`humanoid_im_vic.py`)

기존 ankle만 대상이던 것을 knee(G1, G4)에도 확대.

변경 전:
```python
l_ankle_ccf = self._last_ccf_raw[:, 2]
r_ankle_ccf = self._last_ccf_raw[:, 5]
l_reward = l_is_stance * clamp(l_ankle_ccf, min=0) + (1-l_is_stance) * clamp(-l_ankle_ccf, min=0)
r_reward = r_is_stance * clamp(r_ankle_ccf, min=0) + (1-r_is_stance) * clamp(-r_ankle_ccf, min=0)
bio_reward = (l_reward + r_reward) * 0.5 * w
```

변경 후:
```python
l_ankle_ccf = self._last_ccf_raw[:, 2]  # G2
r_ankle_ccf = self._last_ccf_raw[:, 5]  # G5
l_knee_ccf = self._last_ccf_raw[:, 1]   # G1 (신규)
r_knee_ccf = self._last_ccf_raw[:, 4]   # G4 (신규)

# Ankle reward
l_ankle_rwd = l_is_stance * clamp(l_ankle_ccf, min=0) + (1-l_is_stance) * clamp(-l_ankle_ccf, min=0)
r_ankle_rwd = r_is_stance * clamp(r_ankle_ccf, min=0) + (1-r_is_stance) * clamp(-r_ankle_ccf, min=0)

# Knee reward (신규)
l_knee_rwd = l_is_stance * clamp(l_knee_ccf, min=0) + (1-l_is_stance) * clamp(-l_knee_ccf, min=0)
r_knee_rwd = r_is_stance * clamp(r_knee_ccf, min=0) + (1-r_is_stance) * clamp(-r_knee_ccf, min=0)

bio_reward = (l_ankle_rwd + r_ankle_rwd + l_knee_rwd + r_knee_rwd) * 0.25 * w
```

### 6-2. Phase Logging 버그 수정 (`humanoid_im_vic.py`) — 중요

**발견 경위**: VIC_PHASE_04 평가 후 knee impedance가 "swing에서 stiff, stance에서 compliant"로 보여 보상 함수가 반대로 되었는지 의심. 조사 결과, 보상 함수는 정상이었으나 **phase logging이 gait cycle phase가 아닌 clip phase를 기록**하고 있었음.

**버그**: 기존 phase logging 코드
```python
# 기존 (버그): clip 내 진행률 = 전체 모션 길이 대비 현재 시간
gait_phase = torch.clamp(curr_time[0] / motion_len[0], 0.0, 1.0).item()
```
- 이 값은 4.4초 모션 클립 내에서 0→1로 한 번만 증가
- Gait cycle phase(한 보행 주기 ~0.87초 내 0→1, 4~5회 반복)와 전혀 다름

**수정**: `_gait_phase_table` 기반 gait cycle phase lookup
```python
# 수정 후: gait cycle phase table에서 정확한 phase 값 조회
if self._vic_phase_obs and hasattr(self, '_gait_phase_table'):
    curr_time = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
    motion_len = self._motion_lib._motion_lengths[self._sampled_motion_ids]
    time_in_clip = curr_time % motion_len
    num_samples = len(self._gait_phase_table)
    frame_idx = (time_in_clip / motion_len * num_samples).long().clamp(0, num_samples - 1)
    gait_phase = self._gait_phase_table[frame_idx[0]].item()
```

**영향**: 이전 모든 phase-CCF 분석(VIC_PHASE, PHASE_02, PHASE_03, PHASE_04 초기)이 clip phase 기준이었으므로, gait cycle 내 stance/swing 분화 해석이 정확하지 않았음. 수정 후 VIC_PHASE, PHASE_03, PHASE_04를 재평가하여 올바른 gait cycle phase 기준 결과를 확보.

### 6-3. 설정 파일 수정

`phc/data/cfg/env/env_im_walk_vic.yaml`:
```yaml
motion_file: "sample_data/amass_isaac_walking_forward_single.pkl"  # long→single 복귀
cycle_motion: False
power_coefficient: 0.000002   # (기존 0.000005)
vic_bio_ccf_reward_w: 0.15    # (기존 0.05)
```

`phc/data/cfg/learning/im_walk_vic.yaml`:
```yaml
name: VIC_PHASE_04  # (기존 VIC_PHASE_02)
```

## 7. 실험 진행 히스토리

### 7-1. 모션 파일 혼동 및 재학습

초기 학습 시작 시 모션 파일을 변경하지 않아 `amass_isaac_walking_forward_long.pkl` (8.77s)로 학습이 시작됨. 의도는 VIC_PHASE(01)과 동일한 `amass_isaac_walking_forward_single.pkl` (4.4s)이었으므로:
1. 진행 중인 학습 프로세스 종료
2. 잘못된 체크포인트 삭제
3. `motion_file`을 `single.pkl`로 변경
4. 처음부터 재학습

이 히스토리로 인해 구현 문서 초기 버전의 termination 항목에 `~262 steps (8.77s)`가 잘못 기재되어 있었음. 정확한 값은 `~131 steps (4.4s)`.

### 7-2. Phase logging 버그 발견 및 수정

평가 결과에서 knee impedance가 "phase 후반에 stiff" 패턴을 보여 보상 함수 역전을 의심. 조사 결과:
1. 보상 함수는 정상 (stance→stiff, swing→compliant 방향)
2. Phase logging이 clip phase를 기록하고 있었음 (gait cycle phase가 아닌)
3. Clip phase 기준 "후반 stiff"는 단순히 클립 진행에 따른 점진적 변화이지 gait cycle 내 분화가 아님
4. `_gait_phase_table` 기반으로 수정 후 재평가 → 실제로는 phase-dependent 분화 없음 확인

## 8. 실험 설정 백업

`exp_config/forward_walking/260315_VIC_PHASE_04/`에 스냅샷:
- env_im_walk_vic.yaml
- im_walk_vic.yaml
- humanoid_im_vic.py

## 9. 설계 의도

1. **Knee 포함**: 인간 보행에서 knee는 stance phase의 weight acceptance에 핵심적. Ankle만으로는 하지 전체의 임피던스 패턴을 재현 불가
2. **Bio reward 3x 강화**: PHASE_03에서 w=0.05는 방향성만 제공할 뿐 변화폭이 부족. 0.15로 올려도 총 reward 대비 여전히 작은 비중이므로 학습 안정성 유지 기대
3. **에너지 페널티 0.4x 완화**: 종종걸음의 근본 원인 해결. Reference motion의 자연스러운 보폭을 따라가기 위해 큰 토크 사용을 허용
4. **3가지 동시 변경의 위험**: 각 변경의 순수 효과를 분리하기 어려운 단점. 하지만 세 문제가 상호 연관되어 있어(에너지 페널티가 knee compliance를 강화하고, knee compliance가 종종걸음을 유발) 동시 조정이 합리적

## 10. 성공 기준 및 평가 결과

| 기준 | 기대 | 실제 결과 | 달성 |
|---|---|---|---|
| Knee stance stiffness 출현 | stance 1.0x+, swing 0.7x- | 전체 ~1.0x, **phase 분화 없음** | **부분 달성** (수준만 상승) |
| Ankle swing compliance | swing 1.5x 미만 | 전체 1.65~1.74x, 상수 | 미달 |
| 종종걸음 해소 | 자연스러운 보폭 | **해소됨** (시각화 확인) | **달성** |
| 성능 유지 | 완주율 90%+ | 100% 완주 | **달성** |
| Phase-dependent 변화폭 | stance/swing 20%+ 차이 | **±2~5% (분화 없음)** | **미달** |

## 11. 리스크 (사전 예측 vs 실제)

| 사전 예측 | 실제 |
|---|---|
| Bio reward weight 너무 높으면 극단 수렴 | 발생 안함 (w=0.15도 전체 대비 미미) |
| 에너지 페널티 완화로 jittering 재발 | 발생 안함 |
| Knee+ankle stiff → rigid 로봇 보행 | 부분적 (knee ~1.0x로 과도하진 않음) |
| (사전 미예측) Phase-dependent 분화 미발생 | **핵심 문제** — reward 구조의 근본 한계 |
