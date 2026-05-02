# Personalized + Diverse Motion Roadmap

**작성일**: 2026-05-02
**컨텍스트**: V2 (`HumanoidImResAMPVCmdV2`) 학습 중 server1 job 4117에서 도출된 design observation 및 향후 V3/V4 확장 방향 정리.

---

## 1. 한 줄 요약

> Captured personal motion (Vicon/H5 → SMPL pkl) 을 base로, AMP discriminator가 style을 보존하면서, multi-clip pool + frozen phc_3 PNN + retime + residual head 조합으로 다양한 locomotion command (forward, backward, sideways, turning)에 일반화되는 personalized humanoid controller 를 구축한다. 최종 목표는 exoskeleton 사용자별 RL 제어기.

---

## 2. 왜 이 architecture가 use case에 맞는가

### 2.1 AMP가 style 보존 메커니즘
- AMP discriminator는 **사용자 captured motion 만** 학습 → "이 사람이 걷고/돌고/움직이는 모양" 이 학습됨.
- Policy는 두 가지 reward 동시 최적화:
  1. **Task reward** (cmd tracking)
  2. **AMP score** (이 사람 style이냐)
- Dataset에 있는 motion → 직접 mimic.
- Dataset에 없는 motion (안 잡힌 turn radius, 안 잡힌 v 조합) → policy가 interpolate, AMP는 "다른 사람 스타일"에 penalty → **personal style 유지 + 일반화**.

### 2.2 PHC는 multi-skill 전제로 설계됨
- PNN column + frozen base는 11k+ AMASS motion 처리하도록 만들어짐.
- Walking 3 clip → (walk + turn + sideways + transition) 으로 가는 건 framework가 native로 지원.
- 이미 phc_3가 "임의의 reference pose 시퀀스를 따라가는 능력" 을 마스터한 상태이므로, 우리는 **reference 자체를 어떻게 고를지 + 어떻게 fine-tune할지** 만 학습하면 된다.

---

## 3. Architecture 전체 그림

```
┌──────────────────────────────────────────────────────────┐
│              Captured Personal Motion Pool               │
│  forward walk (slow/fast), turn-L, turn-R (다양한 radius),│
│  lateral step, backward walk, transitions ...            │
│  각 clip pre-labeled with (v_x, v_y, ω) signature        │
└────────────────────────┬─────────────────────────────────┘
                         │ user cmd → nearest clip
                         ↓
┌──────────────────────────────────────────────────────────┐
│  Selected Clip + Retime → Reference pose at time t       │
│  (clip selection = coarse speed/direction)               │
│  (retime ratio = fine speed adjustment)                  │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ├── IM reward (humanoid pose ↔ reference)
                         │
                         ↓
┌──────────────────────────────────────────────────────────┐
│           Frozen phc_3 PNN (학습 안 함)                  │
│   input:  (humanoid_state, reference_pose)               │
│   output: base_action — "reference 따라가는 torque"     │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────┐
│        Residual Head (학습됨, MLP [256, 256])            │
│   input:  (humanoid_state, command_vector)               │
│   output: residual_action — fine-tune                    │
└────────────────────────┬─────────────────────────────────┘
                         │
                         ↓
        final_action = base_action + residual_action
                         │
                         ↓ physics step
                         │
                         ├── Task reward (cmd tracking)
                         │
                         ↓
┌──────────────────────────────────────────────────────────┐
│       AMP Discriminator (학습됨, MLP [1024, 512])        │
│   input:  (state, action)                                │
│   output: "이 사람 style이냐" score                      │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 각 component 상세

| Component | 역할 | 학습 여부 | Input | Output |
|---|---|---|---|---|
| Motion clip pool | 어떻게 움직여야 하나 reference 제공 | 고정 (capture된 데이터) | — | reference pose 시퀀스 |
| Clip selector | cmd → 가장 가까운 clip | 룰 기반 또는 학습 가능 | command | clip_id |
| Retime | clip을 cmd 속도에 맞게 재생속도 조절 | 룰 기반 (ratio = v_cmd / v_natural) | command, clip_id | retimed reference |
| **Frozen phc_3 PNN** | 임의의 reference 따라가는 일반 능력 | **고정** (frozen base) | state + reference | base_action |
| Residual head | cmd 받아서 fine-tune | **학습** | state + command | residual_action |
| AMP discriminator | style 점수 매김 | **학습** | state + action | style_score |

### 4.1 Multi-clip + retime의 분업

| 기능 | 담당 |
|---|---|
| Coarse speed (어느 clip?) | **Clip selection** (예: 0.897 / 0.975 / 1.068 중 가장 가까운) |
| Fine speed (그 clip 내에서) | **Retime** (ratio 0.85 ~ 1.15 범위) |
| Direction / turn 종류 | **Clip selection only** (retime은 방향 못 바꿈) |
| Style | **AMP** (구조가 아닌 "어떻게 움직이는지") |

#### 예시: `v_cmd = 0.92 m/s` 요청
- Closest clip = 0 (v_natural=0.897, 차이 0.023) ← 선택
- Retime ratio = 0.92 / 0.897 = 1.026 → 살짝 빠르게 재생
- 다른 옵션: clip 1 (0.975, 차이 0.055), ratio = 0.944
- Clip 0 + 약간 빠른 retime이 더 자연스러움 (small ratio = inertialization 거의 불필요)

### 4.2 Retime이 잘 먹는 곳 / 안 먹는 곳

**잘 먹는 곳**
- Speed scaling (forward/backward/sideways의 m/s 조절)
- 같은 gait 패턴 내에서 cadence 조절 (사람 자연 walking은 cadence와 stride가 함께 변화하므로 retime이 맞음)

**안 먹는 곳**
- **Turn radius**: 같은 turn clip을 1.5배 빠르게 재생해도 yaw rate만 1.5배, radius는 그대로. 다른 radius 원하면 별도 clip 또는 spatial warp 필요.
- **Step length만 조절** (cadence는 유지): retime은 둘을 같이 바꾸므로 불가.
- **Curve interpolation**: 30°/s turn과 60°/s turn 사이 45°/s 원하면, 두 clip blend가 retime보다 나음.

---

## 5. Mid-episode transition: double-stance gating + inertialization

### 5.1 문제
사용자 cmd가 mid-episode에 변경되어 active_clip이 바뀌어야 할 때, 즉시 switch하면:
- Mid-stride에 있을 수 있음 → humanoid 발이 공중인데 새 reference는 mid-stance라고 함
- IM reward 폭발 → 학습 destabilize / inference 시 fall

### 5.2 Biomechanics 통찰
**Double-support phase가 transition에 가장 안전**:
- 두 발 모두 ground contact = anchor 역할
- COM이 두 발 support polygon 안 → 정적 stability 최대
- Joint velocity가 자연스럽게 작아짐 → inertialization (smooth blend) 잘 먹음
- 사람도 walking 중 방향 전환 시 무의식적으로 이 phase에 맞춤

### 5.3 구현
```python
def maybe_apply_clip_switch(env):
    # 새 cmd로 다른 clip이 desired인 상황
    if desired_clip != active_clip:
        if both_feet_in_contact(env):     # foot_force > threshold for both
            apply_switch_with_inertialization()  # 0.2 초 blend
        else:
            queue_pending_switch()         # 다음 double-stance 기다림
```

PHC에 이미 있음: `_contact_forces`, `_foot_indices`. 추가 비용 ≈ 0.

### 5.4 Edge case
- **Running / jumping** 처럼 flight phase 자주 있는 motion: takeoff 또는 landing impact를 trigger로 사용 (PHP parkour 방식)
- **Standing → walking 시작**: standing 자체가 double-support이므로 자연스러움
- **Walking → standing 정지**: 정지하려는 cmd 들어오면 다음 double-stance에서 standing clip으로 전환

---

## 6. 단계별 Roadmap

| 단계 | 내용 | Risk | Prerequisite |
|---|---|---|---|
| **V2 (현재)** | Scalar v_cmd tracking, walking 3 clip, frozen phc_3 + residual + AMP | 낮음 — 거의 완료 (server job 4117 진행 중) | — |
| **V3** | V2의 clip-master design을 v_cmd-master로 reverse. Mid-episode v_cmd schedule 도입해서 `_apply_clip_switches` 가 실제로 trigger되도록 | 중간 | V2 final ckpt 분석 |
| **V4** | Command 2D (v_x, v_y) — lateral walking clip 추가 capture, Vector cmd 입력에 맞게 residual head input 확장, reward shaping 다시 | 중간 | V3 검증 + 새 capture data |
| **V5** | ω_yaw 추가 → 3D command, turning clip 추가, **double-stance gating** + **inertialization** 추가 | 높음 — mid-episode transition이 진짜 시작되는 단계 | V4 검증 + turning capture |
| **V6** | 비-locomotion (sit-to-stand, stair stepping) 추가 → exoskeleton 적용에 필요한 다양한 motion type 지원 | 매우 높음 — 별도 skill head 필요할 수도 (PHC PNN의 새 column?) | V5 검증 + 추가 capture |

### 6.1 V3 상세 (다음 단계)

**목표**: V2의 design issue (clip-master) 수정.

V2 현 구현 (`humanoid_im_res_amp_vcmd_v2.py:_reset_envs` line 187-216):
```
1. parent (HumanoidIm)가 motion_lib.sample_motions() 로 RANDOM clip 선택
2. parent가 선택한 clip을 READ → self._active_clip
3. v_cmd를 그 clip의 v_natural ± 0.05로 OVERWRITE
   → 즉, clip이 master, v_cmd가 follower
```

**V3 fix**: v_cmd를 먼저 sampling하고, 그에 맞는 clip을 강제:
```python
# 1. v_cmd uniform sampling
new_v = torch.empty(...).uniform_(V_CMD_MIN, V_CMD_MAX)
# 2. v_cmd에 가장 가까운 clip 선택
target_clip = pick_nearest_clip(new_v)        # signature 기반
# 3. motion_lib에 강제하고 그 clip 기준으로 state init
self._sampled_motion_ids[env_ids_t] = target_clip
super()._reset_envs(env_ids)                   # forced clip으로 reset
self._active_clip[env_ids_t] = target_clip
self._v_cmd_target[env_ids_t] = new_v
```

**추가**: episode 내내 v_cmd schedule (linear ramp 또는 random walk) 도입 → `_apply_clip_switches` 가 실제로 trigger됨.

### 6.2 V4 상세 (vector command)

**Command space**: scalar v → (v_x, v_y) ∈ ℝ²
- Forward + lateral 동시 가능
- Sideways walking clip 새로 capture 필요

**Reward shaping**: VIC4_VCMD failure mode (S10-S19B) 가 high-dim에서 재발 가능 → curriculum 필요
- Stage 1: pure forward (v_y = 0) 만 학습
- Stage 2: 작은 v_y 추가
- Stage 3: full 2D cmd
- 각 stage gate에서 eps_len cap 도달 + AMP score threshold

**Clip selector**: 2D space에서 nearest neighbor (Euclidean 또는 weighted L2 — v_x와 v_y의 sensitivity 다를 수 있음)

### 6.3 V5 상세 (turning + double-stance gating)

**Turning clip** capture 필요
- Multiple radius (0.5m, 1.0m, 2.0m, ∞=straight)
- Multiple speed at each radius
- Each labeled with (v_x, v_y, ω_yaw) signature

**Double-stance gating**: §5.3 구현 추가

**Inertialization**: clip transition 시 0.2초 동안 reference pose를 cubic blend
- 첫 frame: old reference
- 중간: blend(old, new, t)
- 마지막: new reference
- 이 동안 IM reward weight 일시 감소 (학습 안정화)

**Risk**: turning은 lateral momentum + balance가 더 sensitive → bunny hop 같은 gait artifact 재발 가능. AMP가 이걸 잡아주길 기대해야 함.

### 6.4 V6 상세 (비-locomotion)

**필요한 motion**
- Sit-to-stand (chair-aware?)
- Stair stepping (up/down)
- Step over obstacle
- Pivot turn (in-place rotation)

**Architecture 변화**
- PHC PNN의 4번째 column 추가? 또는 별도 skill head?
- Locomotion vs. non-locomotion mode switch
- High-level planner 필요 (어느 skill을 언제 호출할지)

**Exoskeleton 연결**
- 사용자 의도 인식 (EMG, force sensor, button) → high-level cmd
- 그 cmd → 해당 skill head 호출
- Skill head가 captured personal motion + AMP로 personalized output

---

## 7. V2 현 설계 한계 (오늘 발견)

### 7.1 Clip-master 문제 (§6.1에서 V3로 수정 예정)
- Training 시 v_cmd가 episode 내내 거의 constant (v_natural ± 0.05)
- → policy가 v_cmd 변화에 응답하는 법을 charging 못함
- → mid-episode `_apply_clip_switches` 가 거의 dead code

### 7.2 Inference 시 motion_lib subsetting 버그
- `num_envs = 1` 로 inference 시, motion_lib는 num_envs 만큼만 motion load
- 즉, 3 clip 중 1개만 load됨 (보통 clip 0)
- 그래서 panel demo에서 clip 0만 사용된 것
- **Workaround**: `num_envs ≥ 3` 으로 inference (이미 sweep_v2_per_clip.py 에 적용)
- **Permanent fix**: `motion_lib.set_minimum_motions(N)` API 추가 또는 inference 시 전체 motion 강제 load

### 7.3 Per-clip bias (Ep 1250 sweep 결과)

| Clip | v_natural | v_actual offset |
|---|---|---|
| 0 | 0.897 | **+0.25 m/s** (over) |
| 1 | 0.975 | **+0.07 m/s** (best) |
| 2 | 1.068 | **−0.10 m/s** (under) |

→ Slope ≈ 1.0 (각 clip 모두 v_cmd에 응답), 하지만 각 clip마다 고유 bias.
→ 학습 진행에 따라 offset 줄어들길 기대 (Ep 5000, 10000 ckpt 비교 필요).

### 7.4 단일 episode = 단일 clip
- §7.1의 결과로, V2는 사실상 "v_natural 부근 v_cmd tracking" 만 학습
- 진정한 multi-clip exploitation은 V3에서야 시작

---

## 8. H5 conversion work와의 connection

지금 진행 중인 H5 → SMPL conversion (CoP-based pelvis IP, foot-lock, …) 이 **이 모든 것의 upstream**:
- Captured motion 품질 ↑ → reference 깨끗 → AMP discriminator가 미세한 style cue 잡음 → personalization 잘 됨
- Personal exoskeleton 사용자 motion capture 시 동일 pipeline 사용
- → **H5 conversion은 critical path 위. Side quest 아님.**

특히 V6 (비-locomotion) 단계에서:
- Sit-to-stand, stair 같은 motion은 vertical 운동이 큼 → pelvis IP가 부정확하면 reference 폭발
- 사용자별 anatomy (height, leg length, mass distribution) 반영도 필요 → H5 SMPL conversion이 정확해야 가능

---

## 9. Risk 분석

| Risk | 단계 | 완화 방법 |
|---|---|---|
| Mid-episode transition fail | V5+ | Double-stance gating + inertialization |
| Command coverage 부족 | V4+ | 충분한 capture 시간 (10-20분), synthesis augmentation |
| Personalization vs. generalization 트레이드오프 | V4+ | AMP weight 조정, dataset 크기 ablation |
| State-reference mismatch at reset | 전 단계 | V3에서 forced clip 기준 state init |
| High-dim cmd Goodharting (VIC4_VCMD 재발) | V4+ | Curriculum, eps_len cap 검증, 양적 video 검증 |
| Bunny hop 같은 gait artifact | V5+ (특히 turning) | AMP discriminator 강화, contact reward 추가 |
| Compute (3+ clip 학습은 envs 더 필요) | V4+ | Server num_envs ↑, motion_lib subset 버그 fix |

---

## 10. 다음 actionable step

1. **V2 server job 4117 끝까지 monitor** (~Ep 20000, ~22h 남음)
2. **Ep 5000 / Ep 10000 / Ep 20000 ckpt에서 per-clip sweep 반복** → bias가 줄어드는지 확인
   - 줄어들면: V2 design은 narrow window지만 학습 진행 시 generalization 가능 → V3는 optimization
   - 안 줄어들면: V2 fundamental limit 확정 → V3 우선
3. **V3 design doc 작성** (V2 fix + mid-episode v_cmd schedule)
4. **V4 capture plan 수립** (lateral walking, 어떤 cmd 조합 필요한지 specification)
5. **`_apply_clip_switches` CUDA assert 버그 수정** (motion_lib subsetting 관련)

---

## 11. 사용자 직관 — design 핵심

오늘 대화에서 사용자님이 instinctively 잡으신 세 가지가 정확히 이 architecture의 핵심 piece:

1. **Multi-clip이 v_cmd 기반으로 선택되어야 함** ← V3가 이걸 fix
2. **Clip 변경은 double-stance에서** ← V5에 추가
3. **Captured motion + frozen network + retime 모두 사용** ← 이미 V2에서 구현 중 (frozen + retime), V3+에서 multi-clip 활용 추가

이 직관들이 V2의 현 design 한계를 정확히 가리키고, 동시에 V3-V6 roadmap의 방향을 정의함.
