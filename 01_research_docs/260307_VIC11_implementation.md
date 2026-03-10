# VIC11 구현 문서 (260307)

## 1. 배경 및 목적

VIC10C에서 에너지 페널티 완화 방향이 유효함을 확인했다 (av reward 377.98, av steps 117.8).
VIC11은 두 가지 구조적 개선을 동시에 적용하여 V4 (av reward ~461) 수준 달성을 시도한다.

- CCF 그루핑: action space 축소로 Stage 2 학습을 더 tractable하게
- Reward Weight 커리큘럼: 초반 보행 학습 → 후반 자연스러운 동작 스타일 유도

## 2. VIC11 변경 사항 (VIC10C 대비)

### 2-1. CCF 그루핑 (8 groups)

기존 VIC10C는 CCF가 per-DOF였다 (69 dims). Stage 1에서는 CCF=0으로 고정하므로 차이 없으나,
Stage 2 전환 시 학습해야 할 CCF 차원이 너무 많아 학습이 어렵다.

VIC11에서는 해부학적으로 의미 있는 8개 그룹으로 묶는다:

| 그룹 인덱스 | 그룹 이름 | DOF 범위 | 비고 |
| :--- | :--- | :--- | :--- |
| G0 | L_Hip | 0-2 | 왼쪽 고관절 |
| G1 | L_Knee | 3-5 | 왼쪽 무릎 |
| G2 | L_Ankle + L_Toe | 6-11 | 왼쪽 발목/발가락 |
| G3 | R_Hip | 12-14 | 오른쪽 고관절 |
| G4 | R_Knee | 15-17 | 오른쪽 무릎 |
| G5 | R_Ankle + R_Toe | 18-23 | 오른쪽 발목/발가락 |
| G6 | Upper-Left | 24-53 | 척추+목+머리+왼팔 |
| G7 | Upper-Right | 54-68 | 오른팔 |

결과적으로 action size: 69 (PD target) + 8 (CCF) = 77 dim (기존 138 → 44% 축소).

### 2-2. Reward Weight 커리큘럼

VIC10C에서 상체가 어색하고 자연스럽지 않은 동작이 관찰됐다.
원인: task_reward_w = disc_reward_w = 0.5 동일 가중치로 disc (스타일) 학습이 불충분.

커리큘럼 전략:
- epoch < 7500: task_reward_w=0.7, disc_reward_w=0.3 (보행 안정화 우선)
- epoch >= 7500: task_reward_w=0.4, disc_reward_w=0.6 (스타일 자연스러움 강화)

후반 disc 가중치 0.6은 VIC10C의 0.5보다 높다. disc를 더 강하게 주어 reference motion에
가까운 자연스러운 상체 동작을 유도하는 것이 목적이다.

## 3. 수정된 파일 목록

1. phc/env/tasks/humanoid_im_vic.py
   - __init__: _vic_ccf_num_groups, reward curriculum 파라미터 추가 (super() 호출 전)
   - super().__init__() 후: _build_ccf_group_dof_map() 호출 추가
   - get_action_size(): ccf_dims = num_groups if grouping else per-DOF
   - _build_ccf_group_dof_map(): 8그룹 정의 메서드
   - _compute_torques(): grouped CCF → full 69 DOF 확장 로직

2. phc/learning/amp_agent.py
   - pre_epoch(): VIC reward curriculum 로직 추가 (# VIC 주석)
   - epoch < switch_epoch → stage1 weights, epoch >= switch_epoch → stage2 weights

3. phc/data/cfg/env/env_im_walk_vic.yaml
   - vic_ccf_num_groups: 8
   - reward_curriculum_switch_epoch: 7500
   - reward_w_stage1_task: 0.7, reward_w_stage1_disc: 0.3
   - reward_w_stage2_task: 0.4, reward_w_stage2_disc: 0.6

4. phc/data/cfg/learning/im_walk_vic.yaml
   - name: VIC11

## 4. 유지 사항 (VIC10C와 동일)

- motion_file: amass_isaac_walking_forward_single.pkl (단일 모션)
- power_coefficient: 0.0000005
- reward_specs: w_pos=0.4, w_rot=0.3, w_vel=0.2, w_ang_vel=0.1
- vic_curriculum_stage: 1 (Stage 1: CCF=0 고정)
- CCF range: [-1.0, 1.0] (Stage 2 전환 후 사용)

## 5. 기대 효과

CCF 그루핑: Stage 1에서는 action size 축소 외 효과 없음. Stage 2 전환 시 학습 tractability 향상.
Reward 커리큘럼: 초반 빠른 보행 학습 + 후반 자연스러운 상체 동작 유도.
두 변화가 독립적이므로 VIC10C 대비 성능 저하 없이 상체 자연스러움이 개선될 것으로 기대.
