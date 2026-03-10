# VIC_CCF_ON 결과 분석 (260310)

## 1. 실험 설정

VIC11 (av reward 947.11) 이후 CCF 학습을 실제로 활성화한 실험.

변경 사항 (VIC11 대비):
- vic_curriculum_stage: 1 → **2** (CCF 학습 활성화)
- reward_curriculum_switch_epoch: 7500 → **10000**
- reward_w_stage2_task: 0.4 → **0.3**, reward_w_stage2_disc: 0.6 → **0.7**
- max_epochs: 25000 → **20000**
- 실험명: VIC_CCF_ON

유지 사항:
- CCF 8그룹, vic_ccf_sigma_init: 없음 (기본 sigma=-2.9 전체 공유)
- motion_file: amass_isaac_walking_forward_single.pkl

## 2. 정량적 평가

평가 환경: num_envs=16, test 모드, 최종 체크포인트 (20000 에폭)

| 지표 | VIC_CCF_ON (20k) | VIC11 (25k) | V4 (PHC baseline) |
| :--- | :--- | :--- | :--- |
| 평가 Avg Reward | **945.30** | 947.11 | ~461 |
| 평가 Avg Steps | **297.98** | 300.1 | ~143 |

VIC11과 사실상 동일한 성능. CCF 활성화에도 불구하고 성능 하락 없음.

## 3. CCF 학습 결과 분석

최종 체크포인트(20k epoch)의 policy mu layer에서 CCF 부분(last 8 dims) 직접 추출:

| 그룹 | CCF bias | impedance_scale |
| :--- | :--- | :--- |
| G0 L_Hip | +0.0048 | 1.003x |
| G1 L_Knee | -0.0413 | 0.972x |
| G2 L_Ankle+Toe | +0.0549 | 1.039x |
| G3 R_Hip | +0.0108 | 1.007x |
| G4 R_Knee | -0.0262 | 0.982x |
| G5 R_Ankle+Toe | +0.0241 | 1.017x |
| G6 Upper-L | -0.0440 | 0.970x |
| G7 Upper-R | +0.0084 | 1.006x |

**모든 그룹에서 impedance_scale ≈ 1.0 (±4%).** 사실상 CCF가 아무것도 학습하지 않았다.

## 4. 원인 분석: 왜 CCF를 못 배웠나

핵심 원인: sigma 공유 문제.

VIC_CCF_ON은 모든 action dim(77개)에 동일한 sigma=-2.9 적용:
- exp(-2.9) ≈ 0.055 → CCF 탐색 std = 0.055
- 학습 중 CCF 샘플 범위: 95% 구간 = ±0.11 → impedance_scale = [0.93, 1.08]

이 범위에서는 임피던스 변화가 너무 작아 reward 차이가 gradient noise에 묻힌다.
PPO 입장에서 CCF≈0(neutral)이나 CCF≠0이나 reward 차이가 없으므로 mu는 0을 유지.

## 5. 한계 및 다음 실험 방향

한계:
- CCF가 실질적으로 작동하지 않았음 (Stage 2인데 Stage 1과 동일 효과)
- sigma 분리 설계 없이 CCF 탐색을 충분히 유도하기 어려움

다음 실험 (VIC_CCF_ON2):
- CCF dims에만 sigma=-1.0 (std=0.37) 별도 적용
- PD dims sigma=-2.9 유지
- 커리큘럼: epoch < 10000 → task=0.7/disc=0.3, epoch ≥ 10000 → task=0.3/disc=0.7
- max_epochs: 20000
