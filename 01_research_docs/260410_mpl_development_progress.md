# 260410 Motion Plan Layer 개발 진행 현황

## 개요

PHC+VIC 구조를 Motion Plan Layer(MPL) 2-모듈 아키텍처로 업그레이드하는 작업.
실측 H5 데이터(Vicon Plug-in Gait)를 motion library로 변환하여 사용.

---

## 데이터 현황

소스 파일: data/combined_data_from_csv.h5 (1.66 GB)

4명 피험자(S001~S004), 7개 task, 100 Hz, 총 90 trials (accel_sine 제외)

mocap/angle 데이터 보유 현황 (joint angle이 있어야 SMPL 변환 가능):
- S001: 13 trials (incline, level 3속도, stopandgo, lv0/lv4/lv7)
- S002~S004: mocap/angle 없음 (forceplate/robot 데이터만 존재)

현재 사용 가능한 데이터: S001 lv0만 (8 trials → 330 clips, 5초 단위)

---

## Phase 1: H5 데이터 수집 + Motion Library 구축

### Phase 1.1 H5 Explorer [완료]
- 파일: scripts/data/h5_explorer.py
- 기능: H5 구조 탐색, 데이터 품질 검증(NaN/outlier), 관절 각도/GRF/CoP/CoM 시각화
- 결과:
  - 관절 각도: degree 단위 확인 (Hip flexion -12~30도, Knee 0~67도)
  - GRF vertical: ~360N mean (체중 절반, 양발 교대)
  - CoM z: ~1030mm
  - NaN/Inf: 없음
  - 플롯 저장: output/h5_explorer/

### Phase 1.2 H5 → Motion Library 변환 [완료]
- 파일: scripts/data/h5_to_motion_lib.py
- 변환 파이프라인:
  1. Vicon PiG Euler XYZ (degree) → quaternion (wxyz) 변환
  2. Biomechanical 12-joint → SMPL 24-joint 매핑
  3. Treadmill speed 적분 → synthetic root translation
  4. 100Hz → 30fps 리샘플링
  5. 5초 단위 clip 분할 (1초 overlap)
  6. Command label 자동 산출 (v_cmd, incline)
- 출력:
  - sample_data/h5_motion_library.pkl (330 clips)
  - sample_data/h5_motion_library.meta.pkl (metadata)
- PKL 형식: AMASS 원본과 동일 (pose_quat_global, pose_quat, trans_orig, pose_aa, beta, gender, fps)

Joint 매핑 테이블:

    Vicon PiG          →  SMPL Joint (Index)
    pelvis (L xyz)     →  Pelvis (0)
    hip (L/R xyz)      →  L_Hip(1) / R_Hip(2)
    knee (L/R xyz)     →  L_Knee(4) / R_Knee(5)
    ankle (L/R xyz)    →  L_Ankle(7) / R_Ankle(8)
    spine (L xyz)      →  Torso(3)
    thorax (L xyz)     →  Spine(6)
    neck (L xyz)       →  Neck(12)
    head (L xyz)       →  Head(15)
    shoulder (L/R xyz) →  L_Shoulder(16) / R_Shoulder(17)
    elbow (L/R xyz)    →  L_Elbow(18) / R_Elbow(19)
    wrist (L/R xyz)    →  L_Wrist(20) / R_Wrist(21)
    (미매핑)           →  L_Toe(10), R_Toe(11), Chest(9), L/R_Thorax(13/14), L/R_Hand(22/23) → identity

### Phase 1.3 MPL Config + Training 검증 [완료]
- phc/data/cfg/env/env_im_walk_mpl.yaml: MPL 전용 환경 설정
  - motion_file: sample_data/h5_motion_library.pkl
  - motion_lib_type: h5
  - numAMPObsSteps: 5 (메모리 절약, RTX 4060 Ti 8GB)
  - num_envs: 64
- phc/data/cfg/learning/im_walk_mpl.yaml: MPL 전용 학습 설정
  - amp_obs_demo_buffer_size: 50000 (메모리 절약)
  - config name: MPL_Walk
- 기존 VIC 환경(HumanoidImVIC)에서 H5 motion library 직접 로드 확인
- h5_to_motion_lib.py: CoM NaN 보간 수정 (stopandgo trial에서 발생)

### Phase 1 검증 [완료]
- 변환된 PKL(330 clips, 76800 frames)을 Isaac Gym에 로드: 성공
- 30 epoch 학습 완료: reward 4.6~4.7 (초기 학습, 정상)
- 체크포인트 저장: output/MPL_Walk.pth (76MB)
- Isaac Gym 시각화: reference pose 추종 동작 확인
- 관찰: eps_len=2.0 (초기 단계, 빠른 낙상) → 장시간 학습 필요

---

## Phase 2: Stability Margin (Alpha) 추가 [미착수]

- phc/utils/stability_utils.py: CoP, XCoM, support polygon, alpha 계산
- scripts/data/compute_h5_alpha.py: H5 실측 데이터에서 alpha ground truth 산출
- humanoid_im_vic.py에 alpha 통합 (obs + reward blending)
- Perturbation curriculum 추가

---

## Phase 3: 2-모듈 분리 (G + R) [미착수]

- phc/env/tasks/humanoid_im_mpl.py: HumanoidImMPL 환경
- phc/learning/amp_network_mpl_builder.py: R policy 네트워크
- 학습 절차: Stage A (G supervised) → Stage B (R RL) → Stage C (joint fine-tune)
- Alpha 기반 연속 task-following ↔ recovery 블렌딩

---

## Phase 4: Command-Conditioned Retrieval + 보간 [미착수]

- phc/utils/motion_retrieval.py: KD-tree 기반 command 검색 + softmax 보간
- CoM velocity trajectory following

---

## Phase 5: 개인 스타일 학습 [미착수]

- phc/utils/style_discriminator.py: subject별 AMP discriminator
- Subject-specific library filtering + style reward
- 현재 S001만 angle 데이터 보유 → S002~S004 데이터 추가 필요

---

## Phase 6: Multi-Skill 확장 [미착수]

- level walking + incline + stopandgo 통합
- Command 확장: (v_cmd, incline_cmd, accel_cmd)

---

## 알려진 이슈

1. S002~S004에 mocap/angle 데이터 없음 → 개인화(Phase 5) 진행 시 데이터 추가 필요
2. decline_5deg, accel_sine에도 angle 데이터 없음
3. Vicon PiG → SMPL axis convention 매핑이 정확한지 시각적 검증 필요 (Phase 1 검증)
4. pose_quat_global을 local rotation으로 제공 중 (v1 근사) → 추후 proper FK 필요
5. Treadmill 위 실험이라 실제 forward displacement가 0 → belt speed 적분으로 합성

---

## 파일 구조

    scripts/data/
        h5_explorer.py          [완료] H5 탐색/시각화
        h5_to_motion_lib.py     [완료] H5 → PKL 변환
        compute_h5_alpha.py     [예정] Alpha ground truth
    phc/utils/
        motion_lib_h5.py        [예정] H5 motion library loader
        stability_utils.py      [예정] CoP/XCoM/alpha 계산
        motion_retrieval.py     [예정] Command 기반 clip 검색
        style_discriminator.py  [예정] 개인 스타일 disc
    phc/env/tasks/
        humanoid_im_mpl.py      [예정] MPL 환경
    phc/learning/
        amp_network_mpl_builder.py [예정] R policy 네트워크
    phc/data/cfg/
        env/env_im_walk_mpl.yaml   [예정] MPL env config
        learning/im_walk_mpl.yaml  [예정] MPL learning config
    sample_data/
        h5_motion_library.pkl      [완료] 변환된 motion library (330 clips)
        h5_motion_library.meta.pkl [완료] Clip metadata
    output/
        h5_explorer/               [완료] 시각화 플롯
