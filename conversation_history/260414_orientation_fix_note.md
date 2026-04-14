# H5 Root Orientation 버그 수정 기록

## 문제 요약
v5~v12까지 모든 학습에서 ep_len ~8-10 (즉시 넘어짐), success rate 0%.
근본 원인: **h5_to_motion_lib.py의 pelvis tilt 부호가 반전**되어 reference pose가 뒤로 기울어져 있었음.

## 발견 경위
1. Isaac Gym에서 v12 policy 시각화 → reference posture 자체가 뒤로 기울어져 보임
2. MuJoCo에서 H5 motion library 직접 재생 → 확인
3. h5_to_motion_lib.py line 311 분석 → 부호 반전 발견

## 기술적 상세

### Vicon PiG Pelvis Tilt 컨벤션
- **Positive tilt = anterior tilt** (상체가 앞으로 기울어짐)
- 정상 보행 시 평균 +4.7° (범위: -5.6° ~ +16.8°)
- 이는 자연스러운 전방 기울어짐으로, CoM이 지지면 위에 있게 해줌

### Z-up World Frame에서 Ry 회전
- Forward = +X, Right = +Y, Up = +Z
- **+Ry** (오른손 법칙, Y축 기준): +Z를 +X 방향으로 회전 → 상체 앞으로 기울어짐
- **-Ry**: 상체 뒤로 기울어짐

### 버그 코드 (이전)
```python
# h5_to_motion_lib.py line 311
R_pelvis_world = sRot.from_euler('XYZ', [
    np.deg2rad(pelvis_obliq[t]),     # Rx: lateral tilt
    np.deg2rad(-pelvis_tilt[t]),     # Ry: ← 부호 반전! 뒤로 기울어짐
    np.deg2rad(pelvis_rot[t]),       # Rz: yaw rotation
])
```

### 수정 코드
```python
R_pelvis_world = sRot.from_euler('XYZ', [
    np.deg2rad(pelvis_obliq[t]),     # Rx: lateral tilt
    np.deg2rad(pelvis_tilt[t]),      # Ry: anterior tilt(+) → +Ry → lean forward
    np.deg2rad(pelvis_rot[t]),       # Rz: yaw rotation
])
```

### 왜 치명적이었는가
- 평균 4.7°의 anterior tilt가 -4.7° posterior tilt로 변환됨
- 보행 중 CoM이 지지면 뒤쪽에 위치하게 됨
- 물리 시뮬레이션에서 0.3초 내 후방 전도 → ep_len ~8 steps
- Reward imitation이 "뒤로 기울어진 자세"를 목표로 하므로, policy가 학습해도 물리적으로 불가능

## 추가 수정: 워밍업 구간 트림
- 트레드밀 벨트 시작: ~6초, 보행 시작: ~5.5-6초 (모든 trial)
- 첫 5초 clip (c000)의 관절각: hip 1.1°, knee -2.7° (정지 상태)
- `--trim_start 10.0` 옵션 추가: 처음 10초 제거
- 10초 시점이면 이미 2-3 gait cycle 완료 후, 안정적 보행 상태

## 좌표계 변환 히스토리 (h5_to_motion_lib.py)

| 세션 | 수정 내용 | 영향 |
|------|----------|------|
| 260413 11:00 | Root에 upright rotation 추가 | 직립 가능 |
| 260413 11:00 | Global rotation: right-multiply by upright_inv | FK 정상화 |
| 260413 11:00 | Translation Y-up → Z-up | 높이 정상 |
| 260413 11:00 | Joint angle sign 전면 수정 | 팔/다리 방향 정상 |
| 260413 19:00 | Per-frame pelvis tilt/obliquity/rotation 반영 | Root rotation 동적 |
| **260414 01:00** | **Pelvis tilt 부호 수정 (+tilt)** | **뒤로 기울어짐 → 앞으로** |
| 260414 01:00 | Warmup trim (--trim_start) | 정지 데이터 제거 |

## 검증
- MuJoCo viewer로 수정 전/후 비교: 뒤로 기울어짐 → 자연스러운 전방 기울어짐 확인
- 첫 clip DOF 값: hip -22°~+5°, knee 55°, ankle ±12° (정상 보행 범위)

---

## AMASS 비교 실험 계획 (v13 실패 시)

### 목적
v13 (수정된 H5 데이터)에서도 학습이 안 되면, 문제가 **H5 데이터 변환 자체**인지 **config 크기**인지 구분 필요.

### 실험 설계
- **동일 config** (128 envs, [512,256] MLP, 10k AMP buffer, VIC stage 2)
- **Motion file만 변경**: `sample_data/amass_isaac_walking_primitive.pkl` (50 clips, walking)
- **1000 epoch 학습** 후 비교

### 해석 기준
| v13 (H5) | AMASS | 결론 |
|-----------|-------|------|
| ep_len flat ~8 | ep_len 상승 | **H5 데이터 변환에 아직 문제 있음** |
| ep_len flat ~8 | ep_len flat ~8 | **Config가 너무 작음** (8GB VRAM 한계) |
| ep_len 상승 | - | **수정 성공**, AMASS 비교 불필요 |

### AMASS 실험 실행 방법
```bash
# learning config에서 name을 AMASS_Compare_v1으로 변경
# env config에서 motion_file을 amass_isaac_walking_primitive.pkl로 변경
cd /home/exolab/Documents/GitHub/PHC
WANDB_MODE=disabled conda run -n phc python phc/run_hydra.py \
  learning=im_walk_mpl \
  env=env_im_walk_mpl \
  env.motion_file=sample_data/amass_isaac_walking_primitive.pkl \
  env.num_envs=64 \
  headless=True
```

### 주의사항
- VIC 설정은 AMASS에서도 동일하게 유지 (공정 비교)
- AMASS는 이미 검증된 데이터이므로, 이걸로 학습이 되면 데이터 문제 확정
- Config가 문제면 gradient accumulation 또는 더 큰 GPU 필요
