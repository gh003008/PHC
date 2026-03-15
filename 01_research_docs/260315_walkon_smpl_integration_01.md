# WalkON Suit + SMPL 인간 모델 통합 계획 (260315)

---

## 1. 목적

PHC의 SMPL 휴머노이드와 WalkON Suit F1 하지 외골격 로봇을 IsaacGym 환경에서 통합하여, VSD(Virtual Spring-Damper) 기반 인간-엑소스켈레톤 상호작용을 구현한다.

---

## 2. 두 모델 개요

### 2-1. SMPL 휴머노이드 (PHC)
- 전신 24관절, 69 DOF
- 총 질량: ~71.8 kg
- 좌표계: Z-up, X-forward
- 포맷: MuJoCo XML (IsaacGym 호환)
- 메쉬: STL (인체 형태) + 충돌 캡슐/박스

### 2-2. WalkON Suit F1
- 하지 전용 외골격, 12 DOF (다리당 6: Hip ABD/ROT/EXT, Knee EXT, Ankle INV/PLA)
- 총 질량: 66.63 kg (+ 페이로드 0~60 kg)
- 좌표계: Z-up (Isaac Gym 로딩 시)
- 포맷: URDF
- 메쉬: SolidWorks CAD STL (고해상도)
- 기존 통합 모델 존재: `walkonsuit_withHuman.urdf` (myobody 인간 모델과 fixed joint 결합)

---

## 3. 치수 비교

### 3-1. SMPL 관절 위치 (Pelvis 기준, Z-up)

| 관절 | X (m) | Y (m) | Z (m) |
|---|---|---|---|
| Pelvis | -0.002 | -0.223 | +0.028 |
| L_Hip | -0.009 | -0.154 | -0.063 |
| R_Hip | -0.006 | -0.291 | -0.062 |
| L_Knee | -0.013 | -0.120 | -0.438 |
| R_Knee | -0.015 | -0.329 | -0.445 |
| L_Ankle | -0.057 | -0.133 | -0.836 |
| R_Ankle | -0.057 | -0.314 | -0.843 |
| Torso | -0.029 | -0.226 | +0.137 |
| Chest | -0.002 | -0.219 | +0.325 |

### 3-2. WalkON 관절 오프셋 (URDF 체인, 우측 다리)

| 관절 | 부모→자식 오프셋 (m) | 의미 |
|---|---|---|
| RH_ABD | [0, -0.115, 0] | BASE→R_BASE (측면) |
| RH_ROT | [-0.246, -0.155, 0.002] | R_BASE→R_HIP |
| RH_EXT | [0.193, 0.032, 0] | R_HIP→R_THIGH |
| RK_EXT | [0.025, **-0.431**, -0.009] | R_THIGH→R_SHANK (대퇴 길이) |
| RA_INV | [0.140, **-0.303**, 0.122] | R_SHANK→R_ANK (하퇴 길이) |
| RA_PLA | [-0.005, 0, **-0.148**] | R_ANK→R_FOOT (발목→지면) |

### 3-3. 핵심 치수 비교

| 세그먼트 | SMPL (m) | WalkON (m) | 비율 | 비고 |
|---|---|---|---|---|
| **대퇴 (hip→knee)** | 0.379 | 0.431 | 1.14 | WalkON 14% 길다 |
| **하퇴 (knee→ankle)** | 0.398 | 0.303 | 0.76 | WalkON 24% 짧다 |
| **발목→지면** | 0.056 | 0.148 | 2.64 | WalkON 발이 훨씬 높다 |
| **고관절 폭 (좌우)** | 0.137 | 0.230 | 1.68 | WalkON 68% 넓다 |
| **고관절→발목 총합** | 0.777 | 0.734 | 0.94 | **전체 다리 길이는 ~6% 차이** |
| **총 서있는 높이** | ~0.87* | ~0.88 | ~1.01 | **거의 동일** |

(*) SMPL: hip→ankle 0.777 + pelvis offset 0.091 ≈ 0.87m

핵심 발견: **전체 다리 길이는 거의 동일하지만, 세그먼트 비율이 다르다.** WalkON은 대퇴가 길고 하퇴가 짧으며, 발이 높다.

---

## 4. 기존 통합 모델 분석

`walkonsuit_withHuman.urdf`에서의 인간-엑소 결합 방식:

```xml
<joint name="human_to_exo_anchor" type="fixed">
    <parent link="LINK_BASE"/>      <!-- 엑소 베이스 -->
    <child link="sacrum"/>           <!-- 인간 골반 -->
    <origin xyz="-0.15 0.0 -0.15" rpy="1.57 0 0"/>
</joint>
```

- 결합 타입: **fixed joint** (강체 결합)
- 결합 위치: 엑소 LINK_BASE ↔ 인간 sacrum
- 오프셋: X=-0.15m (후방), Z=-0.15m (하방), Roll=90° (좌표계 정렬)
- 한계: closed kinematic chain 없이 단순 강체 결합 → 인간-엑소 간 상대 운동 불가

---

## 5. 통합 구현 방안

### 5-1. 전체 구조

```
[IsaacGym 환경]
├─ Actor 1: SMPL 인간 모델 (MuJoCo XML → IsaacGym asset)
│   └─ 69 DOF, VIC 제어 (kp * 2^ccf * pos_error)
│
├─ Actor 2: WalkON 엑소 모델 (URDF → IsaacGym asset)
│   └─ 12 DOF, RL 토크 제어
│
└─ VSD Coupling (코드 레벨 매 스텝 계산)
    ├─ 골반 결합: SMPL Torso ↔ WalkON LINK_BASE
    ├─ 정강이 결합: SMPL L/R_Knee child ↔ WalkON LINK_L/R_SHANK
    └─ 발 결합: SMPL L/R_Ankle ↔ WalkON LINK_L/R_FOOT
```

### 5-2. 스폰 방식

1. 두 모델을 같은 env에 별도 actor로 로딩
2. 초기 자세: 둘 다 직립 (SMPL은 T-pose, WalkON은 default)
3. 초기 위치: 관절이 정렬되도록 배치
   - WalkON LINK_BASE 높이 = SMPL Pelvis 높이에 맞춤
   - 좌우 중심 정렬

### 5-3. VSD 결합 지점 (3곳)

#### (1) 골반 결합: SMPL 상체 ↔ WalkON BASE
- SMPL 쪽: Torso 또는 Spine body (Pelvis 바로 위)
- WalkON 쪽: LINK_BASE
- 타입: 6DOF VSD (병진 3 + 회전 3)
- 이유: 상체 무게를 엑소 프레임이 지지. 가장 강한 결합.

```python
# 병진 VSD
F_trans = K_trans * (pos_human_torso - pos_exo_base) + D_trans * (vel_diff)
# 회전 VSD
T_rot = K_rot * orientation_error + D_rot * (angvel_diff)
```

예상 파라미터: K_trans=5000 N/m, D_trans=500 N·s/m, K_rot=500 N·m/rad, D_rot=50 N·m·s/rad (높은 강성, 거의 rigid)

#### (2) 정강이 결합: SMPL Knee child ↔ WalkON SHANK
- SMPL 쪽: L_Knee / R_Knee body의 하단 (대략 shank 중간)
- WalkON 쪽: LINK_L_SHANK / LINK_R_SHANK
- 타입: 병진 3DOF VSD (회전은 관절이 처리)
- 이유: 무릎 아래 하퇴 부분에서 피부-엑소 접촉 모사

예상 파라미터: K=2000 N/m, D=200 N·s/m (중간 강성, 약간의 상대 미끄러짐 허용)

#### (3) 발 결합: SMPL Ankle ↔ WalkON FOOT
- SMPL 쪽: L_Ankle / R_Ankle body
- WalkON 쪽: LINK_L_FOOT / LINK_R_FOOT
- 타입: 6DOF VSD (발판에 발이 고정)
- 이유: 발이 엑소 풋플레이트에 단단히 고정

예상 파라미터: K_trans=8000 N/m, D_trans=800 N·s/m (높은 강성, 발판 고정)

### 5-4. VSD 구현 방법 (IsaacGym)

```python
# 매 시뮬레이션 스텝에서:
# 1. 두 actor의 rigid body state 읽기
human_body_states = gym.get_actor_rigid_body_states(env, human_actor, gymapi.STATE_ALL)
exo_body_states = gym.get_actor_rigid_body_states(env, exo_actor, gymapi.STATE_ALL)

# 2. 결합 지점별 위치/속도 차이 계산
delta_pos = human_pos[attach_idx] - exo_pos[attach_idx]
delta_vel = human_vel[attach_idx] - exo_vel[attach_idx]

# 3. VSD 힘 계산
F_vsd = K * delta_pos + D * delta_vel

# 4. 양쪽에 반대 방향으로 적용
gym.apply_rigid_body_force_tensors(sim,
    human_forces,   # +F_vsd (인간 쪽으로 당김)
    exo_forces,     # -F_vsd (엑소 쪽으로 당김)
    gymapi.ENV_SPACE)
```

핵심: `apply_rigid_body_force_tensors()`는 GPU 텐서로 작동하므로 다수 환경 병렬 처리 가능.

---

## 6. 관절 정렬 분석

### 6-1. 관절 대응 관계

| SMPL (3DOF per joint) | WalkON | 대응 품질 |
|---|---|---|
| L_Hip_x/y/z | LH_ABD, LH_ROT, LH_EXT | 우수 (3:3) |
| R_Hip_x/y/z | RH_ABD, RH_ROT, RH_EXT | 우수 (3:3) |
| L_Knee_x/y/z | LK_EXT | 부분적 (3:1, WalkON은 굴신만) |
| R_Knee_x/y/z | RK_EXT | 부분적 (3:1) |
| L_Ankle_x/y/z | LA_INV, LA_PLA | 양호 (3:2) |
| R_Ankle_x/y/z | RA_INV, RA_PLA | 양호 (3:2) |
| L/R_Toe | 없음 | WalkON에 toe joint 없음 |
| 상체 전체 | 없음 | WalkON은 하지 전용 |

### 6-2. 세그먼트 길이 불일치 해결

전체 다리 길이는 ~6%만 차이나서 큰 문제가 아니지만, 세그먼트 비율이 다르다:
- WalkON 대퇴가 14% 길고 하퇴가 24% 짧다.

해결 옵션:

**(A) SMPL beta 조정 (권장)**
SMPL beta 파라미터로 다리 비율을 WalkON에 맞추는 방법. beta[0]~beta[9]의 조합으로 대퇴를 늘리고 하퇴를 줄이는 효과를 찾아야 한다. 장점: 물리적으로 일관된 인체 모델 유지. 단점: 원하는 비율을 정확히 맞추기 어려울 수 있음.

```python
# beta 탐색 예시
for b0 in np.linspace(-2, 2, 10):
    for b2 in np.linspace(-2, 2, 10):
        betas = torch.zeros(1, 10)
        betas[0, 0] = b0  # 전체 키
        betas[0, 2] = b2  # 다리 비율 관련
        robot.load_from_skeleton(betas=betas, gender=[0])
        # 생성된 XML에서 대퇴/하퇴 길이 측정
```

**(B) VSD 유연성으로 흡수**
세그먼트 길이 차이를 VSD의 compliance로 자연스럽게 흡수. 정강이·발 결합의 스프링 강성을 적절히 조절하면 ~2~3cm 차이는 VSD가 처리 가능. 장점: 인간 모델 변경 불필요. 단점: 큰 불일치 시 비현실적 힘 발생.

**(C) 하이브리드 (A+B)**
beta로 대략 맞추고, 남는 차이는 VSD compliance로 처리. 가장 현실적.

### 6-3. 고관절 폭 불일치

SMPL 고관절 폭(0.137m) vs WalkON(0.230m): 68% 차이.
- WalkON은 기계적 구조물(액추에이터, 프레임)이 측면으로 나와있어서 넓다.
- 실제 착용 시에도 엑소 hip joint는 인간 hip보다 바깥에 위치한다.
- 이것은 VSD 병진 결합으로 자연스럽게 처리 가능 (엑소가 바깥에서 감싸는 형태).

---

## 7. 구현 순서 (제안)

### Phase 1: 정적 시각화 (코딩 검증)
1. 두 모델을 같은 env에 로딩 (둘 다 fix_base_link=True)
2. 위치 정렬 확인 (관절 높이 매칭)
3. SMPL beta 탐색으로 세그먼트 비율 근사

### Phase 2: VSD 기본 구현
4. 골반 VSD 결합 (가장 중요, 무게 지지)
5. 정강이 + 발 VSD 결합
6. fix_base_link=False로 전환, 중력 하 안정성 확인

### Phase 3: RL 통합
7. 인간 쪽: VIC 기반 보행 (기존 학습된 policy 활용 가능)
8. 엑소 쪽: 토크 제어 policy 학습
9. 보상 함수 설계 (보행 추종 + 에너지 효율 + 결합력 최소화)

---

## 8. 기술적 제약 및 리스크

### 8-1. IsaacGym (PhysX 4) 한계
- closed kinematic chain 미지원 → VSD로 우회 (본 계획의 핵심)
- 같은 env에 서로 다른 포맷(MuJoCo XML + URDF) 로딩 가능 여부 확인 필요
- IsaacGym은 URDF 직접 로딩 지원함 (`gym.load_asset()`에 .urdf 경로 전달)

### 8-2. 두 모델 간 충돌 처리
- 인간과 엑소가 물리적으로 겹치면 충돌력이 폭발할 수 있음
- 해결: 두 actor 간 충돌 필터링 비활성화 (collision group 분리)
- VSD가 상호작용력을 담당하므로, 직접 접촉 충돌은 불필요

### 8-3. 질량 비율
- SMPL: ~71.8 kg, WalkON: ~66.6 kg → 거의 1:1
- 실제로는 WalkON 질량이 과대 (실제 엑소는 20~30 kg 수준)
- 필요시 WalkON URDF의 link 질량을 현실적으로 조정

### 8-4. 좌표계 정렬
- SMPL MuJoCo XML: Z-up 확인됨
- WalkON URDF: URDF 자체는 관례 없지만, 기존 Isaac Gym 설정에서 Z-up으로 사용
- 로딩 시 `asset_options`에서 `up_axis` 설정으로 통일 가능

---

## 9. 참고 파일

| 경로 | 내용 |
|---|---|
| `walkon_model_normalization/human_robot_integrated_model/robots/walkonsuit/urdf/walkonsuit.urdf` | WalkON 메인 URDF |
| `walkon_model_normalization/human_robot_integrated_model/robots/walkonsuit/meshes/` | WalkON STL 메쉬 |
| `walkon_model_normalization/human_robot_integrated_model/robots/walkonsuit_human/urdf/walkonsuit_withHuman.urdf` | 기존 인간+엑소 통합 URDF |
| `walkon_model_normalization/resources/robots/walkonsuit/urdf/walkonsuit_withPayload_*.urdf` | 페이로드 변형 |
| `/tmp/smpl/view_average.xml` | SMPL average 체형 XML (뷰어에서 생성) |
| `scripts/view_smpl_model.py` | SMPL 뷰어 (통합 시각화 확장 가능) |

---

## 10. 결론

전체 다리 길이가 ~6% 차이로 거의 동일하여 스케일 면에서 통합은 충분히 가능하다. 세그먼트 비율 차이(대퇴 +14%, 하퇴 -24%)는 SMPL beta 조정 + VSD compliance 조합으로 해결 가능. 고관절 폭 차이(68%)는 엑소가 인간 바깥을 감싸는 구조이므로 자연스럽다.

IsaacGym에서 두 모델을 별도 actor로 로딩하고 VSD로 결합하는 방식은 closed kinematic chain 없이도 구현 가능하며, GPU 병렬 처리(`apply_rigid_body_force_tensors`)를 활용할 수 있다.

우선 Phase 1 (정적 시각화)로 관절 정렬과 beta 최적화를 검증한 후, VSD 구현 → RL 통합 순서로 진행하는 것을 권장한다.
