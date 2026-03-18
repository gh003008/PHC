# WalkON Suit + SMPL 인간 모델 통합 계획 (260315)

---
모델 경로:
로봇: /home/gunhee/workspace/walkon_model_normalization/.../walkonsuit/urdf/walkonsuit.urdf (맞음)
인간: /tmp/smpl/view_average.xml (런타임 생성, SMPL_Robot으로 만든 메쉬 XML)

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

---

## 11. 구현 진행 기록

스크립트: `scripts/view_human_exo.py`

### 11-1. Phase 1 완료: 두 모델 동시 로딩

- SMPL (MuJoCo XML) + WalkON (URDF)을 같은 env에 별도 actor로 로딩 성공
- IsaacGym이 두 포맷 혼용을 지원함 확인

문제 1: **충돌 폭발** — 두 모델이 겹치면서 PhysX 충돌력 폭발, 모델이 튕겨 나감
해결: `create_actor()`에서 collision_group을 분리 (human=0, exo=1). 서로 간 충돌 비활성화.

문제 2: **`gym.apply_body_force()` 함수 미존재** — IsaacGym API 이름 오류
해결: 올바른 API는 `gym.apply_body_forces(env, rigidHandle, force, torque, space)`. rigid handle은 `gym.get_actor_rigid_body_handle(env, actor, body_idx)`로 취득.

### 11-2. Phase 2 완료: VSD 결합 구현

5개 커플링 포인트 구현:
- Pelvis: SMPL Torso[9] ↔ WalkON LINK_BASE[0]
- L/R Shank: SMPL L/R_Knee[2,6] ↔ WalkON LINK_L/R_SHANK[4,10]
- L/R Foot: SMPL L/R_Ankle[3,7] ↔ WalkON LINK_L/R_FOOT[6,12]

VSD 수식: `F = K * (pos_exo - pos_human) + D * (vel_exo - vel_human)`, force clamping 적용.

### 11-3. VSD 게인 튜닝 과정

초기 계획 vs 실제 사용 가능 게인:

| 파라미터 | 초기 계획 | 1차 시도 | 2차 시도 | 현재 (안정) |
|---|---|---|---|---|
| Pelvis K (N/m) | 5000 | 5000 | 500 | 200 |
| Pelvis D (N·s/m) | 500 | 500 | 100 | 80 |
| Limb K (N/m) | 2000~8000 | 2000 | 300 | 80 |
| Limb D (N·s/m) | 200~800 | 200 | 60 | 40 |
| Max force (N) | 없음 | 없음 | 200 | 100 |

문제: **VSD 활성화 시 하체 진동/발산** — 초기 게인이 10~50배 과다
원인:
1. 초기 모델 간 오프셋이 수 cm 존재 → 높은 K에서 수백 N 힘 즉시 발생
2. 힘 → 가속 → 오버슈트 → 반대 방향 힘 → 진동
3. 특히 가벼운 shank/foot body에서 심함 (질량 대비 힘이 큼)

해결:
1. K를 계획 대비 1/25~1/100로 대폭 감소
2. Damping ratio를 높여 과감쇠 (D/2√(Km) > 1)
3. Force clamping 추가 (body당 100N 상한)
4. VSD ramp-up 추가 (처음 2초간 0→1 선형 증가, 초기 충격 방지)

### 11-4. Rotational VSD 추가

문제: **발이 패시브하게 처지고 안으로 회전** — translational VSD만으로는 body orientation 제어 불가
해결: shank/foot에 rotational VSD 추가
- quaternion 기반 orientation error 계산: `ang_err = 2 * vec(q_exo * q_human^-1)`
- `torque = K_rot * ang_err + D_rot * ang_vel_err`
- K_rot=10 N·m/rad, D_rot=8 N·m·s/rad, max_torque=10 N·m (과감쇠)

### 11-5. 인간 모델 DOF 제어 설정

하반신 마비 환자 시나리오:

| 관절 그룹 | DOF Mode | K | D | 비고 |
|---|---|---|---|---|
| Spine (Torso, Spine, Chest, Neck, Head) | POS | 200 | 20 | 척추 직립 유지 |
| Shoulder (L/R_Shoulder) | POS | 300 | 30 | 팔 내림 자세 |
| Lower body (Hip~Toe) | NONE | 0 | 5 | 마비: passive damping만 |
| Arms (Elbow, Hand) | NONE | 0 | 1 | 자유 (중력) |

문제: **팔이 T-pose에서 안 내려옴** — `DOF_MODE_NONE`에서 gravity만으로는 damping 저항에 의해 매우 느리게 떨어짐
해결: Shoulder를 `DOF_MODE_POS`로 변경, target을 L_Shoulder_x=-80°, R_Shoulder_x=+80° (arm-down)으로 설정. K=300, D=30.

문제: **고개가 숙여짐** — Neck/Head가 제어 없이 중력에 처짐
해결: Neck, Head를 SPINE_JOINTS 그룹에 추가 → PD 제어로 직립 유지

### 11-6. 기타 설정

- 인간 모델 주황색 렌더링: `gym.set_rigid_body_color()` 사용, 간섭 시각 확인용
- VSD 시각화: coupling point 간 컬러 라인 + 십자 마커 (Pelvis=빨강, Shank=녹색, Foot=파랑)
- 콘솔에 2초마다 VSD 상태 출력 (거리, 힘 크기, 방향, 클램핑 여부)
- 초기 스폰 위치: exo (0, 0, 1.2), human (-0.25, 0, 1.05) — 엑소가 인간 배를 감싸는 구조

### 11-7. 현재 남은 이슈

1. VSD 활성화 시 여전히 약간의 진동 존재 → ramp-up + 추가 감쇠로 개선 중
2. SMPL-WalkON 세그먼트 비율 불일치 (대퇴 +14%, 하퇴 -24%) → beta 탐색 미착수
3. fix_base_link=True 상태 → 자유 낙하 테스트 미진행
