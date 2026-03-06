# SMPL Humanoid MJCF 모델 분석

원본 파일: mjcf/smpl_humanoid.xml
백업 참고: exp_config/forward_walking/260225_v4/smpl_humanoid.xml

## 1. 전체 구조

좌표계: coordinate="local" (모든 위치가 부모 body 기준 상대 좌표)

바디 계층 (Kinematic Chain):
```
Pelvis (루트, freejoint - 6DOF 자유이동)
├── L_Hip → L_Knee → L_Ankle → L_Toe
├── R_Hip → R_Knee → R_Ankle → R_Toe
└── Torso → Spine → Chest
    ├── Neck → Head
    ├── L_Thorax → L_Shoulder → L_Elbow → L_Wrist → L_Hand
    └── R_Thorax → R_Shoulder → R_Elbow → R_Wrist → R_Hand
```

총 20개 body. Pelvis를 제외한 각 body에 x, y, z 축 hinge joint 3개씩 배치.
총 DOF: 60 (hinge) + 6 (Pelvis freejoint) = 66 DOF


## 2. 축 방향 설정

모든 joint는 독립 hinge 3개로 ball joint를 분해한 구조:
- x축 (axis="1 0 0"): 좌우 방향 축. 굴곡/신전 (예: 무릎 구부리기)
- y축 (axis="0 1 0"): 상하 방향 축. 내전/외전 (예: 다리 벌리기)
- z축 (axis="0 0 1"): 전후 방향 축. 내회전/외회전

SMPL 기준 Y-up 좌표계. Pelvis 기준 위치 pos="-0.0018 -0.2233 0.0282"에서 y가 음수인 것은 원점 대비 아래에 위치함을 의미.


## 3. 관절별 PD 게인 및 가동범위 (ROM)

joint의 user 속성 첫 번째 값이 kp (Proportional gain).
이 값이 VIC에서 impedance_scale에 의해 스케일링되는 기본 강성이다.

kp 분포:
- 코어 (Torso, Spine, Chest): 500 (가장 높음)
- 하지 대관절 (Hip, Knee): 250
- 상지 근위 (Thorax, Shoulder): 200
- 중간 (Ankle, Toe, Neck, Head, Elbow): 150
- 상지 원위 (Wrist): 100
- 말단 (Hand): 50 (가장 낮음)

하지 관절 가동범위:
- Hip: x[-30, 120], y[-45, 45], z[-45, 30](L) / z[-30, 45](R) -- 좌우 z축 비대칭
- Knee: x[0, 145], y[-5.6, 5.6], z[-5.6, 5.6] -- y, z축 거의 잠김. 실질적으로 x축 굴곡만 가능
- Ankle: x[-50, 25], y[-35, 15](L) / y[-15, 35](R), z[-20, 20] -- y축 좌우 비대칭
- Toe: x[-90, 90], y[-45, 45], z[-45, 45]

체간 관절 가동범위:
- Torso, Spine, Chest: 전 축 [-60, 60]

상지 관절 가동범위:
- Neck: 전 축 [-180, 180] (제한 없음)
- Head: 전 축 [-90, 90]
- Thorax, Shoulder: 전 축 [-180, 180]
- Elbow: 전 축 [-720, 720] (사실상 무제한)
- Wrist: 전 축 [-180, 180]
- Hand: 전 축 [-180, 180]


## 4. user 속성 구조

각 joint의 user 속성은 6개 값으로 구성:
user="kp kd_factor scale kp_alt kd_alt scale_alt"

예시:
- Hip: user="250 2.5 1 500 10 2" → 기본 kp=250, kd=2.5 / 대안 kp=500, kd=10
- Torso: user="500 5 1 500 10 2" → 기본 kp=500, kd=5
- Hand: user="50 1 1 150 1 1" → 기본 kp=50, kd=1

VIC에서 이 kp 값이 impedance_scale = 2^ccf 로 0.5배~2.0배 스케일링된다.


## 5. 형상 및 질량 분포

각 body의 geometry 종류와 밀도(density):

밀도 높은 순서:
- Pelvis: sphere (r=0.094), 밀도 4630 -- 가장 무거움, 질량 중심
- Hip(대퇴), Torso, Spine, Chest: capsule, 밀도 2041 -- 코어 + 대퇴
- Knee(하퇴): capsule, 밀도 1235
- 상지 전체 (Thorax~Wrist): capsule, 밀도 1000 -- 균일
- Ankle: box, 밀도 ~440
- Toe: box, 밀도 ~410
- Hand: box, 밀도 ~400 -- 가장 가벼움

질량 분포가 생체역학적으로 현실적: 몸통이 무겁고 말단으로 갈수록 가벼움.


## 6. 액추에이터

60개 motor, 각 joint DOF 하나당 하나씩 대응.
모두 gear="1" (토크 = 제어 신호 x 1). 별도 토크 스케일링 없음.
실제 토크 한계는 MJCF가 아니라 코드에서 torque_limits로 관리한다.


## 7. 접촉 제외 (Self-collision Exclusion)

인접 body 간:
- Torso ↔ Chest
- Head ↔ Chest

다리 교차 방지:
- L/R_Knee ↔ 반대쪽 Ankle, Toe

팔-몸통:
- L/R_Shoulder ↔ Chest


## 8. VIC 관점 핵심 포인트

(1) 무릎 y/z ROM이 거의 잠겨있음 (±5.6도)
무릎 CCF는 실질적으로 x축(굴곡/신전)에만 의미가 있다.
y, z축 CCF 학습은 불필요한 탐색 공간.

(2) 코어 kp가 가장 높음 (500)
보행 안정성의 핵심 부위. CCF로 이 값이 변하면 자세 안정성에 큰 영향.

(3) 상지 ROM이 매우 넓음 (-180~720도)
제약이 거의 없어 CCF 학습 시 불필요한 탐색 공간이 될 수 있음.
보행 과제에서 상지 CCF의 기여도는 낮을 것으로 예상.

(4) 좌우 ROM은 mirror 대칭
Hip z축(L: -45~30, R: -30~45)과 Ankle y축(L: -35~15, R: -15~35)은
숫자만 보면 비대칭처럼 보이지만, 동일 축 방향에서 좌우 body가 반대편에 있으므로
부호가 뒤집혀 있을 뿐 실제 물리적 ROM은 좌우 동일하다.
