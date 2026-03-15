# SMPL 모델 뷰어 구현 보고서 (260315)

---

## 1. 목적

IsaacGym 환경에서 SMPL 휴머노이드 모델을 단독으로 시각화하고, 관절 조작·체형 변경·외력 인가 등 인터랙티브 기능을 제공하는 뷰어 스크립트 개발.

파일 경로: `scripts/view_smpl_model.py`

---

## 2. 최종 기능

- 3가지 체형 (Average / Thin / Heavy) 키보드 전환 (1/2/3 키)
- 관절별 각도 조절 (UP/DOWN: 관절 선택, LEFT/RIGHT: ±5도 증감)
- 외력 인가 (F: 전방, B: 후방, SPACE: 상방)
- 전체 자세 리셋 (0 키)
- 인간형 STL 메쉬 시각화 (캡슐/박스 아닌 실제 인체 형태)
- 팔 내림 초기 자세 (L_Shoulder_x=-80°, R_Shoulder_x=+80°)

---

## 3. 구현 과정 및 시행착오

### 3-1. 초기 구현: 캡슐 모델 문제

처음에는 `phc/data/assets/mjcf/smpl_humanoid.xml`을 로딩했는데, 이 파일은 메쉬 없는 충돌 지오메트리(캡슐/박스)만 포함하는 기본 XML이다. 결과적으로 인간이 아닌 캡슐 조합 형태로 렌더링되었고, Z-up 좌표계 미적용으로 모델이 하늘을 보고 누워있었다.

해결: `/tmp/smpl/` 디렉토리의 런타임 생성 메쉬 XML 사용으로 전환.

### 3-2. 메쉬 XML 자동 생성

`smpl_sim.smpllib.smpl_local_robot.SMPL_Robot`을 사용하여 3가지 체형 XML을 직접 생성하는 `generate_body_xmls()` 함수 구현.

```python
robot_cfg = {
    'mesh': True, 'replace_feet': True, 'rel_joint_lm': True,
    'upright_start': True, 'remove_toe': False, 'freeze_hand': True,
    'real_weight_porpotion_capsules': True, 'real_weight_porpotion_boxes': True,
    'real_weight': True, 'masterfoot': False, 'big_ankle': True, 'box_body': True,
    'model': 'smpl', 'sim': 'isaacgym'
}

# beta[1]로 체형 조절: 0.0=average, -2.0=thin, 2.0=heavy
for name, beta1_val in [("average", 0.0), ("thin", -2.0), ("heavy", 2.0)]:
    betas = torch.zeros(1, 10)
    betas[0, 1] = beta1_val
    robot.load_from_skeleton(betas=betas, gender=[0])
    robot.write_xml(paths[name])
```

### 3-3. Segfault #1: 메쉬 디렉토리 삭제 문제

`SMPL_Robot.write_xml()` 후 `robot.remove_geoms()`를 호출하면 생성된 STL 메쉬 디렉토리가 통째로 삭제된다. XML 파일은 남아있지만 참조하는 메쉬가 없어서 IsaacGym이 segfault로 크래시.

해결: `robot.remove_geoms()` 호출을 완전히 제거하고 주석으로 경고 명시.

```python
# NOTE: do NOT call robot.remove_geoms() — it deletes the STL mesh directories
```

### 3-4. Segfault #2: 캐시 검증 미흡

`generate_body_xmls()`에서 이미 생성된 XML의 캐시 존재 여부만 확인하고 메쉬 디렉토리 유효성은 체크하지 않았다. 이전 세션에서 메쉬가 삭제된 상태에서 XML만 남아있으면 다시 segfault.

해결: 캐시 검증에 XML 내부의 메쉬 경로 파싱 + 디렉토리 존재 확인 추가.

```python
with open(p, 'r') as f:
    match = re.search(r'file="([^/]+)/geom/', f.read())
if not match or not os.path.isdir(os.path.join("/tmp/smpl", match.group(1), "geom")):
    all_valid = False
```

### 3-5. 팔 자세 시행착오

목표는 T-pose(팔 수평)에서 자연스러운 팔 내림 자세로 변경하는 것.

시도 1: `L_Shoulder_z = -80°` → 팔이 앞으로 나감 (Z축은 수직축이라 다른 평면 회전)
시도 2: `L_Shoulder_y = ±90°` → 팔이 비틀림 (Y축은 측면축이라 internal/external rotation)
시도 3: `L_Shoulder_x = -80°`, `R_Shoulder_x = +80°` → 정상 동작

이유: SMPL 메쉬 XML에서 Shoulder 관절의 axis 정의가 다음과 같다:
- x축: axis="1 0 0" → 전방축. L_Shoulder가 +Y 방향으로 뻗어있으므로, X축 기준 음의 회전이 +Y→-Z (아래)로 이동.
- R_Shoulder는 -Y 방향이므로 양의 X 회전이 -Y→-Z (아래)로 이동.

```python
for i, name in enumerate(dof_names):
    if name == "L_Shoulder_x":
        targets[i] = math.radians(-80)
    elif name == "R_Shoulder_x":
        targets[i] = math.radians(80)
```

### 3-6. 발 메쉬가 직육면체인 문제

SMPL 모델의 발 메쉬가 인간 형태가 아닌 직육면체인 이유: `smpl_sim` 패키지의 `skeleton_mesh_local.py`에서 `replace_feet=True` 설정 시 발 메쉬를 convex hull → box로 변환하는 로직이 하드코딩되어 있다. `GEOM_TYPES` 딕셔너리에서 L/R_Ankle, L/R_Toe가 'box'로 고정. `smpl_sim` 패키지 자체를 수정해야 하므로 현재는 보류.

---

## 4. 기술 구현 상세

### 4-1. SMPL 모델 이중 지오메트리 구조

SMPL XML은 각 body에 두 종류의 geom을 가진다:
- 충돌용 (contype=1, conaffinity=1): 캡슐/박스 형태. 물리 시뮬레이션에 사용.
- 시각용 (contype=0, conaffinity=0): STL 메쉬. 렌더링에만 사용.

체형(beta) 변경 시 둘 다 재계산되지만, 충돌 지오메트리는 근사값이다.

### 4-2. 체형 전환 구현

IsaacGym은 런타임에 asset을 변경할 수 없으므로, 체형 전환 시 env를 파괴하고 새로 생성한다:

```python
if needs_reload:
    gym.destroy_env(env)
    xml_path = body_xmls[body_names_list[current_body_idx]]
    env, actor, num_dofs, dof_names, dof_targets = load_humanoid(xml_path)
```

### 4-3. DOF 구성

Average 체형 기준: 24 joints, 69 DOFs (모든 관절 3DOF hinge)
- 하체: Pelvis(free joint 아님, fix_base_link=True), L/R_Hip(3), L/R_Knee(3), L/R_Ankle(3), L/R_Toe(3)
- 상체: Torso(3), Spine(3), Chest(3), Neck(3), Head(3), L/R_Shoulder(3), L/R_Elbow(3), L/R_Hand(3, frozen)

---

## 5. 사용법

```bash
conda activate phc

# 기본 실행 (3가지 체형 자동 생성)
python scripts/view_smpl_model.py

# 특정 XML 지정
python scripts/view_smpl_model.py --xml /tmp/smpl/smpl_humanoid_XXXX.xml
```

### 키보드 조작
| 키 | 기능 |
|---|---|
| 1/2/3 | Average / Thin / Heavy 체형 전환 |
| UP/DOWN | 이전/다음 관절 선택 |
| LEFT/RIGHT | 선택 관절 각도 -5°/+5° |
| 0 | 전체 자세 리셋 (팔 내림) |
| F | 전방 힘 200N |
| B | 후방 힘 200N |
| SPACE | 상방 힘 600N |

---

## 6. 참고 파일

- `scripts/view_smpl_model.py`: 뷰어 스크립트 (본 보고서 대상)
- `scripts/joint_monkey_smpl.py`: 참고한 원본 DOF 애니메이션 스크립트
- `smpl_sim/smpllib/smpl_local_robot.py`: SMPL_Robot 클래스 (메쉬 XML 생성)
- `smpl_sim/smpllib/skeleton_mesh_local.py`: 발 box 변환 로직
