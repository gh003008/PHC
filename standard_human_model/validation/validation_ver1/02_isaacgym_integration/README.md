# 02_isaacgym_integration — IsaacGym 통합 검증

## 목적

01_muscle_layer에서 수식 정확성을 검증한 파이프라인이
IsaacGym PhysX 시뮬레이터 안에서 실제로 물리적 효과를 내는지 확인한다.
`gym.set_dof_actuation_force_tensor()`를 통한 토크 주입이 실제로 관절을 움직이는지,
환자 프로파일별 차이가 물리 시뮬레이션에서 재현되는지 검증한다.

## 검증 항목

| ID  | 시나리오 | 합격 기준 |
|-----|----------|-----------|
| I01 | +50 Nm 상수 토크를 L_Knee에 2초 주입 | 각도 증가 > 6° (굴곡 방향) |
| I02 | +50 Nm vs -50 Nm 동시 비교 | 양의 토크=굴곡(증가), 음의 토크=신전(감소) |
| I03 | 3 프로파일 knee pendulum 5초 | Spastic > Healthy > Flaccid 최종 각도 순서, 각 차이 > 5° |
| I04 | 80° 굴곡 정적 자세에서 bio-토크 측정 | Spastic > Healthy > Flaccid 저항 토크, Healthy > 0 Nm |

## 실행 방법

```bash
conda activate phc
cd /home/gunhee/workspace/PHC
python standard_human_model/validation/02_isaacgym_integration/run_validation.py --headless --pipeline cpu
```

## 테스트 설명

### I01: 토크 주입 방향
`gym.set_dof_actuation_force_tensor()`가 PhysX에 실제로 반영되는지 가장 기본적으로 확인한다.
양의 토크를 L_Knee DOF에 주입하면 각도가 증가(굴곡)해야 한다.
이것이 실패하면 이후 모든 테스트가 의미 없다.

### I02: 부호 규약
SMPL DOF 정의에서 L_Knee의 양의 방향이 굴곡인지 신전인지 확인한다.
+/-를 동시에 적용해서 방향이 반대로 움직이면 PASS.
이 규약을 알아야 bio-torque 계산 결과를 올바르게 해석할 수 있다.

### I03: 프로파일 분화
실제 물리 시뮬레이션에서 환자 프로파일이 다른 운동 패턴을 만들어내는지 확인한다.
경직(Spastic)은 저항이 강해 초기 자세 근처에 머물고,
이완(Flaccid)은 저항이 없어 중력에 의해 완전 신전 방향으로 이동한다.

### I04: 중력 균형
80° 굴곡 정적 자세에서 중력이 신전 방향으로 작용할 때
bio-torque가 반대(굴곡, 양수) 방향으로 발생하는지 확인한다.
Spastic이 Healthy보다 강한 저항을 보이면 환자 파라미터가 물리적으로 의미 있게 작동하는 것이다.

## 공통 설정

- 테스트 관절: L_Knee (DOF index: JOINT_DOF_RANGE["L_Knee"][0])
- 비테스트 관절: DOF_MODE_POS (kp=500, kd=50으로 중립 고정)
- 에셋: `standard_human_model/isaacgym_validation/smpl_humanoid_fixed.xml` (freejoint 없음)
- 모두 headless 실행 (시각화는 Step 3에서)

## 결과 해석

### I01 플롯
각도가 단조 증가하면 PASS. 각도가 오히려 감소하거나 움직이지 않으면 토크 주입 연결 문제.

### I02 플롯
파란 선(+토크)이 위에, 빨간 선(-토크)이 아래에 있으면 PASS.
두 선이 같은 방향으로 움직이면 부호 규약 또는 DOF 인덱스 문제.

### I03 플롯
상단 각도 플롯에서 빨간(Spastic)이 가장 높고 초록(Flaccid)이 가장 낮으면 PASS.
세 선이 구분되지 않으면 프로파일 파라미터 적용 문제.

### I04 플롯
상단 토크 플롯에서 Spastic > Healthy > Flaccid 이고 모두 양수 방향이면 PASS.
Healthy 토크가 음수(신전 방향)이면 인대 파라미터 또는 reflex 설정 문제.

## 다음 단계

`02_isaacgym_integration` 검증 완료 후:
- `../03_visualization/` — IsaacGym 뷰어를 통한 시각적 확인
- `blend_alpha > 0` 테스트: VIC-MSK 하이브리드에서 bio-torque 비중 증가 실험