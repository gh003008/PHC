# 01_muscle_layer — 근육 레이어 단위 검증

## 목적

IsaacGym 없이 순수 PyTorch로 근골격계 모듈의 수식 정확성을 검증한다.
각 테스트는 독립적이며, 하나가 FAIL이어도 나머지는 계속 실행된다.

## 검증 항목

| ID   | 대상 모듈          | 시나리오                          | 합격 기준 |
|------|--------------------|-----------------------------------|-----------|
| T01  | HillMuscleModel    | F-L 커브 (active) l_norm sweep    | 피크 @1.0 ± 0.01 |
| T02  | HillMuscleModel    | F-V 커브 Hill curve               | f_FV(0)=1.0, 단조성 유지 |
| T03  | HillMuscleModel    | F-L 커브 (passive)                | l_norm≤1.0 → 0, 이상 지수 증가 |
| T04  | HillMuscleModel    | activation 0/0.5/1.0 × l_norm    | active force 선형 비례 |
| T05  | MomentArmMatrix    | 근육 × 관절 토크 히트맵            | 이관절근 ≥2 관절에 토크 |
| T06  | MomentArmMatrix    | L/R 대칭 입력 → 대칭 출력         | 오차 < 1e-4 Nm |
| T07  | LigamentModel      | soft-limit 경계 초과 토크          | 상한 초과=음수, 하한=양수 |
| T08  | ReflexController   | healthy(gain=1) vs spastic(gain=8) | spastic > healthy, v=0에서 0 |
| T09  | HumanBody (전체)   | cmd=0, q=0, vel=0 → tau ≈ 0      | max\|tau\| < 10 Nm |

## 실행 방법

```bash
conda activate phc
cd /home/gunhee/workspace/PHC
python standard_human_model/validation/01_muscle_layer/run_validation.py
```

## 결과 해석

### T01: Hill F-L Active
가우시안 곡선이 l_norm=1.0에서 피크여야 한다.
빨간 점선이 현재 YAML 파라미터(`l_opt=1.0`, `l_slack=0.30m`)의 실제 동작점이다.
동작점이 0.30으로 낮게 나오면 → **l_opt를 실제 최적 근섬유 길이(m 단위)로 수정 필요**.

### T02: Hill F-V
- v=0(등척): f_FV = 1.0 (기준점)
- 수축(v<0): 0을 향해 감소
- 신장(v>0): 최대 1.8배까지 증가

### T03: Hill F-L Passive
l_norm > 1.0 초과 시 지수적 증가. 현재 YAML에서 passive force가 거의 발생하지 않는 이유: 근육이 항상 최적 길이 이하에서 동작 (l_opt 파라미터 이슈).

### T04: Activation Linearity
세 곡선이 세로 방향으로 정확히 2배 비례하면 active force가 선형적으로 활성화를 따름을 의미한다.

### T05: Moment Arm 히트맵
- 노란 테두리 행: 이관절근 (2개 이상 관절에 색이 있어야 함)
- 색의 부호: 빨강=양의 토크, 파랑=음의 토크
- L/R 패턴: 위쪽과 아래쪽이 거울 반전

### T06: L/R 대칭
오차 막대 그래프에서 모든 막대가 1e-4 Nm 아래에 있으면 완전 대칭.

### T07: Ligament
빨간 점선(상한) 오른쪽에서 토크가 음수, 파란 점선(하한) 왼쪽에서 양수. 지수 곡선 형태여야 한다.

### T08: Stretch Reflex
경직(Spastic) 곡선이 정상(Healthy)보다 더 일찍, 더 강하게 반응해야 한다. 비율 그래프에서 8배 선 근처이면 PASS.

### T09: Full Pipeline Zero
모든 막대가 ±5 Nm 이내이면 PASS. 큰 값이 나오면 passive force 또는 ligament 파라미터 점검.

## 알려진 파라미터 이슈

현재 `healthy_baseline.yaml`의 `l_opt: 1.0`은 **단위 혼용** 문제가 있다:
- `l_slack`: meters 단위 (예: 0.30m)
- `l_opt`: 1.0 (무차원으로 해석됨)
- 결과: `l_norm = l_slack / l_opt = 0.30` → 최적 길이의 30%에서 동작 → passive force 미발생

수정 방향: `l_opt`를 해부학적 최적 근섬유 길이(m)로 설정 (예: quadriceps → 0.08m, gastrocnemius → 0.05m).
이 문제는 코드 버그가 아니라 **파라미터 튜닝 이슈**.

## 다음 단계

`01_muscle_layer` 검증 완료 후:
- `../02_isaacgym_integration/` — IsaacGym 물리 시뮬레이터와의 연결 검증
- `../03_visualization/` — IsaacGym 뷰어를 통한 시각적 검증
