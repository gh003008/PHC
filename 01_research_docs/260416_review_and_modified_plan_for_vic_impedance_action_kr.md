# VIC / Impedance 계획에 대한 검토 및 권장 수정안
**Date:** 2026-04-16
**Basis:** `260416_phase0_vic_feasibility_and_next_steps.md`, 더 넓은 WalkON 통합 연구 프로그램, 그리고 최근의 compliance / locomotion RL 문헌.

---

## 1. 집행 결정 (Executive decision)

현재 Phase 0 결과는 다음과 같이 해석되어야 한다:

> **기존 계획은 여전히 유효하나, impedance action의 역할은 좁혀지고 재배치되어야 한다.**

나의 해석은 다음과 같다:

1. **Phase 0는 이미 실제 역할을 완수했다**: VIC 배관(plumbing)이 작동하며 impedance 경로가 연결된 상태에서 안정적인 보행이 가능함을 보였다.
2. **Phase 0는 impedance의 태스크 가치(task value)를 보이지 못했다**, 왜냐하면 정상 트레드밀 보행은 impedance 조절에 대한 reward gradient를 거의 주지 않기 때문이다.
3. 따라서, **impedance action은 정상 보행 동안의 주된 gait-generation action으로 다루어져서는 안 된다**.
4. 대신, 외란 반응, 상호작용 처리, 추후 개인화를 위한 **보조적(secondary), 상황 트리거 기반(context-triggered) 적응 채널**로 다루어져야 한다.

따라서 올바른 다음 단계는 **추가적인 Phase 0 아키텍처 탐색이 아니다**.
올바른 다음 단계는 **최선의 정상 locomotion 정책을 고정하고, impedance 모듈을 residual / gated / perturbation-conditioned 형태로 이동시켜 외란 하에서 테스트하는 것**이다.

---

## 2. Phase 0 결과의 진짜 의미

문서의 깨끗한 비교는 H5 S001 사례이다:

- **No VIC**가 정상 보행에서 최선의 성능.
- **VIC 4-group**은 더 나쁘지만 VIC 8-group보다는 낫다.
- **VIC 8-group**은 H5 변형 중 최악.

이는 **모터 제어(motor control)**와 **impedance 적응(impedance adaptation)**을 분리하면 놀라운 결과가 아니다.

### 나의 해석
정상 트레드밀 보행에서:

- pose-tracking 정책은 이미 명확한 목적을 가지고 있고,
- impedance action은 **강한 태스크-정렬된 신호가 없으며**,
- 그럼에도 여전히 토크 생성에 탐색을 주입한다.

따라서 추가 impedance 차원은 환경이 다음 중 하나를 포함하지 않는 한 **구조화된 노이즈(structured noise)**처럼 행동한다:
- 외부 push,
- 갑작스런 벨트 변화,
- 외골격 상호작용 힘,
- 불균일 지형,
- 또는 명시적 안정성 / compliance 목적.

즉 핵심 교훈은 "VIC가 틀렸다"가 아니다.
핵심 교훈은:

> **Impedance는 태스크에 힘 불확실성 또는 균형 회복 압력이 포함될 때 유용하다. 정상 imitation 보행에서 자동으로 유용한 것은 아니다.**

이것은 Phase 0가 너무 많은 것을 증명하도록 강요하는 것보다, 오히려 원래의 장기 프로그램과 더 잘 부합한다.

---

## 3. 기존 계획에서 변하지 않는 것

나는 연구 방향 전체를 **다시 쓰지는 않을 것이다**. 넓은 계획은 여전히 타당하다.

### 여전히 유효한 부분
- 먼저 강한 정상 locomotion prior를 구축한다.
- 다음으로 외란과 안정성 신호를 도입한다.
- Robustness와 상호작용이 중요한 곳에 impedance / co-contraction 아이디어를 사용한다.
- 이후 다음과 연결:
  - 상체 agency,
  - 하이브리드 인간-로봇 시뮬레이션,
  - 외골격 상호작용,
  - 피험자별 개인화.

### 프로그램 수준의 재정의
넓은 WalkON 프로그램은 이미 다음을 강조한다:
- 순수 rigid-body 제어가 아닌 구조화된 인간-로봇 상호작용,
- 외란 인식 학습,
- 하이브리드 인간 모델링,
- 개인화.

새로운 Phase 0 결과가 말하는 것은 단순히:

> **정상 트레드밀 보행에서 impedance의 가치를 증명하려고 더 시간을 쓰지 마라.
> Impedance가 본래 속한 프로그램의 영역(외란, 상호작용, 개인화)에서 유용하게 만드는 데 시간을 쓰라.**

---

## 4. 좁은 VIC 로드맵에서 바꾸어야 할 것

## 4.1 Phase 0의 의미를 바꾸기
**기존 암묵적 해석:** "VIC는 이미 보행을 개선해야 한다."
**권장 해석:** "VIC 경로는 locomotion을 망가뜨리지 않고 연결될 수 있다."

### 새로운 Phase 0 종료 기준
Phase 0은 다음 조건이 충족되면 완료된 것이다:
1. no-VIC 정상 보행이 안정적이다,
2. VIC 정상 보행 또한 안정적이다,
3. impedance 출력이 수치적으로 건전하다,
4. 로깅 / 그룹핑 / sigma / 토크 배관이 검증되었다.

이것으로 충분하다.

Phase 0에서 VIC가 no-VIC를 **능가할 것을 요구하지 마라**.

---

## 4.2 No-VIC 정책을 mainline baseline으로 승격
최선의 정상 보행자는 **population locomotion prior**가 되어야 한다.

현재 no-VIC H5 정책의 권장 역할:
- base gait generator,
- 이후 외란 fine-tuning의 parent 체크포인트,
- 모든 향후 VIC 비교의 reference controller.

이는 더 어려운 action space로 정상 보행을 재학습하는 시간 낭비를 방지한다.

---

## 4.3 8-group 버전 최적화를 당분간 중단
8-group 버전은 현재 그 가치에 비해 탐색 부담이 너무 크다.

### 권장
- **Mainline:** 하체 4-group VIC, 상체 고정.
- **Ablation 전용:** 8-group VIC.
- **Fallback ablation:** 4-group이 여전히 외란 하에서 실패한다면 2-group 또는 ankle-dominant 버전.

---

## 5. Impedance action을 다루는 방법

이것이 핵심 설계 권장사항이다.

## 5.1 개념적 재정의
Impedance action을 다음과 같이 취급하라:

> **base locomotion action이 아닌, residual 적응 action**

다시 말해:

- locomotion 정책은 **어떻게 걸을지**를 결정하고,
- impedance head는 **조건이 요구할 때 얼마나 stiff/compliant 해질지**를 결정한다.

즉 impedance action은:
- 낮은 차원,
- 낮은 주파수,
- neutral 근처에서 강하게 정규화됨,
- 그리고 주로 외란 / 상호작용 증거가 있을 때 활성화.

---

## 5.2 권장 파라미터화

다음 대신:
```text
policy -> q_ref + xi (every timestep, same exploration style)
```

이것을 사용하라:
```text
policy_main -> q_ref
policy_imp  -> delta_xi
gate(alpha, wrench, contact anomaly, pelvis accel, exo torque) -> g in [0, 1]

xi_eff = xi_nominal(phase) + g * LPF(delta_xi)
K_eff  = K0 * 2^(xi_eff)
```

### 각 항의 의미
- `q_ref`: 주된 정상 모터 명령.
- `delta_xi`: 잔차(residual) impedance 변화.
- `g`: 상황 게이트. 정상 보행에서는 닫힘, 외란 / 힘 상호작용 시 열림.
- `LPF`: step별 토크 노이즈를 방지하는 저역 통과 필터.
- `xi_nominal(phase)`: 선택적 정상 위상 prior. 0에서 시작; 분석이 정당화하면 이후 추가.

### 왜 이게 더 좋은가
원래의 impedance 조절 아이디어를 보존하면서도, Phase 0의 주요 실패 모드를 제거한다:
- impedance가 필요하지 않은 상태에서의 무작위 impedance 탐색.

---

## 5.3 권장 action 구조

### 나의 권장 기본값
**4-group, 좌/우 분리, 상체 고정**

해석:
- 좌측 근위(proximal) (hip + knee)
- 좌측 원위(distal) (ankle + toe)
- 우측 근위
- 우측 원위

### 왜 좌우 대칭을 강제하지 않는가?
정상 보행에서는 대칭이 매력적이다.
외란 회복에서는 대칭이 종종 틀리다.

Slip, push, trip은 보통 타이밍과 하중에서 **비대칭적**이다.
따라서:
- 좌/우를 분리 유지하고,
- mirror augmentation 또는 약한 대칭 정규화를 사용하되,
- 두 쪽을 **강하게 묶지는 않는다**.

---

## 5.4 권장 탐색(exploration) 처리

현재 문제는 차원성만이 아니다.
Impedance 탐색이 너무 일찍부터 너무 자유롭다는 점도 있다.

### 권장
1. **Impedance 전용 actor head 분리**
   - 모터 head와 정확히 동일한 확률적 처리를 공유하지 말 것.

2. **Impedance 초기 sigma를 낮게**
   - 모터 action head보다 훨씬 좁게.
   - 시작점: `sigma_init ≈ -2.5 to -3.0` (현재의 `-1.0` 근처가 아닌).

3. **Neutral impedance 근처에서 시작**
   - `delta_xi = 0`으로 초기화.

4. **더 강한 정규화 사용**
   - neutral prior: `||delta_xi||^2`
   - 시간적 부드러움: `||delta_xi_t - delta_xi_{t-1}||^2`
   - 선택적 위상 부드러움

5. **낮은 업데이트 주파수 사용**
   - impedance는 매 제어 step마다 변할 필요 없음.
   - 3–5 제어 step마다, 또는 5–10 Hz 정도로 업데이트.

### 권장 시작 범위
이것들은 시작점일 뿐이다:
- **Phase 0 / nominal:** `delta_xi ∈ [-0.25, 0.25]` 또는 `[-0.3, 0.3]`
- **Phase 1 / perturbation:** 점진적으로 `[-0.7, 0.7]`까지 확장
- 이후 증거가 뒷받침할 때만 전체 `[-1, 1]` 사용

---

## 5.5 권장 보상(reward) 처리

Impedance는 태스크-정렬된 gradient를 받아야 한다.

### 정상 전용 보행 중에는
다음만 사용:
- neutral prior,
- smoothness prior,
- 선택적 phase plausibility prior.

성능 이득을 기대하지 **말라**.

### 외란 학습 중에는
이제 impedance는 다음을 통해 보상받을 수 있다:
- 생존 / 낙상 회피,
- 회복 시간,
- step 회복 품질,
- WBAM / 각운동량 제어,
- 발 배치 회복,
- 감소된 충돌 임펄스,
- 낮은 토크 스파이크,
- 개선된 balance margin.

### 중요 경고
만약 `alpha`가 거의 항상 낮다면, alpha-blended 설계는 "impedance를 결코 사용하지 않음"으로 붕괴할 수 있다.

따라서 학습은 다음을 통해 충분한 고-안정성-요구 상태를 보장해야 한다:
- 명시적 외란 스케줄링,
- 혼합 nominal/perturbed 에피소드,
- 또는 전용 회복 단계.

---

## 5.6 권장 nominal-vs-reactive 분해

내가 생각하는 가장 깔끔한 분해는:

### Layer 1 — nominal gait
- no-VIC 또는 사실상 neutral-VIC
- imitation 품질과 장기 안정성 최적화

### Layer 2 — reactive impedance
- 균형/상호작용이 요구할 때만 활성
- nominal stiffness로부터 일시적 편차를 학습

### Layer 3 — 이후 개인화
- 타이밍, gain, 지속시간, 비대칭의 피험자별 스케일링
- 특히 외골격 배치에 관련

이 분해는 더 넓은 프로그램의 임상/개인화 부분과도 더 잘 부합한다.

---

## 6. 제안하는 실험 사다리 (experiment ladder)

## 6.1 Phase 0의 즉시 종료
새로운 학습 전에, 최소한의 마무리 작업을 끝내라:

### A. 현재 run 완료
- Cell D를 최종 체크포인트까지 완료
- Cell E를 최종 체크포인트까지 완료

### B. CCF 위상 분석 실행
C, E 체크포인트에 대해:
- 보행 위상에 걸친 그룹별 CCF plot
- 진폭, 부드러움, 비대칭 비교

### C. 정상 CCF가 유용한 것을 학습했는지 결정
세 가지 가능성:
1. **0 근처에서 평평**
   -> 정상 보행은 능동적 impedance 조절이 필요 없음을 확인.
2. **작지만 부드러운 ankle-dominant 위상 패턴**
   -> 이후 `xi_nominal(phase)` prior로 사용.
3. **노이즈가 많거나 불안정한 패턴**
   -> 현재 actionization이 너무 비제약적이라는 증거.

이 분석은 저렴하며 더 이상의 아키텍처 분기 전에 수행되어야 한다.

---

## 6.2 수정된 Phase 1 (권장 mainline)

### Stage 1 — 정상 baseline 고정
- 최선의 no-VIC 체크포인트를 parent로 freeze
- 고정된 정상 평가 배터리에서 평가

### Stage 2 — residual VIC head 부착
- 4-group만
- 상체 고정
- 0 근처에서 초기화
- 정상 에피소드에서 gate 닫힘 또는 거의 닫힘

목표:
- VIC head가 **정상 보행을 과도하게 해치지 않고** 존재할 수 있음을 증명

### Stage 3 — 혼합 커리큘럼
혼합 에피소드로 시작:
- 실용적 시작점으로 **70% nominal / 30% perturbed**
- 필요하면 이후 **50% / 50%**로 이동

외란:
- pelvis pushes
- 무작위 onset
- 무작위 방향
- 무작위 지속시간
- 무작위 크기 커리큘럼

목표:
- impedance에 실제 gradient를 주면서 정상 보행을 유지

### Stage 4 — 안정성 신호 추가
도입:
- `alpha`
- 가능하다면 외력 추정
- pelvis 가속도 / contact anomaly 특징

다음 용도로 사용:
- impedance gate 입력
- reward blending
- 이후 분석

### Stage 5 — 외골격 관련 외란 추가
단순 push가 작동한 후:
- 관절 레벨 assistance/resistance 토크
- onset/offset 서프라이즈
- 가능하다면 벨트 속도 외란
- 상호작용 하중

목표:
- 일반적 외란 robustness에서 외골격 관련 robustness로 이동

---

## 6.3 Phase 1 평가 기준
VIC 모델은 정상 보상이 높다는 이유로 **수락되어서는 안 된다**.
올바른 메트릭을 개선할 때 수락되어야 한다.

### 주된 수락 기준
VIC 모델은 다음을 보이면 통과한다:
1. no-VIC baseline 대비 **더 높은 외란 회복 성공률**
2. **더 나은 회복 품질** (회복 시간 / step 배치 / WBAM 등)
3. **치명적 정상 보행 손실 없음**

### 실용적 임계값
좋은 첫 성공 조건은:

- no-VIC baseline보다 나은 외란 메트릭, 그리고
- 정상 보상 / 정상 에피소드 길이 저하가 대략 **10–15%** 이내로 유지됨.

정상 손실이 반응적 이득보다 훨씬 크다면, impedance 설계는 여전히 작업이 필요하다.

---

## 7. 관리 계획: 여기서부터 어떻게 관리하고 수정할 것인가

## 7.1 작업을 4개 branch로 조직
모든 것을 하나의 실험 스트림에 섞는 대신:

### Branch A — 정상 locomotion baseline
담당 목표:
- 가장 강한 no-VIC 보행 정책을 안정적으로 유지
- 정상 eval 배터리 유지

### Branch B — VIC residual 모듈
담당 목표:
- action head 분리
- gating
- low-pass
- sigma 분리
- 4-group actionization

### Branch C — 외란 커리큘럼 + alpha
담당 목표:
- push 생성기
- 회복 단계
- alpha 계산
- 외란 평가 배터리

### Branch D — 분석 / 툴링
담당 목표:
- phase-CCF plot
- perturbation-conditioned impedance log
- nominal vs perturbed 메트릭 대시보드

이 분리가 끝없는 아키텍처 churn을 방지한다.

---

## 7.2 parent-child 실험 구조를 고정
매 새로운 run마다:
- 정확히 하나의 parent 체크포인트를 정의,
- 한 번에 하나의 주요 가설만 변경,
- 한 줄의 config diff 기록,
- 동일한 배터리로 평가.

### 권장 네이밍 규칙
다음을 인코딩한 run 이름 사용:
- parent
- VIC mode
- perturbation mode
- gate status
- group count
- sigma version

예시:
```text
H5Base_D20k__VIC4_residual_gateA_sigma3__PushCurr_v1
```

---

## 7.3 중단 규칙 정의
중단 규칙 없이는 이 주제가 너무 많은 시간을 흡수할 수 있다.

### 권장 중단 규칙
- 외란 배터리로 평가하기 전 **최대 2–3개 아키텍처 변형**만.
- 4-group이 이미 명확한 반응적 이득을 보이지 않는 한 8-group을 재방문하지 말 것.
- 하체 외란 가치가 입증되기 전까지 상체 impedance를 추가하지 말 것.
- 다음을 모두 시도한 후에도 VIC가 이득을 보이지 않으면:
  - residual head,
  - low sigma,
  - gating,
  - perturbation curriculum,
  - alpha 입력,
  VIC 확장을 일시 중단하고 더 단순한 fallback을 시도.

---

## 7.4 VIC가 여전히 부족할 경우의 권장 fallback 경로
Residual 4-group VIC가 여전히 실패하면, 다음 fallback 중 하나를 사용:

### Fallback A — 고정된 정상 위상 prior + 학습된 impedance 없음
- 수작업 설계 또는 추출된 ankle-dominant 위상 곡선 사용
- 확률적 impedance head 없음

### Fallback B — 트리거 기반 reflex stiffness
- 외란 지표가 임계값을 넘을 때만 impedance 증가
- 전체 정책 출력보다 단순

### Fallback C — ankle 전용 impedance
- 데이터가 유용한 신호가 거의 전적으로 원위(distal)에 있음을 보이면

### Fallback D — motion-library / retrieval-first 전략
- perturbation-conditioned retrieval 또는 exemplar conditioning 사용
- motion prior가 회복 구조를 처리하는 동안 impedance는 단순하게 유지

---

## 8. 더 넓은 WalkON 프로그램과의 관계

더 넓은 프로그램은 여전히 일관적이다.
이 Phase 0 결과는 주로 방향이 아닌 **순서와 강조점**을 바꾼다.

## 8.1 Topic A (상태 추정 / 변형 인식 제어)
이후 impedance gate는 더 나은 상태 신호에 의존해야 할 것이다:
- pelvis/trunk 상태,
- 상호작용 효과,
- latent 변형 / 정렬 오차,
- balance margin.

따라서 Topic A는 덜 중요해지는 것이 아니라 더 중요해진다.

## 8.2 Topic B / D (상체 agency 및 하이브리드 인간 모델)
상체 자발적 움직임을 아직 VIC 질문에 가져오지 **말라**.

먼저 보여라:
- 하체 impedance가 시뮬레이션에서 외란 회복을 개선함.

그 후에:
- 상체 motion prior / 하이브리드 인간 반응 추가,
- 왜냐하면 반응적 균형과 상체 보상은 결합되어 있기 때문.

## 8.3 Topic E (개인화)
이 결과는 사실 개인화 스토리를 강화한다.

왜?
정상 보행은 약한 impedance 조절만 필요할 수 있지만, **외란 회복과 어시스턴스 타이밍은 훨씬 더 피험자 특이적**이기 때문이다.

따라서 개인화는 다음을 대상으로 해야 한다:
- 타이밍,
- 지속시간,
- 비대칭,
- 그리고 외란 하 stiffness 스케일링,

단순한 정상 kinematic tracking이 아니라.

---

## 9. 구체적 구현 제안

## 9.1 내가 우선순위를 둘 최소 코드 변경
1. **정책 head 분리**
   - 모터 head
   - impedance head

2. **impedance sigma 분리**
   - 독립적 초기화 및 스케줄

3. **impedance 저역 통과 필터**
   - step별 노이즈 주입 방지

4. **impedance gate**
   - 정상 에피소드에서 닫히거나 약함
   - 외란/안정성 증거에 의해 열림

5. **통합 로깅**
   - `delta_xi`의 mean / std
   - 그룹별 위상 곡선
   - perturbation-conditioned 통계
   - gate activation 히스토그램

6. **평가 배터리**
   - nominal 보행
   - push 회복
   - 비대칭 외란
   - assistance onset/offset 외란

---

## 9.2 제안하는 config 시작점
예시용:

```yaml
vic:
  enabled: true
  mode: residual
  groups: 4
  upper_body_fixed: true

  xi_range_nominal: [-0.30, 0.30]
  xi_range_perturb: [-0.70, 0.70]

  update_every_n_steps: 4
  lpf_tau_sec: 0.15

  sigma_init: -2.8
  smoothness_weight: 0.01
  neutral_prior_weight: 0.01

  gate:
    enabled: true
    inputs: [alpha, pelvis_acc, contact_anomaly, ext_force_est, exo_torque]
    alpha_threshold: 0.15
    init_bias_closed: true

training:
  curriculum:
    nominal_episode_ratio: 0.70
    perturb_episode_ratio: 0.30
    perturb_force_N: [20, 200]
    perturb_duration_sec: [0.10, 0.50]
```

다시 말하지만, 이 값들은 **시작점**이지 최종 진리가 아니다.

---

## 10. 최종 권장

모든 것을 한 문장으로 압축하면:

> **기존 연구 방향은 유지하되, impedance action을 풀타임 보행 action에서 gated residual 적응 모듈로 강등하고, 실제로 중요해야 할 곳(외란, 상호작용, 개인화)에서 테스트하라.**

### 오늘부터의 권장 mainline
1. D와 E를 완료한다.
2. phase-CCF 분석을 실행한다.
3. no-VIC 정상 baseline을 freeze한다.
4. VIC를 4-group residual action으로, 낮은 sigma와 smoothing과 함께 재도입한다.
5. 즉시 혼합 nominal + perturbation 학습으로 이동한다.
6. 정상 보상만이 아닌 외란 메트릭으로 성공을 판단한다.
7. 긍정적 증거가 있을 때만 더 풍부한 impedance 구조를 고려한다.

이 경로는 원래 계획을 보존하고, 낭비되는 탐색을 줄이며, impedance 아이디어에 공정한 테스트를 부여한다.

---

## 11. 참고 기반

### 내부 문서
- `260416_phase0_vic_feasibility_and_next_steps.md`
- `walkon_integrated_research_program_detailed.md`

### 추론에 사용된 외부 문헌
1. Roberto Martín-Martín et al., **Variable Impedance Control in End-Effector Space: An Action Space for Reinforcement Learning in Contact-Rich Tasks**, IROS 2019 / arXiv:1906.08880.
2. Adrian Hartmann et al., **Deep Compliant Control for Legged Robots**, ICRA 2024.
3. Botian Xu et al., **FACET: Force-Adaptive Control via Impedance Reference Tracking for Legged Robots**, 2025.
4. Antonia Bronars, Younghyo Park, Pulkit Agrawal, **Tune to Learn: How Controller Gains Shape Robot Policy Learning**, arXiv:2604.02523, 2026.
5. Oliver Hausdörfer et al., **Latent Action Priors for Locomotion with Deep Reinforcement Learning**, arXiv:2410.03246, 2025.
6. Abdel-Rahman Akl et al., **Muscle Co-Activation around the Knee during Different Walking Speeds in Healthy Females**, Sensors 2021.
7. Nancy T. Nguyen et al., **Co-contraction about the ankle increases with the threat of a walking perturbation**, Journal of Electromyography and Kinesiology, 2025.
8. Stacie A. Chvatal and Lena H. Ting, **Voluntary and Reactive Recruitment of Locomotor Muscle Synergies during Perturbed Walking**, Journal of Neuroscience, 2012.
9. Maria T. Tagliaferri and Inseung Kang, **Systematic Evaluation of Hip Exoskeleton Assistance Parameters for Enhancing Gait Stability During Ground Slip Perturbations**, arXiv:2601.15056, 2026.
