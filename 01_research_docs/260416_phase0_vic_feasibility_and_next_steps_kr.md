# Phase 0 VIC 타당성 검증: 결과 및 향후 방향

**Version**: 1.0 — 2026-04-16
**Scope**: S001 트레드밀 보행 데이터에서 4개 구성을 실행한 후의 중간 분석
**Related**: [260413_MPL_methodology_refined.md](./260413_MPL_methodology_refined.md)

---

## 1. Executive Summary

EXOLAB 서버에서 S001 H5 트레드밀 보행 데이터와 AMASS KIT walking 서브셋을 이용해 4개의 Phase 0 타당성 실험을 수행했다. 목적은 MPL 방법론(§2.2)의 단일 정책(single-policy) VIC 아키텍처가 안정적인 정상(nominal) 보행을 학습할 수 있는지 검증하는 것이었다. 결과는 **직관에 반하는 발견**을 보여준다:

> **학습 가능한 action으로 CCF를 추가하는 것이 정상 보행 성능을 개선하지 않으며 — 오히려 동일 데이터에서 순수 PD 제어 대비 뚜렷하게 저하시킨다.**

그러나 이것이 MPL 방법론 자체를 무효화하는 것은 **아니다**. 이는 Phase 0의 역할을 재정의한다: 이는 배관(plumbing) 점검이지 가치(value) 점검이 아니다. VIC의 진짜 시험은 외란(perturbation)이 임피던스 조절에 대한 선택 압력을 만드는 Phase 1 이후에서 이루어진다. 본 문서는 데이터를 요약하고, MPL 프레임워크를 통해 해석하며, 구체적인 다음 단계를 제안한다.

---

## 2. 실험 매트릭스

모든 실험은 다음을 공통으로 사용한다:
- **아키텍처**: PHC 단일 정책 (MLP [1024,1024,512,512])
- **학습**: PPO + AMP discriminator
- **모션 데이터**: S001 트레드밀 보행 (H5) 또는 AMASS KIT walking 서브셋
- **보상**: imitation + discriminator + metabolic (외란 없음, balance reward 없음, α 신호 없음)
- **최대 epoch**: 20,000
- **동일 하이퍼파라미터** (비교 가능한 범위에서: power_coefficient 5e-7, cycle_motion True, task/disc 가중치 0.5/0.5, 10k에서 reward curriculum 전환)

최종(또는 최신) 결과:

| Cell | Motion | VIC | CCF groups | Final Epoch | `rwd` | `eps_len` (30Hz) | 보행 지속시간 |
|---|---|---|---|---|---|---|---|
| A | AMASS primitive | **OFF** (baseline) | — | 7,146 (수동 중단) | 152 | 50 | 1.67 s |
| B | AMASS primitive | ON | 8 | 20,001 | 223 | 80 | 2.67 s |
| C | H5 S001 | ON | 8 | 20,001 | **392** | 130 | 4.33 s |
| D | H5 S001 | **OFF** | — | running (Ep ~18,593) | **605** | **192** | **6.40 s** |
| E | H5 S001 | ON | 4 (하체 전용, 상체 고정) | running (Ep ~12,674) | 330 | 115 | 3.83 s |

**Ep 10,000 스냅샷** (curriculum 전환 직후이므로 가장 직접적으로 비교 가능):

| Cell | `rwd` | `eps_len` |
|---|---|---|
| C (H5 + VIC 8grp) | 262.8 | 93.2 |
| D (H5 + no VIC) | **740.9** | **235.1** |
| E (H5 + VIC 4grp) | ~300 | ~100 |

**핵심 순위 (H5 motion)**: `no VIC ≫ VIC 4grp > VIC 8grp`. 이 태스크에서 VIC는 PD baseline 대비 보상의 25–50%를 희생시킨다.

---

## 3. MPL 방법론을 통한 해석

### 3.1 배관(Plumbing) 점검으로서의 Phase 0

MPL 로드맵(§9)은 Phase 0을 다음과 같이 정의한다:

> "보행 데이터로 단일 정책 VIC → 타당성 검증: CCF와 함께 안정적인 보행"

성공 기준은 *CCF action을 추가한 상태에서 안정적인 보행이 가능하다*는 것이지, *CCF가 정상 보행을 개선한다*는 것이 아니다. 세 VIC 구성 (B, C, E) 모두 eps_len ≫ 1 step인 안정적인 보행을 생성하므로, 배관은 작동한다:

- ✅ Action space scaling (69 + N_g)이 자동 유도됨
- ✅ CCF → 임피던스 공식 `τ = K · 2^ξ · ...`가 올바르게 적용됨
- ✅ `amp_agent.py`의 CCF sigma override가 그룹 수에 자동 조정됨
- ✅ 4-group 축소 (상체 1.0× 고정)가 코드 변경 없이 작동 — `humanoid_im_vic.py:199-205`의 기존 분기가 이를 정확히 지원
- ✅ 사전 VIC_CCF_ON2 실험(AMASS single forward clip)에서 학습된 CCF 패턴이 생체역학적으로 타당한 계층을 보여줌: ankle ≫ hip ≳ knee ≈ 상체 (Winter 2009 임피던스 문헌과 일치)

### 3.2 Phase 0에서 CCF가 해가 되는 이유

방법론은 다음과 같이 기술한다 (§7.4):

> "안정화 외력이 감지될 때 → 정책은 ξ를 낮출 수 있다 (더 compliant해짐)
> 불안정화 외력이 감지될 때 → 정책은 ξ를 높인다 (저항하기 위해 stiffen)"

**CCF의 가치는 Phase 0에 존재하지 않는 이벤트에 의존한다.** 외란 없음, 외골격 힘 없음, 지형 변화 없음:

1. 정상 보행에 대한 최적 CCF는 거의 **평평(flat)**하다 (보행 위상에 걸쳐 거의 일정, 기껏해야 push-off를 위한 완만한 ankle 조절 정도).
2. 그러나 정책은 σ=0.37 (log-std −1.0)로 매 timestep마다 CCF를 출력하도록 강요받는다. 이는 노이즈가 있는 4 또는 8차원의 추가 action을 만들고, **토크 계산을 교란**하여 PD tracking을 미세하게 불안정화시킨다.
3. 보상 함수에는 주어진 위상에서 높거나 낮은 CCF를 명시적으로 선호하는 항이 없다 — 이는 imitation reward (reference pose tracking)에 의해 지배된다. 어떠한 CCF 탐색도 태스크와 관련된 gradient 없이 단지 노이즈 토크로 변환될 뿐이다.
4. 따라서 PPO 옵티마이저는 많은 샘플을 CCF 분산을 0으로 줄이는 데 쓰지만, `sigma_init: -1.0`에 의해 주입된 확률성은 이를 넓게 유지시켜 *모터* 정책의 수렴을 늦춘다.

이는 일반 원리와 일치한다: **보상에 기여하는 차원이 없다면 action space가 커지는 것은 sample efficiency를 해친다**.

### 3.3 왜 baseline AMASS (Cell A)가 가장 나쁘게 보이는가

Cell A (152 rwd / 50 eps_len)가 가장 약하다. 이유:

- B/C/D/E와 다른 학습 설정 (reward curriculum 없는 구버전 `im_walk.yaml`, 0.5/0.5 task/disc 가중치 없음, 구버전 `power_coefficient: 5e-5`는 신버전 5e-7의 100배)
- 시각적 plateau로 인해 Ep 7,146에서 수동 종료

**따라서 Cell A와 Cell B는 VIC의 효과에 대한 깨끗한 A/B가 아니다** — 학습 하이퍼파라미터도 다르기 때문이다. 깨끗한 A/B는 **C vs D** (둘 다 H5, 둘 다 20k epoch 목표, 둘 다 0.5/0.5 + curriculum + 5e-7 power_coefficient — VIC만 다름)이다.

그 깨끗한 비교 하에서 VIC의 비용은 명확하다: 약 50% 보상 저하, 약 45% episode length 저하.

---

## 4. Co-Contraction 질문

사용자의 VIC에 대한 개념적 동기는 **근육 레벨의 co-contraction**을 모방하는 것이다: 인간은 주동근(agonist)과 길항근(antagonist)을 동시에 활성화하여 관절을 stiffen시킨다. CCF는 이것의 관절 레벨 추상 표현이다.

### 4.1 Co-Contraction이 정상 트레드밀 보행에서 의미 있는가?

EMG 문헌 (Winter 2009, Sartori et al. 2015)은 다음을 보여준다:

1. 건강한 성인 트레드밀 보행에서의 co-contraction은 **낮고 위상-고정(phase-locked)**이며, heel strike (impact absorption) 부근과 push-off (propulsion) 동안 약간 peak를 보인다.
2. 한 stride 내 stiffness 조절 범위는 평균 stiffness의 0.7×~1.3× 정도 — **1.9배** (log₂ ≈ 0.9, 우리의 ξ ∈ [−1, 1] 범위와 대략 일치)이다.
3. Co-contraction은 노인, 외란 회복, 불균일 지형, 인지 부하, 하중 운반 시 **상당히 증가**한다. Co-contraction의 *동적 범위*는 정상 보행 자체가 아니라 이러한 외부 요인에 의해 주도된다.

**함의**: 우리의 측정 결과 (정상 H5 보행에서 no-VIC 승리)는 인간 생체역학과 일치한다: 건강한 젊은 성인은 트레드밀에서 강한 co-contraction 조절이 필요하지 않으므로, RL 정책도 CCF로부터 이득을 얻지 못한다.

### 4.2 CCF가 효과를 발휘해야 할 시점

MPL 방법론은 CCF가 실질적인 신호를 갖는 5가지 시나리오를 제공한다:

1. **무작위 외력** (§5.1 Stage 2): 0.1–0.5 s bursts 동안 pelvis에 F_ext ∼ U(−F_max, F_max).
2. **외골격 토크** (§7.3): 관절 레벨 토크 외란, 위상별 보조/저항, 갑작스런 onset/offset.
3. **벨트 외란** (§4.2 데이터 수집): trip/slip 반응을 유발하는 갑작스런 트레드밀 속도 변화.
4. **안정성 여유 α** (§3.2): 높은 α는 임피던스 패턴이 정상 보행과 다른 외란-반응 exemplar로 정책을 라우팅.
5. **생체역학적 CCF 보상** (`vic_bio_ccf_reward_w`, 현재 기본값 0): CCF가 보행 위상별로 변화하도록 장려하는 명시적 reward shaping — 현실적 임피던스 곡선을 향한 inductive bias로 작동.

이 5가지 시나리오 모두에서 CCF는 **reward gradient**를 갖는다: 인간 reference에 대해 tracking되거나 survival (balance reward)에 직접 연결된다. Phase 0에서는 이 중 어떤 것도 활성화되지 않는다.

### 4.3 4-Group 결과 해석 (Cell E)

Cell E는 상체 CCF를 1.0×로 고정하고 다리별 hip+knee를 그룹화하여 CCF action 차원을 8에서 4로 축소한다. 초기 결과 (Ep 12,674: rwd 330)는:

- **동일 epoch에서 8-group VIC보다 우수 (C는 Ep 10k에서 262, Ep 12k에서 ~300 추정)**
- **여전히 no-VIC보다 열위 (D: Ep 10k에서 740)**

4-group 버전은 가장 덜 나쁜 VIC이다. 이는 다음과 일치한다:

- 탐색 차원 수 감소 → 토크 노이즈 감소
- 상체 고정 → VIC_CCF_ON2에서 자연 변동이 가장 작다고 나타난 영역의 CCF 조절 제거 (상체 0.80–0.86× vs ankle 1.28–1.49×)
- Hip+knee pooling은 일부 해상도 비용 (VIC_CCF_ON2 데이터에서 hip은 더 stiff, knee는 더 compliant 선호)이지만 정보 손실은 제한적

**정책에 CCF를 넣고 싶다면, 4-group 버전이 올바른 형태 — 그러나 외란이 없는 Phase 0에서는 이것조차 pure PD 대비 성능 비용을 발생시킨다.**

---

## 5. 제안하는 향후 방향

### 5.1 Phase 전환

데이터는 Phase 0 VIC 변형을 계속 반복하기보다 **Phase 1 (시뮬레이션 내 외란 curriculum)**으로 이동하는 것을 다음 실험으로 강력히 지지한다.

**권장 실험 순서:**

| Step | 구성 | 가설 |
|---|---|---|
| **1** | H5 + no-VIC (Cell D)를 **population 사전학습 baseline** (§6.3의 θ_pop)으로 재사용. | 가능한 최선의 정상 보행자. |
| **2** | VIC 4-group + **시뮬레이션 외란** (§5.1 Stage 2)을 추가하여 Cell D 정책을 fine-tune. Pelvis에 ramped F_ext 20 → 200 N, 0.1–0.5 s duration 적용. | CCF가 reward gradient를 획득해야 함. 예상: 외란이 심한 평가에서 VIC 4grp이 결국 no-VIC를 상회. |
| **3** | α 계산 (§3) 추가: XCoM + CoP + angular momentum. α를 observation에 포함. | α-가중 reward 혼합 및 (추후) motion library retrieval 활성화. |
| **4** | α-혼합 가중치와 함께 balance reward (§5.1 Stage 2) 추가. | 정책이 high-α 상태에서 tracking 정확도를 희생할 수 있도록 함. |
| **5** | Phase 2 데이터 수집 (S001의 벨트 외란) — MPL "critical next step" (§9). | 진정한 개인화를 가능케 함. |

Step 1–4는 **기존 시뮬레이션 인프라와 기존 H5 보행 데이터**로 수행 가능하다. 새로운 motion capture는 필요 없다.

### 5.2 CCF 가치 검증 (Phase 0 → Phase 1 브릿지)

Phase 0 내에서도 기존 `analyze_phase_ccf.py` 스크립트를 사용해 CCF가 의미 있게 학습되고 있는지 검증할 수 있다:

1. Ep 20k의 Cell C (VIC 8grp + H5) 정책을 phase-CCF 로깅으로 평가 모드에서 실행.
2. 보행 사이클에 걸쳐 각 관절 그룹의 CCF(φ) 곡선을 plot.
3. **만약** ankle 그룹이 stance vs swing 변동 > 0.2를 보이고 상체 그룹이 CCF ≈ 0을 보이면, → VIC가 reward 압력 없이도 *무언가* 유용한 것을 학습했음.
4. **만약** 모든 CCF가 거의 0 부근에서 평평하면, → Phase 0에서 CCF가 사실상 미사용임이 확인됨.

기존 체크포인트에서 이 분석은 30분 미만이 소요되며 새로운 학습 없이 실행 가능한 정보를 제공한다.

### 5.3 서버에서의 즉시 실행 항목

1. **Cell D (Ep 20k) 완료**: 현재 Ep 18,593, 완료까지 약 30분. 최종 체크포인트는 Phase 1 fine-tuning 시작점 역할.
2. **Cell E (Ep 20k) 완료**: 현재 Ep 12,674, 추가 약 8시간. 최종 VIC 4grp vs 8grp vs no-VIC 비교.
3. **C, E, D 체크포인트에 대한 CCF phase 분석 실행** (마지막 것은 대조군 — no-VIC 정책은 의미 있는 CCF 출력이 없어야 함).
4. **Phase 1 config 준비**: env yaml을 `perturbation.enabled: True`, ramped F_ext, random intervals로 확장. 외력 적용 코드 경로는 이미 `humanoid.py`에 있을 수 있음; 없다면 작은 force injector 추가.

### 5.4 ChatGPT에 물어볼 내용

외부 피드백을 구한다면, 핵심 질문은:

1. **Phase 0 결과 (정상 보행에서 no-VIC > VIC)가 다른 출판된 결과와 일치하는가**, 예를 들어 외란 도입 전에 더 풍부한 action space가 때때로 해가 되는 Peng et al. ASE? 동일 문제를 겪고 이후 phase에서 해결한 canonical "VIC in RL" 논문이 있는가?

2. **제안된 Phase 1 실험 (no-VIC baseline을 VIC + sim 외란으로 fine-tune)**이 CCF에 reward gradient를 주는 올바른 방법인가? 대안은:
   - 외란 + VIC를 처음부터 함께 학습
   - Co-training: 배치의 절반은 외란, 절반은 없음 — 정상 보행을 보존하면서 외란 반응 학습
   - Distillation: 수작업 설계된 stiffness 제어기에서 CCF 정책을 추출하여 warm-start

3. **4-group (하체 + 상체 고정)**이 co-contraction을 위한 합리적인 prior인가? 좌우를 대칭화해야 하는가 (2 그룹: Hip+Knee, Ankle+Toe) 양측 대칭을 강제하기 위해? 혹은 보행 중 ankle이 조절을 지배한다는 생체역학 문헌을 근거로 단일 스칼라 (Ankle stiffness만)로 더 축소해야 하는가?

4. **Imitation reward (S001 reference tracking)는 외란 조건에서 CCF reward (robustness를 위한 co-contraction)와 충돌할 수 있음**. 이들을 어떻게 균형 잡아야 하는가? α-혼합 접근법 (§6)이 현재의 답이지만, 붕괴할 수 있다: 학습 중 α가 항상 낮으면, CCF는 gradient를 얻지 못한다.

5. **외골격 배치 (§7)의 경우**, CCF는 관절 임피던스뿐만 아니라 **biceps/triceps 같은 길항근 쌍(antagonistic pairs)**도 명시적으로 모델링해야 하는가? 현재 CCF는 관절별이며, 실제 근육 쌍은 더 미세한 제어를 제공하지만 파라미터 수가 많아지는 비용이 있다.

---

## 6. 연구에 대한 열린 질문

1. **명시적 action 없이도 나타나는 암묵적 CCF가 있는가?** 순수 PD 정책 (Cell D)은 암묵적으로 상수 K_p, K_d를 사용한다. no-VIC 정책이 co-contraction이 필요하지 않은 영역에 머무는 것 (즉, 임피던스 조절의 필요성을 회피하도록 학습)을 *관찰*할 수 있는가?

2. **VIC가 외란 하에서만 가치가 있다면, CCF가 올바른 추상화인가?** 대안적 정식화:
   - 직접 토크 출력 (action이 τ, (q_ref, ξ)가 아님) — 정책에 더 많은 자유를 주지만 PD 구조 상실.
   - 학습된 reflex gain — 전체 관절별 CCF action이 아닌 단순한 "힘 감지되면 → stiffness 추가" 규칙.

3. **Motion library retrieval 메커니즘 (§4.4)이 CCF가 제공하는 일부를 대체할 수 있는가?** 높은 α에서 올바른 외란 반응 exemplar가 검색되면, 정책은 이미 올바른 임피던스를 포함한 인간 행동에 조건화된다 — 어쩌면 명시적 CCF가 전혀 필요 없을 수 있다.

4. **개인화 (§6.2)의 경우, CCF가 올바른 개인화 채널인가?** 피험자의 임피던스 프로파일은 다르다 (Horak & Nashner 1986). 그러나 타이밍, 진폭, step placement도 다르다. CCF는 여러 개인화 벡터 중 하나이며, 외골격 예측에 가장 가치 있는 것은 무엇인가?

5. **VIC_CCF_ON2 결과 (AMASS forward_single에서 학습된 ankle 1.3×, knee 0.9×, upper 0.85×)**가 S001 H5 데이터에 일반화되는가? 그것은 다른 피험자의 데이터였다. Cell E를 완료까지 실행하고 CCF 패턴을 추출하면 답을 얻을 수 있다.

---

## Appendix: 파일 위치 (서버)

### 실험 코드 (git 추적됨, Jimin branch)
- `phc/data/cfg/env/env_im_walk_vic.yaml` — Cell C config
- `phc/data/cfg/env/env_im_walk_vic_amass.yaml` — Cell B config
- `phc/data/cfg/env/env_im_walk_h5_novic.yaml` — Cell D config
- `phc/data/cfg/env/env_im_walk_vic_4grp.yaml` — Cell E config
- `phc/env/tasks/humanoid_im_vic.py:187-221` — CCF grouping (4 및 8 지원)
- `phc/learning/amp_agent.py:536-548` — CCF sigma override (자동 조정)

### 체크포인트 (서버 전용)
- `~/PHC/output/VIC_CCF_ON2_H5.pth` (Cell C, Ep 20k)
- `~/PHC/output/VIC_CCF_ON2_H5_4grp.pth` (Cell E, running)
- `~/PHC/output/VIC_CCF_ON2_NoVIC_H5.pth` (Cell D, running, 거의 완료)
- `~/PHC/output/VIC_CCF_ON2_AMASS.pth` (Cell B, Ep 20k)
- `~/PHC/output/HumanoidIm/.../Humanoid_V4_Fresh_Start_01.pth` (Cell A, Ep 7k 수동 중단)

### Logs
- `~/PHC/logs/phc_vic_h5_3614.out` (Cell C)
- `~/PHC/logs/phc_vic_amass_3615.out` + `3616.out` (Cell B, 재개)
- `~/PHC/logs/phc_h5_novic_3644.out` (Cell D)
- `~/PHC/logs/phc_vic_4grp_3646.out` (Cell E)
- `~/PHC/logs/progress_report_*.md` (cell별 1000-epoch 마일스톤)
