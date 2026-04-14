아래는 그대로 복붙해서 저장할 수 있는 **Markdown 문서 초안**이다.
요청한 범위대로 **MPL 방법론만** 다루고, 특히 **reflex-aware 업그레이드**, **G+R 구조의 한계와 개선**, **perturbation data의 활용 방식**, **개인화와 위기상황 반응 구현**에 집중해서 정리했다.

---

# Reflex-Aware Motion Plan Layer (MPL) 방법론 고도화 제안

## CoM 입력 대응, 개인화, 외란·위기상황 반응을 위한 인간 agent 상위제어기 설계

## 0. 문서 목적

본 문서는 기존 Motion Plan Layer(MPL) 개념을 더 구체화하여, 아래 세 가지 요구를 동시에 만족하는 **simulation-ready human agent 상위제어기** 방법론을 제안한다.

1. **시뮬레이션에서 임의의 CoM 움직임 입력이 주어져도 넘어지지 않고 동작을 생성할 수 있어야 한다.**
2. **한 명의 다양한 동작 데이터로부터 학습되어, 그 사람 특유의 gait style, 습관, 반응 양식이 드러나야 한다.**
3. **로봇 힘, 외력, 예기치 않은 disturbance 같은 위기상황에서의 반응이 핵심적으로 구현되어야 한다.**

기존 MPL 문서가 제안한 핵심 아이디어는, 근골격/반사/생리학적 하위 모델 위에 coordination-level 상위제어를 얹고, 상위 출력으로 **reference motion + low-dimensional impedance**를 생성하는 것이다. 이 구조는 직접 muscle activation을 action으로 두는 방식보다 개인화와 해석 가능성에 유리하고, fixed-PD trajectory control보다 stiffness/compliance 표현력이 높다. 또한 intent → coordination/timing/stiffness → reflex/synergy → muscle activation이라는 인간 운동제어 계층과도 잘 맞는다.   

하지만 현재의 **G + R** 구조만으로는 “사람다운 nominal walking”과 “작은 residual correction”은 가능해도, **인간의 반사적 균형 조절**과 **위기상황에서의 step relocation / recovery strategy**까지 충분히 다루기 어렵다. 따라서 본 문서는 MPL을 단순한 generator-residual 구조가 아니라, **balance-aware, reflex-aware, recovery-aware 계층형 구조**로 업그레이드하는 방안을 제안한다.

---

## 1. 현재 MPL의 장점과 한계

기존 내부 문서에서 MPL은 크게 두 모듈로 정리되어 있다.

* **G (Nominal Gait Generator)**: state, command, reference window를 받아 nominal trajectory와 impedance prior를 생성
* **R (Residual / Impedance Policy)**: nominal 출력에 대한 보정량을 생성하여 물리적 안정성과 외란 대응을 담당

또한 출력은 q_ref와 5축 impedance latent로 구성되며, 이 latent는 stance sagittal stiffness, swing compliance, landing damping, lateral stabilization, overall stiffness scale 같은 축으로 해석 가능하게 설계되어 있다. 학습은 Stage A(offline supervised), Stage B(freeze G + RL for R), Stage C(joint fine-tuning)로 나뉜다.    

이 구조의 장점은 분명하다.

* nominal gait를 데이터에서 배우고 residual을 RL로 보정하므로 학습이 단일 monolithic 정책보다 효율적이다. 
* 개인차를 coordination-level style과 lower-layer physiology로 분리할 수 있어 personalization에 유리하다. 
* low-dimensional impedance 표현으로 인간의 능동적 stiffness/compliance 조절을 모델링할 수 있다. 

그러나 한계도 뚜렷하다.

### 1.1 G + R만으로는 reflex를 충분히 설명하기 어렵다

Residual 정책 R이 작은 perturbation에 대해 반응할 수는 있지만, 그 자체가 곧 **인간의 반사 작용**을 의미하지는 않는다. 인간의 반응은 단순한 “policy residual”이 아니라, 짧은 지연과 위상 의존성을 가지는 structured feedback에 가깝다. 내부 연구 프로그램에서도 wearable humanoid simulator는 passive mechanics, reflexive response, voluntary motion을 모두 포함해야 하며, explicit reflex structure가 pure black-box human model보다 perturbation fidelity에 유리하다고 정리하고 있다.  

### 1.2 nominal gait library와 perturbation clips를 한데 섞으면 style이 흐려질 수 있다

명령 조건부 motion library는 nominal gait family를 학습하는 데 유리하지만, 여기에 perturbation response나 crisis recovery를 그대로 섞으면 “평상시 보행 style”과 “위기상황 response style”이 한 분포로 뭉개질 위험이 있다. 즉, subject style이 보존되는 것이 아니라 평균화될 수 있다.

### 1.3 CoM 입력을 그대로 추종하는 것은 위험하다

사용자가 말한 “임의의 CoM 움직임 입력”은 상위 command로서는 유용하지만, 이를 곧바로 tracking target으로 삼으면 물리적으로 불가능한 명령을 따라가다가 넘어질 수 있다. 실제로 centroidal dynamics 기반 접근에서는 upper policy가 “where to move”를 생성하되, lower physical layer가 dynamics mismatch와 feasibility를 보정해야 한다. payload/generalization 연구에서도 nominal intention layer와 physical compensation layer를 분리해야 out-of-distribution dynamics와 push recovery가 안정화된다고 보고된다.   

---

## 2. 핵심 제안: G + R를 넘는 Reflex-Aware MPL

본 문서는 MPL을 다음 4계층으로 재정의할 것을 제안한다.

[
\boxed{
C ;\rightarrow; G ;\rightarrow; R_f ;\rightarrow; R_s
}
]

여기서 각 모듈은 다음 역할을 가진다.

### 2.1 C: CoM Feasibility / Balance Filter

상위에서 들어온 CoM-related command 또는 locomotion intention을 그대로 추종하지 않고, 현재 support condition과 XCoM 상태를 고려하여 **실행 가능한 intention**으로 정규화한다.

즉, 사용자가 원하는 “움직이고 싶은 방향/CoM 변화”와 “현재 넘어지지 않을 수 있는 범위” 사이를 연결하는 필터다.

### 2.2 G: Nominal Style-Conditioned Gait Generator

subject-specific nominal gait manifold를 생성한다.
평상시 보행 스타일, cadence, stride pattern, baseline impedance bias를 주로 담당한다.

### 2.3 (R_f): Fast Reflex Layer

짧은 지연을 가진 structured feedback module이다.
주요 역할은 ankle/stance stiffness, early corrective foot placement bias, contact-aware stabilization 등 **빠른 반응**이다.

이 모듈은 “큰 신경망이 다 알아서 하도록” 두는 것이 아니라, **위상 의존적, 지연 포함, 저차원 structured feedback**으로 설계하는 것이 핵심이다.

### 2.4 (R_s): Strategic Recovery Layer

큰 perturbation이나 crisis 상황에서 **step relocation, timing reset, cadence change, temporary bracing** 등을 담당한다.
이는 reflex보다 느리지만 더 큰 범위의 회복 전략을 담당한다.

즉, 제안 구조는 다음과 같이 해석된다.

* **C**: “이 CoM 의도는 지금 가능한가?”
* **G**: “평소 이 사람이라면 이렇게 걷는다.”
* **(R_f)**: “갑자기 밀리면 먼저 이렇게 반응한다.”
* **(R_s)**: “그래도 위험하면 다음 발을 이렇게 옮겨 살린다.”

---

## 3. 왜 이런 분해가 필요한가

### 3.1 인간 균형 제어는 단일 메커니즘이 아니다

보행 중 balance는 하나의 제어기가 아니라 여러 메커니즘의 결합이다. 특히 lateral perturbation에 대해서는 **lateral ankle mechanism**, **foot placement mechanism**, 그리고 더 늦은 **push-off correction**이 상호보완적으로 작동한다는 점이 반복적으로 보고되었다. 초기 correction은 ankle/CoP 쪽이 더 빠르고, 이후에는 foot placement가 크게 작용하며, 이들 메커니즘은 서로 trade-off를 이룬다. ([The Company of Biologists][1])

따라서 하나의 residual 네트워크가 모든 것을 end-to-end로 처리하게 하기보다는, 시간척도와 기능이 다른 보정 메커니즘을 분리하는 것이 더 타당하다.

### 3.2 reflex는 “joint residual”보다 “phase-dependent delayed feedback”에 가깝다

walking balance control은 CoM position/velocity 기반의 delayed feedback으로 잘 설명되며, 이 feedback gain은 gait phase에 따라 달라진다. 이런 구조는 neuromechanical model을 확장하는 데도, exoskeleton-like assistance를 다루는 데도 유용하다고 제안되어 왔다. ([PLOS][2])

즉, reflex를 구현하려면 단순한 `R(x)`보다 아래 형태가 더 자연스럽다.

[
\Delta u_t^{f}
==============

M(\phi_t),K_f(z_{\text{reflex}}),\bigl(y_{t-\tau_f} - y_{t-\tau_f}^{nom}\bigr)
]

여기서

* (\phi_t): gait phase
* (M(\phi_t)): phase-dependent masking/modulation
* (K_f): subject-specific reflex gains
* (y_t): CoM/XCoM/contact/step-state 기반 feedback state
* (\tau_f): reflex delay

이 구조는 black-box residual보다 해석 가능하고, perturbation data로 parameter identification도 가능하다.

### 3.3 nominal intention과 physical compensation을 분리하면 generalization이 좋아진다

내부의 physics-augmented RL 자료는 nominal intention layer와 physical action/compensation layer를 분리했을 때, payload 변화와 external perturbation에 대한 push recovery가 end-to-end RL보다 크게 향상된다고 보고한다. 특히 nominal policy는 “what gait pattern to adopt”만 담당하고, lower layer가 dynamics mismatch를 보정할 때, unseen condition에 대한 generalization이 커진다.    

MPL에도 같은 논리를 적용할 수 있다.
즉, **G는 nominal style/intention**, **(R_f)와 (R_s)는 physical/reactive compensation**을 담당하게 해야 한다.

---

## 4. 요구 조건을 만족시키기 위한 제안 구조

## 4.1 Requirement 1: 임의의 CoM 입력에서도 넘어지지 않아야 함

이 요구는 “CoM trajectory를 무조건 따라라”가 아니라, **CoM intention을 받고도 물리적으로 feasible한 방식으로 balance-preserving motion을 생성하라**로 해석해야 한다.

이를 위해 먼저 XCoM/ICP 기반 stability state를 정의한다.

[
\omega_t = \sqrt{\frac{g}{h_t}}
]

[
\xi_t = p^{CoM}_t + \frac{\dot p^{CoM}_t}{\omega_t}
]

여기서 (\xi_t)는 XCoM 또는 capture-like state다.
support polygon을 (S_t)라 하면, 불안정도 스칼라 (\alpha_t)를 다음과 같이 둔다.

[
\alpha_t
========

\mathrm{clip}
\left(
\frac{\max(0, d(\xi_t, S_t))}{d_{\mathrm{ref}}},
0,1
\right)
]

* (d(\xi_t, S_t)): XCoM가 support polygon 밖으로 얼마나 벗어났는지 나타내는 signed distance
* (\alpha_t=0): 안정
* (\alpha_t \to 1): 넘어질 위험 큼

이 개념은 내부 MPL 계획에서도 stability margin alpha와 XCoM 기반 blending signal로 이미 제안되어 있다.  

이제 상위 입력 (u_t^{user})를 feasibility filter (C)를 통해 실행 가능한 형태로 바꾼다.

[
\tilde u_t
==========

\arg\min_{u \in \mathcal U_{\mathrm{feas}}(x_t)}
|u-u_t^{user}|*Q^2
+
\lambda*{\mathrm{stab}} R_{\mathrm{stab}}(x_t,u)
]

의미는 간단하다.

* 사용자가 원하는 CoM movement intention을 가능한 한 존중하되
* 현재 안정성 조건을 깨지 않는 가장 가까운 command로 projection한다.

따라서 requirement 1을 만족하려면, **CoM input을 그대로 G에 넣는 것이 아니라 C를 통해 정규화한 뒤 G에 넣어야 한다.**

---

## 4.2 Requirement 2: 한 명의 다양한 동작 데이터로 개인화되어야 함

개인화는 단순히 gait clip 몇 개를 subject-filtered library에 넣는 수준보다 더 구조적으로 다뤄야 한다.

기존 문서도 subject-filtered library와 style discriminator를 통해 per-subject style preservation이 가능하다고 정리한다. 
하지만 여기서는 perturbation response까지 포함해야 하므로 style을 다음처럼 factorized representation으로 분리하는 것이 더 적절하다.

[
z_{\mathrm{style}}
==================

[z_{\mathrm{gait}},;
z_{\mathrm{imp}},;
z_{\mathrm{reflex}},;
z_{\mathrm{crisis}}]
]

* (z_{\mathrm{gait}}): cadence, stride, preferred timing
* (z_{\mathrm{imp}}): baseline stiffness/compliance bias
* (z_{\mathrm{reflex}}): short-latency feedback gains, delays, asymmetry
* (z_{\mathrm{crisis}}): crisis 상황에서 step으로 버티는지, stiffness로 버티는지, 회복 보폭을 크게 쓰는지 등의 선호

즉, “개인의 style”은 nominal gait 하나가 아니라, **평상시 gait + baseline impedance + reflex profile + recovery preference**의 묶음으로 봐야 한다.

### 데이터 라이브러리도 분리해야 한다

모든 데이터를 하나의 motion library에 섞지 말고 최소 세 묶음으로 나누는 것이 좋다.

[
\mathcal L = \mathcal L_{nom} \cup \mathcal L_{bal} \cup \mathcal L_{pert}
]

* (\mathcal L_{nom}): steady walking, speed variation, turning, start-stop
* (\mathcal L_{bal}): voluntary CoM shift, narrow-base control, in-place stepping
* (\mathcal L_{pert}): external push/pull, robot torque pulse, asymmetric assistance, unexpected release 등 perturbation-response 데이터

이 분해가 중요한 이유는 다음과 같다.

* **G**는 주로 (\mathcal L_{nom})에서 nominal style을 배운다.
* **(R_f)**는 (\mathcal L_{bal}) + (\mathcal L_{pert})의 초기 response window에서 짧은 지연 반응을 배운다.
* **(R_s)**는 (\mathcal L_{pert}) 전체 window에서 recovery strategy를 배운다.

이렇게 해야 perturbation data를 넣더라도 nominal style이 흐려지지 않는다.

---

## 4.3 Requirement 3: 로봇 힘·외력·위기상황 반응이 핵심이어야 함

이 요구는 사실 MPL에서 가장 중요한 부분이다.
일반적인 nominal gait imitation은 robot-human interaction에서 충분하지 않다. 위기상황 response는 작은 residual correction이 아니라, 때로는 **새로운 step placement**, **contact reorganization**, **timing reset**을 필요로 한다.

최근 humanoid recovery 연구도, fall recovery를 명시적으로 state distribution에 포함시키고, capture-point/CoM/momentum 같은 balance-aware structure를 critic이나 reward에 넣어야 robust recovery가 가능하다고 보고한다. 또한 단순 tracking만으로는 highly dynamic maneuver와 contact-rich recovery에 한계가 있으며, fail-state recovery를 함께 학습시키면 robustness가 크게 좋아진다. ([arXiv][3])

따라서 crisis response는 아래 두 레벨로 분리해야 한다.

### Fast level: (R_f)

* stance ankle/hip stiffness 급상승
* CoP shift
* swing leg immediate flexion bias
* early foot-placement bias
* contact-aware damping modulation

### Strategic level: (R_s)

* next footstep relocation
* step timing reset
* cadence acceleration or braking
* temporary stop / brace action
* large step-width expansion
* recovery-specific impedance reconfiguration

즉, requirement 3을 위해서는 **R을 하나로 두지 말고 (R_f + R_s)**로 나누는 것이 더 적절하다.

---

## 5. 제안하는 최종 수식 구조

상위 실행 변수는 아래처럼 둔다.

[
u_t = [q^{ref}_t,; \kappa_t,; p^{step}_t,; T^{step}_t]
]

* (q^{ref}_t): reference joint target
* (\kappa_t): low-dimensional impedance latent
* (p^{step}_t): next step placement bias
* (T^{step}_t): next step timing adjustment

전체 제어는 다음처럼 합성한다.

[
u_t
===

u_t^G
+
g_f(\alpha_t),\Delta u_t^{f}
+
g_s(\alpha_t),\Delta u_t^{s}
]

여기서

* (u_t^G = G(x_t^{hist}, \tilde u_t, z_{\mathrm{gait}}, r_{t:t+K}^{nom}))
* (\Delta u_t^f = R_f(\cdot)): fast reflex correction
* (\Delta u_t^s = R_s(\cdot)): strategic recovery correction
* (g_f, g_s): instability에 따른 게이팅 함수

예를 들어

[
g_f(\alpha) = \alpha
]

[
g_s(\alpha) = \sigma(k(\alpha-\alpha_s))
]

처럼 두면,

* 약간 흔들릴 때는 (R_f)만 주로 작동
* 많이 위험할 때는 (R_s)까지 크게 개입

하는 구조가 된다.

---

## 6. Reflex를 실제로 어떻게 구현할 것인가

## 6.1 RL만으로 reflex를 “알아서” 배우게 두는 것은 추천하지 않는다

RL residual이 perturbation에 반응할 수는 있다. 하지만 그것만으로는

* 짧은 latency
* phase dependence
* contact-conditioned response
* person-specific response magnitude

를 안정적으로 얻기 어렵다.

내부 reflex controller 개선 문서도 reflex는 stretch, reciprocal inhibition, load reflex, delay differentiation, spindle nonlinearity, phase modulation처럼 **구조화된 요소**가 필요하다고 정리한다. 특히 delay=0은 수치 불안정을 만들고, position component가 없는 velocity-only reflex는 정적 안정화가 어렵다고 명시되어 있다.   

따라서 reflex-aware MPL에서는 (R_f)를 다음처럼 설계하는 것이 좋다.

### (R_f)의 입력

* gait phase (\phi_t)
* pelvis/CoM position and velocity
* XCoM / capture margin
* current contact state
* recent external wrench / robot interaction torque
* nominal state deviation (e_t = y_t - y_t^{nom})

### (R_f)의 내부 구조

* phase mask (M(\phi_t))
* subject-specific gain matrix (K_f(z_{reflex}))
* explicit delay buffers (\tau_f)
* small decoder to joint-space corrections and impedance channels

### (R_f)의 권장 저차원 출력 채널

1. lateral ankle / CoP correction
2. stance sagittal stiffness boost
3. swing compliance / clearance bias
4. push-off modulation
5. immediate foot-placement bias

즉, (R_f)는 큰 unconstrained policy가 아니라 **delay-aware structured correction head**여야 한다.

---

## 6.2 Perturbation data는 필요한가?

### 결론

* **generic disturbance rejection만 원하면** simulation RL만으로도 어느 정도 가능하다.
* **subject-specific human-like perturbation response를 원하면**, 실제 perturbation-response data가 **강하게 권장된다**.

이는 내부 parameter identification 문서와도 잘 맞는다. 해당 문서는 perturbation data로부터 reflex delay와 stretch gain을 추정하는 `PerturbationAnalyzer`를 제안하고 있으며, perturbation 분석이 단순 gait time-series fitting보다 reflex parameter 식별에 더 직접적일 수 있다고 적고 있다.  

즉, “반사 작용까지 개인화된 agent”를 만들고 싶다면, 보행 데이터만으로는 부족하고 다음이 필요하다.

* perturbation onset 시점
* perturbation 방향/크기
* response onset latency
* response magnitude
* recovery step placement
* re-stabilization timing

이 정보가 있어야 (z_{reflex})와 (z_{crisis})를 실제 subject에 맞게 맞출 수 있다.

---

## 6.3 그렇다면 perturbation 데이터를 nominal motion library에 다 넣어야 하나?

그대로 한데 섞는 것은 추천하지 않는다.

그 이유는 nominal gait prior가 “걷는 방식”을 배우는 모듈인데, perturbation clip은 평상시 gait가 아니라 **event-conditioned recovery fragment**이기 때문이다. 이 둘을 동일 분포로 취급하면, nominal generator가 이상하게 보수적이거나 흔들리는 gait를 평균적으로 출력할 수 있다.

따라서 추천 방식은 다음과 같다.

### 방식 A: library 분리

* `L_nom`: 평상 gait
* `L_pert`: perturbation-response fragment
* `L_bal`: voluntary balance control fragment

### 방식 B: discriminator도 분리

* (D_{nom}): nominal style 보존용
* (D_{rec}): recovery style 보존용

### 방식 C: retrieval key도 분리

* nominal retrieval key: speed, turning rate, gait phase
* perturbation retrieval key: phase, support leg, perturbation direction, perturbation magnitude, CoM/XCoM state

즉, perturbation data는 “같은 library 안의 또 다른 gait clip”이 아니라,
**event memory / recovery prior**로 다루는 것이 맞다.

---

## 7. G와 R를 나누는 것이 정말 효율적인가?

이 질문은 매우 중요하다.
답은 **“그냥 G+R이면 무조건 효율적”이 아니라, 무엇을 G에 맡기고 무엇을 R에 맡기느냐에 달려 있다”**이다.

### 7.1 G가 담당해야 할 것

G는 low-entropy, 반복적, subject-specific한 nominal manifold를 담당해야 한다.

* cadence
* stride pattern
* preferred joint coordination
* nominal impedance bias
* speed/turning에 따른 평상 gait family

이 부분은 데이터를 통해 offline으로 먼저 잡는 것이 sample-efficient하다. 내부 MPL 문서도 G는 command-conditioned motion library로부터 offline supervised로 학습하고, command label은 motion 자체로부터 계산한다고 명시한다.  

### 7.2 R가 담당하면 안 되는 것

R가 nominal gait 자체를 다 새로 만들게 하면 효율이 떨어진다.
그러면 residual이 아니라 사실상 monolithic policy가 되어버린다.

### 7.3 residual around reference는 실제로 exploration을 줄인다

최근 humanoid motion tracking 연구에서도 reference pose 주변에서 residual을 학습하는 것은 exploration space를 줄여 sample efficiency를 높이는 데 유리하다고 보고된다. 또한 physically consistent reference를 먼저 정리한 후 policy를 학습하는 것이 stable long-horizon control에 유리하다는 흐름도 있다. ([arXiv][3])

따라서 **G는 강한 prior**, **R은 제한된 correction**으로 두는 것이 맞다.
다만 여기서 R을 다시 (R_f)와 (R_s)로 나눠야 효율이 유지된다.

* (R_f): 작은 수의 gain/delay/channel만 학습
* (R_s): crisis subset에서만 주로 학습

이렇게 해야 한 모듈이 모든 반응을 다 책임지는 비효율을 피할 수 있다.

---

## 8. 권장 학습 파이프라인

## Stage 0. Lower execution layer 준비

현재 v1에서는 joint-space + low-dimensional impedance로 시작하는 것이 가장 현실적이다. 내부 MPL 계획도 IsaacGym/PHC 상에서 q_ref와 5축 impedance latent를 출력하는 구조로 정리되어 있다.  

장기적으로는 CALM 같은 neuromuscular lower layer로 확장할 수 있지만, v1에서는 joint-space가 적절하다.

---

## Stage 1. Subject-specific nominal prior (G) 학습

데이터: (\mathcal L_{nom})

입력:

* state history
* feasible command (\tilde u_t)
* future reference window

출력:

* (q_t^G)
* (\kappa_t^G)

loss:

* reconstruction
* temporal smoothness
* command consistency
* style consistency

이 단계는 기존 MPL의 Stage A와 동일하되, command를 단순 speed/yaw뿐 아니라 CoM-intention normalization 결과까지 포함하도록 확장한다.

---

## Stage 2. Fast reflex module (R_f) 학습

데이터: (\mathcal L_{bal}) + (\mathcal L_{pert})의 초기 response windows

목표:

* perturbation onset 후 짧은 구간의 corrective response를 학습
* 가능하면 subject-specific gain/delay identification 병행

학습 방식:

1. perturbation 데이터에서 response latency와 magnitude를 추출
2. (z_{reflex}), (K_f), (\tau_f)를 fitting
3. structured decoder로 joint correction / impedance correction 채널에 매핑

이 부분은 내부 reflex identification 및 controller enhancement 방향과 직접 연결된다.   

---

## Stage 3. Strategic recovery module (R_s) 학습

데이터: (\mathcal L_{pert}) 전체 + simulation disturbance curriculum

목표:

* step relocation
* timing reset
* cadence adjustment
* recovery-specific impedance reconfiguration

학습 방식:

* imitation + RL 혼합
* RL critic에는 privileged balance state 제공 가능

  * CoM
  * XCoM / capture margin
  * support polygon distance
  * centroidal momentum

최근 recovery RL 논문들도 capture-point, CoM-state, momentum 같은 balance-aware 정보가 recovery 학습에 중요하다고 보고한다. ([arXiv][4])

---

## Stage 4. Joint fine-tuning

전체를 end-to-end로 조금만 조정한다.

단, 이때는 nominal style이 깨지지 않도록

* (G)에는 작은 learning rate
* (D_{nom}), (D_{rec}) style regularization
* motion prior regularization
* crisis subset oversampling

을 함께 넣어야 한다.

내부 문서도 G를 먼저 freeze하고 R을 학습한 뒤, 마지막에 작은 LR로 G까지 unfreeze하는 Stage C를 권장한다. 

---

## 9. 실제 데이터 수집 권장안 (하체만)

상지는 현재 제외한다고 했으므로, lower-body only 기준으로 정리하면 다음이 적절하다.

### 9.1 Nominal set

* 여러 속도의 level walking
* start / stop
* turning
* stride length variation
* cadence variation

### 9.2 Balance set

* in-place weight shift
* voluntary CoM shift
* narrow-base stepping
* small AP / ML sway control

### 9.3 Perturbation set

* pelvis AP/ML push
* treadmill belt acceleration / deceleration
* robot assist torque pulse
* asymmetric robot assistance
* unexpected assistance release
* stance / swing phase별 perturbation

### 9.4 권장 센서

* lower-body kinematics
* pelvis IMU
* foot contact / FSR / GRF
* robot joint torque / interaction force
* optional EMG

여기서 중요한 것은 **EMG가 있으면 좋지만 v1 필수는 아니라는 점**이다. subject-specific exoskeleton response 연구들에서는 kinematic response는 상당히 예측 가능하지만 myoelectric response는 훨씬 어렵게 예측된다는 결과가 있어, v1에서는 kinematics/contact/interaction-based reflex style 학습이 현실적이다. EMG는 v2 muscle-space/CALM 통합 때 더 중요해진다. ([PMC][5])

---

## 10. V1과 V2의 구현 로드맵

## V1: Joint-space Reflex-Aware MPL

가장 먼저 구현할 버전이다.

구성:

* C: CoM/XCoM feasibility filter
* G: subject-specific nominal generator
* (R_f): delayed CoM/contact feedback + impedance/foot-bias channels
* (R_s): RL-based recovery planner
* lower layer: joint-space PD/impedance execution

실행식:

[
\tau_t = K_p(\kappa_t)(q_t^{ref} - q_t) - K_d(\kappa_t)\dot q_t
]

장점:

* PHC/IsaacGym 위에서 바로 구현 가능
* 현재 internal MPL plan과 가장 잘 이어짐
* perturbation data가 없어도 baseline 구축 가능
* perturbation data가 들어오면 (R_f, R_s)만 업데이트 가능

---

## V2: Neuromuscular / CALM-integrated MPL

장기 버전이다.

구성:

* 상위 구조는 동일
* lower layer만 CALM/ReflexController 기반으로 교체
* (R_f)는 q_ref residual뿐 아니라 reflex gain, delay, synergy activation bias도 출력 가능

즉,

[
a_{human} = f_{passive} + f_{reflex} + \pi_{voluntary}
]

라는 내부 hybrid human model 관점을 실제 실행 구조에 연결하는 버전이다.  

이때는 internal reflex enhancement 문서의 spindle nonlinearity, delay differentiation, load reflex, phase-dependent modulation을 직접 사용할 수 있다. 

---

## 11. 검증 지표

## 11.1 Requirement 1 검증: 임의 CoM command에도 넘어지지 않는가

* no-fall success rate
* recovery success rate
* XCoM/support-margin violation ratio
* feasible command projection ratio
* recovery time
* step placement error

## 11.2 Requirement 2 검증: 개인 style이 보존되는가

* cadence / stride / stance-swing ratio consistency
* baseline impedance profile similarity
* perturbation response latency similarity
* recovery step preference similarity
* subject ID classifier accuracy on generated clips
* nominal vs recovery style 분리 유지 여부

## 11.3 Requirement 3 검증: crisis response가 구현되는가

* robot force / external push에 대한 recovery success
* time-to-restabilization
* foot placement correction magnitude
* interaction torque spike
* contact consistency
* crisis 후 nominal gait 복귀 시간

내부 연구 계획서도 perturbation robustness, balance recovery time, interaction smoothness를 핵심 평가 항목으로 두고 있다. 

---

## 12. 최종 정리

### 핵심 답변 1

**기존의 G + R만으로는 MPL의 목표를 완전히 달성하기 어렵다.**
그 구조는 nominal gait와 작은 residual correction에는 적합하지만, reflex와 crisis recovery까지 다루기에는 부족하다.

### 핵심 답변 2

**인간의 반사 작용을 구현하려면, R을 단일 residual policy로 두기보다 (R_f)와 (R_s)로 나누고, (R_f)는 structured delayed feedback module로 설계하는 것이 맞다.**

### 핵심 답변 3

**subject-specific perturbation data는 strongly recommended다.**
다만 그 데이터를 nominal gait library에 단순 혼합하는 것은 좋지 않고, `L_nom / L_bal / L_pert`로 분리하여 각각 G, (R_f), (R_s) 학습에 다르게 사용해야 한다.

### 핵심 답변 4

**개인의 style은 gait 하나가 아니라 gait style + baseline impedance + reflex profile + crisis recovery preference의 묶음으로 정의해야 한다.**

### 핵심 답변 5

**“임의의 CoM 입력”은 직접 tracking target이 아니라 intention이다.**
이를 feasibility filter C로 정규화한 후에 G와 recovery layers가 받아 처리해야만 안정성과 일반성이 동시에 나온다.

---

## 결론

따라서 본 문서가 제안하는 업그레이드된 MPL은 다음과 같이 요약할 수 있다.

> **MPL은 더 이상 단순한 nominal generator + residual policy가 아니라,
> subject-specific nominal style, structured reflex, strategic recovery를 분리한
> balance-aware human agent 상위제어기여야 한다.**

즉 최종 구조는 다음이 적절하다.

[
\boxed{
\text{CoM feasibility filter } C
;\rightarrow;
\text{Nominal generator } G
;\rightarrow;
\text{Fast reflex } R_f
;\rightarrow;
\text{Strategic recovery } R_s
}
]

이 구조는

* 시뮬레이션에서 arbitrary CoM intention을 받더라도 안정성을 유지하고,
* 한 사람의 다양한 nominal/perturbation 데이터를 통해 그 사람의 style과 reflex profile을 개인화하며,
* 로봇 힘 및 외력에 대한 crisis response를 핵심 기능으로 구현하는

MPL의 보다 현실적이고 확장 가능한 방법론이라고 볼 수 있다.

---

원하면 다음 답변에서 이 문서를 이어서 **논문 proposal 형식**으로 바꾸거나, **수식/모듈/학습 단계만 더 정제한 버전**으로 다시 다듬어줄게.

[1]: https://journals.biologists.com/jeb/article/213/15/2655/9730/Balance-responses-to-lateral-perturbations-in "https://journals.biologists.com/jeb/article/213/15/2655/9730/Balance-responses-to-lateral-perturbations-in"
[2]: https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008369 "https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008369"
[3]: https://arxiv.org/html/2601.23080v1 "https://arxiv.org/html/2601.23080v1"
[4]: https://arxiv.org/html/2603.08619v1 "https://arxiv.org/html/2603.08619v1"
[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7653383/ "https://pmc.ncbi.nlm.nih.gov/articles/PMC7653383/"
