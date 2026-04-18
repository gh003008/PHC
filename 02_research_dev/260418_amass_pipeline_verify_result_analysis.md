# AMASS 파이프라인 검증 결과 분석 (260418)

## 1. 실험 배경

### 왜 이 실험을 했는가
Phase 0 H5 4-cell 매트릭스(C/D/E/F)에서 최대 reward ~605 / eps_len ~192(≈50% 성공률)로 정체되어, **gunhee의 VIC_PHASE(932/299, ≈100%)와 큰 격차**를 보였다. 이 격차의 원인을 두 가설로 분해:

1. **H5 데이터 변환 품질 문제** (상체 7개 관절 zero-mapping 등)
2. **프레임워크 드리프트** (amp_agent.py, humanoid_im_vic.py, motion_lib 등 최근 변경)

H5 데이터 수정 작업 전에 **프레임워크가 gunhee 시점 수준으로 여전히 작동하는지** 확인하기 위해, gunhee가 사용했던 AMASS `amass_isaac_walking_forward_single.pkl` 한 클립으로 동일 세팅 실험을 재현.

### 운영
- 서버 GPU 2장(idx 2/3)에서 병렬 실행 (Slurm)
- 환경/학습 config: `exp_config/forward_walking/260417_AMASS_VERIFY/`
- 총 20000 epochs, num_envs=512, PPO+AMP, VIC Stage 2, phase_obs=True
- 두 variant: 8-group (gunhee 오리지널 재현) vs 4-group (하체+상체 고정)

---

## 2. 학습 결과 (Training, stochastic rollout)

### Training log 마지막 지표 (epoch 19999 근방)

| Metric | AMASS_VERIFY_VIC8 | AMASS_VERIFY_VIC4 |
|---|---|---|
| av_reward | ~588 | ~552 |
| av_eps_len | ~194 | ~180 |
| 학습 시간 | ~11h 35m (종료 08:00) | ~11h 00m (종료 07:25) |

**초기 해석 착오 (자체 정정)**: 이 수치만 보고 "gunhee의 932/299 대비 63% / 65%로 크게 회귀"라고 진단했으나, gunhee의 VIC_PHASE(260312) 분석 문서 4장을 재확인한 결과:
> "학습 마지막 reward: ~590–660, 학습 마지막 eps_len: ~195–220 / Headless 평가 첫 에피소드: 932.44 reward, 299 steps"

즉 gunhee의 training-stochastic 수치 자체가 **~590–660 / ~195–220** 수준. 우리의 588/194는 정상 범위다.

---

## 3. Test-Greedy 평가 결과 (deterministic action)

로컬에서 `--test --epoch -1 --num_envs 1 --no_virtual_display`로 직접 실행.

| Metric | AMASS_VERIFY_VIC8 | AMASS_VERIFY_VIC4 | gunhee VIC_PHASE (참고) |
|---|---|---|---|
| 에피소드 수 (집계) | 7 | 15 | 1+ |
| eps_len | **299 / 299 (100%)** | **299 / 299 (100%)** | 299 |
| reward mean | **940.71** | **951.19** | 932.44 |
| reward std | 1.55 | **0.35** | — |
| reward min/max | 938.07 / 942.15 | 950.65 / 951.67 | — |

### 핵심 관찰
1. **두 variant 모두 100% 에피소드 완주** (299/299 steps).
2. **두 variant 모두 gunhee 기준치 재현/초과**. reward는 VIC8 +8.27, VIC4 +18.75.
3. **VIC4가 VIC8보다 일관되게 우수**: mean +10.48, std 4.4× 작음.
4. **파이프라인/프레임워크 건강함 확정**. gunhee 대비 회귀 없음.

### Training vs Test 메트릭의 괴리 (학습된 교훈)
- 학습 중 policy sigma: PD(-2.9, std=0.055) + CCF(-1.0, std=0.37, Stage 2)
- 학습 로그의 `av_reward` / `av_eps_len`는 **512 envs × stochastic exploration** 평균
- Test 모드는 deterministic (sigma=0) → "진짜" policy 성능
- **격차 원인**: Stage 2 CCF sigma=0.37이 큰 탐색 노이즈를 주입. 일부 env에서 조기 실패 → training 평균 눌림
- **향후 실험 분석 시 주의**: nominal 보행에서 training log는 policy의 실제 성능을 **상당히 과소평가**함. 최종 판단은 반드시 test-greedy로

---

## 4. VIC8 vs VIC4 비교

### 숫자 상의 차이
- VIC4: mean=951.19, std=0.35 (매우 일관됨)
- VIC8: mean=940.71, std=1.55 (variance 4.4×)
- **에피소드 처리 속도**: VIC4 15 eps / VIC8 7 eps (동일 ~4분) → **2× 빠름**

### 해석
8-group은 상체 2 dims(Upper_L, Upper_R) × CCF가 추가로 있는데, 전진 단순 보행 task에서는 상체 임피던스 조절이 reward 신호와 연결되지 않음. 즉 이 2 dims가:
- 추가 탐색 부담을 주고 (training 로그의 약한 신호)
- 학습 후에도 보상 없는 방향으로 약간의 variance를 계속 생성

**결론**: nominal 보행 한정으로 **4-group이 mainline, 8-group은 ablation 전용**이라는 260416 권장이 데이터로 뒷받침됨.

### 260416 계획 참조
`01_research_docs/260416_review_and_modified_plan_for_vic_impedance_action_kr.md` §4.3:
> **Mainline:** 하체 4-group VIC, 상체 고정. **Ablation 전용:** 8-group VIC.

---

## 5. 결론

1. **Phase 0 파이프라인 검증 종결** ✅
   - 프레임워크 드리프트 없음
   - AMASS forward_single에서 gunhee 수준 성능 재현

2. **H5 Phase 0 데이터의 50% 성공률은 프레임워크 탓이 아님**
   - H5 motion library 품질 (상체 zero-mapping 등)이 주요 원인으로 거의 확정
   - 단, 본 연구는 **H5 수정을 우선순위에서 내림** — 260416 계획대로 AMASS 기반 Phase 1으로 진입

3. **Nominal 보행에서 4-group이 8-group보다 우수**
   - std 4.4× 작고, 에피소드당 reward 10.48 높음
   - Phase 1 mainline actionization 확정: 4-group

4. **Training log 수치의 재해석 주의**
   - Stage 2 탐색 노이즈로 인해 training av_reward는 실제 성능을 ~35% 저평가
   - 앞으로 모든 VIC 실험의 합/불 판단은 test-greedy 기준

---

## 6. 다음 단계 — Phase 1 (260416 계획 기반)

**원칙** (260416 §10 요약):
> **기존 연구 방향은 유지하되, impedance action을 풀타임 보행 action에서 gated residual 적응 모듈로 강등하고, 실제로 중요해야 할 곳(외란, 상호작용, 개인화)에서 테스트하라.**

### 6.1 즉시 착수할 최소 변경
1. **Policy head 분리** (amp_network 수정): motor head + impedance head
2. **Impedance sigma 분리**: motor head `-2.9` 유지, impedance head `-2.5 ~ -3.0`로 낮춤
3. **Impedance 저역 통과 필터**: `delta_xi` LPF tau ≈ 0.15s
4. **Impedance gate `g`**: 외란/balance margin 입력, 정상 보행 시 닫힘
5. **Residual 형식**: `xi_eff = xi_nominal(phase) + g · LPF(delta_xi)`, `delta_xi ∈ [-0.3, 0.3]` (nominal)

### 6.2 Phase 1 실험 사다리 (AMASS 데이터 기반)
- **Stage 1**: AMASS_VERIFY_VIC4를 **parent checkpoint**로 freeze (951 reward / 299 steps 기준선)
- **Stage 2**: Residual VIC head 부착, 0 근처 초기화, 정상 에피소드 한정 검증 → **기존 성능 치명적 손실 없음** 확인
- **Stage 3**: Perturbation curriculum 도입 (pelvis push, 70% nominal / 30% perturbed)
- **Stage 4**: alpha / pelvis_acc / contact_anomaly 입력으로 gate 구동
- **Stage 5**: 외골격 관련 외란 (joint-level assistance/resistance torque)

### 6.3 Phase 1 성공 기준 (260416 §6.3)
1. no-VIC baseline 대비 **외란 회복 성공률 증가**
2. **회복 품질 개선** (회복 시간 / step 배치 / WBAM)
3. **정상 보행 손실 ≤ 10–15%**

### 6.4 현재 준비 상태
- ✅ Parent checkpoint 확보: `output/AMASS_VERIFY_VIC4.pth`
- ✅ Motion data: `sample_data/amass_isaac_walking_forward_single.pkl`
- ⬜ Perturbation 생성기 구현 필요
- ⬜ Residual policy head 분리 구현 필요
- ⬜ Gate 모듈 구현 필요
- ⬜ alpha / balance margin 계산 필요

---

## 7. 산출물

| 파일 | 위치 |
|---|---|
| VIC8 test log | `output/amass_verify_test_logs/vic8_test.log` |
| VIC4 test log | `output/amass_verify_test_logs/vic4_test.log` |
| VIC8 checkpoint | `output/AMASS_VERIFY_VIC8.pth` (server: `~/PHC/output/AMASS_VERIFY_VIC8.pth`) |
| VIC4 checkpoint | `output/AMASS_VERIFY_VIC4.pth` (Phase 1 parent 후보) |
| Training logs | server: `~/PHC/logs/amass_verify_vic{4,8}_3733,3734.out` |
| 학습 config 스냅샷 | `exp_config/forward_walking/260417_AMASS_VERIFY/` |

---

## 8. 참고

- `01_research_docs/260416_phase0_vic_feasibility_and_next_steps_kr.md` — Phase 0 H5 결과 분석
- `01_research_docs/260416_review_and_modified_plan_for_vic_impedance_action_kr.md` — Phase 1 재설계 계획 (Residual/gated VIC)
- `02_research_dev/260312_forward_walk_vic_phase_result_analysis.md` — gunhee의 원본 VIC_PHASE 결과 (참조 기준)
- `exp_config/forward_walking/260417_AMASS_VERIFY/README_FOR_SERVER.md` — 본 실험 서버 실행 가이드
