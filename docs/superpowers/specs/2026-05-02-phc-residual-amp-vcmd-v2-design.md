# PHC Residual + AMP + v_cmd Walker — v2 Design

**Date:** 2026-05-02
**Goal:** v_cmd ∈ [0.76, 1.23] m/s를 추종하며 자연스럽게 걷는 휴머노이드 정책 학습. v1 실패 (`02_research_dev/260502_res_amp_vcmd_failure_analysis.md`) 의 4가지 lesson을 반영한 재설계.
**Architecture:** Frozen phc_3 PNN + learnable residual (single 또는 soft-gated 3-MoE). HumanoidImMCP base (검증된 PNN 통합).
**Tech Stack:** IsaacGym, rl_games, PyTorch, AMP discriminator, Hydra config.

---

## 1. v1 vs v2 핵심 변경 (왜 다시 하는가)

| 영역 | v1 (실패) | v2 (이번) |
|------|-----------|----------|
| Base task | `HumanoidIm` + obs override 직접 구현 | **`HumanoidImMCP`** (검증된 multi-PNN 통합) |
| `r_survive` | 1.0 fixed (reward의 66%) → Goodharting | **제거** |
| Reward weights | r_track 0.5 / r_amp 0.3 / r_im 0.2 + r_survive 1.0 | r_track 0.5 / r_amp 0.3 / r_im 0.2 (사용자 명시: walking 안 풀린 직후라 walking 자연스러움이 우선, 속도 추종은 다음) |
| Episode cap (학습) | 600 | 600 (변경 없음) |
| Episode cap (test) | 99999 (run.py가 강제) | **600** (학습/test 동일하게 강제) |
| Sanity check | 5500 epoch 후 처음 시도 | **1000 epoch마다 자동** |
| delta_clip | ±0.2 | **±0.1** (보수적) |
| Network width | 256-256 | **256-256** (15GB / 7.6GB 한도) |

---

## 2. 3개 Trial (병렬)

### Trial A — Single residual (Server idx0)

phc_3 column 2 frozen + 단일 residual MLP.

```
obs (935-D)
  ├─→ phc_3.actors[2] (frozen) → a_base (69-D)
  └─→ ResMLPHead (256-256) → Δa (69-D, clamp ±0.1)
                                       ↓
                                   a = a_base + Δa
```

가설: 단일 residual로 [0.76, 1.23] 전 범위 cover 가능. v1의 실패 원인 (r_survive, episode cap, base task)이 진짜 원인이었다면 이번엔 학습 잘 됨.

### Trial B — Soft-gated 3-residual MoE (Server idx2)

phc_3 column 2 frozen + 속도별 3개 residual + smooth gating.

```
v_cmd → gating weights w0, w1, w2 (sum=1, smooth)
        |
        ↓
        ├─→ ResMLPHead_low  (256-256) → Δa_0
        ├─→ ResMLPHead_mid  (256-256) → Δa_1
        └─→ ResMLPHead_high (256-256) → Δa_2
                ↓
        Δa = w0·Δa_0 + w1·Δa_1 + w2·Δa_2  (soft mix, clamp ±0.1)
                ↓
        a = a_base + Δa
```

**Soft gating 함수** (속도 → 가중치):

3개 motion clip의 자연 속도 V_NATURAL = (0.897, 0.975, 1.068) 기준.
- v_cmd ∈ [0.76, 0.94]: low residual dominant (w0 ≈ 1)
- v_cmd ∈ [0.94, 1.02]: low/mid 혼합 (linear interp)
- v_cmd ∈ [1.02, 1.10]: mid/high 혼합
- v_cmd ∈ [1.10, 1.23]: high dominant (w2 ≈ 1)

각 transition zone의 너비 ~0.08 m/s. v_cmd 변화 시 보행이 부드럽게 전환 (v1의 hysteresis hop 문제 해결).

**Gating 수식** (간단 trapezoidal):
```python
def soft_gate(v_cmd, centers=(0.897, 0.975, 1.068), width=0.08):
    """Returns (w0, w1, w2) summing to 1, with linear blend in transition zones."""
    raw = [max(0, 1 - abs(v_cmd - c) / width) for c in centers]
    s = sum(raw)
    return [r / s for r in raw] if s > 0 else [1/3, 1/3, 1/3]
```

가설: 속도 specialized residuals이 단일 residual보다 capacity 활용 ↑, 보행 안정성 ↑. Soft gating으로 transition 부드러움.

### Trial C — phc_3 column ablation (Local, no training)

학습 없이 phc_3의 column 0/1/2 각각에서 inference. residual=0, 우리 env 사용. 어느 column이 우리 env에서 가장 잘 걷는지 측정.

| Run | training_prim | base used | residual |
|-----|---------------|-----------|----------|
| C0 | 0 | column 0 | 0 |
| C1 | 1 | column 1 | 0 |
| C2 | 2 | column 2 | 0 |

각 run: num_envs=8, episode_length=600, 30 episode 평균 eps_len 비교. 가장 긴 eps_len의 column이 (A)/(B) 학습의 base column이 되어야 함 (현재는 col 2 가정).

**중요**: 이게 "phc_3가 우리 env에서 잘 걷는지" 자체의 sanity check. 만약 col 0, 1, 2 모두 짧으면 (~15) → 우리 env 자체가 phc_3에 OOD → (A)/(B) 시작 전에 env 추가 fix 필요.

---

## 3. 새 Task Class — `HumanoidImResAMPVCmdV2`

`phc/env/tasks/humanoid_im_res_amp_vcmd_v2.py` (새 파일)

```python
class HumanoidImResAMPVCmdV2(HumanoidImMCP):
    """v_cmd-conditioned imitation on top of frozen phc_3 PNN.

    상속 chain:
      HumanoidIm → HumanoidImMCP → HumanoidImResAMPVCmdV2

    HumanoidImMCP가 PNN 통합 / multi-prim 처리하고, 우리는 위에 v_cmd만 추가.
    """
    def __init__(self, cfg, ...):
        super().__init__(cfg, ...)
        # v_cmd state
        self._v_cmd_target = torch.full((self.num_envs,), 1.0, device=self.device)
        self._v_cmd_ramped = torch.full((self.num_envs,), 1.0, device=self.device)

    def get_obs_size(self):
        return super().get_obs_size() + 1   # +1 for v_cmd

    def _compute_observations(self, env_ids=None):
        obs = super()._compute_observations(env_ids)   # MCP가 정상 obs 만들어줌
        v_cmd_norm = (self._v_cmd_ramped - 0.995) / 0.235   # to [-1, 1]
        return torch.cat([obs, v_cmd_norm.unsqueeze(-1)], dim=-1)

    def _compute_reward(self, actions):
        v_act = torch.linalg.norm(self._humanoid_root_states[:, 7:9], dim=-1)
        r_track = torch.exp(-5.0 * (v_act - self._v_cmd_ramped) ** 2)
        # r_im은 super()가 계산 (HumanoidImMCP의 imitation reward)
        super()._compute_reward(actions)
        r_im = self.rew_buf.clone()    # super가 채운 값
        r_amp = 0.0   # AMP agent가 별도로 더함 (yaml의 disc_reward_w)
        # NOTE: r_survive 제거 (v1의 Goodharting 원인)
        self.rew_buf[:] = 0.5 * r_track + 0.2 * r_im
        # r_amp는 amp_agent가 disc_reward_w * disc_reward로 더함

    def pre_physics_step(self, actions):
        self._ramp_v_cmd()
        super().pre_physics_step(actions)   # MCP의 standard pre_physics_step

    def _ramp_v_cmd(self):
        delta_max = 0.5 * self.dt   # max_accel = 0.5 m/s²
        diff = self._v_cmd_target - self._v_cmd_ramped
        step_size = torch.clamp(diff, -delta_max, delta_max)
        self._v_cmd_ramped += step_size
        # NOTE: v1의 random re-sample 제거. v_cmd_target은 reset 시에만 변경.
```

**v1 대비 변경 핵심:**
- `_compute_observations` 직접 reimplement 안 함 → super()(MCP)가 정상 obs 보장
- `_compute_reward`에서 r_survive 제거
- `_ramp_v_cmd`에 random re-sample 없음 (S2 ramp는 reset 시에만)
- Per-step retime 제거 (`_motion_start_times_offset` 안 건드림). MCP의 standard 행동에 맡김. Velocity tracking은 residual의 책임.

---

## 4. Network — `res_amp_network_v2.py` (새 파일)

### 4.1 단일 residual (Trial A)

```python
class ResAMPVCmdNetworkV2(AMPPNNBuilder.Network):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        # phc_3 frozen base (col 2)
        self.pnn = load_frozen_phc_3(...)
        self.training_prim = 2   # col 2 (Trial C 결과로 변경 가능)
        # Single residual head
        self.residual_head = ResMLPHead(in_dim=935, action_dim=69, hidden=(256, 256), sigma_init=-2.5)
        self.delta_clip = 0.1

    def eval_actor(self, obs_dict):
        obs = obs_dict['obs']
        base_obs = obs[:, :-1]    # drop v_cmd dim
        with torch.no_grad():
            a_base, _ = self.pnn(self.actor_cnn(base_obs).flatten(1), idx=2)
        delta_mu, delta_logstd = self.residual_head(obs)
        delta_mu = torch.clamp(delta_mu, -0.1, 0.1)
        return a_base + delta_mu, delta_logstd
```

### 4.2 Soft-gated 3-MoE (Trial B)

```python
class ResAMPVCmdNetworkV2_MoE(AMPPNNBuilder.Network):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.pnn = load_frozen_phc_3(...)
        self.training_prim = 2
        # Three residual heads, one per velocity range
        self.residual_low  = ResMLPHead(935, 69, (256, 256), sigma_init=-2.5)
        self.residual_mid  = ResMLPHead(935, 69, (256, 256), sigma_init=-2.5)
        self.residual_high = ResMLPHead(935, 69, (256, 256), sigma_init=-2.5)
        self.delta_clip = 0.1
        self.gate_centers = torch.tensor([0.897, 0.975, 1.068])
        self.gate_width = 0.08

    def eval_actor(self, obs_dict):
        obs = obs_dict['obs']
        base_obs = obs[:, :-1]
        v_cmd_norm = obs[:, -1]
        v_cmd = v_cmd_norm * 0.235 + 0.995    # [-1, 1] → [0.76, 1.23]

        with torch.no_grad():
            a_base, _ = self.pnn(self.actor_cnn(base_obs).flatten(1), idx=2)

        # Soft gate
        raw = torch.clamp(1.0 - torch.abs(v_cmd[:, None] - self.gate_centers[None, :]) / self.gate_width, min=0.0)
        weights = raw / raw.sum(dim=-1, keepdim=True).clamp(min=1e-6)   # (B, 3)

        d_low,  std_low  = self.residual_low(obs)
        d_mid,  std_mid  = self.residual_mid(obs)
        d_high, std_high = self.residual_high(obs)

        delta = (weights[:, 0:1] * d_low
               + weights[:, 1:2] * d_mid
               + weights[:, 2:3] * d_high)
        delta = torch.clamp(delta, -0.1, 0.1)
        # logstd: same blend (or just use mid's; doesn't matter as much for sampling)
        logstd = std_mid
        return a_base + delta, logstd
```

---

## 5. Sanity Check Hook (1000 epoch마다)

`phc/learning/res_amp_agent_v2.py`에 PPO update 후 1000 epoch마다 자동 실행:

```python
class ResAMPVCmdAgentV2(AmpAgent):
    def train_epoch(self):
        super().train_epoch()
        if self.epoch_num % 1000 == 0 and self.epoch_num > 0:
            self._sanity_check_walk()

    def _sanity_check_walk(self):
        """Inference smoke: freeze running_mean_std, sample stochastic actions,
        run 8 envs × 600 steps, measure mean eps_len with v_cmd=1.0 fixed."""
        # 5초 inference, 결과 출력
        with torch.no_grad():
            self.model.eval()
            # ... (구현 detail은 plan 단계에서)
            mean_eps_len = ...
            mean_v_actual = ...
        log_str = (f"[SANITY ep={self.epoch_num}] inference eps_len={mean_eps_len:.1f}/600 "
                   f"v_actual={mean_v_actual:.3f} (target=1.0) — "
                   f"{'OK' if mean_eps_len > 200 else 'GOODHART_WARNING'}")
        print(log_str)
        self.model.train()
```

기준: inference eps_len > 200 (33% of cap)이면 정책 진짜로 학습 중. 100 이하면 Goodharting 의심 → 학습 중단 + 디버그.

---

## 6. Reward 설계

```python
# Trial A and B 공통
self.rew_buf[:] = 0.5 * r_track + 0.2 * r_im
# r_amp은 AmpAgent가 self.disc_reward_w * disc_reward로 더함 (yaml: disc_reward_w=0.3)
# r_survive 명시적으로 제거
```

각 component:
- `r_track = exp(-5 × (v_act - v_cmd_ramped)²)` ∈ [0, 1] (v_err < 0.45 → r > 0.36)
- `r_im` = HumanoidImMCP의 imitation reward (pose match) — super() 위임
- `r_amp` = AmpAgent의 discriminator reward — Δ로 더해짐

총 max reward per step ≈ 0.5 + 0.2 + 0.3 = **1.0** (v1의 ~2.0 대비 정확히 절반). 학습 시 cumulative return이 더 작지만 PPO에는 무관 (advantage normalized).

---

## 7. Episode cap 학습/test 동일

`phc/run.py` line 249-250 패치 제거 또는 우회. 가장 깨끗한 방법: **`--test` 시 episodeLength override 안 함**. CLI에서 명시적으로 600 전달.

이번 sbatch 스크립트에서 `--episode_length 600` 전달 + 우리 sanity check 코드도 600 cap 유지. 학습/test consistency 보장.

---

## 8. 새 yaml 파일

### env: `phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml`

기본은 v1 yaml과 비슷하지만:
- `task: HumanoidImResAMPVCmdV2`
- `has_pnn: True` (MCP가 사용)
- `num_prim: 3`
- `training_prim: 2` (Trial C 결과로 변경 가능)
- `models: ["output/HumanoidIm/phc_3/Humanoid.pth"]` (MCP가 자동 로드)
- `cycle_motion: True` (계속)
- `terminationDistance: 0.25` (phc_3 학습 환경 매칭)
- `terminationHeight: 0.15` (phc_3 학습 환경 매칭)
- 나머지 (v_cmd 설정 등)는 v1과 동일

### learning: 두 yaml

- `phc/data/cfg/learning/im_res_amp_vcmd_v2_A.yaml` — Trial A (single residual)
- `phc/data/cfg/learning/im_res_amp_vcmd_v2_B.yaml` — Trial B (3-MoE)

차이는 `params.network.name`만 (`amp_pnn_residual_v2` vs `amp_pnn_residual_v2_moe`).

---

## 9. 파일 inventory (만들 것 / 수정할 것)

**새로 만들기:**
1. `phc/env/tasks/humanoid_im_res_amp_vcmd_v2.py` — 새 task class
2. `phc/learning/res_amp_network_v2.py` — Trial A network (single)
3. `phc/learning/res_amp_network_v2_moe.py` — Trial B network (3-MoE soft gated)
4. `phc/learning/res_amp_agent_v2.py` — Sanity check hook 포함
5. `phc/data/cfg/env/env_im_res_amp_vcmd_v2.yaml` — env config
6. `phc/data/cfg/learning/im_res_amp_vcmd_v2_A.yaml` — A learning config
7. `phc/data/cfg/learning/im_res_amp_vcmd_v2_B.yaml` — B learning config
8. `train_phc_res_amp_vcmd_v2_A.sh` — Slot A sbatch
9. `train_phc_res_amp_vcmd_v2_B.sh` — Slot B sbatch
10. `scripts/test_phc3_columns.py` — Trial C local script

**수정:**
1. `phc/utils/parse_task.py` — `HumanoidImResAMPVCmdV2` 등록
2. `phc/run.py` — algo / network 등록 (`amp_pnn_residual_v2`, `amp_pnn_residual_v2_moe`)

**그대로 (재사용):**
- `output/HumanoidIm/phc_3/Humanoid.pth` — frozen base
- `sample_data/amass_walking_3clips_seamless_60s_v9.pkl` — motion data
- `phc/learning/amp_agent.py` — AMP discriminator (변경 없음)

---

## 10. 통과 기준 (Stage 2)

| 기준 | 목표 | 검증 |
|------|------|------|
| 학습 metric (rwd, eps_len) | 단조 상승 | 학습 로그 |
| **1000 epoch sanity check** | inference eps_len > 200 | 자동 출력, GOODHART_WARNING 없음 |
| 최종 (20K epoch) | inference eps_len > 400 | 학습 후 eval |
| 시각 검증 | 휴머노이드가 진짜 걷고 v_cmd 따라 속도 변화 | 비디오 |
| Smooth transition | v_cmd 변경 시 fall 없이 부드럽게 가속/감속 | 비디오 (Trial B) |

v1과 다르게 **inference eps_len을 진짜 metric으로** 사용. 학습 metric 부풀려졌어도 sanity check가 잡음.

---

## 11. 컴퓨트 예산

- Trial A: idx0, ~24-36h, 20K epoch, num_envs=512
- Trial B: idx2, ~30-42h, 20K epoch, num_envs=512 (3 networks라 살짝 느림)
- Trial C: local 4060 Ti, ~20분 총 (3 columns × ~7분)

총 학습 시간: ~40 시간 (병렬). 이전 13h 학습 + 12h 디버그 (총 25h)와 비슷한 규모.

---

## 12. 리스크 + Mitigation

| 리스크 | 가능성 | Mitigation |
|--------|--------|-----------|
| `HumanoidImMCP` 위에 v_cmd append 시 obs dim mismatch | 중 | 1000 epoch 전 부팅 smoke로 발견 |
| Trial C에서 모든 column이 우리 env에서 못 걸음 | 중 | env 추가 fix 또는 phc_3 fine-tune 필요 결정 |
| Trial B의 soft gating이 학습 신호 분산시켜 학습 안 됨 | 저-중 | Sanity check가 1000 epoch에 잡음 → 단일 residual로 회귀 |
| 학습/test cap consistency가 PHC의 다른 코드 경로에서 깨짐 | 저 | `flags.test`가 활성화하는 다른 경로 (motion_times=0 등) 가 inference에 영향 줄 수 있음. Sanity check가 이걸 측정 |
| r_survive 제거가 학습 초기 PPO를 너무 약하게 만듦 | 저 | r_im 0.2가 base 역할 — 0이 아닌 small reward는 보장됨 |

---

## 13. 다음 step (이 spec 이후)

1. `docs/superpowers/plans/2026-05-02-phc-residual-amp-vcmd-v2.md` — 24-task 단위 plan (이 spec을 atomic task로 분해)
2. 구현 (subagent-driven-development 또는 직접 — user 결정)
3. Trial C 먼저 (~20분, 학습 결정에 영향)
4. C 결과 보고 A/B base column 결정 후 sbatch 제출
5. 1000 epoch에 첫 sanity check 결과 확인 → 진행 여부 결정
