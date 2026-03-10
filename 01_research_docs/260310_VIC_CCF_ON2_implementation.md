# VIC_CCF_ON2 구현 문서 (260310)

## 핵심 변경 사항 (한 줄 요약)

> **CCF 8개 그룹의 탐색 sigma를 PD target과 분리: -2.9 (std=0.055) → -1.0 (std=0.37)**
> 이를 통해 CCF 학습이 실제로 일어나도록 한다.

---

## 1. 배경 및 목적

VIC_CCF_ON에서 CCF가 학습되지 않은 원인 확인:
- sigma=-2.9 (std=0.055)를 PD target(69 dims)와 CCF(8 dims)가 동일하게 공유
- CCF 탐색 범위가 ±0.11에 불과 → impedance_scale 변화 ±8% → reward gradient에 묻힘
- 20k epoch 학습 후에도 CCF bias ≈ 0, impedance_scale ≈ 1.0×로 수렴

VIC_CCF_ON2는 CCF dims에만 sigma=-1.0 (std=0.37)을 별도 적용한다.
이렇게 하면 CCF 탐색 범위가 ±0.74까지 확장되고 impedance_scale [0.6, 1.7]를 충분히 탐색한다.

---

## 2. sigma란?

PPO에서 policy는 Gaussian 분포로 action을 샘플링: `a ~ N(mu(obs), exp(sigma)^2)`

- mu: 네트워크가 출력하는 action 평균값
- sigma: log-std (고정값, 탐색 폭 결정)
- `fixed_sigma: True, learn_sigma: False` → sigma는 상수로 유지됨
- 평가(test mode)에서는 `a = mu` (결정론적, sigma 무관)

| sigma | std | CCF 95% 탐색 범위 | impedance_scale 범위 |
| :--- | :--- | :--- | :--- |
| -2.9 (기존) | 0.055 | ±0.11 | [0.93, 1.08] |
| **-1.0 (신규)** | **0.37** | **±0.74** | **[0.60, 1.68]** |

---

## 3. VIC_CCF_ON2 변경 사항 (VIC_CCF_ON 대비)

### 3-1. CCF sigma 분리 (핵심 변경)

학습 시작 직후 CCF dims(last 8)의 sigma를 -1.0으로 덮어쓴다:
- PD target(69 dims): sigma=-2.9 유지
- CCF(8 dims): sigma=-1.0으로 override

`amp_agent.py`의 `pre_epoch()`에서 최초 1회만 실행:
```python
# VIC: Override CCF sigma for better exploration (runs once at first epoch)
if (not getattr(self, '_vic_ccf_sigma_overridden', False)
        and hasattr(humanoid_env, '_vic_ccf_sigma_init')
        and humanoid_env._vic_ccf_sigma_init is not None):
    num_pd = humanoid_env._num_actions
    with torch.no_grad():
        self.model.a2c_network.sigma.data[num_pd:] = humanoid_env._vic_ccf_sigma_init
    self._vic_ccf_sigma_overridden = True
```

### 3-2. 파라미터

| 파라미터 | VIC_CCF_ON | VIC_CCF_ON2 |
| :--- | :--- | :--- |
| sigma (PD, 69 dims) | -2.9 | **-2.9** (유지) |
| sigma (CCF, 8 dims) | -2.9 | **-1.0** (신규 분리) |
| vic_curriculum_stage | 2 | 2 (유지) |
| reward_curriculum_switch_epoch | 10000 | 10000 (유지) |
| reward_w_stage1_task/disc | 0.7/0.3 | 0.7/0.3 (유지) |
| reward_w_stage2_task/disc | 0.3/0.7 | 0.3/0.7 (유지) |
| max_epochs | 20000 | 20000 (유지) |

---

## 4. 수정된 파일 목록

1. **phc/env/tasks/humanoid_im_vic.py**
   - `__init__`: `_vic_ccf_sigma_init` 파라미터 추가

2. **phc/learning/amp_agent.py**
   - `pre_epoch()`: CCF sigma override 로직 추가 (최초 1회)

3. **phc/data/cfg/env/env_im_walk_vic.yaml**
   - `vic_ccf_sigma_init: -1.0` 추가

4. **phc/data/cfg/learning/im_walk_vic.yaml**
   - `name: VIC_CCF_ON2`

---

## 5. 기대 효과

CCF 탐색 폭 확대로 policy가 impedance를 실질적으로 변화시키는 행동을 발견할 것.

- 관절별 stiffness 차별화 가능: 발목(높은 stiffness)과 상체(낮은 stiffness)
- wandb `vic/ccf_g0~g7`으로 그룹별 CCF 변화 모니터링 가능
- 성능 유지 or 향상 기대 (CCF=0이 이미 최적이면 동일, 아니면 향상)
