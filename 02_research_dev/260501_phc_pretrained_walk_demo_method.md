# PHC Pretrained Walk Demo — 방법 정리 및 v1 → v2 변경사항

작성일: 2026-05-01
관련 코드: `scripts/phc_walk_demo.py` (v2), `scripts/phc_walk_demo_v1.py` (v1 백업), `scripts/phc_walk_demo_panel.py`
관련 문서:
- 디자인 스펙: `docs/superpowers/specs/2026-04-30-phc-pretrained-walk-demo-design.md`
- 구현 플랜: `docs/superpowers/plans/2026-04-30-phc-pretrained-walk-demo.md`

---

## 1. 한 줄 요약

PHC `phc_3` 사전학습 모델을 그대로 쓰면서, **3개의 자연 보행 클립(약 0.90 / 0.97 / 1.07 m/s)** 을 미리 준비해두고 **사용자가 키보드 또는 Tk 패널로 입력한 v_cmd**에 따라 (1) 클립을 선택하고 (2) 그 클립의 재생 속도를 매 step retiming 하는 방식의 **추가 학습 없는(zero-training) 인터랙티브 워킹 데모**.

기존 PHC 추론 파이프라인은 전혀 손대지 않고 `Humanoid.pre_physics_step` / `Humanoid.render` 두 메서드만 monkey-patch 해서 만든다.

---

## 2. 왜 이 방식인가 (D-2 결정 배경)

| 후보 | 요지 | 채택 여부 |
|---|---|---|
| (A) phc_3 + 단일 클립을 학습 없이 그대로 재생 | 가장 단순, 그러나 v_cmd 변화 불가 | X |
| (B) phc_3 위에 residual policy를 짧게 학습 | 학습 비용 + 학습 yaml 호환 문제 | X |
| (C) phc_3 + 단일 클립 retime | 한 클립을 stretch만 함 → 빠를수록 보폭은 그대로/케이던스만 빨라지는 비현실적 보행 | X |
| **(D-2) phc_3 + 다중 자연속도 클립 + 매 step retime** | **클립 자체를 v_cmd에 가장 가까운 자연속도 것으로 골라주고, 그 위에서 ±10% 정도만 retime → 운동학·동역학적으로 합리적인 범위 유지** | **채택** |

핵심 통찰: **사람이 빨리 걸으면 단순히 같은 패턴을 빠르게 재생하는 게 아니라 보폭·관절각이 같이 바뀐다.** 단일 클립 retime만 쓰면 이 점을 놓친다. 3개의 자연 속도 클립을 두고 mid-point 근처에서 hysteresis로 스위칭하면 retime 비율은 항상 ~[0.90, 1.10] 안에 머물러 자연스러움이 유지된다.

---

## 3. 시스템 구조

```
                ┌────────────────────────────────────────────────────┐
                │   scripts/phc_walk_demo.py (메인 드라이버)         │
                │                                                    │
                │   1) Humanoid.pre_physics_step  ← monkey-patch     │
                │      ├─ v_cmd_ramped 갱신 (max_accel=0.5 m/s²)     │
                │      └─ _motion_start_times_offset                 │
                │           +=  dt * (v_cmd_ramped/v_natural - 1)    │
                │                                                    │
                │   2) Humanoid.render            ← monkey-patch     │
                │      ├─ 키보드 이벤트 처리 (↑↓, 1-9, R/P/Q)         │
                │      ├─ 패널 input JSON 폴링                        │
                │      ├─ _maybe_switch_clip (60 step 주기, hyst.)    │
                │      ├─ forward arrow 그리기 (색=v_cmd)              │
                │      ├─ v_actual 측정, 카메라 follow                │
                │      └─ 패널 state JSON 쓰기 (3 step 마다)           │
                │                                                    │
                │   3) runpy.run_path('phc/run_hydra.py')            │
                │      → 표준 PHC 추론 루프 시작 (test=True)           │
                └────────────────────────────────────────────────────┘
                         │
                         │ /tmp/phc_walk_state.json   (demo → panel)
                         │ /tmp/phc_walk_input.json   (panel → demo)
                         ▼
                ┌────────────────────────────────────────────────────┐
                │   scripts/phc_walk_demo_panel.py (Tk 패널)         │
                │   - 상태 표시 (target/ramped/actual/clip 등)        │
                │   - Entry로 v_cmd 직접 입력 + Apply                  │
                │   - Slider [V_MIN, V_MAX]                          │
                │   - Reset / Pause / Quit 버튼                       │
                └────────────────────────────────────────────────────┘
```

### 3.1 주요 상수 (`phc_walk_demo.py` 상단)

| 상수 | 값 | 역할 |
|---|---|---|
| `EXP_NAME` | `"phc_3"` | 사용 체크포인트 이름 |
| `LEARNING` | `"im_pnn_big"` | phc_3와 매칭되는 PNN 아키텍처 yaml |
| `MOTION_FILE` | `amass_walking_3clips_seamless_60s_v9.pkl` | 3개 자연 속도 보행 클립 |
| `V_NATURAL` | `(0.897, 0.975, 1.068)` | 각 클립의 평균 자연 보속(m/s) |
| `V_CMD_INIT / MIN / MAX` | `1.0 / 0.76 / 1.23` | v_cmd 입력 범위 |
| `V_CMD_KEY_STEP` | `0.02` | ↑↓ 키 한번에 변하는 양 |
| `V_CMD_MAX_ACCEL` | `0.5 m/s²` | v_cmd 변화 ramp 상한 |
| `HYSTERESIS` | `0.02` | 클립 스위칭 데드밴드 |
| `MIDPOINT_AB / BC` | `~0.936 / ~1.022` | 클립 0↔1, 1↔2 전환 임계 |
| `NUM_ENVS / ENV_SPACING` | `2 / 50` | 보조 env는 멀리 두어 안 보이게 |
| `CLIP_SWITCH_INTERVAL` | `60` render step (~2 s) | **v2 신규** — 클립 스위치 폴링 주기 |
| `EPISODE_LENGTH` | `99999` | **v2 신규** — 사실상 무한 에피소드 |

### 3.2 부팅 패턴 (record_phc_pretrained.py 와 동일)

`phc/run_hydra.py`를 직접 import 하면 Hydra의 `config_path`가 호출 스크립트 위치 기준으로 풀려 깨진다. 따라서:

```python
sys.argv = ["run_hydra.py", "learning=im_pnn_big", "env=env_im_pnn", ...]
_install_render_hook()
_install_pre_physics_patch()
import runpy
runpy.run_path("phc/run_hydra.py", run_name="__main__")
```

— `runpy.run_path` 가 `phc/run_hydra.py` 파일을 그 위치의 `__main__`으로 실행해주므로 hydra가 정상적으로 `phc/data/cfg/`를 잡는다.

---

## 4. 핵심 알고리즘

### 4.1 Per-step retime (`pre_physics_step` 패치)

PHC는 매 시뮬 step마다 reference motion 시점을
```
motion_time = progress_buf * dt + _motion_start_times + _motion_start_times_offset
```
로 계산한다. 우리는 마지막 항을 매 step

```python
self._motion_start_times_offset += self.dt * (v_cmd_ramped / v_natural - 1.0)
```

만큼 누적시킨다. 결과적으로 reference motion이 `v_cmd_ramped/v_natural` 비율로 더 빨리/느리게 흐르는 셈이고, phc_3는 이 retimed reference를 그대로 imitation 한다.

### 4.2 클립 선택 (hysteresis)

```python
# midpoint = (V_NATURAL[i] + V_NATURAL[i+1]) / 2
# 현재 클립 1에서 v_cmd_ramped 가
#   < MIDPOINT_AB - HYSTERESIS → 클립 0 으로
#   > MIDPOINT_BC + HYSTERESIS → 클립 2 로
```

이 hysteresis (0.02 m/s) 덕분에 v_cmd 가 midpoint 부근에서 흔들려도 클립이 ping-pong 하지 않는다.

### 4.3 v_cmd 램프 (smoothing)

키보드/패널은 v_cmd_target 만 갱신한다. 실제 데모가 쓰는 값은 매 sim step

```python
delta = clip(v_cmd_target - v_cmd_ramped, ±V_CMD_MAX_ACCEL * dt)
v_cmd_ramped += delta
```

로 0.5 m/s² 이내에서만 변한다 → 클립 스위칭과 retime 비율이 모두 부드럽게 따라간다.

---

## 5. v1 → v2 변경 (이번 세션의 본 작업)

v1에서 사용자가 두 가지 문제를 발견:
1. **클립이 안 바뀐다** — v_cmd 를 0.76 까지 내려도 active_clip 이 1 에서 안 움직임.
2. **5초 마다 모션이 끊긴다** — 보행은 잘 하는데 일정 주기로 reference motion 의 random start 가 새로 뽑혀서 휴머노이드가 점프하듯 위치가 바뀜.

원인을 추적해서 v2 에서 두 가지 수정을 가했다. v1 은 `scripts/phc_walk_demo_v1.py` 로 그대로 보존했다 (보행 자체는 잘 되므로 비교용).

### 5.1 Bug 1 — 클립 스위칭이 안 됨

**v1 의 방식 (`_maybe_switch_clip` in `phc_walk_demo_v1.py:350-375`):**

```python
motion_time = progress_buf * dt + _motion_start_times + _motion_start_times_offset
cycle_count = int(motion_time // motion_len)   # motion_len = 60 s
if cycle_count > last_cycle_count:
    # 새 cycle이 시작될 때만 클립 재선택
```

문제:
- 클립 길이가 60 초 → 첫 cycle 경계가 거의 영원히 안 옴.
- 게다가 episode reset 마다 `progress_buf` 가 0 으로 리셋되어 `cycle_count` 도 같이 리셋되니, 누적 비교 자체가 의미 없음.

**v2 의 수정 (`_maybe_switch_clip` in `phc_walk_demo.py:356-370`):**

```python
def _maybe_switch_clip(env):
    if not hasattr(env, "_sampled_motion_ids"):
        return
    if _STATE["step"] == 0 or _STATE["step"] % CLIP_SWITCH_INTERVAL != 0:
        return                          # 60 render step ≈ 2초 마다 한번 폴링
    desired = _select_clip(_STATE["v_cmd_ramped"], _STATE["active_clip"])
    if desired != _STATE["active_clip"]:
        _STATE["active_clip"] = desired
        _STATE["v_natural"] = V_NATURAL[desired]
        env._sampled_motion_ids[:] = desired
        print(f"[demo] clip switch → {desired} (v_natural={V_NATURAL[desired]:.3f})")
```

→ cycle 경계가 아니라 **단순 시간 간격** 으로 폴링. hysteresis 가 ping-pong 을 막아주므로 더 단순하고 견고함.

검증: 패널로 v_cmd 를 0.85 → 1.10 → 0.80 으로 흔들었을 때 로그에 `clip switch → 0`, `→ 1`, `→ 2` 가 차례로 찍힘.

### 5.2 Bug 2 — 5초 마다 끊김

**원인**: 부팅 시 `sys.argv` 에 `env.episode_length=300` 을 박았었음. PHC 의 episode_length 는 sim step 수 → `300 step × ~0.0167 s/step ≈ 5 s` 마다 자동 reset 발동, reset 시 reference motion 의 random start time 도 새로 뽑히므로 휴머노이드가 점프함.

**v2 수정**:

```python
EPISODE_LENGTH = 99999      # 사실상 무한
...
sys.argv = [..., f"env.episode_length={EPISODE_LENGTH}"]
```

→ 에피소드가 끝나지 않으니 reference motion 도 끊기지 않고 계속 retime 된 시점에서 자연스럽게 흐른다. 휴머노이드가 진짜로 넘어지면 PHC 의 fall termination 이 reset 을 걸 수 있으나, phc_3 가 안정적으로 잘 걸어서 실측 상 거의 발동하지 않음.

### 5.3 v2 추가 코멘트

`phc_walk_demo.py` 의 docstring 상단에 v1 → v2 차이를 7-13 줄에 직접 명시해서 향후 유지보수자가 즉시 파악할 수 있게 했다. v1 파일도 `phc_walk_demo_v1.py` 로 보존되어 직접 비교 가능.

---

## 6. 사용 방법

```bash
conda activate phc

# 기본 실행 (Tk 패널이 자동 spawn 됨)
python scripts/phc_walk_demo.py

# 30초 영상 녹화 후 종료
python scripts/phc_walk_demo.py --record_seconds 30 --out_name my_demo

# v1 (참고용 — 보행은 잘 되지만 클립 안 바뀜 / 5초마다 점프)
python scripts/phc_walk_demo_v1.py
```

### 키보드 (IsaacGym 윈도우)
| 키 | 동작 |
|---|---|
| ↑ / ↓ | v_cmd ±0.02 |
| 1..9 | v_cmd 를 [V_MIN, V_MAX] 사이 9 단계로 스냅 |
| R | 에피소드 reset |
| P | pause / resume |
| Q | quit |

### Tk 패널
- 상태 표시: v_cmd_target, v_cmd_ramped, v_actual, active_clip, v_natural, retime_ratio, step, paused
- 입력: Entry + Apply / Slider / Reset / Pause / Quit
- 통신: `/tmp/phc_walk_state.json` (demo → panel, 3 step 주기) / `/tmp/phc_walk_input.json` (panel → demo, single-shot, atomic write)

---

## 7. 파일 구조

```
scripts/
├── phc_walk_demo.py        # v2 (현재) — 479 LOC
├── phc_walk_demo_v1.py     # v1 백업 — 484 LOC (cycle-기반 스위칭)
└── phc_walk_demo_panel.py  # Tk 패널 — 197 LOC

sample_data/
├── amass_walking_3clips_seamless_60s_v9.pkl       # 3개 자연속도 보행 클립
└── amass_walking_3clips_seamless_60s_v9_dirmeta.json

docs/superpowers/
├── specs/2026-04-30-phc-pretrained-walk-demo-design.md   # 디자인 스펙
└── plans/2026-04-30-phc-pretrained-walk-demo.md          # 11-task 구현 플랜

output/HumanoidIm/phc_3/
└── Humanoid.pth             # 사전학습 체크포인트 (~419 MB)
```

PHC base class는 일절 수정하지 않았다. 모든 변경은 두 monkey-patch (`pre_physics_step`, `render`) 안에 격리됨.

---

## 8. 한계 및 추후 과제

1. **클립 전환 순간의 미세 점프** — `_sampled_motion_ids` 만 바꾸므로 `_motion_start_times_offset` 의 위상은 그대로 이어진다. 새 클립의 stance/swing 위상과 어긋날 가능성 존재. 현재는 hysteresis 와 인접 클립끼리의 보폭 차가 작아 시각적으로 거의 안 보임.
2. **v_cmd 가 클립 자연속도에서 벗어날수록 retime 비율이 1 에서 멀어져** 동역학적 일관성이 떨어짐. v_cmd 범위를 (0.76, 1.23) 으로 좁힌 이유.
3. **VIC 와의 결합** — 이 데모는 imitation 만 한다. CCF 학습은 여기에 포함 안 됨. 추후 phc_3 위에 small residual VIC head 를 distillation 할 수 있음 (D-1 옵션이었던 것).
4. **단일 envspec/multi-env 동기화** — `_motion_start_times_offset` 을 broadcast 로 전체 env 에 같은 값 더함. v_cmd 가 여러 env 마다 달라지는 시나리오는 미지원.

---

## 9. 변경 이력 (commit 단위)

```
03ae0a7  phc_walk_demo: v2 — clip switching + continuous walking
6fa2241  phc_walk_demo: cleanup — drop dead _STATE keys, narrow exception handlers
9f086e4  phc_walk_demo: optional --record_seconds N flag
cb983f5  phc_walk_demo: spawn Tkinter panel as subprocess
8801e86  phc_walk_demo_panel: bidirectional Tkinter panel
2db607b  phc_walk_demo: v_actual measurement + camera follow
2fde1f3  phc_walk_demo: forward arrow visualization (color = v_cmd)
3e7900d  phc_walk_demo: clip selection at cycle boundary with hysteresis
cb25569  phc_walk_demo: per-step retime via pre_physics_step + _motion_start_times_offset
bb68357  phc_walk_demo: keyboard handlers + render hook
3ec2ee4  phc_walk_demo: state dict + panel JSON I/O
13e09ae  phc_walk_demo: skeleton + boot via runpy
```

— 11 tasks (subagent-driven-development) + v2 사용자 피드백 fix.
