# Matplotlib 실시간 모니터 디버깅 기록 (260319)

## 문제 상황

`standard_human_model/validation/03_visualization/run_visualization.py` 실행 시
IsaacGym 뷰어 창은 뜨지만 matplotlib 실시간 모니터 창이 **전혀 나타나지 않는** 문제.

headless 모드로 실행해도 플롯이 저장되지 않음.

---

## 원인 분석

### 핵심 원인: matplotlib 백엔드 충돌

`run_visualization.py`는 `from run_validation import ...`로 `02_isaacgym_integration/run_validation.py`를 임포트한다.

해당 파일 상단에 다음 코드가 있었다:

```python
# run_validation.py (문제가 된 코드)
import matplotlib
matplotlib.use("Agg")     # ← 무조건 Agg로 고정
import matplotlib.pyplot as plt
```

이 import가 실행되는 순간 matplotlib 백엔드가 `Agg`(비표시 백엔드)로 고정된다.
이후 `run_visualization.py`에서 `matplotlib.use("TkAgg")`를 호출해도 이미 plt가 초기화되어 있어 **무시**된다.

결과적으로 `plt.show()`나 `plt.ion()`을 호출해도 화면에 아무것도 표시되지 않음.

### 진단 방법

```bash
conda run -n phc python -c "
import sys
sys.path.insert(0, '.')
import matplotlib
print('before import:', matplotlib.get_backend())

from standard_human_model.validation.run_validation import make_human_body
print('after import:', matplotlib.get_backend())
"
```

결과:
```
before import: TkAgg   (또는 Qt5Agg 등 시스템 기본값)
after import: agg       ← run_validation.py가 강제 변경
```

---

## 시도했던 잘못된 접근들

| 시도 | 결과 |
|------|------|
| `run_visualization.py` 상단에 `matplotlib.use("TkAgg")` 추가 | plt가 이미 초기화되어 무시 |
| `plt.switch_backend("TkAgg")` | ImportError 또는 무시 |
| IsaacGym import 순서 조정 | 관계 없음 |
| `SmartDisplay` 사용 | 다른 오류 발생 |

---

## 최종 수정 사항

### 1. `run_validation.py` — 조건부 백엔드 설정

```python
# 수정 전 (문제):
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 수정 후:
import matplotlib
if matplotlib.get_backend().lower() in ("agg", ""):
    matplotlib.use("Agg")     # 이미 비표시 모드인 경우만 Agg 유지
import matplotlib.pyplot as plt
```

이미 TkAgg 등 표시 백엔드가 설정된 경우에는 건드리지 않음.

### 2. `run_visualization.py` — import 순서 + `force=True`

```python
# 핵심: isaacgym + run_validation을 먼저 import한 뒤 matplotlib 설정
import sys
from isaacgym import gymapi, gymtorch, gymutil
import torch
import numpy as np
import os
from collections import deque

import matplotlib
_is_headless = "--headless" in sys.argv
# sys.path 설정 ...
from run_validation import create_sim_and_envs, make_human_body   # ← 먼저 실행

# run_validation의 Agg 설정을 force=True로 덮어씀
matplotlib.use("Agg" if _is_headless else "TkAgg", force=True)   # ← force 필수
import matplotlib.pyplot as plt
```

`force=True` 없이는 이미 초기화된 백엔드를 변경할 수 없음.

---

## 두 번째 문제: 모니터 창이 뜨지만 플롯이 한 곳에 뭉쳐서 표시되는 현상

IsaacGym 뷰어가 있으면 matplotlib ion() 모드의 draw/pause가 정상 작동하지 않는 경우가 있다.
특히 x축이 계속 누적되어 초기 데이터들이 압축되어 왼쪽에 뭉치는 현상.

### 원인

`ax.relim()` + `ax.autoscale_view()` 만으로는 x축이 0부터 현재까지 전체를 표시하려 해서
초반부가 매우 작게 표시되고 최신 데이터만 오른쪽 끝에 몰림.

### 수정: 롤링 시간 윈도우

```python
MONITOR_WINDOW_SECS = 10.0   # 최근 N초만 표시

def draw(fig, axes, ts, signals):
    t_now = ts[-1]
    t_lo  = max(ts[0], t_now - MONITOR_WINDOW_SECS)

    for ax, (key, data) in zip(axes, signals.items()):
        ax.clear()
        ax.plot(ts, data, lw=1.8, color=MONITOR_SIGNALS[key][2])
        ax.set_ylabel(f"{MONITOR_SIGNALS[key][0]} ({MONITOR_SIGNALS[key][1]})")
        ax.grid(True, alpha=0.3)

    # x축 범위를 현재 기준 슬라이딩 윈도우로 고정
    for ax in axes:
        ax.relim()
        ax.autoscale_view(scalex=False)   # y만 자동 스케일
        ax.set_xlim(t_lo, t_now + 0.5)   # x는 수동으로 슬라이딩 윈도우

    fig.canvas.draw()
    fig.canvas.flush_events()
```

---

## 최종 실행 결과

- IsaacGym 뷰어: Healthy / Spastic / Flaccid 세 프로파일 무릎 진자 시연
- matplotlib 모니터: L_Knee 각도 / Bio-Torque / 각속도 실시간 표시, 10초 슬라이딩 윈도우
- Spastic 프로파일에서 높은 stretch gain에 의한 진동(pseudo-clonus) 관찰 가능

---

## 정리: IsaacGym + Matplotlib 공존 시 주의사항

1. `run_validation.py` 등 분석 모듈을 import 할 때 해당 파일의 `matplotlib.use()` 호출이 백엔드를 오염시킬 수 있음
2. import 순서: `isaacgym` → 기타 모듈 → `matplotlib.use(..., force=True)` → `import matplotlib.pyplot`
3. GPU pipeline (`use_gpu_pipeline=True`) 필수 (뷰어 모드에서 CPU pipeline이면 `step_graphics` crash)
4. `plt.pause()` 대신 `fig.canvas.draw()` + `fig.canvas.flush_events()` 조합이 더 안정적
