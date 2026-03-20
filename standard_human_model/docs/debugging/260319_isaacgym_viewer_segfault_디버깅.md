# IsaacGym 뷰어 Segfault 디버깅 기록 (260319)

## 문제 상황

`standard_human_model/validation/03_visualization/run_visualization.py` 실행 시
뷰어 창이 열리자마자 즉시 segmentation fault로 종료되는 문제.

```
뷰어 켜짐. 'Q' 또는 창 닫기로 종료.
Segmentation fault (core dumped)
```

## 원인 분석 과정

### 단계 1: 정확한 crash 위치 특정 (faulthandler)

```python
import faulthandler
faulthandler.enable()
```

결과: line 176 `gym.step_graphics(sim)` 에서 crash 확인.

### 단계 2: 테스트 스크립트로 재현 시도

동일한 로직을 /tmp/test_*.py 로 분리해서 실행 → 600 스텝 전부 통과.
실제 스크립트만 step 0에서 crash.

유일한 코드 차이점 발견:

```python
# run_visualization.py에만 있었던 코드 (lines 88-90):
num_bodies = gym.get_asset_rigid_body_count(
    gym.get_actor_asset(envs[0], actor_handles[0])
) if hasattr(gym, "get_actor_asset") else 24
```

이 변수는 선언 후 **전혀 사용되지 않는 dead code**였다.

### 단계 3: 원인 확인

`gym.get_actor_asset()` 호출이 IsaacGym 내부 그래픽 상태를 오염시켜
이후 `gym.step_graphics(sim)` 호출 시 segfault 발생.

## 시도했던 잘못된 접근들

| 시도 | 결과 |
|------|------|
| CPU pipeline (`use_gpu_pipeline=False`) | `step_graphics` 자체가 CPU pipeline에서 crash |
| SmartDisplay (visible=True) + CPU pipeline | "NV-GLX missing on display :1" + segfault |
| SmartDisplay + GPU pipeline | segfault (원인 다름) |
| graphics_device=-1 하드코딩 제거 | viewer 창 생성 자체 실패 |
| `fetch_results` 조건부 호출 | 필요한 수정이지만 crash 원인 아님 |
| `query_viewer_has_closed` 위치 변경 | 안전장치지만 crash 원인 아님 |

## 최종 수정 사항

### 1. `gym.get_actor_asset()` 제거 (핵심)

```python
# 제거된 코드:
num_bodies = gym.get_asset_rigid_body_count(
    gym.get_actor_asset(envs[0], actor_handles[0])
) if hasattr(gym, "get_actor_asset") else 24
```

### 2. `time.sleep` → `gym.sync_frame_time`

```python
# 변경 전:
time.sleep(DT * SLOW_FACTOR)

# 변경 후:
gym.sync_frame_time(sim)   # IsaacGym 내장 타이밍 동기화
```

`gym.sync_frame_time`은 실시간(60Hz) 기준으로 sync. `time.sleep` 대비 안정적.

### 3. `query_viewer_has_closed` 위치 변경

```python
# 변경 전 (closed viewer에 step_graphics 가능):
gym.step_graphics(sim)
gym.draw_viewer(viewer, sim, True)
if gym.query_viewer_has_closed(viewer):
    break

# 변경 후 (step_graphics 전에 체크):
if gym.query_viewer_has_closed(viewer):
    max_steps = step + 1
    break
gym.step_graphics(sim)
gym.draw_viewer(viewer, sim, True)
gym.sync_frame_time(sim)
```

### 4. `run_validation.py` GPU pipeline 분기

뷰어 사용 시 `use_gpu_pipeline=True` 필수.

```python
def create_sim_and_envs(gym, args, num_envs, initial_knee_angles, graphics_device=None):
    ...
    if graphics_device is None:
        graphics_device = -1  # headless: CPU pipeline
        sim_params.use_gpu_pipeline = False
    else:
        sim_params.use_gpu_pipeline = True  # 뷰어: GPU pipeline 필수
```

## GPU vs CPU Pipeline 동작 차이

| 상황 | pipeline | `step_graphics` | `fetch_results` |
|------|----------|-----------------|-----------------|
| headless | CPU | 불필요 (crash 유발) | 필요 |
| 뷰어 | GPU | 필수 | 불필요 (crash 유발) |

코드:
```python
gym.simulate(sim)
if dof_pos_all.device.type == "cpu":
    gym.fetch_results(sim, True)  # GPU pipeline이면 호출하지 않음
```

## 실행 방법

```bash
conda activate phc
cd /home/gunhee/workspace/PHC

# 뷰어 모드 (GPU pipeline 자동 선택)
python standard_human_model/validation/03_visualization/run_visualization.py

# headless (플롯만 저장)
python standard_human_model/validation/03_visualization/run_visualization.py --headless
```

## 시각화 내용

- **Healthy (파랑)**: 적절한 감쇠, 적당히 진동
- **Spastic (빨강)**: 높은 저항, 초기 자세 근처 유지
- **Flaccid (초록)**: 저항 거의 없음, 중력에 따라 자유 진자

결과 플롯: `standard_human_model/validation/03_visualization/results/knee_pendulum_profiles.png`
