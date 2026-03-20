# IsaacGym Viewer 디버깅 정리 (260320)

이전 세션(260319)에서 발생한 IsaacGym viewer 문제와 해결책을 정리한다.
06_integration_test/run_visualization.py 에서도 동일 패턴이 재발하여 보완했다.


## 1. 뷰어가 바로 꺼지는 문제 (segfault / immediate close)

증상: python 실행 후 IsaacGym 뷰어가 열리자마자 꺼짐.

원인 및 해결 (커밋 5a04b7c 기준):

- gym.get_actor_asset() 호출이 step_graphics 내부 상태를 오염시킴 → 해당 호출 제거
- query_viewer_has_closed를 step_graphics 이후에 호출하면 이미 닫힌 뷰어에 step_graphics 호출됨 → query_viewer_has_closed를 step_graphics 앞으로 이동
- time.sleep → gym.sync_frame_time으로 교체 (안정적 타이밍)
- 시뮬레이션 완료 후 루프를 추가하여 뷰어가 Q 또는 창 닫기 전까지 유지

올바른 뷰어 루프 패턴:
```
gym.simulate(sim)
if dof_pos.device.type == "cpu":
    gym.fetch_results(sim, True)    # CPU pipeline만

if viewer is not None:
    if gym.query_viewer_has_closed(viewer):   # 먼저 체크
        break
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# 시뮬 완료 후 뷰어 유지
if viewer is not None:
    while not gym.query_viewer_has_closed(viewer):
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        gym.sync_frame_time(sim)
    gym.destroy_viewer(viewer)
```


## 2. GPU pipeline으로 인한 복합 문제 (CPU pipeline으로 해결)

증상:
  1) viewer 모드로 실행하면 humanoid가 완전히 고정(frozen)된 채로 움직이지 않음.
  2) "expected a GPU tensor, received a CPU tensor" 에러 반복 출력.
  3) HumanBody compute_torques에 CUDA tensor 전달 시 즉시 종료.
  4) 시뮬 완료 후 hold loop에서 segmentation fault.

원인: use_gpu_pipeline=True 사용 시 dof_pos가 CUDA tensor. HumanBody는 CPU 전용이므로
      CUDA tensor를 넘기면 크래시. torques도 CPU이면 set_dof_actuation_force_tensor 실패.

핵심 해결: GPU pipeline을 사용하지 않는다. CPU pipeline은 viewer 모드에서도 정상 동작.
(02_isaacgym_integration, 03_visualization 모두 CPU pipeline으로 viewer 사용 중)

```python
# create_viz_sim: GPU/headless 분기 없이 항상 CPU pipeline
gfx = graphics_device if graphics_device is not None else -1
sim_params.use_gpu_pipeline = False   # 항상 CPU

# dof_pos는 cpu tensor → HumanBody에 그대로 사용 가능
# torques도 cpu tensor → set_dof_actuation_force_tensor 정상 동작
# fetch_results(sim, True) 항상 호출
```


## 3. HumanBody (CPU) 에 CUDA tensor 입력 시 크래시

증상: GPU pipeline에서 HumanBody.compute_torques 호출 시 즉시 종료.

원인: dof_pos가 CUDA tensor이므로 pos_i = dof_pos[i].unsqueeze(0)도 CUDA.
HumanBody는 device="cpu"로 생성되어 CUDA 입력을 처리하지 못함.

해결:
```python
pos_i = dof_pos[i].cpu().unsqueeze(0)
vel_i = dof_vel[i].cpu().unsqueeze(0)
```


## 4. 실시간 모니터 x축 스케일이 계속 변하는 문제

증상: matplotlib 실시간 모니터에서 시간이 지남에 따라 x축이 계속 우측으로 확장됨.
롤링 윈도우처럼 일정 폭을 유지해야 하는데 스케일이 계속 바뀜.

원인: MONITOR_SECS(=8.0) > DURATION(=6.0)이면 시뮬레이션이 끝날 때까지 deque가
꽉 차지 않아서 t_lo = ts[0] = 0으로 고정되고 t_hi = t_now가 계속 커짐.

해결 1: MONITOR_SECS를 DURATION보다 작게 설정 (예: 4.0).
해결 2: xlim을 항상 고정 폭으로 설정.

```python
MONITOR_SECS = 4.0   # DURATION(6.0)보다 작아야 스크롤 시작됨

# draw() 내부
t_hi = t_now + 0.1
t_lo = t_hi - MONITOR_SECS
for ax in axes:
    ax.relim()
    ax.autoscale_view(scalex=False)
    ax.set_xlim(t_lo, t_hi)   # 고정 폭 유지
```


## 5. GPU vs CPU Pipeline 정리 (결론: 항상 CPU 사용)

| 항목 | CPU pipeline (headless) | CPU pipeline (viewer) | GPU pipeline (viewer) — 비권장 |
|---|---|---|---|
| compute_device_id | 0 | 0 | 0 |
| graphics_device_id | -1 | GPU id | GPU id |
| use_gpu_pipeline | False | **False** | True |
| dof_pos device | cpu | cpu | cuda |
| torques device | cpu | cpu | cuda (필수) |
| fetch_results | 필요 | 필요 | 불필요 |
| step_graphics | 불필요 | 필요 | 필요 |
| HumanBody 입력 | 그대로 | 그대로 | .cpu() 필수 |

→ viewer를 쓸 때도 CPU pipeline을 유지하면 코드가 단순해지고 에러가 없다.
   IsaacGym에서 viewer 렌더링과 physics pipeline은 독립적이다.


## 파일별 적용 현황

| 파일 | 적용 여부 | 비고 |
|---|---|---|
| 02_isaacgym_integration/run_validation.py | 완료 | CPU pipeline 전용, viewer 없음 |
| 03_visualization/run_visualization.py | 완료 | 260319 커밋 5a04b7c |
| 06_integration_test/run_visualization.py | 완료 | 260320 수정 |
