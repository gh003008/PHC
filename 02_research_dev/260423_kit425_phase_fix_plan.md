# KIT_425 단일-피험자 Multi-Clip + Phase-Obs Per-Clip 수정 + 3-Slot Ablation 계획 (2026-04-23)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 22-clip MULTICLIP_FWD 의 0 % 성공률 원인으로 지목된 phase-obs silent-corruption 문제를 해결하고, 단일-피험자 (KIT_425, 3 clip) 로 3-slot ablation (main / phase_obs off / cycle_motion on) 을 병렬 실행해 각 knob 의 기여도를 측정한다.

**Architecture:** (1) `HumanoidImVIC._precompute_gait_phase()` 의 `motion_id=0` 하드코딩을 per-clip lookup table `[num_clips, 1000]` 로 확장. (2) KIT_425 피험자 3 clip 서브셋을 학습 데이터로 사용 (단일-피험자 원칙). (3) 서버 GPU 3 슬롯 병렬로 `(cycle_motion, vic_phase_obs) ∈ {(F,T), (F,F), (T,T)}` 2×2 minus 1 factorial 실행. (4) 평가 metric 은 raw (av_steps / av_time_s / fall-free rate) — 단일 정량값 안 만듦.

**Tech Stack:** Python 3.8 + PyTorch 2.1 (local 검증), IsaacGym (서버 학습), rl-games, argparse yaml configs, sbatch on Slurm.

---

## 0. 현재 문제 (해결 대상)

| # | 문제 | 증거 | 해결 |
|---|---|---|---|
| P1 | `_precompute_gait_phase` 가 `motion_id=0` 에만 테이블 계산 | `humanoid_im_vic.py:1003-1004` | Task 4 (코드 수정) |
| P2 | 22 clip 중 14 개가 HS 검출 "성공" 이지만 stride CoV > 0.3 (silent corruption) | `check_hs_detection_22clips.py` 결과 | Task 1 (KIT_425 로 교체하면 CoV < 0.25 전원 통과) |
| P3 | 22 clip 의 피험자 다양성이 policy 학습 부담 (11 subject × 6 family) — 개인화 연구 원칙 위배 | 피험자 breakdown | Task 1 (KIT_425 단일-피험자로 고정) |
| P4 | KIT_314 에 0.34 → 0.75 구간 0.41 m/s gap → retrieval 오차 | per-subject 분석 | Task 1 (KIT_425 는 gap 0.28/0.20 고름) |
| P5 | `cycle_motion: True` 가 clip wrap 불연속을 만들어 imitation reward 오염 | reward 함수 구조 분석 | **Task 6 의 Slot 1/2 가 cycle=False 로 설정하고 Slot 3 로 T/F 비교** |
| P6 | 초기 motion 샘플링이 clip 내 전구간 random → cycle=False 에서 episode 조기 종료 | logic 분석 | Task 5 (첫 30% 제한) |
| P7 | Canonical ckpt 이 save-on-best 실패로 last-epoch state | `last_mean_rewards: -100500` sentinel | **이번엔 건드리지 않음**. Task 9 평가 시 milestone 중 best 손수 선택 |
| P8 | Phase-CCF 분석기 4-vs-8 group broadcasting 에러 | test-greedy 종료 로그 | **범위 밖** — 학습 성능 무관 |

**범위 밖** (follow-up iteration): P8 (분석기 post-hoc fix), retime-aware phase lookup, HS 검출 amplitude-threshold 개선 (KIT_425 는 필요 없음), save-on-best 로직 자체 수정.

---

## 1. 해결 전략 (high-level)

1. **데이터 축소 (KIT_425 3 clip)** 로 P2/P3/P4 일괄 해결. Dry-run 에서 HS CoV 0.19~0.23 으로 깨끗.
2. **Per-clip phase fix** 로 P1 제거.
3. **3 slot ablation 병렬 실행** — 피험자/데이터/코드 전부 동일하고 `(cycle_motion, vic_phase_obs)` 만 knob 으로 변경:
   - Slot 1 (MAIN): `(cycle=F, phase=T)` — 깨끗한 기본안.
   - Slot 2 (NOPHASE): `(cycle=F, phase=F)` — phase_obs 자체가 유용한가?
   - Slot 3 (LEGACY): `(cycle=T, phase=T)` — cycle_motion wrap 의 실제 피해량 정량화.
4. **cycle_motion=False 주의사항**: episode 가 clip_end 에서 자연 종료 → av_steps 상한이 clip 길이. 초기 motion 샘플링을 clip 첫 30% 로 제한하여 조기 종료 왜곡 방지.
5. **Metric 은 raw 값 병렬 보고**: av_steps / av_time_s / fall-free rate / clip_end rate. 단일 정량값 없음 — 세 slot 결과를 병렬 표로 사람이 판단.
6. **실험은 NR variant 만.** §1-3 결과로 RT ≈ NR 확정이므로 GPU 예산 절반 절약.

## 2. File Structure

**새 파일**:

| Path | 역할 |
|---|---|
| `scripts/data/build_kit425_subset.py` | 22-clip json 에서 KIT_425 3 clip 추출 |
| `sample_data/amass_isaac_walking_primitive_kit425_only.json` | 위 스크립트 산출물 |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_main.yaml` | Slot 1 env (cycle=F, phase=T) |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_nophase.yaml` | Slot 2 env (cycle=F, phase=F) |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_legacy.yaml` | Slot 3 env (cycle=T, phase=T) |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/im_walk_vic_kit425.yaml` | 공통 learning yaml (experiment_name 은 sbatch 에서 override) |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_main_gpu0.sh` | Slot 1 sbatch |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_nophase_gpu1.sh` | Slot 2 sbatch |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_legacy_gpu2.sh` | Slot 3 sbatch |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_main.sh` | Slot 1 test-greedy sbatch |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_nophase.sh` | Slot 2 test-greedy sbatch |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_legacy.sh` | Slot 3 test-greedy sbatch |
| `exp_config/forward_walking/260423_KIT425_PHASEFIX/README.md` | 3 슬롯 구성 및 명명 규칙 문서 (다른 Claude 세션 참조용) |
| `tests/test_gait_phase_per_clip.py` | local 단위 테스트 |

**수정 파일**:

| Path | 변경 내용 |
|---|---|
| `phc/env/tasks/humanoid_im_vic.py` | `_precompute_gait_phase` per-clip 화 (Task 4); 두 phase lookup site 의 인덱싱 변경 |
| `phc/env/tasks/humanoid_im.py` | termination reason 플래그 추가 + `_sample_time` 의 초기 motion 샘플링 첫 30% 제한 (Task 5) |
| `phc/learning/im_amp_players.py` | 평가 시 termination reason 집계 + fall-free/clip_end rate 출력 (Task 5) |

---

## Task 1: KIT_425 서브셋 JSON 생성

**Files:**
- Create: `scripts/data/build_kit425_subset.py`
- Create: `sample_data/amass_isaac_walking_primitive_kit425_only.json`

- [ ] **Step 1: 서브셋 추출 스크립트 작성**

Path: `scripts/data/build_kit425_subset.py`

```python
"""Extract KIT_425 single-subject subset from the 22-forward-clip JSON."""
import json
import re
from pathlib import Path

SRC = Path("sample_data/amass_isaac_walking_primitive_fwd_only.json")
OUT = Path("sample_data/amass_isaac_walking_primitive_kit425_only.json")
SUBJECT = "425"


def main():
    with open(SRC) as f:
        meta = json.load(f)

    kept = {}
    for key, v in meta.items():
        mm = re.match(r"\d+-KIT_(\d+)_", key)
        if mm and mm.group(1) == SUBJECT:
            kept[key] = v

    kept_sorted = dict(sorted(kept.items(), key=lambda kv: abs(kv[1]["v_x_mean_mid"])))
    with open(OUT, "w") as f:
        json.dump(kept_sorted, f, indent=2)

    print(f"Extracted {len(kept_sorted)} clips for KIT_{SUBJECT}:")
    for k, v in kept_sorted.items():
        print(f"  |v_x|={abs(v['v_x_mean_mid']):.3f}  dur={v['duration_s']:.2f}s  {k}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스크립트 실행**

Run: `python scripts/data/build_kit425_subset.py`

Expected output:
```
Extracted 3 clips for KIT_425:
  |v_x|=0.371  dur=6.50s  0-KIT_425_walking_03_poses
  |v_x|=0.652  dur=5.40s  0-KIT_425_walking_medium08_poses
  |v_x|=0.845  dur=4.23s  0-KIT_425_walking_medium05_poses
Wrote sample_data/amass_isaac_walking_primitive_kit425_only.json
```

- [ ] **Step 3: 커밋**

```bash
git add scripts/data/build_kit425_subset.py \
        sample_data/amass_isaac_walking_primitive_kit425_only.json
git commit -m "data: KIT_425 3-clip forward-walking subset for per-subject experiment"
```

---

## Task 2: KIT_425 에 대한 HS 검출 품질 재확인

**Files:**
- Modify: `scripts/data/check_hs_detection_22clips.py` (CLI 인자화)

- [ ] **Step 1: 스크립트 일반화**

`scripts/data/check_hs_detection_22clips.py` 의 `main()` 을:

```python
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="sample_data/amass_isaac_walking_primitive_fwd_only.json",
                   help="Forward-subset metadata JSON path")
    p.add_argument("--pkl", default="sample_data/amass_isaac_walking_primitive.pkl",
                   help="Source motion pkl")
    args = p.parse_args()

    pkl_path = Path(args.pkl)
    fwd_json = Path(args.json)
    # ... 기존 로직 그대로 ...
```

- [ ] **Step 2: KIT_425 서브셋에 대해 실행**

Run: `PYTHONIOENCODING=utf-8 python scripts/data/check_hs_detection_22clips.py --json sample_data/amass_isaac_walking_primitive_kit425_only.json`

Expected: `3/3 detection success (100.0%)`, stride CoV 각 0.19~0.23 범위.

CoV > 0.35 면 regression — 스크립트 diff 점검.

- [ ] **Step 3: 커밋**

```bash
git add scripts/data/check_hs_detection_22clips.py
git commit -m "tools: make HS-detection dry-run script take --json arg"
```

---

## Task 3: 로컬 단위 테스트 작성 (heel-strike detector)

**Files:**
- Create: `tests/test_gait_phase_per_clip.py`

- [ ] **Step 1: 실패할 테스트 먼저 작성**

Path: `tests/test_gait_phase_per_clip.py`

```python
"""Local unit test for gait-phase detector (CPU-only, no IsaacGym)."""
import math
import pytest
import torch

from phc.env.tasks.humanoid_im_vic import (
    _detect_heel_strikes_static,
    _build_phase_from_hs_static,
)


def _make_walking_signal(stride_s=1.0, duration_s=5.0, fps=30):
    t = torch.linspace(0, duration_s - 1e-4, int(duration_s * fps))
    z = 0.5 - 0.2 * torch.cos(2 * math.pi * t / stride_s)
    return z, t


def test_detect_heel_strikes_finds_expected_count():
    z, t = _make_walking_signal(stride_s=1.0, duration_s=5.0, fps=30)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    assert 4 <= len(hs) <= 5


def test_detect_heel_strikes_filters_close_minima():
    t = torch.linspace(0, 5 - 1e-4, 150)
    z = 0.5 - 0.2 * torch.cos(2 * math.pi * t) + 0.05 * torch.cos(2 * math.pi * t / 0.1)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    assert len(hs) <= 6


def test_build_phase_from_hs_monotone_in_cycle():
    z, t = _make_walking_signal(stride_s=1.0, duration_s=5.0, fps=30)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    phase = _build_phase_from_hs_static(t, hs, num_samples=len(t), device="cpu")
    assert phase.min() >= -1e-4
    assert phase.max() <= 1.0 + 1e-4
    for i in range(len(hs) - 1):
        idx_lo, idx_hi = hs[i], hs[i + 1]
        seg = phase[idx_lo:idx_hi]
        if len(seg) >= 2:
            diffs = seg[1:] - seg[:-1]
            assert (diffs >= -1e-3).all()


def test_detect_heel_strikes_returns_empty_on_flat():
    t = torch.linspace(0, 5 - 1e-4, 150)
    z = torch.full_like(t, 0.5)
    hs = _detect_heel_strikes_static(z, t, min_interval_s=0.5)
    assert len(hs) == 0
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `pytest tests/test_gait_phase_per_clip.py -v`

Expected: `ImportError: cannot import name '_detect_heel_strikes_static'`.

---

## Task 4: `humanoid_im_vic.py` 를 per-clip phase-table 로 리팩터링

**Files:**
- Modify: `phc/env/tasks/humanoid_im_vic.py:973-1073` (obs lookup + precompute method)
- Modify: `phc/env/tasks/humanoid_im_vic.py:1471-1488` (phase_ccf_log lookup)

- [ ] **Step 1: Module-level helper 함수 2 개 추가** (테스트에서 import 가능)

`phc/env/tasks/humanoid_im_vic.py` 파일 import block 바로 아래 (첫 class 정의 직전) 에 추가:

```python
def _detect_heel_strikes_static(z, sample_times, min_interval_s=0.5):
    """Detect right heel strikes as local minima of foot-height z with min separation.

    Pure function so it can be unit-tested without IsaacGym.
    """
    import torch
    if z.shape[0] < 3:
        return []
    local_min = (z[1:-1] < z[:-2]) & (z[1:-1] <= z[2:])
    min_indices = torch.where(local_min)[0] + 1
    if len(min_indices) == 0:
        return []
    filtered = [min_indices[0].item()]
    for i in range(1, len(min_indices)):
        if sample_times[min_indices[i]] - sample_times[filtered[-1]] > min_interval_s:
            filtered.append(min_indices[i].item())
    return filtered


def _build_phase_from_hs_static(sample_times, hs_indices, num_samples, device):
    """Build per-sample gait phase in [0,1] given heel strike indices."""
    import torch
    gait_phase = torch.zeros(num_samples, device=device)
    if len(hs_indices) < 2:
        return sample_times / (sample_times[-1] + 1e-6)

    hs_idx_tensor = torch.tensor(hs_indices, dtype=torch.long, device=device)
    hs_times = sample_times[hs_idx_tensor]

    for i in range(len(hs_times) - 1):
        t0 = hs_times[i].item()
        t1 = hs_times[i + 1].item()
        mask = (sample_times >= t0) & (sample_times < t1)
        if mask.any():
            gait_phase[mask] = (sample_times[mask] - t0) / (t1 - t0)

    stride_period = hs_times[1] - hs_times[0]
    mask_before = sample_times < hs_times[0]
    if mask_before.any():
        gait_phase[mask_before] = (
            (sample_times[mask_before] - hs_times[0] + stride_period) / stride_period
        ) % 1.0

    last_stride = hs_times[-1] - hs_times[-2]
    mask_after = sample_times >= hs_times[-1]
    if mask_after.any():
        gait_phase[mask_after] = torch.clamp(
            (sample_times[mask_after] - hs_times[-1]) / last_stride, 0.0, 1.0
        )
    return gait_phase
```

- [ ] **Step 2: 테스트 통과 확인**

Run: `pytest tests/test_gait_phase_per_clip.py -v`

Expected: 4 tests PASS.

- [ ] **Step 3: `_precompute_gait_phase` 를 per-clip 으로 리팩터링**

Replace `humanoid_im_vic.py:995-1073` (기존 method body 전체) 를:

```python
    def _precompute_gait_phase(self):
        """Pre-compute per-clip gait cycle phase tables.

        For each motion loaded into self._motion_lib, detect right heel strikes
        as local minima of R_Ankle z-height and build a 0→1 phase array sampled
        at 1000 points across the clip duration. Tables are stacked into
        self._gait_phase_table of shape [num_clips, num_samples].
        """
        num_samples = 1000
        num_clips = self._motion_lib._motion_lengths.shape[0]
        device = self.device
        r_ankle_id = self._build_key_body_ids_tensor(["R_Ankle"]).item()
        min_interval_s = getattr(self, "_vic_gait_hs_min_interval_s", 0.5)

        phase_table = torch.zeros(num_clips, num_samples, device=device)
        motion_lengths = self._motion_lib._motion_lengths.clone().to(device)
        n_hs_per_clip = torch.zeros(num_clips, dtype=torch.long, device=device)
        stride_mean_per_clip = torch.full((num_clips,), float("nan"), device=device)

        offsets = torch.zeros(num_samples, 3, device=device)

        for cid in range(num_clips):
            motion_length = motion_lengths[cid].item()
            if motion_length <= 0.1:
                phase_table[cid] = torch.linspace(0.0, 1.0, num_samples, device=device)
                continue

            sample_times = torch.linspace(0, motion_length - 1e-4, num_samples, device=device)
            motion_ids = torch.full((num_samples,), cid, dtype=torch.long, device=device)
            motion_res = self._motion_lib.get_motion_state(motion_ids, sample_times, offset=offsets)
            r_ankle_height = motion_res["rg_pos"][:, r_ankle_id, 2]

            hs_indices = _detect_heel_strikes_static(r_ankle_height, sample_times, min_interval_s)
            n_hs_per_clip[cid] = len(hs_indices)

            if len(hs_indices) >= 2:
                phase_table[cid] = _build_phase_from_hs_static(
                    sample_times, hs_indices, num_samples, device
                )
                hs_times = sample_times[torch.tensor(hs_indices, dtype=torch.long, device=device)]
                strides = hs_times[1:] - hs_times[:-1]
                stride_mean_per_clip[cid] = strides.mean()
            else:
                phase_table[cid] = sample_times / motion_length

        self._gait_phase_table = phase_table
        self._gait_phase_num_samples = num_samples
        self._gait_motion_lengths = motion_lengths

        n_ok = int((n_hs_per_clip >= 2).sum().item())
        strides_ok = stride_mean_per_clip[~torch.isnan(stride_mean_per_clip)]
        stride_range = (
            f"{strides_ok.min().item():.3f}-{strides_ok.max().item():.3f}s"
            if len(strides_ok) > 0 else "n/a"
        )
        print(
            f"[{self.__class__.__name__}] Gait phase precomputed: "
            f"{num_clips} clips, {n_ok}/{num_clips} HS-detected, "
            f"{num_clips - n_ok} linear fallback. "
            f"HS count range: [{n_hs_per_clip.min().item()}, {n_hs_per_clip.max().item()}], "
            f"stride period range: {stride_range}"
        )
```

- [ ] **Step 4: 메인 obs 경로 lookup 변경**

Replace `humanoid_im_vic.py:983-985`:

```python
            # Lookup gait cycle phase from pre-computed table
            frame_idx = (time_in_clip / self._gait_motion_length * self._gait_phase_num_samples).long()
            frame_idx = frame_idx.clamp(0, self._gait_phase_num_samples - 1)
            gait_phase = self._gait_phase_table[frame_idx]
```

with:

```python
            # Lookup per-clip gait cycle phase from pre-computed table
            cur_motion_ids = self._sampled_motion_ids[env_ids]                     # [N]
            cur_motion_len = self._gait_motion_lengths[cur_motion_ids]             # [N]
            frame_idx = (time_in_clip / cur_motion_len * self._gait_phase_num_samples).long()
            frame_idx = frame_idx.clamp(0, self._gait_phase_num_samples - 1)
            gait_phase = self._gait_phase_table[cur_motion_ids, frame_idx]         # [N]
```

- [ ] **Step 5: phase_ccf_log 경로 lookup 변경**

Replace `humanoid_im_vic.py:1480-1482`:

```python
                num_samples = len(self._gait_phase_table)
                frame_idx = (time_in_clip / motion_len * num_samples).long().clamp(0, num_samples - 1)
                gait_phase = self._gait_phase_table[frame_idx[0]].item()
```

with:

```python
                num_samples = self._gait_phase_num_samples
                frame_idx = (time_in_clip / motion_len * num_samples).long().clamp(0, num_samples - 1)
                env0_mid = self._sampled_motion_ids[0]
                gait_phase = self._gait_phase_table[env0_mid, frame_idx[0]].item()
```

- [ ] **Step 6: 커밋**

```bash
git add phc/env/tasks/humanoid_im_vic.py tests/test_gait_phase_per_clip.py
git commit -m "feat(vic): per-clip gait phase table (fix motion_id=0 hardcoding)"
```

---

## Task 5: Termination reason 로깅 + 초기 motion 샘플링 첫 30% 제한

**Files:**
- Modify: `phc/env/tasks/humanoid_im.py` (reset reason 플래그 + `_sample_time` 수정)
- Modify: `phc/learning/im_amp_players.py` (평가 loop 에서 reason 집계 + 출력)

- [ ] **Step 1: reset reason 플래그 추가**

`phc/env/tasks/humanoid_im.py:1492~` 의 `_compute_reset()` 메서드를 찾고, return 직전에 다음 라인들을 추가:

```python
        # Record reset reason for evaluation (per env):
        # 0: no reset, 1: fall, 2: clip_end, 3: max_episode
        if not hasattr(self, '_last_reset_reason'):
            self._last_reset_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._last_reset_reason[:] = 0
        # fall mask is the per-env boolean that flags imitation-error or z-threshold termination.
        # In PHC's existing code, this appears as a combined reset condition before cycle logic.
        # The exact variable name depends on the class — grep "reset_buf" and find the line
        # where pure-fall conditions are combined, and save that mask here as 'fall_mask_local'.
        # Then:
        self._last_reset_reason[fall_mask_local] = 1
        self._last_reset_reason[pass_time_motion_len & ~fall_mask_local] = 2
        self._last_reset_reason[pass_time_max & ~fall_mask_local & ~pass_time_motion_len] = 3
```

**주의**: `fall_mask_local` 의 정확한 변수명은 `_compute_reset` 본문을 읽고 확정해야 함 — 이미 fall 조건이 `self.reset_buf` 에 병합돼 있어 별도 추출이 필요. 구체 구현은 다음 step 에서.

- [ ] **Step 2: fall mask 변수명 확정**

`humanoid_im.py` 의 `_compute_reset` 메서드 전체를 읽고 (대략 line 1492~1550), reset 이 발생하는 원인별 boolean tensor 들이 어떻게 조립되는지 파악:

```bash
grep -n "reset_buf\|pass_time\|compute_humanoid_reset" phc/env/tasks/humanoid_im.py | head -30
```

`compute_humanoid_reset` (common util) 에서 오는 `reset_buf` 가 전체 reset 을 주지만, 안에 fall 조건만 따로 꺼내야 할 수도. 필요시 새 함수를 작성해 fall-only mask 를 계산. 대안으로 더 단순한 분류:

- `pass_time_motion_len` = clip 종료
- `pass_time_max` = max episode
- 나머지 reset = fall (모든 다른 실패 원인)

이 단순 분류를 쓰면:

```python
        fall_mask_local = self.reset_buf & ~pass_time_motion_len & ~pass_time_max
        self._last_reset_reason[:] = 0
        self._last_reset_reason[pass_time_motion_len] = 2
        self._last_reset_reason[pass_time_max & ~pass_time_motion_len] = 3
        self._last_reset_reason[fall_mask_local] = 1
```

이 구현으로 Step 1 코드를 정리.

- [ ] **Step 3: 초기 motion 샘플링 첫 30% 제한**

`humanoid_im.py` 의 `_sample_time` 또는 equivalent 를 찾기:

```bash
grep -n "_sample_time\|motion_start_times" phc/env/tasks/humanoid_im.py | head -20
```

대부분의 PHC variant 는 `_sample_time(self, motion_ids)` 같은 메서드에서 `torch.rand(N) * motion_lengths[motion_ids]` 형태로 랜덤 시간 샘플. 이를 cycle_motion 이 False 일 때만 `0.3 *` 곱해서 첫 30% 로 제한:

```python
    def _sample_time(self, motion_ids):
        motion_lengths = self._motion_lib._motion_lengths[motion_ids]
        if self.cycle_motion:
            # full clip range (기존 동작)
            return torch.rand(motion_ids.shape[0], device=self.device) * motion_lengths
        else:
            # cycle=False: episode 가 clip_end 에서 자연 종료되므로,
            # 시작 시간을 clip 첫 30% 로 제한해서 최소 70% 의 clip 을 재생 보장.
            return torch.rand(motion_ids.shape[0], device=self.device) * motion_lengths * 0.3
```

만약 기존 `_sample_time` 이 다른 클래스에 override 돼 있으면 (예: `humanoid_im_vic.py` 에서), 거기도 동일 로직 적용.

- [ ] **Step 4: 평가 loop 에서 termination reason 집계**

`phc/learning/im_amp_players.py` 에서 평가 메인 루프를 찾기:

```bash
grep -n "av_reward\|av_steps\|done\|reset" phc/learning/im_amp_players.py | head -20
```

done event 발생 시 `env._last_reset_reason` 을 읽어 누적하는 로직 추가. 대략:

```python
        # 집계용 카운터 초기화 (평가 시작 전)
        reason_counts = {'fall': 0, 'clip_end': 0, 'max_episode': 0, 'other': 0}

        # 매 step 평가 루프 안에서, done 이 True 인 env 마다:
        done_ids = torch.where(dones)[0]
        if len(done_ids) > 0 and hasattr(env, '_last_reset_reason'):
            for env_id in done_ids.tolist():
                r = env._last_reset_reason[env_id].item()
                if r == 1:
                    reason_counts['fall'] += 1
                elif r == 2:
                    reason_counts['clip_end'] += 1
                elif r == 3:
                    reason_counts['max_episode'] += 1
                else:
                    reason_counts['other'] += 1

        # 루프 종료 후 출력 추가:
        total_eps = sum(reason_counts.values())
        if total_eps > 0:
            print(f"Termination reasons over {total_eps} episodes:")
            for reason, cnt in reason_counts.items():
                print(f"  {reason}: {cnt} ({100*cnt/total_eps:.1f}%)")
            print(f"Fall-free rate: {100*(1 - reason_counts['fall']/total_eps):.1f}%")
            if reason_counts['clip_end'] > 0:
                print(f"Clip-end rate:  {100*reason_counts['clip_end']/total_eps:.1f}%")
```

정확한 변수명 (`dones`, `env`) 은 평가 함수 내 실제 변수에 맞춰 조정.

- [ ] **Step 5: 커밋**

```bash
git add phc/env/tasks/humanoid_im.py phc/learning/im_amp_players.py
git commit -m "feat: termination reason logging + init motion sampling 30% clamp for cycle=False"
```

---

## Task 6: 3 슬롯 실험 설정 (yaml + sbatch + README) 생성

**Files:**
- Create: 3 × env yaml, 1 × learning yaml, 6 × sbatch, 1 × README (§2 File Structure 참조)

- [ ] **Step 1: 디렉토리 생성 + 기존 NR yaml 로부터 base env 작성**

```bash
mkdir -p exp_config/forward_walking/260423_KIT425_PHASEFIX
```

`env_im_walk_vic_kit425_main.yaml` (Slot 1 — MAIN, cycle=False, phase_obs=True):

```bash
cp exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/env_im_walk_vic_multiclip_fwd_NR.yaml \
   exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_main.yaml
```

그리고 yaml 필드를 다음과 같이 편집:

| 필드 | 값 |
|---|---|
| `project_name` | `"PHC_Walk_KIT425_PHASEFIX"` |
| `notes` | `"KIT_425 Slot 1 MAIN: cycle=False, phase_obs=True, per-clip phase fix."` |
| `env.cycle_motion` | `False` |
| `env.vic_phase_obs` | `True` |
| `env.multiclip_v_cmd_range` | `[0.40, 0.82]` |
| `env.multiclip_v_nat_path` | `"sample_data/amass_isaac_walking_primitive_kit425_only.json"` |
| `env.multiclip_retime_enabled` | `False` |
| 기타 (`vic_ccf_num_groups`, network, reward, etc.) | 기존 NR yaml 그대로 유지 |

- [ ] **Step 2: Slot 2 env 작성 (NOPHASE)**

```bash
cp exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_main.yaml \
   exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_nophase.yaml
```

변경 필드:

| 필드 | 값 |
|---|---|
| `notes` | `"KIT_425 Slot 2 NOPHASE: cycle=False, phase_obs=False (ablation)."` |
| `env.vic_phase_obs` | `False` |

(cycle_motion 은 main 과 같이 False 유지)

- [ ] **Step 3: Slot 3 env 작성 (LEGACY)**

```bash
cp exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_main.yaml \
   exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_legacy.yaml
```

변경 필드:

| 필드 | 값 |
|---|---|
| `notes` | `"KIT_425 Slot 3 LEGACY: cycle=True, phase_obs=True (wrap 피해량 측정)."` |
| `env.cycle_motion` | `True` |

(phase_obs 는 main 과 같이 True 유지)

- [ ] **Step 4: 공통 learning yaml 작성**

```bash
cp exp_config/forward_walking/260421_AMASS_MULTICLIP_FWD/im_walk_vic_multiclip_fwd_NR.yaml \
   exp_config/forward_walking/260423_KIT425_PHASEFIX/im_walk_vic_kit425.yaml
```

기존 `full_experiment_name` 필드는 지우거나 비워두고, 각 sbatch 스크립트가 `--experiment_name` CLI 로 override. 네트워크 / PPO / AMP 하이퍼는 기존 그대로 유지.

- [ ] **Step 5: 3 × training sbatch 작성**

`train_kit425_main_gpu0.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=kit425_main
#SBATCH --output=logs/kit425_main_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

source ~/.bashrc
conda activate phc

# 서버 repo 경로에 맞게 편집
cd /path/to/PHC

python phc/run.py \
    --task HumanoidImVICCmdMultiClip \
    --cfg_env exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_main.yaml \
    --cfg_train exp_config/forward_walking/260423_KIT425_PHASEFIX/im_walk_vic_kit425.yaml \
    --headless --num_envs 512 \
    --experiment_name KIT425_MAIN
```

`train_kit425_nophase_gpu1.sh`: 위 스크립트 복제, `--cfg_env ..._nophase.yaml`, `--experiment_name KIT425_NOPHASE`, `--job-name=kit425_nophase`.

`train_kit425_legacy_gpu2.sh`: 동일 패턴으로 `..._legacy.yaml`, `KIT425_LEGACY`, `kit425_legacy`.

- [ ] **Step 6: 3 × test-greedy sbatch 작성**

`test_kit425_main.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=testg_kit425_main
#SBATCH --output=logs/testg_kit425_main_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate phc
cd /path/to/PHC

python phc/run.py \
    --task HumanoidImVICCmdMultiClip \
    --cfg_env exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_main.yaml \
    --cfg_train exp_config/forward_walking/260423_KIT425_PHASEFIX/im_walk_vic_kit425.yaml \
    --num_envs 1 --test --epoch -1 --no_virtual_display \
    --experiment_name KIT425_MAIN
```

`test_kit425_nophase.sh`, `test_kit425_legacy.sh`: 패턴 동일.

- [ ] **Step 7: README.md 작성**

Path: `exp_config/forward_walking/260423_KIT425_PHASEFIX/README.md`

```markdown
# KIT_425 Phase-Fix Experiment (2026-04-23)

KIT_425 피험자 3-clip 으로 multi-clip policy 학습. 2×2 factorial 의 3 cell 실행.

## 슬롯 구성

| Slot | experiment_name | yaml                                  | 특이사항 |
|------|-----------------|---------------------------------------|----------|
| #1   | KIT425_MAIN     | env_im_walk_vic_kit425_main.yaml      | cycle=False, phase_obs=True  (메인) |
| #2   | KIT425_NOPHASE  | env_im_walk_vic_kit425_nophase.yaml   | cycle=False, phase_obs=False (phase obs ablation) |
| #3   | KIT425_LEGACY   | env_im_walk_vic_kit425_legacy.yaml    | cycle=True,  phase_obs=True  (legacy wrap 비교) |

공통: KIT_425 3 clip, per-clip phase fix 코드, NR variant, v_cmd [0.40, 0.82], 20k epoch, 512 envs.

Wandb project: `PHC_Walk_KIT425_PHASEFIX` (공유).

## 평가 Metric (단일 정량값 안 만듦, 병렬 보고)

- `av_steps`          (raw, 평균 에피소드 길이, frames)
- `av_time_s`         (`av_steps × dt`)
- fall-free rate      (fall 아닌 이유로 종료된 에피소드 비율)
- clip_end rate       (cycle=False 에서 자연 clip 종료 비율)
- `max_possible_avg`  (cycle=False 시 clip 평균 길이, 참고값)

## 참고 문서

- Plan: `02_research_dev/260423_kit425_phase_fix_plan.md`
- Results (TBD): `02_research_dev/YYMMDD_kit425_phasefix_results.md`
```

- [ ] **Step 8: 실행 권한 + 커밋**

```bash
chmod +x exp_config/forward_walking/260423_KIT425_PHASEFIX/*.sh
git add exp_config/forward_walking/260423_KIT425_PHASEFIX/
git commit -m "config: KIT_425 3-slot ablation (main/nophase/legacy) setup"
```

---

## Task 7: 서버 smoke-test

**Goal:** 3 슬롯 각각의 `_precompute_gait_phase` 가 "3/3 HS-detected" 를 찍는지 확인.

- [ ] **Step 1: 서버에서 3 개 test sbatch 제출 (학습 ckpt 없으므로 load 실패 상태에서 precompute 까지만 실행)**

```bash
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_main.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_nophase.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_legacy.sh
```

- [ ] **Step 2: 각 로그에서 precompute 출력 확인**

`logs/testg_kit425_main_<jobid>.out` 등 3 개 로그에서:

```
[HumanoidImVICCmdMultiClip] Gait phase precomputed: 512 clips, 3/3 HS-detected, 509 linear fallback. HS count range: [0, 6], stride period range: 0.700-0.900s
```

(512 = motion_lib slot 수, 3 = eligible KIT_425 clip 수, 509 = non-eligible fallback)

Slot 2 (NOPHASE) 는 `_precompute_gait_phase` 가 호출되지 않아 이 라인이 안 뜸 — 정상. 대신 obs dim 이 2 적게 설정됐는지 rl-games init 로그에서 확인.

- [ ] **Step 3: 이상 있으면 rollback**

3/3 이 아닌 수치가 나오면 Task 4 의 코드 재검토. 모든 슬롯이 동일 숫자여야 함 (같은 JSON 을 쓰므로).

**No commit** — 실행만.

---

## Task 8: 본 학습 3 슬롯 병렬 제출

- [ ] **Step 1: 3 개 training sbatch 제출**

```bash
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_main_gpu0.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_nophase_gpu1.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/train_kit425_legacy_gpu2.sh
```

예상 실행 시간: 각 ~8-10 시간.

- [ ] **Step 2: 초기 1000 epoch 진행 상황 확인**

각 로그 (`logs/kit425_{main,nophase,legacy}_<jobid>.out`) 에서:
- Epoch 1000 시점 av_reward 가 -50 이상, eps_len 이 30+ 로 상승 중.
- Slot 2 (NOPHASE) 는 obs 차원이 작아 초기 수렴이 조금 다를 수 있음 — 문제는 아님.
- Slot 3 (LEGACY) 는 cycle=True 라 eps_len 가 다른 둘보다 더 높게 올라감 (구조적으로).

진행이 평탄 (eps_len 10 부근) 이면 config 오타 의심.

- [ ] **Step 3: 20k epoch 완료 대기**

결과 ckpts: `output/KIT425_{MAIN,NOPHASE,LEGACY}.pth` + milestones 8 개씩.

**No commit** — 학습 산출물은 `output/` (.gitignore).

---

## Task 9: 3 슬롯 test-greedy 평가

- [ ] **Step 1: 3 개 test sbatch 제출**

```bash
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_main.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_nophase.sh
sbatch exp_config/forward_walking/260423_KIT425_PHASEFIX/test_kit425_legacy.sh
```

- [ ] **Step 2: 각 로그에서 수치 추출**

Task 5 에서 추가한 termination reason 출력 확인:

```
av_reward: X
av_steps:  Y
Termination reasons over N episodes:
  fall: ... (X %)
  clip_end: ... (Y %)
  max_episode: ... (Z %)
  other: ...
Fall-free rate: ...
Clip-end rate:  ...
```

`av_time_s = av_steps * dt` 는 수동 계산 (dt = 1/30 ≈ 0.0333).

- [ ] **Step 3: canonical ckpt 가 best 아닐 수 있음 주의**

`torch.load('output/KIT425_*.pth')['last_mean_rewards']` 가 `-100500` sentinel 이면, numbered milestone 중 best 을 손수 선택해서 재평가:

```bash
for ckpt in output/KIT425_MAIN_00002500.pth \
            output/KIT425_MAIN_00005000.pth \
            output/KIT425_MAIN_00007500.pth \
            ...; do
    python phc/run.py \
        --task HumanoidImVICCmdMultiClip \
        --cfg_env exp_config/forward_walking/260423_KIT425_PHASEFIX/env_im_walk_vic_kit425_main.yaml \
        --cfg_train exp_config/forward_walking/260423_KIT425_PHASEFIX/im_walk_vic_kit425.yaml \
        --num_envs 1 --test --epoch $(basename $ckpt .pth | sed 's/.*_0*//') \
        --no_virtual_display --experiment_name KIT425_MAIN
done
```

가장 성능 좋은 milestone 을 Slot 1 의 "대표 수치" 로. Slot 2/3 동일 처리.

**No commit** — 로그는 logs/.

---

## Task 10: 3 슬롯 결과 분석 md 작성

**Files:**
- Create: `02_research_dev/<YYMMDD>_kit425_phasefix_results.md`  (평가 완료일 기준 YYMMDD)

- [ ] **Step 1: 문서 템플릿 작성**

구조 (한글):

```markdown
# KIT_425 3-Slot Ablation 결과 (YYYY-MM-DD)

## 1. 설정 요약
- Task: HumanoidImVICCmdMultiClip
- Clips: KIT_425 3 개 (0.37 / 0.65 / 0.85 m/s)
- v_cmd range: [0.40, 0.82]
- Variant: NR (no retime)
- Code: per-clip phase fix (commit <hash>), termination reason log (commit <hash>)

## 2. 슬롯 구성
| Slot | experiment_name | cycle_motion | phase_obs |
|---|---|---|---|
| #1 | KIT425_MAIN    | False | True  |
| #2 | KIT425_NOPHASE | False | False |
| #3 | KIT425_LEGACY  | True  | True  |

## 3. Training trajectory (milestone sampling)
| Epoch | MAIN rwd / eps_len | NOPHASE rwd / eps_len | LEGACY rwd / eps_len |
|---|---|---|---|
| 1000 | | | |
| 2500 | | | |
| ... | | | |
| 20000 | | | |

## 4. Test-greedy 결과 (best milestone 기준)
| 지표 | MAIN | NOPHASE | LEGACY |
|---|---|---|---|
| best epoch | | | |
| av_reward | | | |
| av_steps | | | |
| av_time_s | | | |
| fall-free rate | | | |
| clip_end rate | N/A (cycle=T) | | |

## 5. 해석
### 5.1 cycle_motion 의 영향 (MAIN vs LEGACY)
(fall-free rate 차이 / 수치 근거 / 결론)

### 5.2 phase_obs 의 영향 (MAIN vs NOPHASE)
(수치 차이 / phase-fix 작업의 정당성 여부)

### 5.3 종합
(가장 유용한 설정 조합 / 다음 iteration 방향)

## 6. 다음 단계
(결과에 따라 작성)
```

- [ ] **Step 2: Task 9 의 수치로 표 채우기**

Training log 에서 milestone reward / eps_len 뽑기 (매 2500 epoch). Test log 에서 §4 표 완성.

- [ ] **Step 3: §5 해석 작성**

각 ablation pair 의 차이를 수치로 근거 들어 서술. 특히:
- MAIN vs LEGACY fall-free rate 차이가 10%p 이상 → cycle_motion 이 명확히 해롭다.
- MAIN vs NOPHASE fall-free rate 차이가 5%p 미만 → phase_obs 의 기여가 미미 → 다음 iteration 에서 phase 가 아닌 다른 쪽 (contact obs, reward shaping) 으로.

- [ ] **Step 4: 커밋**

```bash
git add 02_research_dev/<YYMMDD>_kit425_phasefix_results.md
git commit -m "docs: KIT_425 3-slot ablation results (main/nophase/legacy)"
```

---

## Self-Review

- **Spec coverage**: P1 (Task 4), P2/P3/P4 (Task 1 — 데이터 교체), P5 (Task 6 의 Slot 1/3 대조), P6 (Task 5), P7 (Task 9 milestone-pick 우회), P8 은 명시적 범위 밖.
- **Placeholder 스캔**: `cd /path/to/PHC` 는 서버 경로라 실행자가 채워야 함 — 주석으로 명시. Task 5 의 `fall_mask_local` 변수명은 Step 2 에서 확정하도록 경로 지정. 나머지 "TODO" 는 없음.
- **Type consistency**: `_gait_phase_table` 이 `[num_clips, num_samples]` 로 일관, 두 lookup site 모두 `motion_ids` 인덱싱. `_gait_motion_length` (scalar) 는 `_gait_motion_lengths` (vector) 로 완전 대체. `_last_reset_reason` 은 `[num_envs]` long tensor.

---

## Execution 선택

**Plan complete and saved to `02_research_dev/260423_kit425_phase_fix_plan.md`.**

**Two execution options:**

**1. Subagent-Driven (recommended)** — task 별로 fresh subagent 배치, 중간 리뷰 포함. Task 4 (phase fix) 와 Task 5 (termination reason) 는 각각 위험 요소 (code 리팩터링) 이므로 독립 agent 로 돌리고 리뷰 권장.

**2. Inline Execution** — 현재 세션에서 `superpowers:executing-plans` 로 batch 실행. Task 사이 checkpoint 로 사용자 검토.

**어느 방식으로 진행할까?**
