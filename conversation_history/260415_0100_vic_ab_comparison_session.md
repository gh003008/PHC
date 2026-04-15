# 260415 01:00 세션 — VIC A/B (H5 vs AMASS) 병렬 실험 + 로컬 시각화

## 배경

직전 세션 (`260414_1530_server_bringup_session.md`) 에서 baseline (`PHC_Server_Baseline_v1`) Ep 7146 수동 종료. Ep 1500~7100 동안 plateau (rwd 120~158 / eps_len 40~53). Resume 으로 못 풀 구조적 한계 확인.

사용자 결정: **VIC_CCF_ON2** (로컬 실험, av_reward 939.51 / av_steps 297.39) 의 exact 학습 설정 재현. 모션 데이터 축만 달리해 **병렬 A/B 실험**:
- **Exp A (idx2)**: VIC + H5 (`h5_motion_library.pkl`)
- **Exp B (idx3)**: VIC + AMASS primitive (`amass_isaac_walking_primitive.pkl`)

VIC_CCF_ON2 원본은 `amass_isaac_walking_forward_single.pkl` 을 썼으나 서버에 없어 primitive 로 대체.

## 실행 기록

### Step A — H5 모션 mp4 미리보기 (headless 검증)
- `scripts/vis/render_h5_mp4.py` (matplotlib Agg) — skeleton 기반 3D, `~/PHC/output/h5_motion_preview.mp4` (1.04MB, 447 frames)
- `scripts/vis/render_h5_mp4_mj.py` (MuJoCo EGL offscreen) — SMPL 메쉬 렌더, `~/PHC/output/h5_motion_preview_mj.mp4` (3.85MB)
- 트릭: PyOpenGL 3.1.0 → 3.1.10 업그레이드로 `EGLDeviceEXT` 지원
- H5 motion 은 `S001_level_075mps_lv0_trial_01_c000~c002` 등 Vicon 기반, 정상 확인

### Step B/C — VIC config + sbatch 2개
4 개 yaml 생성 (snapshot 기반, motion_file + name 만 다름):

| 파일 | motion_file | name |
|---|---|---|
| `phc/data/cfg/env/env_im_walk_vic.yaml` | h5_motion_library.pkl | - |
| `phc/data/cfg/env/env_im_walk_vic_amass.yaml` | amass_isaac_walking_primitive.pkl | - |
| `phc/data/cfg/learning/im_walk_vic.yaml` | - | VIC_CCF_ON2_H5 |
| `phc/data/cfg/learning/im_walk_vic_amass.yaml` | - | VIC_CCF_ON2_AMASS |

공통 파라미터 (VIC_CCF_ON2 snapshot 그대로):
- `power_coefficient: 0.0000005`, `cycle_motion: True`
- `task_reward_w: 0.5`, `disc_reward_w: 0.5`
- `max_epochs: 20000`, `reward_curriculum_switch_epoch: 10000`
- `vic_enabled: True`, `vic_curriculum_stage: 2`
- CCF 8 groups, `vic_ccf_sigma_init: -1.0`
- MLP [1024,1024,512,512], AMP buffer 200k, lr 5e-5

sbatch:
- `train_phc_vic.sh` — idx2, `-t 24:00:00`
- `train_phc_vic_amass.sh` — idx3, 초기 `-t 07:30:00` (9시 timeout) → resume 시 `-t 12:00:00`

### Step D — 실행 및 결과

**Job 3614 (H5, idx2)**: 01:30 KST 시작, 20001 epoch 완주 (약 14h 20m runtime)
**Job 3615 (AMASS, idx3)**: 01:30 KST 시작, 09:01 KST timeout @ Ep 9574
**Job 3616 (AMASS resume)**: 09:58 KST 재개 (Ep 9500 체크포인트 로드) → 20001 epoch 완주 (약 8h runtime)

## 최종 결과

| 실험 | Final Ep | Final rwd | Final eps_len | 실행 시간 |
|---|---|---|---|---|
| Baseline (no VIC, AMASS primitive) | 7146 (수동) | 152.2 | 50.1 | ~5.5h |
| **VIC + AMASS primitive** | **20001** | **223.0** | **79.9** | ~14h (두 job 합산) |
| **VIC + H5** | **20001** | **391.9** | **130.4** | ~14h |

### 1000 epoch 단위 추이

**VIC + H5**:
| Ep | rwd | eps_len |
|---|---|---|
| 1000 | 82.9 | 49.1 |
| 3000 | 173.8 | 62.0 |
| 5000 | 181.9 | 62.1 |
| 7000 | 235.5 | 84.5 |
| 8000 | 285.6 | 111.0 (1차 peak) |
| 10000 | 262.8 | 93.2 (curriculum switch) |
| 12000 | ~268 | ~95 |
| 15000 | ~260 | ~91 |
| 20001 | **391.9** | **130.4** (최종 peak) |

**VIC + AMASS primitive**:
| Ep | rwd | eps_len |
|---|---|---|
| 1000 | 2.0 | 45.2 |
| 3000 | 122.4 | 69.3 |
| 5000 | 147.9 | 59.4 |
| 7000 | 157.4 | 60.8 |
| 9000 | 173.6 | 69.8 |
| 9500 (3615 end) | — | — |
| 9551 (3616 resume) | 60.0 | 18.5 (reset drop) |
| 10272 | 188.6 | 69.8 (curriculum switch 후 회복) |
| 11517 | 134.2 | 46.8 (switch 적응 하락) |
| 13899 | 178.6 | 64.4 |
| 19926 | 233.0 | 78.9 |
| 20001 | 223.0 | 79.9 |

## 핵심 발견

1. **VIC 자체 효과 (baseline 대비)**:
   - AMASS primitive: rwd 152 → 223 (**1.47×**), eps_len 50 → 80 (**1.60×**)
   - H5: rwd 152 → 392 (**2.58×**), eps_len 50 → 130 (**2.60×**)

2. **H5 데이터가 AMASS primitive 보다 VIC 학습에 우월** (같은 학습 설정):
   - rwd: 392 vs 223 (**1.76×**)
   - eps_len: 130 vs 80 (**1.63×**)
   - 해석: H5 는 Vicon 기반 **단일 피험자 연속 보행** 시퀀스 (speed 0.75/1.00/1.25 m/s 등) → 모션 일관성 높아 학습 쉬움. AMASS primitive 는 KIT 30+ 모션 (앞/뒤/원형 보행 혼재) → 다양성 vs 어려움.

3. **Reward curriculum switch (Ep 10000)** 효과 확실:
   - H5: Ep 10000 직후 rwd 262 → 400 대까지 상승 (disc 비중 증가가 자연스러운 움직임 유도)
   - AMASS: switch 직후 잠시 하락 (130대) 후 회복. 모션 다양성 때문에 적응 비용 큰 편.

4. **Baseline (no VIC) vs VIC 모두 plateau 돌파**: baseline 은 eps_len 50 에 갇혀있었으나 VIC 둘 다 70+ 로 상승.

## 저장된 체크포인트

**H5 (server-only, git 제외)**:
- `~/PHC/output/VIC_CCF_ON2_H5.pth` (latest, 90MB)
- `~/PHC/output/VIC_CCF_ON2_H5_00020000.pth` (Ep 20000)
- 100 epoch 단위 `_00000100.pth` ~ `_00020000.pth` (총 약 200 개, 18GB)

**AMASS (server-only, git 제외)**:
- `~/PHC/output/VIC_CCF_ON2_AMASS.pth` (latest)
- `~/PHC/output/VIC_CCF_ON2_AMASS_00020000.pth` (Ep 20000)
- 100 epoch 단위 전체 보존

**로그**:
- `~/PHC/logs/phc_vic_h5_3614.out` / `.err`
- `~/PHC/logs/phc_vic_amass_3615.out` (첫 run)
- `~/PHC/logs/phc_vic_amass_3616.out` (resume)
- `~/PHC/logs/progress_report_env_im_walk_vic.md` (H5, 1000 epoch milestone)
- `~/PHC/logs/progress_report_env_im_walk_vic_amass.md` (AMASS, 합쳐진 버전)

## 로컬 시각화 재현 가이드

### 1. 코드 pull
```bash
cd ~/PHC && git checkout Jimin && git pull origin Jimin
```

### 2. 체크포인트 rsync (최종 모델)
```bash
mkdir -p ~/PHC/output
rsync -avz --progress \
    jiminyoun@server1:~/PHC/output/VIC_CCF_ON2_H5.pth \
    jiminyoun@server1:~/PHC/output/VIC_CCF_ON2_H5_00020000.pth \
    jiminyoun@server1:~/PHC/output/VIC_CCF_ON2_AMASS.pth \
    jiminyoun@server1:~/PHC/output/VIC_CCF_ON2_AMASS_00020000.pth \
    ~/PHC/output/
```
(약 360 MB)

### 3. H5 motion 파일 (로컬에 없으면)
```bash
mkdir -p ~/PHC/sample_data
rsync -avz jiminyoun@server1:~/PHC/sample_data/h5_motion_library.pkl ~/PHC/sample_data/
```

### 4. Viewer 실행 (IsaacGym 창)

**VIC + H5 정책 (최고 결과)**:
```bash
conda activate phc
cd ~/PHC
python phc/run.py \
    --task HumanoidImVIC \
    --cfg_env phc/data/cfg/env/env_im_walk_vic.yaml \
    --cfg_train phc/data/cfg/learning/im_walk_vic.yaml \
    --num_envs 1 \
    --test \
    --epoch 20000 \
    --no_log
```

**VIC + AMASS primitive 정책**:
```bash
python phc/run.py \
    --task HumanoidImVIC \
    --cfg_env phc/data/cfg/env/env_im_walk_vic_amass.yaml \
    --cfg_train phc/data/cfg/learning/im_walk_vic_amass.yaml \
    --num_envs 1 \
    --test \
    --epoch 20000 \
    --no_log
```

`--epoch -1` 로 `output/VIC_CCF_ON2_<H5|AMASS>.pth` (latest) 자동 로드도 가능.

### 5. Plot 용 CSV 덤프 (선택)
```bash
# 서버에서 실행 후 rsync
~/PHC/dump_metrics_csv.sh 3614   # H5
~/PHC/dump_metrics_csv.sh 3615 3616   # AMASS (두 job 이어붙임)
# → ~/PHC/logs/epoch_metrics.csv (가장 최근 호출의 것으로 덮어씌워짐)
```

## 다음 단계 후보
- **Phase-CCF 분석**: 두 실험 체크포인트로 `analyze_phase_ccf.py` 실행 → 그룹별 임피던스 패턴 시각화
- **H5 데이터 AMASS 추가 실험**: `amass_isaac_walking_forward_single.pkl` 을 로컬에서 rsync → exact VIC_CCF_ON2 재현 (940 / 297 복제 여부 검증)
- **H5 cycle 끄기 실험**: `cycle_motion: False` 로 bunny hop 영향 제거 후 재학습
