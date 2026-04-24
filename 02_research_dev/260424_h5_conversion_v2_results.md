# H5 변환 v2 결과 분석 (2026-04-24)

## 1. 실행 요약

적용 fix 3종 (+ 기존 armfix shoulder sign fix):

1. Baseline 차감 (`--baseline_s 3.0`, pelvis만 적용 — 사용자 승인 편차)
2. Spine/Thorax y,z 축 매핑 (`--spine_3axis`, left channel only for y,z)
3. Neck/Head 매핑 (`--upper_body`, left channel only for y,z)

최종 pkl: `sample_data/h5_walk_S001_lv0_trial01_v2_full.pkl`

비교 artifact:
- `output/h5_visual_check/v2_full_vs_armfix_metrics.png` (Task 5에서 생성)
- `output/h5_visual_check/v2_full_vs_armfix_compare.mp4`
- `output/h5_visual_check/v2_full_vs_amass_metrics.png` (Task 6 Step 1)
- `output/h5_visual_check/v2_full_vs_amass_compare.mp4`
- `output/h5_visual_check/armfix_vs_v2_full_metrics.png` (Task 6 Step 2)
- `output/h5_visual_check/armfix_vs_v2_full_compare.mp4`

---

## 2. 정량 비교

AMASS 클립: `0-KIT_11_WalkingStraightForwards05_poses` (L=5.40s, 측정 구간 0.3–3.8s)
H5 클립: `S001_level_100mps_lv0_trial_01` (L=182.63s, 측정 구간 10–30s)

| 지표 | armfix (pre-v2) | v2_full | AMASS 참조 | 판정 |
|---|---|---|---|---|
| Trunk lean mean (sagittal) | +13.94° | **+4.32°** | +3.18° | AMASS에 근접 (개선폭 −9.6°) |
| Trunk lean std | 1.69° | 1.57° | 2.39° | 유사 범위 |
| Head-rel-pelvis x std | 14.2 mm | 16.4 mm | 22.9 mm | v2_full < AMASS (합리적) |
| Head-rel-pelvis y std | 24.3 mm | **43.6 mm** | 8.9 mm | v2_full > AMASS (lateral 과대) |
| Head-rel-pelvis z std | 5.0 mm | 3.4 mm | 2.7 mm | 미세 감소, AMASS 근접 |
| Shoulder-line axial std | 2.60° | **5.41°** | 5.18° | AMASS와 거의 동일 |
| Walking speed | 1.00 m/s | 1.00 m/s | 0.64 m/s | H5는 트레드밀 속도 고정 |
| Gait period | 1.18 s | 1.18 s | 1.40 s | H5 cadence 더 빠름 |
| Pelvis bob | 4.28 cm | 4.28 cm | 4.99 cm | 유사 |
| LK↔LW correlation | −0.88 | −0.88 | −0.88 | 세 PKL 모두 동일 (arm swing 정상) |

참고: shoulder axial std는 joint 14(L_shoulder) – joint 19(R_shoulder) 벡터 수평각 기준 계산.
Task 3에서 보고된 2.59° → 5.02° 수치는 동일 신호에 다른 angular wrapping 방식 사용.

상세 플롯: `output/h5_visual_check/v2_full_vs_armfix_metrics.png` / `v2_full_vs_amass_metrics.png` 참고.

---

## 3. 각 Fix별 기여 분석

### Fix 1: Baseline 차감 (pelvis만)

목적: PiG standing calibration offset 제거 (standing trial 3.0s 평균 차감).

결과: trunk lean +13.94° → +5.83° (pelvis-only 적용 시, Task 2 측정값). v2_full 최종에서는 +4.32°로 추가 개선.

승인된 편차: 원 계획은 모든 PiG 채널에 적용이었으나, spine x의 경우 standing = −9.6° (자연스러운 요추 전만), walking = +4.2° (굴곡)으로 부호가 달라 평균 차감 시 오히려 lean이 악화 (+14.64°). Pelvis만 적용으로 결정.

### Fix 2: Spine/Thorax 3-axis

목적: H5의 y/z 축 spine motion (~7–13°) 반영 — 기존엔 x축(sagittal)만 매핑.

결과: shoulder-line axial std 2.60° → 5.41° (+2.81°), AMASS 참조값 5.18°와 거의 일치.

구현 편차: PiG가 좌/우 midline joint를 mirror convention으로 리포트 → L+R 평균 시 y,z가 ~0으로 상쇄. Left 채널만 사용. 이로 인해 부호 방향 검증 필요 (영상 시각 확인 항목 참고).

### Fix 3: Neck/Head mapping

목적: PiG neck(평균 −13°, 머리 숙임) / head 채널을 SMPL joint 12/15에 매핑 — 기존엔 identity.

결과: head-rel-pelvis y std 24.3 mm → 43.6 mm (lateral head motion 증가), x std 14.2 → 16.4 mm.

주의: AMASS 참조의 head y std는 8.9 mm로, v2_full(43.6 mm)이 약 5배 크다. Lateral head motion 과대 가능성 있음 — 영상에서 시각 확인 필요.

Spine 3-axis와 동일 패턴: x는 L+R 평균, y,z는 left only.

---

## 4. 미검증 항목 (시각 확인 필요)

- [ ] v2_full 영상에서 armfix 대비 실제로 (a) trunk가 덜 굽혀 보이는지, (b) 상체 lateral sway가 자연스러운지, (c) head-down 자세가 보이는지
- [ ] head-rel-pelvis y std가 AMASS(8.9 mm) 대비 v2_full(43.6 mm)로 약 5배 큰 것이 영상에서 눈에 띄는 이상 동작으로 나타나는지
- [ ] spine3d / upper_body의 y,z 부호가 해부학적으로 올바른지 — left 채널만 사용한 것이 좌우 반대로 나오지 않는지 mp4에서 확인
- [ ] AMASS 대비 남아 있는 차이 (예: arm swing amplitude, stance foot contact, walking speed 차이로 인한 gait pattern 비교 한계)

---

## 5. Step C 진행 여부 결정

주요 수치 요약:
- Trunk lean: armfix +13.94° → v2_full +4.32° (AMASS +3.18°에 근접, 개선 명확)
- Shoulder axial rotation: armfix 2.60° → v2_full 5.41° (AMASS 5.18°와 거의 동일, 개선 명확)
- Head lateral motion: v2_full 43.6 mm (AMASS 8.9 mm 대비 과대 — 추가 검토 여지 있음)

- [ ] v2_full로 VIC 학습 시도할 준비 완료 — 선생님 시각 검토 후 체크
- [ ] 추가 수정 필요 — 무엇: (head y std 과대 원인 조사 / upper_body fix y,z 스케일 조정)

(Task 7 준비됨: `exp_config/forward_walking/260424_VIC_H5_V2/` 스냅샷 + train.sh 대기. 시각 검토 후 실행 판단.)
