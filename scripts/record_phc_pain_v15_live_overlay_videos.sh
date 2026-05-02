#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="analysis/plots/phc_pain_v15_trials/videos"
LOG_DIR="analysis/plots/phc_pain_v15_trials/videos/live_overlay_logs"
mkdir -p "${OUT_DIR}"
mkdir -p "${LOG_DIR}"

BASELINE="output/HumanoidIm/phc_shape_pnn_iccv/Humanoid.pth"
MOTION="sample_data/phc_pain_overlay_single_walk.pkl"
DISPLAY_ID="${DISPLAY:-:1}"
FPS="${PHC_SCREEN_CAPTURE_FPS:-30}"
WINDOW_WARMUP_SECONDS="${PHC_ISAAC_WINDOW_WARMUP_SECONDS:-12}"
RECORD_SECONDS="${PHC_SCREEN_CAPTURE_SECONDS:-10}"
WINDOW_WAIT_SECONDS="${PHC_ISAAC_WINDOW_WAIT_SECONDS:-45}"
WINDOW_CLOSE_WAIT_SECONDS="${PHC_ISAAC_WINDOW_CLOSE_WAIT_SECONDS:-10}"

find_isaac_window() {
  xwininfo -root -tree 2>/dev/null | awk '/Isaac Gym/ {print $1; exit}'
}

wait_for_isaac_window() {
  local waited=0
  local window_id=""
  while [[ "${waited}" -lt "${WINDOW_WAIT_SECONDS}" ]]; do
    window_id="$(find_isaac_window)"
    if [[ -n "${window_id}" ]]; then
      printf '%s\n' "${window_id}"
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

wait_for_isaac_window_closed() {
  local waited=0
  while [[ "${waited}" -lt "${WINDOW_CLOSE_WAIT_SECONDS}" ]]; do
    if [[ -z "$(find_isaac_window)" ]]; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

terminate_process_tree() {
  local root_pid="$1"
  local signal="$2"
  local child
  while read -r child; do
    [[ -n "${child}" ]] || continue
    terminate_process_tree "${child}" "${signal}"
  done < <(pgrep -P "${root_pid}" 2>/dev/null || true)
  kill "-${signal}" "${root_pid}" 2>/dev/null || true
}

stop_play() {
  local play_pid="$1"
  if kill -0 "${play_pid}" 2>/dev/null; then
    terminate_process_tree "${play_pid}" INT
    sleep 1
  fi
  if kill -0 "${play_pid}" 2>/dev/null; then
    terminate_process_tree "${play_pid}" TERM
    sleep 1
  fi
  if kill -0 "${play_pid}" 2>/dev/null; then
    terminate_process_tree "${play_pid}" KILL
  fi
  wait "${play_pid}" 2>/dev/null || true
  wait_for_isaac_window_closed || true
}

run_play() {
  local exp_name="$1"
  PHC_COMPARE_SECONDARY_CHECKPOINT="${BASELINE}" \
  PHC_VIEW_CAMERA="${PHC_VIEW_CAMERA:-sagittal_right}" \
  PHC_VIEW_ENV="${PHC_VIEW_ENV:-0}" \
  PHC_VIEW_HEIGHT="${PHC_VIEW_HEIGHT:-0.45}" \
  PHC_VIEW_TARGET_Z="${PHC_VIEW_TARGET_Z:-0.55}" \
  PHC_VIEW_DISTANCE="${PHC_VIEW_DISTANCE:-3.2}" \
  conda run --no-capture-output -n phc python phc/run_hydra.py \
    learning=im_pnn \
    exp_name="${exp_name}" \
    epoch=-1 \
    test=True \
    env=env_im_pain_v1 \
    env.num_prim=4 \
    robot=smpl_humanoid_shape \
    robot.has_shape_variation=False \
    robot.freeze_hand=True \
    robot.box_body=False \
    robot.has_shape_obs_disc=True \
    env.motion_file="${MOTION}" \
    env.num_envs=2 \
    env.env_spacing=0.0 \
    headless=False \
    no_virtual_display=True \
    learning.params.config.player.games_num=50000000 \
    +learning.params.config.player.render_sleep=0.033
}

record_trial() {
  local label="$1"
  local exp_name="$2"
  local out_name="$3"
  local out_path="${OUT_DIR}/${out_name}"
  local log_path="${LOG_DIR}/${out_name%.mp4}.log"

  echo "=== Live screen recording ${label}: blue=PHC pretrained, red=${exp_name} ==="
  run_play "${exp_name}" > "${log_path}" 2>&1 &
  local play_pid=$!

  local window_id
  if ! window_id="$(wait_for_isaac_window)"; then
    stop_play "${play_pid}"
    echo "Could not find Isaac Gym window for ${exp_name}; see ${log_path}" >&2
    return 1
  fi

  sleep "${WINDOW_WARMUP_SECONDS}"

  set +e
  ffmpeg -y -v error \
    -f x11grab \
    -draw_mouse 0 \
    -framerate "${FPS}" \
    -window_id "${window_id}" \
    -i "${DISPLAY_ID}" \
    -t "${RECORD_SECONDS}" \
    -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" \
    -c:v libx264 \
    -preset ultrafast \
    -crf 18 \
    -pix_fmt yuv420p \
    "${out_path}"
  local ffmpeg_status=$?
  set -e

  stop_play "${play_pid}"
  if [[ "${ffmpeg_status}" -ne 0 ]]; then
    echo "Screen recording failed for ${exp_name}; see ${log_path}" >&2
    return "${ffmpeg_status}"
  fi
  echo "Wrote ${out_path}"
}

record_trial \
  "v1.5 OA sanity/long" \
  "phc_shape_pnn_iccv_pain_v15_oa_sanity_idx3" \
  "01_v15_oa_sanity_vs_pretrained_sagittal_right.mp4"

record_trial \
  "aggressive pain" \
  "phc_shape_pnn_iccv_pain_v15_oa_aggressive_idx3" \
  "02_aggressive_vs_pretrained_sagittal_right.mp4"

record_trial \
  "middle pain" \
  "phc_shape_pnn_iccv_pain_v15_oa_middle_idx3" \
  "03_middle_vs_pretrained_sagittal_right.mp4"

record_trial \
  "walk-gated contact regularized" \
  "phc_shape_pnn_iccv_pain_v15_oa_walkgate_idx3" \
  "04_walkgate_vs_pretrained_sagittal_right.mp4"

record_trial \
  "moment-biased pain" \
  "phc_shape_pnn_iccv_pain_v15_oa_momentbias_idx3" \
  "05_momentbias_vs_pretrained_sagittal_right.mp4"
