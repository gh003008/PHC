
import glob
import json
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
import torch
from isaacgym import gymapi
from phc.utils.flags import flags
from rl_games.algos_torch import torch_ext
from rl_games.common.player import BasePlayer

import learning.amp_players as amp_players
from tqdm import tqdm
import joblib
import time
from smpl_sim.smpllib.smpl_eval import compute_metrics_lite
from rl_games.common.tr_helpers import unsqueeze_obs
from datetime import datetime
import copy
COLLECT_Z = False

class IMAMPPlayerContinuous(amp_players.AMPPlayerContinuous):
    def __init__(self, config):
        super().__init__(config)
        self._compare_overlay_enabled = False
        self._compare_model = None
        self._compare_running_mean_std = None

        self.terminate_state = torch.zeros(self.env.task.num_envs, device=self.device)
        self.terminate_memory = []

        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.curr_stpes = 0

        if COLLECT_Z:
            self.zs, self.zs_all = [], []

        humanoid_env = self.env.task
        humanoid_env._termination_distances[:] = 0.5 # if not humanoid_env.strict_eval else 0.25 # ZL: use UHC's termination distance
        humanoid_env._recovery_episode_prob, humanoid_env._fall_init_prob = 0, 0
        self._apply_play_visual_flags()
        self._apply_base_body_color(humanoid_env)
        self._setup_compare_overlay(humanoid_env)

        if humanoid_env.collect_dataset:
            self.obs_buf, self.obs_buf_all = [], []
            self.env_actions, self.actions_all = [], []
            self.motion_length_all = []
            self.clean_actions, self.clean_actions_all = [], []
            self.keys_all = []
            self.reset_buf, self.reset_buf_all = [], []

        if flags.im_eval:
            self.success_rate = 0
            self.pbar = tqdm(range(humanoid_env._motion_lib._num_unique_motions // humanoid_env.num_envs))
            humanoid_env.zero_out_far = False
            humanoid_env.zero_out_far_train = False
            
            if len(humanoid_env._reset_bodies_id) > 15:
                humanoid_env._reset_bodies_id = humanoid_env._eval_track_bodies_id  # Following UHC. Only do it for full body, not for three point/two point trackings. 
            
            humanoid_env.cycle_motion = False
            self.print_stats = False
        
        # joblib.dump({"mlp": self.model.a2c_network.actor_mlp, "mu": self.model.a2c_network.mu}, "single_model.pkl") # ZL: for saving part of the model.
        return

    def _apply_play_visual_flags(self):
        show_traj = os.environ.get("PHC_SHOW_TRAJ")
        if show_traj is not None:
            flags.show_traj = show_traj.lower() not in ("0", "false", "no", "off")

    def _apply_base_body_color(self, humanoid_env):
        raw_color = os.environ.get("PHC_VIS_BASE_BODY_COLOR")
        if not raw_color:
            return
        try:
            color = np.array([float(v) for v in raw_color.split(",")], dtype=np.float32)
        except ValueError:
            return
        if color.shape[0] != 3:
            return
        humanoid_env.set_char_color(color, list(range(humanoid_env.num_envs)))

    def _setup_compare_overlay(self, humanoid_env):
        secondary_path = os.environ.get("PHC_COMPARE_SECONDARY_CHECKPOINT")
        if not secondary_path:
            return

        if humanoid_env.num_envs < 2:
            raise RuntimeError(
                "PHC_COMPARE_SECONDARY_CHECKPOINT requires env.num_envs >= 2 "
                "so env0 and env1 can show separate policies."
            )
        if self.is_rnn:
            raise RuntimeError("PHC compare overlay does not support RNN players.")

        self._compare_model = copy.deepcopy(self.model)
        self._compare_model.to(self.device)
        self._compare_model.eval()
        if self.normalize_input:
            self._compare_running_mean_std = copy.deepcopy(self.running_mean_std)
            self._compare_running_mean_std.to(self.device)
            self._compare_running_mean_std.eval()

        checkpoint = amp_players._adapt_expanded_obs_checkpoint(
            self, torch_ext.load_checkpoint(secondary_path)
        )
        self._compare_model.load_state_dict(checkpoint["model"])
        if self.normalize_input:
            self._compare_running_mean_std.load_state_dict(checkpoint["running_mean_std"])

        humanoid_env.set_char_color(np.array([0.12, 0.38, 0.95]), [0])
        humanoid_env.set_char_color(np.array([0.95, 0.12, 0.08]), [1])

        self._compare_overlay_enabled = True
        print(
            "PHC_COMPARE_OVERLAY: env0 blue=secondary/pretrained, "
            "env1 red=primary/current."
        )

    def _preproc_obs_with_stats(self, obs_batch, running_mean_std):
        if type(obs_batch) is dict:
            return {
                key: self._preproc_obs_with_stats(value, running_mean_std)
                for key, value in obs_batch.items()
            }

        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
        if self.normalize_input:
            obs_batch_proc = obs_batch[:, : running_mean_std.mean_size]
            obs_batch_out = running_mean_std(obs_batch_proc)
            obs_batch = torch.cat(
                [obs_batch_out, obs_batch[:, running_mean_std.mean_size :]], dim=-1
            )
        return obs_batch

    def _get_action_from_model(
        self, obs, model, running_mean_std, is_determenistic=False
    ):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs_with_stats(obs, running_mean_std)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "rnn_states": self.states,
        }

        with torch.no_grad():
            res_dict = model(input_dict)

        current_action = res_dict["mus"] if is_determenistic else res_dict["actions"]
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return amp_players.rescale_actions(
                self.actions_low,
                self.actions_high,
                torch.clamp(current_action, -1.0, 1.0),
            )
        return current_action

    def get_action(self, obs_dict, is_determenistic=False):
        if not self._compare_overlay_enabled:
            return super().get_action(obs_dict, is_determenistic)

        obs = obs_dict["obs"]
        if obs.shape[0] < 2:
            raise RuntimeError("PHC compare overlay expected at least two env observations.")

        secondary_action = self._get_action_from_model(
            obs[:1], self._compare_model, self._compare_running_mean_std, is_determenistic
        )
        primary_action = self._get_action_from_model(
            obs[1:], self.model, self.running_mean_std, is_determenistic
        )
        return torch.cat([secondary_action, primary_action], dim=0)

    def _post_step(self, info, done):
        super()._post_step(info)
        
        
        # modify done such that games will exit and reset.
        if flags.im_eval:

            humanoid_env = self.env.task
            
            termination_state = torch.logical_and(self.curr_stpes <= humanoid_env._motion_lib.get_motion_num_steps() - 1, info["terminate"]) # if terminate after the last frame, then it is not a termination. curr_step is one step behind simulation. 
            # termination_state = info["terminate"]
            self.terminate_state = torch.logical_or(termination_state, self.terminate_state)
            if (~self.terminate_state).sum() > 0:
                max_possible_id = humanoid_env._motion_lib._num_unique_motions - 1
                curr_ids = humanoid_env._motion_lib._curr_motion_ids
                if (max_possible_id == curr_ids).sum() > 0: # When you are running out of motions. 
                    bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                    if (~self.terminate_state[:bound]).sum() > 0:
                        curr_max = humanoid_env._motion_lib.get_motion_num_steps()[:bound][~self.terminate_state[:bound]].max()
                    else:
                        curr_max = (self.curr_stpes - 1)  # the ones that should be counted have teimrated
                else:
                    curr_max = humanoid_env._motion_lib.get_motion_num_steps()[~self.terminate_state].max()

                if self.curr_stpes >= curr_max: curr_max = self.curr_stpes + 1  # For matching up the current steps and max steps. 
            else:
                curr_max = humanoid_env._motion_lib.get_motion_num_steps().max()

            if humanoid_env.collect_dataset:
                self.obs_buf.append(info['obs_buf'])
                self.clean_actions.append(info['clean_actions'])
                self.env_actions.append(info['actions'])
                self.reset_buf.append(info['reset_buf'])

            self.mpjpe.append(info["mpjpe"])
            self.gt_pos.append(info["body_pos_gt"])
            self.pred_pos.append(info["body_pos"])
            if COLLECT_Z: self.zs.append(info["z"])
            self.curr_stpes += 1

            if self.curr_stpes >= curr_max or self.terminate_state.sum() == humanoid_env.num_envs:
                
                self.terminate_memory.append(self.terminate_state.cpu().numpy())
                self.success_rate = (1 - np.concatenate(self.terminate_memory)[: humanoid_env._motion_lib._num_unique_motions].mean())

                # MPJPE
                all_mpjpe = torch.stack(self.mpjpe)
                try:
                    assert(all_mpjpe.shape[0] == curr_max or self.terminate_state.sum() == humanoid_env.num_envs) # Max should be the same as the number of frames in the motion.
                except:
                    import ipdb; ipdb.set_trace()
                    print('??')

                all_mpjpe = [all_mpjpe[: (i - 1), idx].mean() for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())] # -1 since we do not count the first frame. 
                all_body_pos_pred = np.stack(self.pred_pos)
                all_body_pos_pred = [all_body_pos_pred[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                all_body_pos_gt = np.stack(self.gt_pos)
                all_body_pos_gt = [all_body_pos_gt[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]

                if COLLECT_Z:
                    all_zs = torch.stack(self.zs)
                    all_zs = [all_zs[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.zs_all += all_zs


                if humanoid_env.collect_dataset:
                    all_obs_buf = np.stack(self.obs_buf) # Time, batch, obs
                    all_obs_buf = [all_obs_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.obs_buf_all += all_obs_buf

                    all_clean_actions = np.stack(self.clean_actions) 
                    all_clean_actions = [all_clean_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.clean_actions_all += all_clean_actions
                    
                    all_actions = np.stack(self.env_actions)
                    all_actions = [all_actions[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.actions_all += all_actions

                    all_reset_buf = np.stack(self.reset_buf)
                    all_reset_buf = [all_reset_buf[: (i - 1), idx] for idx, i in enumerate(humanoid_env._motion_lib.get_motion_num_steps())]
                    self.reset_buf_all += all_reset_buf
                    
                    self.keys_all += humanoid_env._motion_lib.curr_motion_keys.tolist()

                    self.motion_length_all += [obs.shape[0] for obs in all_obs_buf]

                self.mpjpe_all.append(all_mpjpe)
                self.pred_pos_all += all_body_pos_pred
                self.gt_pos_all += all_body_pos_gt
                

                if (humanoid_env.start_idx + humanoid_env.num_envs >= humanoid_env._motion_lib._num_unique_motions):
                    terminate_hist = np.concatenate(self.terminate_memory)
                    succ_idxes = np.nonzero(~terminate_hist[: humanoid_env._motion_lib._num_unique_motions])[0].tolist()

                    pred_pos_all_succ = [(self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]
                    gt_pos_all_succ = [(self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions])[i] for i in succ_idxes]

                    pred_pos_all = self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]
                    gt_pos_all = self.gt_pos_all[: humanoid_env._motion_lib._num_unique_motions]

                    # np.sum([i.shape[0] for i in self.pred_pos_all[:humanoid_env._motion_lib._num_unique_motions]])
                    # humanoid_env._motion_lib.get_motion_num_steps().sum()

                    failed_keys = humanoid_env._motion_lib._motion_data_keys[terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    success_keys = humanoid_env._motion_lib._motion_data_keys[~terminate_hist[: humanoid_env._motion_lib._num_unique_motions]]
                    # print("failed", humanoid_env._motion_lib._motion_data_keys[np.concatenate(self.terminate_memory)[:humanoid_env._motion_lib._num_unique_motions]])
                    if flags.real_traj:
                        pred_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all]
                        gt_pos_all = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all]
                        pred_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in pred_pos_all_succ]
                        gt_pos_all_succ = [i[:, humanoid_env._reset_bodies_id] for i in gt_pos_all_succ]
                        
                        
                        
                    metrics = compute_metrics_lite(pred_pos_all, gt_pos_all)
                    metrics_succ = compute_metrics_lite(pred_pos_all_succ, gt_pos_all_succ)

                    metrics_all_print = {m: np.mean(v) for m, v in metrics.items()}
                    metrics_print = {m: np.mean(v) for m, v in metrics_succ.items()}

                    print("------------------------------------------")
                    print("------------------------------------------")
                    print(f"Success Rate: {self.success_rate:.10f}")
                    print("All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()]))
                    print("Succ: "," \t".join([f"{k}: {v:.3f}" for k, v in metrics_print.items()]))
                    # print(1 - self.terminate_state.sum() / self.terminate_state.shape[0])
                    print(self.config['network_path'])
                    if COLLECT_Z:
                        zs_all = self.zs_all[:humanoid_env._motion_lib._num_unique_motions]
                        zs_dump = {k: zs_all[idx].cpu().numpy() for idx, k in enumerate(humanoid_env._motion_lib._motion_data_keys)}
                        joblib.dump(zs_dump, osp.join(self.config['network_path'], "zs_run.pkl"))

                    if humanoid_env.collect_dataset:
                        motion_file = humanoid_env.cfg.env.motion_file.split('/')[-1].split('.')[0]
                        dump_dir = osp.join(self.config['network_path'], "phc_act", motion_file, f"noise_{humanoid_env.add_action_noise}_{humanoid_env.action_noise_std}_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.pkl")
                        os.makedirs(osp.join(self.config['network_path'], "phc_act", motion_file), exist_ok=True)
                        print("Dumping to: ", dump_dir)
                        joblib.dump({
                                "obs": self.obs_buf_all, 
                                "clean_action": self.clean_actions_all, 
                                "env_action": self.actions_all,
                                "key_names": np.array(self.keys_all),
                                "motion_lengths": np.array(self.motion_length_all),
                                "reset": np.concatenate(self.reset_buf_all), 
                                "running_mean": self.running_mean_std.state_dict(),
                                "config": humanoid_env.cfg,
                                }, dump_dir, compress=True)
                        exit()

                    import ipdb; ipdb.set_trace()

                    joblib.dump(failed_keys, osp.join(self.config['network_path'], "failed.pkl"))
                    joblib.dump(success_keys, osp.join(self.config['network_path'], "long_succ.pkl"))
                    print("....")

                done[:] = 1  # Turning all of the sequences done and reset for the next batch of eval.

                humanoid_env.forward_motion_samples()
                self.terminate_state = torch.zeros(
                    self.env.task.num_envs, device=self.device
                )

                self.pbar.update(1)
                self.pbar.refresh()
                self.mpjpe, self.gt_pos, self.pred_pos,  = [], [], []
                if humanoid_env.collect_dataset: 
                    self.obs_buf, self.env_actions, self.clean_actions, self.reset_buf, self.keys = [], [], [], [], []
                if COLLECT_Z: self.zs = []
                self.curr_stpes = 0


            update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_stpes} | Start: {humanoid_env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}"
            self.pbar.set_description(update_str)

        return done
    
    def get_z(self, obs_dict):
        obs = obs_dict['obs']
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None,
            'obs': obs,
            'rnn_states': self.states
        }
        with torch.no_grad():
            z = self.model.a2c_network.eval_z(input_dict)
            return z

    def _apply_pain_view_camera(self, humanoid_env):
        camera_mode = os.environ.get("PHC_VIEW_CAMERA")
        if not camera_mode or not getattr(humanoid_env, "viewer", None):
            return

        try:
            humanoid_env.gym.refresh_actor_root_state_tensor(humanoid_env.sim)
            focus_env = int(os.environ.get("PHC_VIEW_ENV", "0"))
            focus_env = max(0, min(focus_env, humanoid_env.num_envs - 1))
            humanoid_env.viewing_env_idx = focus_env
            root = humanoid_env._humanoid_root_states[focus_env, 0:3].detach().cpu().numpy()
            body_pos = humanoid_env._rigid_body_pos[focus_env].detach().cpu().numpy()
        except Exception:
            return

        def _body_side_offset(sign):
            names = getattr(humanoid_env, "_body_names", [])
            left = names.index("L_Hip")
            right = names.index("R_Hip")
            side = body_pos[right, :3] - body_pos[left, :3]
            side[2] = 0.0
            norm = np.linalg.norm(side)
            if norm < 1e-5:
                side = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            else:
                side = side / norm
            distance = float(os.environ.get("PHC_VIEW_DISTANCE", "3.0"))
            height = float(os.environ.get("PHC_VIEW_HEIGHT", "0.85"))
            return sign * distance * side + np.array([0.0, 0.0, height], dtype=np.float32)

        if camera_mode == "sagittal_right":
            offset = _body_side_offset(1.0)
        elif camera_mode == "sagittal_left":
            offset = _body_side_offset(-1.0)
        elif camera_mode == "world_x_plus":
            offset = np.array([3.0, 0.0, 0.85], dtype=np.float32)
        elif camera_mode == "world_x_minus":
            offset = np.array([-3.0, 0.0, 0.85], dtype=np.float32)
        elif camera_mode == "world_y_plus":
            offset = np.array([0.0, 3.0, 0.85], dtype=np.float32)
        elif camera_mode == "world_y_minus":
            offset = np.array([0.0, -3.0, 0.85], dtype=np.float32)
        elif camera_mode == "front":
            offset = np.array([0.0, -3.0, 0.85], dtype=np.float32)
        else:
            raw_offset = os.environ.get("PHC_VIEW_CAMERA_OFFSET")
            if not raw_offset:
                return
            offset = np.array([float(v) for v in raw_offset.split(",")], dtype=np.float32)
            if offset.shape[0] != 3:
                return

        target_z = float(os.environ.get("PHC_VIEW_TARGET_Z", "0.9"))
        cam_pos = gymapi.Vec3(root[0] + offset[0], root[1] + offset[1], root[2] + offset[2])
        cam_target = gymapi.Vec3(root[0], root[1], root[2] + target_z)
        humanoid_env.gym.viewer_camera_look_at(humanoid_env.viewer, None, cam_pos, cam_target)
        if hasattr(humanoid_env, "_cam_prev_char_pos"):
            humanoid_env._cam_prev_char_pos[:] = root
        if hasattr(humanoid_env, "recorder_camera_handles") and focus_env < len(humanoid_env.recorder_camera_handles):
            humanoid_env.gym.set_camera_location(
                humanoid_env.recorder_camera_handles[focus_env],
                humanoid_env.envs[focus_env],
                cam_pos,
                cam_target,
            )

    def _maybe_start_auto_record(self, humanoid_env):
        frames = int(os.environ.get("PHC_AUTO_RECORD_FRAMES", "0"))
        if frames <= 0:
            return 0
        humanoid_env.recording = True
        humanoid_env.recording_state_change = True
        print(f"PHC_AUTO_RECORD_FRAMES: {frames}")
        return frames

    def _finish_auto_record(self, humanoid_env):
        humanoid_env.recording = False
        humanoid_env.recording_state_change = True
        self._apply_pain_view_camera(humanoid_env)
        humanoid_env.render()
        video_path = getattr(humanoid_env, "curr_video_file_name", None)
        states_path = getattr(humanoid_env, "curr_states_file_name", None)
        if video_path:
            print(f"PHC_AUTO_RECORD_VIDEO: {video_path}")
        if states_path:
            print(f"PHC_AUTO_RECORD_STATES: {states_path}")

    def _apply_pain_knee_visual(self, humanoid_env):
        if os.environ.get("PHC_VIS_RIGHT_KNEE_PAIN", "0").lower() in (
            "0", "false", "no", "off"
        ):
            return
        if not hasattr(humanoid_env, "knee_drive"):
            return
        if "right" not in humanoid_env.knee_drive:
            return
        try:
            knee_body_id = humanoid_env._body_names.index("R_Knee")
            source = os.environ.get("PHC_VIS_RIGHT_KNEE_PAIN_SOURCE", "load")
            if source == "drive":
                tensor = humanoid_env.knee_drive["right"]
            elif source == "state":
                channel_idx = humanoid_env._pain_channel_to_idx.get("right_knee")
                if channel_idx is None:
                    return
                tensor = humanoid_env.pain_body_state[:, channel_idx]
            else:
                tensor = humanoid_env.knee_load_proxy["right"]
            values = tensor.detach().cpu().numpy()
            scale = float(os.environ.get("PHC_VIS_RIGHT_KNEE_PAIN_SCALE", "1.5"))
        except Exception:
            return

        for env_id in range(min(humanoid_env.num_envs, values.shape[0])):
            intensity = float(np.clip(values[env_id] * scale, 0.0, 1.0))
            if intensity <= 1e-4:
                color = gymapi.Vec3(0.05, 0.35, 0.95)
            else:
                color = gymapi.Vec3(1.0, 0.08 + 0.12 * (1.0 - intensity), 0.04)
            humanoid_env.gym.set_rigid_body_color(
                humanoid_env.envs[env_id],
                humanoid_env.humanoid_handles[env_id],
                knee_body_id,
                gymapi.MESH_VISUAL,
                color,
            )

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        probe_path = os.environ.get("PHC_PAIN_PROBE_JSON")
        probe_scalars = {}
        timeseries_path = os.environ.get("PHC_PAIN_TIMESERIES_JSON")
        timeseries_scalars = {}
        auto_record_frames = 0
        auto_record_count = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn
        for t in range(n_games):
            if games_played >= n_games:
                break
            obs_dict = self.env_reset()
            humanoid_env = self.env.task
            self._apply_pain_view_camera(humanoid_env)
            if auto_record_frames == 0:
                auto_record_frames = self._maybe_start_auto_record(humanoid_env)

            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

            print_game_res = False

            done_indices = []

            with torch.no_grad():
                for n in range(self.max_steps):
                    obs_dict = self.env_reset(done_indices)


                    if COLLECT_Z: z = self.get_z(obs_dict)
                        

                    if has_masks:
                        masks = self.env.get_action_mask()
                        action = self.get_masked_action(obs_dict, masks, is_determenistic)
                    else:
                        action = self.get_action(obs_dict, is_determenistic)

                    obs_dict, r, done, info = self.env_step(self.env, action)
                    self._apply_pain_knee_visual(humanoid_env)

                    cr += r
                    steps += 1

                    if COLLECT_Z: info['z'] = z
                    done = self._post_step(info, done.clone())
                    if probe_path and isinstance(info, dict):
                        for key, value in info.items():
                            if not (
                                key.startswith("pain_v1_")
                                or key.startswith("lower_limb_")
                                or key in ("pain_mean", "pain_max")
                            ):
                                continue
                            if isinstance(value, (int, float, bool)):
                                probe_scalars.setdefault(key, []).append(float(value))
                    if timeseries_path and isinstance(info, dict):
                        for key, value in info.items():
                            if not (
                                key.startswith("lower_limb_")
                                or key.startswith("pain_v1_")
                                or key in ("pain_mean", "pain_max")
                            ):
                                continue
                            if isinstance(value, (int, float, bool)):
                                timeseries_scalars.setdefault(key, []).append(float(value))

                    if render:
                        self._apply_pain_view_camera(humanoid_env)
                        self.env.render(mode="human")
                        time.sleep(self.render_sleep)

                    if auto_record_frames > 0:
                        auto_record_count += 1
                        if auto_record_count >= auto_record_frames:
                            self._finish_auto_record(humanoid_env)
                            return
                        
                    all_done_indices = done.nonzero(as_tuple=False)
                    done_indices = all_done_indices[:: self.num_agents]
                    done_count = len(done_indices)
                    games_played += done_count

                    if done_count > 0:
                        if self.is_rnn:
                            for s in self.states:
                                s[:, all_done_indices, :] = (
                                    s[:, all_done_indices, :] * 0.0
                                )

                        cur_rewards = cr[done_indices].sum().item()
                        cur_steps = steps[done_indices].sum().item()

                        cr = cr * (1.0 - done.float())
                        steps = steps * (1.0 - done.float())
                        sum_rewards += cur_rewards
                        sum_steps += cur_steps

                        game_res = 0.0
                        if isinstance(info, dict):
                            if "battle_won" in info:
                                print_game_res = True
                                game_res = info.get("battle_won", 0.5)
                            if "scores" in info:
                                print_game_res = True
                                game_res = info.get("scores", 0.5)
                        if self.print_stats:
                            if print_game_res:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count, "w:", game_res,)
                            else:
                                print("reward:", cur_rewards / done_count, "steps:", cur_steps / done_count,)

                        sum_game_res += game_res
                        # if batch_size//self.num_agents == 1 or games_played >= n_games:
                        if games_played >= n_games:
                            break

                    done_indices = done_indices[:, 0]

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )
        if probe_path:
            os.makedirs(os.path.dirname(probe_path), exist_ok=True)
            with open(probe_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        key: {
                            "mean": float(np.mean(values)),
                            "last": float(values[-1]),
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "n": len(values),
                        }
                        for key, values in sorted(probe_scalars.items())
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
            print("PHC_PAIN_PROBE_JSON:", probe_path)
        if timeseries_path:
            os.makedirs(os.path.dirname(timeseries_path), exist_ok=True)
            with open(timeseries_path, "w", encoding="utf-8") as f:
                json.dump(timeseries_scalars, f, indent=2, sort_keys=True)
            print("PHC_PAIN_TIMESERIES_JSON:", timeseries_path)

        return
