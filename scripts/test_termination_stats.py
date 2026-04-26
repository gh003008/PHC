"""Run greedy test on KIT_425 PERSONAL ckpt and report termination-reason stats.

Reasons (per humanoid_im.py:1200-1206):
  1 = early termination (any body > terminationDistance from reference)
  2 = clip_end (motion finished — SUCCESS)
  3 = max_episode (hit episodeLength cap — almost never since cycle=F)

Usage:
  python scripts/test_termination_stats.py --slot B [--epoch -1] [--num_envs 32] [--max_resets 1500]
"""
from __future__ import annotations
import argparse
import os
import sys
from collections import Counter

# run.py uses both `from env...` and `from phc.env...` — so both PHC root and PHC/phc/ need to be on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, 'phc')
for p in (_PHC_PKG, _PHC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401  (must precede torch)
import torch


REASON_NAMES = {
    1: 'early_term  (body > 0.25m from ref — fall/drift)',
    2: 'clip_end    (motion finished — SUCCESS)',
    3: 'max_episode (hit episodeLength cap)',
}


def print_summary(stats, lengths, total, header=''):
    if header:
        print('\n' + '=' * 72)
        print(header)
        print('=' * 72)
    else:
        print(f"\n=== Termination summary @ {total} resets ===")
    for r in (1, 2, 3):
        n = stats.get(r, 0)
        pct = 100 * n / total if total else 0
        ep_lens = lengths.get(r, [])
        if ep_lens:
            mean_len = sum(ep_lens) / len(ep_lens)
            print(f"  [{r}] {REASON_NAMES[r]:<48} : {n:5d} ({pct:5.1f}%) | ep_len mean={mean_len:5.1f} min={min(ep_lens)} max={max(ep_lens)}")
        else:
            print(f"  [{r}] {REASON_NAMES[r]:<48} : {n:5d} ({pct:5.1f}%)")
    print()


def install_patch(max_resets: int, print_every: int):
    """Wrap HumanoidImVIC._compute_reset (which shadows the base instrumented version)
    so we can re-classify termination reasons. We pre-compute pass_time_* before the
    original runs (since _compute_reset does not expose them), then read reset_buf after.
    """
    from phc.env.tasks.humanoid_im_vic import HumanoidImVIC
    orig = HumanoidImVIC._compute_reset

    def patched(self):
        # IMAmpPlayer.restore() overrides _termination_distances to 0.5 unconditionally
        # (im_amp_players.py:41). For S1/S2/S3 this masks the env yaml's terminationDistance=100.
        # Restore env-config value so termination matches training.
        if not getattr(self, '_term_dist_restored', False):
            cfg_dist = float(self.cfg["env"].get("terminationDistance", 0.5))
            self._termination_distances[:] = cfg_dist
            self._term_dist_restored = True
            print(f"[term-stats] _termination_distances overridden back to {cfg_dist} (from env yaml)")

        # Re-derive pass-time markers (mirrors humanoid_im_vic.py:1531-1534).
        time = self.progress_buf * self.dt + self._motion_start_times + self._motion_start_times_offset
        pass_time_max = self.progress_buf >= self.max_episode_length - 1
        pass_time_motion_len = time >= self._motion_lib._motion_lengths

        orig(self)  # populates self.reset_buf

        if not hasattr(self, '_term_stats'):
            self._term_stats: Counter = Counter()
            self._term_lens: dict = {1: [], 2: [], 3: []}
            self._last_print = 0

        reset_mask = self.reset_buf.bool()
        # Classification (matches humanoid_im.py:1200-1206 semantics):
        # 1 = early_term, 2 = clip_end, 3 = max_episode
        clip_end = reset_mask & pass_time_motion_len
        max_ep = reset_mask & pass_time_max & ~pass_time_motion_len
        early_term = reset_mask & ~pass_time_motion_len & ~pass_time_max

        for env_i in torch.where(early_term)[0].tolist():
            self._term_stats[1] += 1
            self._term_lens[1].append(int(self.progress_buf[env_i].item()))
        for env_i in torch.where(clip_end)[0].tolist():
            self._term_stats[2] += 1
            self._term_lens[2].append(int(self.progress_buf[env_i].item()))
        for env_i in torch.where(max_ep)[0].tolist():
            self._term_stats[3] += 1
            self._term_lens[3].append(int(self.progress_buf[env_i].item()))

        total = sum(self._term_stats.values())
        if total - self._last_print >= print_every:
            self._last_print = total
            print_summary(self._term_stats, self._term_lens, total)
        if total >= max_resets:
            print_summary(self._term_stats, self._term_lens, total,
                          header=f'FINAL — reached max_resets={max_resets}')
            sys.exit(0)

    HumanoidImVIC._compute_reset = patched
    print(f"[term-stats] HumanoidImVIC._compute_reset patched (max_resets={max_resets}, print_every={print_every})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slot', choices=['A', 'B', 'C', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6'], required=True)
    ap.add_argument('--epoch', type=int, default=-1)
    ap.add_argument('--num_envs', type=int, default=32)
    ap.add_argument('--max_resets', type=int, default=1500)
    ap.add_argument('--print_every', type=int, default=200)
    args = ap.parse_args()

    if args.slot in ('A', 'B', 'C'):
        src_env = f'exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/env_im_walk_vic_kit425_personal_{args.slot}.yaml'
        cfg_train = 'exp_config/forward_walking/260424_KIT425_PERSONAL_VCMD/im_walk_vic_kit425_personal.yaml'
        experiment = f'KIT425_PERSONAL_{args.slot}'
        task_name = 'HumanoidImVICCmdMultiClip'
    elif args.slot in ('S1', 'S2', 'S3'):
        slot_suffix = {'S1': 'S1_termoff', 'S2': 'S2_cyclebase', 'S3': 'S3_vcmdhop'}[args.slot]
        src_env = f'exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_{slot_suffix}.yaml'
        cfg_train = 'exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml'
        experiment = f'KIT425_CONT_{args.slot}'
        task_name = 'HumanoidImVICCmdMultiClip'
    else:  # S4, S5, S6
        src_env = f'exp_config/forward_walking/260427_VIC4_VCMD/env_im_walk_vic_{args.slot}.yaml'
        cfg_train = 'exp_config/forward_walking/260427_VIC4_VCMD/im_walk_vic.yaml'
        experiment = f'VIC4_VCMD_{args.slot}'
        task_name = 'HumanoidImVICCmdRetime' if args.slot == 'S4' else 'HumanoidImVICCmdMultiClip'

    cfg_env = f'/tmp/env_kit425_{args.slot}_termstats.yaml'
    with open(src_env) as f:
        env_cfg = f.read()
    env_cfg = env_cfg.replace('num_envs: 512', f'num_envs: {args.num_envs}')
    env_cfg = env_cfg.replace('numEnvs: 512', f'numEnvs: {args.num_envs}')
    with open(cfg_env, 'w') as f:
        f.write(env_cfg)

    sys.argv = [
        'run.py',
        '--task', task_name,
        '--cfg_env', cfg_env,
        '--cfg_train', cfg_train,
        '--num_envs', str(args.num_envs),
        '--test', '--epoch', str(args.epoch),
        '--no_virtual_display', '--headless',
        '--experiment', experiment,
    ]

    install_patch(max_resets=args.max_resets, print_every=args.print_every)

    print(f"[term-stats] Running slot={args.slot} epoch={args.epoch} num_envs={args.num_envs}")
    from phc import run as phc_run
    phc_run.main()


if __name__ == '__main__':
    main()
