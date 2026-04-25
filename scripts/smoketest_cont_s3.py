"""Smoke test for Slot 3 cycle-boundary v_cmd resampling.

Verifies that with multiclip_resample_on_cycle=True, env._current_cmd[0, 0]
changes value at least once during a 200-step rollout (covering >= one
clip cycle of ~5.4s = 162 control steps).
"""
from __future__ import annotations
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PHC_ROOT = os.path.dirname(_THIS_DIR)
_PHC_PKG = os.path.join(_PHC_ROOT, 'phc')
for p in (_PHC_PKG, _PHC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(_PHC_ROOT)

import isaacgym  # noqa: F401
import torch


def main():
    cfg_env = 'exp_config/forward_walking/260426_KIT425_CONT/env_im_walk_vic_kit425_cont_S3_vcmdhop.yaml'
    cfg_train = 'exp_config/forward_walking/260426_KIT425_CONT/im_walk_vic_kit425_cont.yaml'

    sys.argv = [
        'run.py',
        '--task', 'HumanoidImVICCmdMultiClip',
        '--cfg_env', cfg_env,
        '--cfg_train', cfg_train,
        '--num_envs', '4',
        '--test', '--epoch', '-1',
        '--no_virtual_display', '--headless',
        '--experiment', 'KIT425_PERSONAL_B',  # reuse existing ckpt for the smoke test
    ]

    from phc import run as phc_run
    from phc.env.tasks.humanoid_im_vic_cmd_multiclip import HumanoidImVICCmdMultiClip

    # Monkey-patch _resample_vcmd_on_cycle to log every call. If the env never
    # calls it, we know cycle_motion + multiclip_resample_on_cycle aren't both
    # active. If it gets called, we know the new code path works.
    if not hasattr(HumanoidImVICCmdMultiClip, '_resample_vcmd_on_cycle'):
        print('[SMOKE TEST] FAIL: _resample_vcmd_on_cycle method missing — code change not applied')
        sys.exit(1)

    orig_resample = HumanoidImVICCmdMultiClip._resample_vcmd_on_cycle
    call_log = []

    def logged_resample(self, env_ids):
        before = self._current_cmd[env_ids, 0].clone() if env_ids.numel() > 0 else None
        orig_resample(self, env_ids)
        after = self._current_cmd[env_ids, 0].clone() if env_ids.numel() > 0 else None
        call_log.append({
            'n_envs': int(env_ids.numel()),
            'env_ids': env_ids.cpu().tolist(),
            'before': before.cpu().tolist() if before is not None else [],
            'after': after.cpu().tolist() if after is not None else [],
        })
        print(f"[SMOKE] _resample_vcmd_on_cycle called: n={int(env_ids.numel())} "
              f"before={before.cpu().tolist() if before is not None else []} "
              f"after={after.cpu().tolist() if after is not None else []}",
              flush=True)

    HumanoidImVICCmdMultiClip._resample_vcmd_on_cycle = logged_resample

    # Patch the player loop to exit after ~200 steps to keep test fast.
    # The player's `for t in range(n_games)` runs n_games (typically large).
    # We override to run only ~3 games.
    from phc.learning import im_amp_players
    orig_run = im_amp_players.IMAMPPlayerContinuous.run

    def short_run(self):
        # Force only 3 games to keep test fast (~10 sec).
        self.games_num = 3
        return orig_run(self)

    im_amp_players.IMAMPPlayerContinuous.run = short_run

    try:
        phc_run.main()
    except SystemExit:
        pass

    print(f"[SMOKE TEST] resample called {len(call_log)} times")
    if len(call_log) == 0:
        print('[SMOKE TEST] FAIL: _resample_vcmd_on_cycle was never called. '
              'Either cycle_motion is False, multiclip_resample_on_cycle is False, '
              'or no env crossed a cycle boundary in the test window.')
        sys.exit(1)

    # Verify v_cmd actually changed in at least one call.
    any_changed = False
    for c in call_log:
        for b, a in zip(c['before'], c['after']):
            if abs(b - a) > 1e-6:
                any_changed = True
                break
    if not any_changed:
        print('[SMOKE TEST] FAIL: resample called but v_cmd did not change '
              '(unexpected: random sample should produce a different value).')
        sys.exit(1)

    print('[SMOKE TEST] PASS — cycle-boundary v_cmd resampling works')
    sys.exit(0)


if __name__ == '__main__':
    main()
