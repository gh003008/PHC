import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

summaries = glob.glob('output/HumanoidIm/humanoid_smpl/MPL_H5_v7_*/summaries/')
if not summaries:
    print("No summaries found yet")
    sys.exit(0)

ea = EventAccumulator(summaries[0])
ea.Reload()

tags_to_check = ['rewards0/iter', 'episode_lengths/iter', 'disc/reward_mean', 'disc/agent_acc', 'rewards/returns']

for tag in tags_to_check:
    try:
        events = ea.Scalars(tag)
        if not events:
            continue
        latest = events[-1]
        # Find values at ~1000 epoch intervals
        milestones = {}
        for e in events:
            epoch_k = (e.step // 1000) * 1000
            if epoch_k > 0 and epoch_k not in milestones:
                milestones[epoch_k] = e.value
        milestones['latest'] = (latest.step, latest.value)

        if tag == 'episode_lengths/iter':
            print(f"\n{'Epoch':>8} | {'Ep Length':>10} | {'Reward':>10} | {'Disc Rwd':>10} | {'Disc AgAcc':>10} | {'Returns':>10}")
            print("-" * 75)
            break
    except:
        pass

# Collect all data at milestones
try:
    ep_len = {e.step: e.value for e in ea.Scalars('episode_lengths/iter')}
    reward = {e.step: e.value for e in ea.Scalars('rewards0/iter')}
    disc_rwd = {e.step: e.value for e in ea.Scalars('disc/reward_mean')}
    disc_acc = {e.step: e.value for e in ea.Scalars('disc/agent_acc')}
    returns = {e.step: e.value for e in ea.Scalars('rewards/returns')}

    all_steps = sorted(ep_len.keys())
    latest_step = all_steps[-1] if all_steps else 0
    print(f"Latest epoch: {latest_step}")
    print(f"\n{'Epoch':>8} | {'Ep Length':>10} | {'Reward':>10} | {'Disc Rwd':>10} | {'Disc AgAcc':>10} | {'Returns':>10}")
    print("-" * 75)

    for target in [1, 100, 500, 1000, 2000, 3000, 4000, 5000]:
        # Find closest step
        closest = min(all_steps, key=lambda s: abs(s - target)) if all_steps else None
        if closest and abs(closest - target) < 100:
            print(f"{closest:>8} | {ep_len.get(closest, 0):>10.1f} | {reward.get(closest, 0):>10.1f} | {disc_rwd.get(closest, 0):>10.4f} | {disc_acc.get(closest, 0):>10.4f} | {returns.get(closest, 0):>10.1f}")

    # Always print latest
    s = latest_step
    print(f"{s:>8} | {ep_len.get(s, 0):>10.1f} | {reward.get(s, 0):>10.1f} | {disc_rwd.get(s, 0):>10.4f} | {disc_acc.get(s, 0):>10.4f} | {returns.get(s, 0):>10.1f}  (latest)")
except Exception as ex:
    print(f"Error: {ex}")
