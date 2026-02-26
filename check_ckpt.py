import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_path = "log/HumanoidImVIC/env_im_walk_vic/VIC01_26-12-46-46/summaries"

ea = EventAccumulator(log_path)
ea.Reload()

print("Available scalar tags:")
print(ea.Tags()["scalars"])
print()

# reward 관련 태그 출력
for tag in ea.Tags()["scalars"]:
    if any(k in tag.lower() for k in ["reward", "episode", "mean"]):
        events = ea.Scalars(tag)
        if events:
            # 구간별로 샘플링
            total = len(events)
            indices = list(range(0, total, max(1, total // 15))) + [total - 1]
            print(f"[{tag}]  (total {total} steps)")
            for i in indices:
                e = events[i]
                print(f"  step={e.step:7d}  value={e.value:.3f}")
            print()
