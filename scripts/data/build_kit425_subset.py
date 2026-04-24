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
