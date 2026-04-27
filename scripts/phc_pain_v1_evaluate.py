#!/usr/bin/env python3
"""Evaluate PHC-Pain-v1.0 mechanism-proof summary metrics.

Input is a JSON file with condition-level aggregate metrics. This script does
not parse raw IsaacGym logs; it is the final gate over extracted metrics so the
mechanism claim stays explicit and auditable.
"""

import argparse
import json
import sys


REQUIRED_CONDITIONS = ("main", "no_obs", "no_reward")
REQUIRED_METRICS = (
    "affected_pain_avg",
    "affected_pain_p95",
    "unaffected_pain_avg",
    "episode_length_avg",
    "no_fall_rate",
    "forward_progress_avg",
)


def _get(condition, metric):
    if metric not in condition:
        raise KeyError(metric)
    return float(condition[metric])


def _check_required(data):
    errors = []
    for name in REQUIRED_CONDITIONS:
        if name not in data:
            errors.append(f"missing condition: {name}")
            continue
        for metric in REQUIRED_METRICS:
            if metric not in data[name]:
                errors.append(f"missing {name}.{metric}")
    return errors


def evaluate(data, args):
    errors = _check_required(data)
    if errors:
        return "BLOCKED", errors

    main = data["main"]
    no_reward = data["no_reward"]
    no_obs = data["no_obs"]

    findings = []

    main_pain = _get(main, "affected_pain_avg")
    no_reward_pain = _get(no_reward, "affected_pain_avg")
    no_obs_pain = _get(no_obs, "affected_pain_avg")

    reduction_vs_no_reward = no_reward_pain - main_pain
    reduction_vs_no_obs = no_obs_pain - main_pain
    pain_pass = (
        reduction_vs_no_reward >= args.min_pain_reduction
        and reduction_vs_no_obs >= args.min_obs_benefit
    )
    findings.append(
        f"pain_reduction: {'PASS' if pain_pass else 'FAIL'} "
        f"(vs_no_reward={reduction_vs_no_reward:.4f}, "
        f"vs_no_obs={reduction_vs_no_obs:.4f})"
    )

    competence_pass = (
        _get(main, "no_fall_rate") >= args.min_no_fall_rate
        and _get(main, "episode_length_avg") >= args.min_episode_length
        and _get(main, "forward_progress_avg") >= args.min_forward_progress
    )
    findings.append(
        f"locomotion_competence: {'PASS' if competence_pass else 'FAIL'}"
    )

    affected_drop = _get(no_reward, "affected_pain_p95") - _get(main, "affected_pain_p95")
    unaffected_drop = _get(no_reward, "unaffected_pain_avg") - _get(main, "unaffected_pain_avg")
    side_pass = affected_drop > max(unaffected_drop, 0.0)
    findings.append(
        f"side_specificity: {'PASS' if side_pass else 'FAIL'} "
        f"(affected_p95_drop={affected_drop:.4f}, "
        f"unaffected_avg_drop={unaffected_drop:.4f})"
    )

    verdict = "PASS" if pain_pass and competence_pass and side_pass else "FAIL"
    return verdict, findings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json")
    parser.add_argument("--min-pain-reduction", type=float, default=0.05)
    parser.add_argument("--min-obs-benefit", type=float, default=0.01)
    parser.add_argument("--min-no-fall-rate", type=float, default=0.95)
    parser.add_argument("--min-episode-length", type=float, default=250.0)
    parser.add_argument("--min-forward-progress", type=float, default=0.0)
    args = parser.parse_args()

    try:
        with open(args.summary_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        print(f"PHC_PAIN_V1_VERDICT=BLOCKED")
        print(f"input_error: {exc}")
        return 2

    verdict, findings = evaluate(data, args)
    print(f"PHC_PAIN_V1_VERDICT={verdict}")
    for finding in findings:
        print(finding)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
