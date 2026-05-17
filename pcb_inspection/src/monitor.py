"""
Monitoring Utilities
====================
- check_class_balance  : Reports label distribution across YOLO annotation files
- score_drift_monitor  : Detects anomaly score distribution shift over time
- harvest_hard_negatives: Collects edge-case images for active learning

Usage:
    python src/monitor.py --check-balance ./data/pcb_labeled/labels/train
    python src/monitor.py --drift-check   ./logs/scores.npy
"""

import argparse
import json
import numpy as np
import redis
from collections import Counter
from pathlib import Path


CLASS_NAMES = [
    "missing_component",
    "solder_bridge",
    "cold_joint",
    "trace_crack",
    "contamination",
]


# ── Class Balance ─────────────────────────────────────────────────────────────

def check_class_balance(label_dir: str) -> Counter:
    """
    Counts instances of each defect class across all YOLO .txt label files.
    Prints a visual bar chart to stdout.

    Aim for balanced counts — severe imbalance hurts mAP on minority classes.
    Target: >= 200 instances per class before Phase 2 training.
    """
    counts: Counter = Counter()
    label_path = Path(label_dir)

    txt_files = list(label_path.glob("*.txt"))
    if not txt_files:
        print(f"No .txt label files found in: {label_dir}")
        return counts

    for f in txt_files:
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                cls_id = int(line.split()[0])
                counts[cls_id] += 1

    total = sum(counts.values())

    print(f"\nClass balance — {label_dir}")
    print(f"Total annotations: {total}")
    print("-" * 52)

    for cls_id, name in enumerate(CLASS_NAMES):
        count = counts.get(cls_id, 0)
        bar   = "█" * min(count // 5, 40)
        warn  = "  ⚠ LOW" if count < 200 else ""
        print(f"  {cls_id} {name:22s}: {count:4d}  {bar}{warn}")

    print("-" * 52)

    # Imbalance warning
    if counts:
        min_cls = min(counts.values())
        max_cls = max(counts.values())
        ratio   = max_cls / (min_cls + 1e-9)
        if ratio > 5:
            print(f"  ⚠ Class imbalance ratio {ratio:.1f}x — consider oversampling"
                  " or weighted loss")
        else:
            print(f"  ✓ Class balance ratio {ratio:.1f}x — acceptable")

    return counts


# ── Score Drift Detection ─────────────────────────────────────────────────────

def score_drift_monitor(redis_host: str = "localhost",
                        redis_port: int = 6379,
                        baseline_mean: float = None,
                        drift_threshold_pct: float = 10.0,
                        sample_size: int = 500) -> dict:
    """
    Checks if anomaly scores on good boards have shifted significantly.
    A shift > 10% indicates camera/lighting drift or PCB design change.

    Args:
        baseline_mean       : Expected mean score for good boards
        drift_threshold_pct : Alert if mean shifts more than this % from baseline
        sample_size         : How many recent Redis entries to sample

    Returns:
        dict with current_mean, baseline_mean, drift_pct, alert
    """
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)

    try:
        entries = r.xrevrange("inspections", count=sample_size)
    except redis.RedisError as e:
        print(f"Redis error: {e}")
        return {}

    if not entries:
        print("No inspection records in Redis")
        return {}

    # Only look at boards that passed (good boards)
    good_scores = []
    for _, e in entries:
        if e.get(b"pass") == b"True":
            try:
                good_scores.append(float(e[b"score"]))
            except (KeyError, ValueError):
                pass

    if not good_scores:
        print("No passing boards in recent history")
        return {}

    current_mean = np.mean(good_scores)
    current_std  = np.std(good_scores)

    result = {
        "sample_size":   len(good_scores),
        "current_mean":  round(current_mean, 4),
        "current_std":   round(current_std, 4),
        "baseline_mean": baseline_mean,
        "alert":         False,
        "drift_pct":     None,
    }

    if baseline_mean is not None:
        drift_pct = abs(current_mean - baseline_mean) / (baseline_mean + 1e-9) * 100
        result["drift_pct"] = round(drift_pct, 2)
        result["alert"]     = drift_pct > drift_threshold_pct

        print(f"\nScore Drift Report")
        print(f"  Baseline mean  : {baseline_mean:.4f}")
        print(f"  Current mean   : {current_mean:.4f}  (±{current_std:.4f})")
        print(f"  Drift          : {drift_pct:.2f}%  (alert threshold: {drift_threshold_pct}%)")

        if result["alert"]:
            print(f"  ⚠ DRIFT ALERT — check camera alignment and lighting!")
        else:
            print(f"  ✓ Score distribution stable")
    else:
        print(f"\nBaseline Score (save this value):")
        print(f"  Mean  : {current_mean:.4f}")
        print(f"  Std   : {current_std:.4f}")
        print(f"\n  Use --baseline {current_mean:.4f} in future drift checks")

    return result


# ── Hard Negative Harvesting ──────────────────────────────────────────────────

def harvest_hard_negatives(redis_host: str = "localhost",
                           redis_port: int = 6379,
                           low: float = 0.4,
                           high: float = 0.6,
                           output_log: str = "./logs/hard_negatives.json") -> list:
    """
    Identifies images near the decision boundary (score between low-high).
    These are the most valuable for human review and model improvement.

    Returns list of hard negative records for labeling.
    """
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
    entries = r.xrevrange("inspections", count=5000)

    hard_negatives = []
    for entry_id, e in entries:
        try:
            score = float(e[b"score"])
            if low <= score <= high:
                hard_negatives.append({
                    "redis_id": entry_id.decode(),
                    "score": score,
                    "pass": e[b"pass"].decode(),
                    "defects": json.loads(e[b"defects"]),
                })
        except (KeyError, ValueError):
            pass

    Path(output_log).parent.mkdir(parents=True, exist_ok=True)
    with open(output_log, "w") as f:
        json.dump(hard_negatives, f, indent=2)

    print(f"\nHard Negatives (score {low}–{high}):")
    print(f"  Found  : {len(hard_negatives)} entries")
    print(f"  Saved  : {output_log}")
    print(f"  Action : Review these images — they improve threshold calibration")

    return hard_negatives


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCB Inspection Monitoring")
    parser.add_argument("--check-balance", metavar="LABEL_DIR",
                        help="Check class balance in YOLO label directory")
    parser.add_argument("--drift-check", action="store_true",
                        help="Run score drift monitor against Redis")
    parser.add_argument("--baseline", type=float, default=None,
                        help="Baseline mean score for drift comparison")
    parser.add_argument("--harvest", action="store_true",
                        help="Harvest hard negatives from Redis log")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    args = parser.parse_args()

    if args.check_balance:
        check_class_balance(args.check_balance)

    if args.drift_check:
        score_drift_monitor(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
            baseline_mean=args.baseline,
        )

    if args.harvest:
        harvest_hard_negatives(
            redis_host=args.redis_host,
            redis_port=args.redis_port,
        )
