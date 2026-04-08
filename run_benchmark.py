"""PhysIQ Benchmark — end-to-end pipeline runner.

Generates scenarios, scores ground-truth answers, and saves all results
to outputs/.  No LLM calls required; this exercises the physics engine,
scenario generation, formatting, and scoring subsystems only.

Usage:
    python run_benchmark.py [--scenarios N] [--seed SEED]

Outputs (written to outputs/):
    scenarios.json          — all generated scenarios with ground truth
    scores.csv              — per-scenario scoring results
    summary_report.txt      — human-readable summary
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Setup ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# Force UTF-8 on stdout/stderr so ASCII art and special chars print cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Scenario generation ───────────────────────────────────────────────────────

def generate_test_scenarios(
    scenarios_per_task: int = 4,
    seed: int = 42,
    difficulties: list[str] | None = None,
) -> list[dict]:
    """Generate a small validated dataset across all 5 task types."""
    from physiq.templates import get_registry
    from physiq.generation import validate_scenario, compute_ground_truth

    if difficulties is None:
        difficulties = ["easy", "medium", "hard"]

    registry = get_registry()
    all_scenarios: list[dict] = []
    stats: dict[str, dict] = {}

    per_diff = max(1, scenarios_per_task // len(difficulties))

    for task_type, templates in registry.items():
        task_stats = {"generated": 0, "attempts": 0, "failed_validate": 0}
        task_seed = seed ^ hash(task_type) & 0xFFFFFFFF
        rng = np.random.RandomState(task_seed & 0x7FFFFFFF)

        for difficulty in difficulties:
            collected: list[dict] = []
            max_attempts = per_diff * 10

            # Shuffle templates so small subsets get diversity across templates
            eligible = [t for t in templates
                        if not hasattr(t, "difficulties") or difficulty in t.difficulties]
            shuffled = list(eligible)
            rng.shuffle(shuffled)

            for attempt in range(max_attempts):
                if len(collected) >= per_diff:
                    break
                task_stats["attempts"] += 1

                s = int(rng.randint(0, 2**31))
                template = shuffled[attempt % len(shuffled)]

                try:
                    scenario = template.generate(difficulty, s)
                except Exception:
                    task_stats["failed_validate"] += 1
                    continue

                if validate_scenario(scenario, task_type):
                    try:
                        scenario["_ground_truth"] = compute_ground_truth(scenario, task_type)
                    except Exception:
                        continue
                    collected.append(scenario)
                else:
                    task_stats["failed_validate"] += 1

            task_stats["generated"] += len(collected)
            all_scenarios.extend(collected)

        stats[task_type] = task_stats

    print(f"Generated {len(all_scenarios)} scenarios total:")
    for task_type, s in stats.items():
        print(f"  {task_type}: {s['generated']} scenarios "
              f"({s['attempts']} attempts, {s['failed_validate']} failed validation)")

    return all_scenarios


# ── Ground-truth scoring ──────────────────────────────────────────────────────

def score_ground_truth(scenario: dict) -> dict:
    """Score a scenario using its own ground truth (upper-bound self-check).

    For trajectory: feed exact position back in.
    For stability: feed exact stable/unstable judgment back in.
    For causal_chain: feed all events and outcome back in.
    For tool_use / replan: simulate a successful run using the goal definition.

    Returns a dict with score, task_type, scenario_id, difficulty.
    """
    from physiq.scoring import (
        score_trajectory, score_stability, score_causal_chain,
        score_tool_use, score_replan,
    )

    task_type = scenario["task_type"]
    gt = scenario.get("_ground_truth", {})
    sid = scenario.get("id", "unknown")
    difficulty = scenario.get("difficulty", "unknown")

    result = {
        "scenario_id": sid,
        "task_type": task_type,
        "difficulty": difficulty,
        "score": 0.0,
        "notes": "",
    }

    try:
        if task_type == "trajectory":
            pos = gt.get("final_position", [0, 0])
            diag = gt.get("world_diagonal", 14.0)
            s = score_trajectory(tuple(pos), tuple(pos), world_diagonal=diag)
            result["score"] = s
            result["notes"] = f"pos={pos}, diag={diag:.2f}"

        elif task_type == "stability":
            stable = gt.get("stable", True)
            failure_events = gt.get("failure_events", [])
            # Build a predicted_failure string from the actual events
            pred_failure = ""
            if not stable and failure_events:
                fe = failure_events[0]
                pred_failure = f"{fe.get('object_id', '')} {fe.get('direction', '')}"
            s = score_stability(stable, pred_failure, stable, failure_events)
            result["score"] = s
            result["notes"] = f"stable={stable}, events={len(failure_events)}"

        elif task_type == "causal_chain":
            events = gt.get("events", [])
            outcome = gt.get("outcome", "")
            # Feed events back as predicted steps
            pred_steps = []
            for evt in events:
                objs = evt.get("objects", [])
                interaction = evt.get("interaction", "collision")
                if objs:
                    pred_steps.append(f"{objs[0]} {interaction}")
            s = score_causal_chain(pred_steps, outcome, events, outcome)
            result["score"] = s
            result["notes"] = f"events={len(events)}, outcome_len={len(outcome)}"

        elif task_type == "tool_use":
            # Simulate perfect run: goal achieved in 1 turn
            s = score_tool_use(
                goal_achieved=True,
                turns_used=1,
                max_turns=10,
                reasoning_valid=True,
            )
            result["score"] = s
            result["notes"] = "perfect run simulated"

        elif task_type == "replan":
            s = score_replan(
                failure_recognized=True,
                recovery_plan_valid=True,
                goal_achieved=True,
                recovery_turns=2,
            )
            result["score"] = s
            result["notes"] = "perfect recovery simulated"

    except Exception as e:
        result["notes"] = f"ERROR: {e}"
        result["score"] = 0.0

    return result


# ── Format coverage check ─────────────────────────────────────────────────────

def check_formats(scenario: dict) -> dict[str, bool]:
    """Verify all three representation formats produce non-empty output."""
    from physiq.formats import format_as_json, format_as_ascii, format_as_nl
    results = {}
    for name, fn in [("json", format_as_json), ("ascii", format_as_ascii), ("nl", format_as_nl)]:
        try:
            out = fn(scenario)
            results[name] = bool(out and len(out) > 10)
        except Exception:
            results[name] = False
    return results


# ── Save outputs ──────────────────────────────────────────────────────────────

def save_scenarios_json(scenarios: list[dict], path: Path) -> None:
    """Serialize scenarios (without internal _ground_truth) to JSON."""
    out = []
    for s in scenarios:
        row = {k: v for k, v in s.items() if k != "_ground_truth"}
        row["ground_truth"] = s.get("_ground_truth", {})
        out.append(row)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved {len(out)} scenarios → {path}")


def save_scores_csv(score_rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(score_rows)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} score rows → {path}")


def save_summary_report(
    scenarios: list[dict],
    score_rows: list[dict],
    format_checks: list[dict],
    path: Path,
    elapsed: float,
) -> None:
    from physiq.scoring import physiq_score, format_robustness_score

    df = pd.DataFrame(score_rows)

    lines = [
        "=" * 60,
        "PhysIQ Benchmark — Ground-Truth Verification Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed: {elapsed:.1f}s",
        "=" * 60,
        "",
        f"Total scenarios generated: {len(scenarios)}",
        "",
        "--- Scenarios by Task ---",
    ]

    by_task = df.groupby("task_type")["score"]
    task_means: dict[str, float] = {}
    for task_type in ["trajectory", "stability", "causal_chain", "tool_use", "replan"]:
        task_df = df[df["task_type"] == task_type]
        if task_df.empty:
            lines.append(f"  {task_type:20s}: no scenarios")
            continue
        by_diff = task_df.groupby("difficulty")["score"].mean()
        mean = task_df["score"].mean()
        task_means[task_type] = mean
        diff_str = "  ".join(f"{d}={v:.3f}" for d, v in by_diff.items())
        lines.append(f"  {task_type:20s}: mean={mean:.3f}  [{diff_str}]  n={len(task_df)}")

    lines += [
        "",
        "--- Scoring functions (ground-truth self-check) ---",
        "  Expected: trajectory=1.0, stability=1.0, causal_chain>=0.5",
        "  tool_use=1.0 (perfect run), replan=1.0 (perfect recovery)",
    ]

    # Aggregate PhysIQ score
    # Map replan -> replanning for scoring module
    mapped = {("replanning" if k == "replan" else k): v for k, v in task_means.items()}
    agg = physiq_score(mapped)
    lines += [
        "",
        f"  Aggregate PhysIQ score (ground truth): {agg:.3f}",
    ]

    # Format coverage
    lines += ["", "--- Format Coverage ---"]
    fmt_ok = {"json": 0, "ascii": 0, "nl": 0}
    fmt_total = len(format_checks)
    for fc in format_checks:
        for fmt in ("json", "ascii", "nl"):
            if fc.get(fmt):
                fmt_ok[fmt] += 1
    for fmt in ("json", "ascii", "nl"):
        lines.append(f"  {fmt:8s}: {fmt_ok[fmt]}/{fmt_total} scenarios OK")

    # Format robustness score (using mean scores per format as proxy)
    lines += ["", "--- Output Files ---"]
    lines.append(f"  outputs/scenarios.json")
    lines.append(f"  outputs/scores.csv")
    lines.append(f"  outputs/summary_report.txt")

    lines += ["", "=" * 60]

    report = "\n".join(lines)
    print(report)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Saved summary → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run PhysIQ benchmark pipeline")
    parser.add_argument("--scenarios", type=int, default=4,
                        help="Scenarios per task type (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed (default: 42)")
    args = parser.parse_args()

    import time
    t0 = time.time()

    print(f"\n{'='*60}")
    print("PhysIQ Benchmark Pipeline")
    print(f"Scenarios per task: {args.scenarios}, seed: {args.seed}")
    print(f"{'='*60}\n")

    # 1. Generate scenarios
    print("--- Step 1: Generating scenarios ---")
    try:
        scenarios = generate_test_scenarios(
            scenarios_per_task=args.scenarios,
            seed=args.seed,
        )
    except Exception as e:
        print(f"FATAL: scenario generation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not scenarios:
        print("FATAL: no scenarios generated")
        sys.exit(1)

    # 2. Score with ground-truth answers
    print("\n--- Step 2: Scoring (ground-truth self-check) ---")
    score_rows = []
    for s in scenarios:
        row = score_ground_truth(s)
        score_rows.append(row)
        flag = "OK" if row["score"] > 0 else "WARN"
        print(f"  [{flag}] {s['id']:35s} {row['task_type']:15s} "
              f"{row['difficulty']:8s} score={row['score']:.3f}")

    # 3. Check format coverage
    print("\n--- Step 3: Format coverage check ---")
    format_checks = []
    for s in scenarios:
        fc = check_formats(s)
        fc["scenario_id"] = s["id"]
        format_checks.append(fc)
        status = "OK" if all(fc[k] for k in ("json", "ascii", "nl")) else "PARTIAL"
        print(f"  [{status}] {s['id']:35s}  json={fc['json']} ascii={fc['ascii']} nl={fc['nl']}")

    # 4. Save outputs
    print("\n--- Step 4: Saving outputs ---")
    save_scenarios_json(scenarios, OUTPUTS / "scenarios.json")
    save_scores_csv(score_rows, OUTPUTS / "scores.csv")

    elapsed = time.time() - t0
    save_summary_report(scenarios, score_rows, format_checks,
                        OUTPUTS / "summary_report.txt", elapsed)

    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
