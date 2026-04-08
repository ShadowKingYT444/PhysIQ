"""Dataset generation pipeline for PhysIQ benchmark.

Seed management, validation, 3x oversampling, DataFrame construction.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from physiq.engine import PhysIQWorld, DT, SCALE

MASTER_SEED = 42


def _task_seed(task_name: str, master_seed: int = MASTER_SEED) -> int:
    h = hashlib.sha256(f"{task_name}{master_seed}".encode()).hexdigest()
    return int(h[:8], 16)


# ── Validation ──────────────────────────────────────────────────────────────

def validate_scenario(scenario: dict, task_type: str) -> bool:
    """Validate that a scenario produces a clear, unambiguous result."""
    try:
        world = PhysIQWorld(scenario)
        sim_time = scenario.get("simulation_time", 5.0)

        if task_type == "trajectory":
            result = world.simulate(sim_time)
            target = scenario.get("target_object", "ball")
            state = result["final_state"]
            if target not in state:
                return False
            pos = state[target]["position"]
            bounds = scenario.get("world", {}).get("bounds", {"width": 10, "height": 10})
            # Object must be within reasonable bounds
            if pos[0] < -1 or pos[0] > bounds["width"] + 1:
                return False
            if pos[1] < -1 or pos[1] > bounds["height"] + 1:
                return False
            # Must have settled somewhat (velocity < 5 m/s)
            vel = state[target]["velocity"]
            if math.hypot(vel[0], vel[1]) > 5.0:
                return False
            return True

        elif task_type == "stability":
            stable, events = world.is_stable(settle_time=3.0)
            expected = scenario.get("expected_stable")
            if expected is not None and stable != expected:
                return False
            # Check stability margin — must not be borderline
            # Re-simulate to check margin
            world2 = PhysIQWorld(scenario)
            world2.simulate(3.0)
            state = world2.get_state()
            max_displacement = 0.0
            for oid, obj_state in state.items():
                orig = None
                for odef in scenario["objects"]:
                    if odef["id"] == oid and odef.get("position"):
                        orig = odef["position"]
                        break
                if orig:
                    dx = obj_state["position"][0] - orig[0]
                    dy = obj_state["position"][1] - orig[1]
                    max_displacement = max(max_displacement, math.hypot(dx, dy))
            # Margin check: clearly stable (< 0.05 displacement) or clearly unstable (> 0.3)
            if 0.05 < max_displacement < 0.3:
                return False  # ambiguous
            return True

        elif task_type == "causal_chain":
            result = world.simulate(sim_time)
            expected_steps = scenario.get("expected_steps", 1)
            # Must have at least the expected number of collisions
            if len(result["events"]) < expected_steps:
                return False
            return True

        elif task_type in ("tool_use", "replan"):
            # For tool use, we just verify the scenario builds cleanly
            # and the goal is well-defined
            goal = scenario.get("goal")
            if not goal:
                return False
            tools = scenario.get("available_tools", [])
            if not tools:
                return False
            return True

        return True

    except Exception:
        return False


# ── Ground truth computation ────────────────────────────────────────────────

def compute_ground_truth(scenario: dict, task_type: str) -> dict:
    """Simulate scenario and extract ground truth answer."""
    world = PhysIQWorld(scenario)
    sim_time = scenario.get("simulation_time", 5.0)

    if task_type == "trajectory":
        result = world.simulate(sim_time)
        target = scenario.get("target_object", "ball")
        state = result["final_state"]
        bounds = scenario.get("world", {}).get("bounds", {"width": 10, "height": 10})
        diag = math.hypot(bounds["width"], bounds["height"])
        return {
            "final_position": state[target]["position"],
            "final_velocity": state[target]["velocity"],
            "world_diagonal": diag,
            "events": result["events"],
        }

    elif task_type == "stability":
        stable, failure_events = world.is_stable(settle_time=3.0)
        state = world.get_state()
        # Identify first failure
        first_failure = None
        if not stable and failure_events:
            first_failure = {
                "object_id": failure_events[0].get("objects", ["unknown"])[0],
                "direction": _infer_direction(failure_events[0]),
            }
        return {
            "stable": stable,
            "failure_events": [
                {
                    "object_id": e.get("objects", ["?"])[0],
                    "direction": _infer_direction(e),
                    "time": e.get("time", 0),
                }
                for e in failure_events[:5]
            ],
            "final_state": state,
        }

    elif task_type == "causal_chain":
        result = world.simulate(sim_time)
        state = result["final_state"]
        target_obj = scenario.get("target_object")
        target_pos = None
        world_diag = None
        if target_obj and target_obj in state:
            pos = state[target_obj]["position"]
            target_pos = pos
            bounds = scenario.get("world", {}).get("bounds", {"width": 10, "height": 8})
            world_diag = math.hypot(bounds["width"], bounds["height"])
        # Sort events by time for ordered scoring
        sorted_events = sorted(result["events"], key=lambda e: e.get("time", 0))
        return {
            "events": sorted_events,
            "final_state": state,
            "outcome": _describe_outcome(state, scenario),
            "target_object": target_obj,
            "target_final_position": target_pos,
            "world_diagonal": world_diag,
        }

    elif task_type in ("tool_use", "replan"):
        # Ground truth for multi-turn is computed during interaction
        return {"goal": scenario.get("goal"), "available_tools": scenario.get("available_tools", [])}

    return {}


def _infer_direction(event: dict) -> str:
    """Infer direction of failure from collision event."""
    impulse = event.get("impulse", (0, 0))
    if abs(impulse[0]) > abs(impulse[1]):
        return "right" if impulse[0] > 0 else "left"
    else:
        return "up" if impulse[1] > 0 else "topples"


def _describe_outcome(state: dict, scenario: dict) -> str:
    """Generate text description of final state for causal chain."""
    parts = []
    for oid, s in state.items():
        pos = s["position"]
        vel = s["velocity"]
        speed = math.hypot(vel[0], vel[1])
        if speed < 0.1:
            parts.append(f"{oid} at rest at ({pos[0]:.1f}, {pos[1]:.1f})")
        else:
            parts.append(f"{oid} moving at ({pos[0]:.1f}, {pos[1]:.1f})")
    return "; ".join(parts) if parts else "no dynamic objects"


# ── Dataset generation ──────────────────────────────────────────────────────

def generate_dataset(master_seed: int = MASTER_SEED) -> list[dict]:
    """Generate all validated scenarios for the benchmark."""
    from physiq.templates import TEMPLATE_REGISTRY, SCENARIO_COUNTS

    all_scenarios = []

    for task_type, templates in TEMPLATE_REGISTRY.items():
        task_seed = _task_seed(task_type, master_seed)
        rng = np.random.RandomState(task_seed)
        counts = SCENARIO_COUNTS[task_type]

        for difficulty in ["easy", "medium", "hard"]:
            target_count = counts[difficulty]
            collected = []
            max_attempts = target_count * 3  # 3x oversampling

            # Shuffle templates so scenarios get diversity across templates
            eligible = [t for t in templates
                        if not hasattr(t, "difficulties") or difficulty in t.difficulties]
            shuffled = list(eligible)
            rng.shuffle(shuffled)

            for attempt in range(max_attempts):
                if len(collected) >= target_count:
                    break

                seed = int(rng.randint(0, 2**31))
                template = shuffled[attempt % len(shuffled)]

                try:
                    scenario = template.generate(difficulty, seed)
                except Exception:
                    continue

                if validate_scenario(scenario, task_type):
                    scenario["_ground_truth"] = compute_ground_truth(scenario, task_type)
                    collected.append(scenario)

            all_scenarios.extend(collected)

    return all_scenarios


# ── DataFrame construction ──────────────────────────────────────────────────

def build_evaluation_dataframes(
    scenarios: list[dict],
) -> dict[str, pd.DataFrame]:
    """Build per-task evaluation DataFrames with all 3 formats.

    Returns dict mapping task_type → DataFrame with columns:
        scenario_id, difficulty, representation, prompt, ground_truth
    """
    from physiq.formats import build_prompt

    task_dfs = {}
    by_task: dict[str, list[dict]] = {}
    for s in scenarios:
        tt = s["task_type"]
        by_task.setdefault(tt, []).append(s)

    for task_type, task_scenarios in by_task.items():
        rows = []
        for s in task_scenarios:
            gt = s.get("_ground_truth", {})
            for fmt in ["json", "ascii", "nl"]:
                prompt = build_prompt(s, fmt, task_type)
                rows.append({
                    "scenario_id": s["id"],
                    "difficulty": s["difficulty"],
                    "representation": fmt,
                    "prompt": prompt,
                    "ground_truth": json.dumps(gt),
                    "scenario_json": json.dumps(
                        {k: v for k, v in s.items() if k != "_ground_truth"}
                    ),
                })
        task_dfs[task_type] = pd.DataFrame(rows)

    return task_dfs
