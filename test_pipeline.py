"""End-to-end smoke test for the PhysIQ benchmark pipeline."""

import os
import sys
import json
import traceback

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[ENV] .env loaded")
except ImportError:
    print("[ENV] python-dotenv not installed")

# ── 1. Physics engine ─────────────────────────────────────────────────────────
print("\n=== 1. Physics Engine ===")
try:
    from physiq.engine import PhysIQWorld, parse_action, parse_coordinates

    scenario = {
        "task_type": "trajectory",
        "id": "test_traj_01",
        "difficulty": "easy",
        "simulation_time": 2.0,
        "target_object": "ball",
        "world": {"gravity": [0, -9.81], "bounds": {"width": 6, "height": 4}, "damping": 0.99},
        "objects": [
            {
                "id": "floor",
                "type": "static_segment",
                "start": [-0.5, 0.0],
                "end": [6.5, 0.0],
                "radius": 0.05,
                "material": {"friction": 0.4, "elasticity": 0.3, "density": 1.0},
            },
            {
                "id": "ball",
                "type": "circle",
                "radius": 0.15,
                "position": [1.0, 3.0],
                "velocity": [2.0, 0.0],
                "material": "rubber",
            },
        ],
    }

    world = PhysIQWorld(scenario)
    result = world.simulate(2.0)
    state = result["final_state"]
    pos = state["ball"]["position"]
    print(f"  [OK] Simulation ran: ball at {pos}, events={len(result['events'])}")

    # Test action parsing
    action = parse_action("ACTION: PUSH ball WITH_FORCE 50 DIRECTION 45")
    assert action and action["type"] == "PUSH" and action["force"] == 50.0
    print(f"  [OK] parse_action: {action}")

    coord = parse_coordinates("The ball lands at ANSWER: [3.14, 0.5]")
    assert coord == (3.14, 0.5)
    print(f"  [OK] parse_coordinates: {coord}")

except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ── 2. Materials ──────────────────────────────────────────────────────────────
print("\n=== 2. Materials ===")
try:
    from physiq.materials import resolve_material
    mat = resolve_material({"material": "rubber"})
    assert mat["friction"] == 0.8 and mat["elasticity"] == 0.8
    print(f"  [OK] rubber: {mat}")
    mat2 = resolve_material({"material": {"friction": 0.3, "elasticity": 0.5, "density": 1.0}})
    assert mat2["friction"] == 0.3
    print(f"  [OK] custom material: {mat2}")
except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ── 3. Format representations ─────────────────────────────────────────────────
print("\n=== 3. Format Representations ===")
try:
    from physiq.formats import format_as_json, format_as_ascii, format_as_nl, build_prompt

    scenario_snap = {
        "world": {"gravity": [0, -9.81], "bounds": {"width": 6, "height": 4}},
        "objects": [
            {"id": "ball", "type": "circle", "radius": 0.15,
             "position": [1.0, 3.0], "velocity": [2.0, 0.0], "material": "rubber"},
            {"id": "floor", "type": "static_segment",
             "start": [-0.5, 0.0], "end": [6.5, 0.0], "radius": 0.05,
             "material": {"friction": 0.4, "elasticity": 0.3, "density": 1.0}},
        ],
    }

    j = format_as_json(scenario_snap)
    assert len(j) > 10
    print(f"  [OK] JSON ({len(j)} chars): {j[:80]}...")

    a = format_as_ascii(scenario_snap)
    assert len(a) > 10
    print(f"  [OK] ASCII ({len(a)} chars): {a[:80].encode('ascii', errors='replace').decode()!r}...")

    nl = format_as_nl(scenario_snap)
    assert len(nl) > 10
    print(f"  [OK] NL ({len(nl)} chars): {nl[:150]}...")

    # build_prompt
    full_scenario = dict(scenario)
    full_scenario["task_type"] = "trajectory"
    prompt = build_prompt(full_scenario, "json", "trajectory")
    assert len(prompt) > 50
    print(f"  [OK] build_prompt (json, trajectory): {len(prompt)} chars")

    prompt_nl = build_prompt(full_scenario, "nl", "trajectory")
    print(f"  [OK] build_prompt (nl, trajectory): {len(prompt_nl)} chars")

except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ── 4. Scoring functions ──────────────────────────────────────────────────────
print("\n=== 4. Scoring ===")
try:
    from physiq.scoring import (
        score_trajectory, score_stability, score_causal_chain,
        score_tool_use, score_replan, physiq_score, format_robustness_score,
        TASK_WEIGHTS
    )

    # Trajectory
    s = score_trajectory((3.14, 0.5), (3.14, 0.5), world_diagonal=7.21)
    assert s == 1.0, f"Expected 1.0 got {s}"
    s2 = score_trajectory((0.0, 0.0), (5.0, 5.0), world_diagonal=7.21)
    assert s2 == 0.0, f"Expected 0.0 got {s2}"
    print(f"  [OK] score_trajectory: perfect=1.0, far=0.0")

    # Stability — new scoring: stable max=0.5, unstable max=1.0
    s = score_stability(True, "", True, [])
    assert s == 0.3, f"Expected 0.3 (binary only, no reasoning), got {s}"
    s_reasoned = score_stability(True, "wide base and symmetric geometry supports the center of mass", True, [])
    assert s_reasoned == 0.5, f"Expected 0.5 (binary + reasoning), got {s_reasoned}"
    s2 = score_stability(False, "ball topples left", False, [{"object_id": "ball", "direction": "left"}])
    assert s2 > 0.5, f"Expected >0.5, got {s2}"
    s_wrong = score_stability(True, "looks stable", False, [{"object_id": "block_0", "direction": "right"}])
    assert s_wrong < 0.5, f"Wrongly predicted stable on unstable scenario should score <0.5, got {s_wrong}"
    print(f"  [OK] score_stability: stable_binary={s:.2f}, stable_with_reasoning={s_reasoned:.2f}, unstable_match={s2:.2f}, wrong_prediction={s_wrong:.2f}")

    # Causal chain
    events = [{"objects": ["A", "B"], "interaction": "collision"}]
    s = score_causal_chain(["A hits B"], "B moves", events, "B moves")
    print(f"  [OK] score_causal_chain: {s:.2f}")

    # Tool use
    s = score_tool_use(True, 3, 10, True)
    print(f"  [OK] score_tool_use goal+efficient: {s:.2f}")
    s2 = score_tool_use(False, 0, 10, False, progress=0.5)
    print(f"  [OK] score_tool_use no goal, 50% progress: {s2:.2f}")

    # Replan
    s = score_replan(True, True, True, 2)
    print(f"  [OK] score_replan full: {s:.2f}")

    # Aggregate
    task_scores = {"trajectory": 0.8, "stability": 0.7, "causal_chain": 0.6, "tool_use": 0.5, "replanning": 0.4}
    agg = physiq_score(task_scores)
    print(f"  [OK] physiq_score: {agg:.3f}")

    frs = format_robustness_score({"json": 0.8, "ascii": 0.7, "nl": 0.6})
    print(f"  [OK] format_robustness_score: {frs:.3f}")

    print(f"  [OK] TASK_WEIGHTS sum = {sum(TASK_WEIGHTS.values()):.1f}")

except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ── 5. Scenario generation (small test set) ───────────────────────────────────
print("\n=== 5. Scenario Generation (2 per task, easy only) ===")
try:
    from physiq.templates import get_registry, SCENARIO_COUNTS
    from physiq.generation import validate_scenario, compute_ground_truth
    import numpy as np

    registry = get_registry()
    print(f"  [OK] Templates loaded: {', '.join(f'{k}={len(v)}' for k, v in registry.items())}")

    generated = {}
    for task_type, templates in registry.items():
        rng = np.random.RandomState(12345)
        collected = []
        attempts = 0
        while len(collected) < 2 and attempts < 30:
            seed = int(rng.randint(0, 2**31))
            template = templates[attempts % len(templates)]
            attempts += 1
            try:
                scenario = template.generate("easy", seed)
            except Exception as ex:
                print(f"    [{task_type}] generate failed: {ex}")
                continue
            if validate_scenario(scenario, task_type):
                scenario["_ground_truth"] = compute_ground_truth(scenario, task_type)
                collected.append(scenario)

        generated[task_type] = collected
        status = f"[OK] {len(collected)}/2" if len(collected) == 2 else f"[WARN] only {len(collected)}/2"
        print(f"  {status} {task_type} scenarios generated")

    print(f"  Total scenarios: {sum(len(v) for v in generated.values())}")

except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ── 6. DataFrame construction ─────────────────────────────────────────────────
print("\n=== 6. DataFrame Construction ===")
try:
    from physiq.generation import build_evaluation_dataframes
    import pandas as pd

    all_scenarios = []
    for task_type, scenarios in generated.items():
        all_scenarios.extend(scenarios)

    dfs = build_evaluation_dataframes(all_scenarios)
    for task, df in dfs.items():
        assert set(df.columns) >= {"scenario_id", "difficulty", "representation", "prompt", "ground_truth"}
        assert len(df) > 0
        print(f"  [OK] {task}: {len(df)} rows, cols={list(df.columns)}")

except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ── 7. Multi-turn: stability check ────────────────────────────────────────────
print("\n=== 7. Multi-turn Action Loop (tool_use) ===")
try:
    from physiq.engine import PhysIQWorld, parse_action

    tool_scenario = {
        "task_type": "tool_use",
        "id": "test_tool_01",
        "difficulty": "easy",
        "world": {"gravity": [0, -9.81], "bounds": {"width": 8, "height": 6}, "damping": 0.99},
        "objects": [
            {"id": "floor", "type": "static_segment",
             "start": [-0.5, 0.0], "end": [8.5, 0.0], "radius": 0.05,
             "material": {"friction": 0.4, "elasticity": 0.3, "density": 1.0}},
            {"id": "target", "type": "circle", "radius": 0.2,
             "position": [7.0, 0.3], "velocity": [0, 0], "material": "rubber"},
        ],
        "available_tools": [
            {"id": "ramp", "type": "static_segment",
             "start": [2.0, 1.0], "end": [4.0, 0.0],
             "radius": 0.05,
             "material": {"friction": 0.3, "elasticity": 0.5, "density": 1.0}},
        ],
        "goal": {"type": "position", "object_id": "target", "target_position": [4.0, 0.3], "tolerance": 2.0},
    }

    world = PhysIQWorld(tool_scenario)

    # Test a PUSH action
    action = parse_action("ACTION: PUSH target WITH_FORCE 100 DIRECTION 180")
    assert action is not None
    result = world.execute_action(action)
    print(f"  [OK] PUSH result: {result}")

    goal_met = world.check_goal()
    progress = world.measure_progress()
    state_desc = world.get_state_description("json")
    print(f"  [OK] goal_met={goal_met}, progress={progress:.2f}")
    print(f"  [OK] state_description ({len(state_desc)} chars)")

    # Test perturbation
    world.apply_perturbation({"type": "position_drift", "object_id": "target", "dx": 0.1, "dy": 0.0})
    print(f"  [OK] perturbation applied")

except Exception as e:
    print(f"  [FAIL] {e}")
    traceback.print_exc()

# ── 8. API key check ──────────────────────────────────────────────────────────
print("\n=== 8. API Key Check ===")
api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
if api_key:
    masked = api_key[:8] + "..." + api_key[-4:]
    print(f"  [OK] API key found: {masked}")
else:
    print("  [WARN] No ANTHROPIC_API_KEY or OPENAI_API_KEY found in environment")
    # List what keys ARE present
    env_keys = [k for k in os.environ if "key" in k.lower() or "api" in k.lower() or "token" in k.lower()]
    print(f"  Available env keys: {env_keys[:10]}")

print("\n=== DONE ===")
