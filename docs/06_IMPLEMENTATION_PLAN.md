# Implementation Plan

## Overview

This document maps PhysIQ to the exact Kaggle Benchmarks SDK workflow. Everything runs in a single Kaggle notebook.

---

## Phase 1: Setup & Physics Engine (Day 1-2)

### Step 1.1: Install dependencies

```python
!pip install pymunk numpy pandas
import kaggle_benchmarks as kbench
import pymunk
import numpy as np
import pandas as pd
import json
```

### Step 1.2: Build the physics simulation core

Create the world builder, simulator, and state extractor as Python functions within the notebook. Key classes:

```python
class PhysIQWorld:
    """Manages a pymunk simulation from a scenario dict."""
    
    def __init__(self, scenario: dict):
        self.space = pymunk.Space()
        self.space.gravity = tuple(scenario["world"]["gravity"])
        self.space.iterations = 20
        self.objects = {}
        self.events = []
        self._build(scenario["objects"])
    
    def _build(self, object_defs):
        """Create pymunk bodies/shapes from scenario definition."""
        ...
    
    def simulate(self, duration: float, dt=1/60) -> dict:
        """Run simulation, return final state + event log."""
        ...
    
    def get_state_description(self, fmt="json") -> str:
        """Serialize current state in requested format."""
        ...
```

### Step 1.3: Validate physics engine

Test with known scenarios (ball drop, pendulum, elastic collision) and verify against analytical solutions. This validation code stays in the notebook as proof of correctness.

---

## Phase 2: Scenario Generation (Day 2-3)

### Step 2.1: Implement template classes

One class per template type. Each generates a scenario dict given a seed.

### Step 2.2: Build the dataset

```python
def generate_dataset(master_seed=42):
    scenarios = []
    rng = np.random.RandomState(master_seed)
    
    for task_type, templates in TEMPLATE_REGISTRY.items():
        for difficulty in ["easy", "medium", "hard"]:
            count = SCENARIO_COUNTS[task_type][difficulty]
            for i in range(count * 3):  # oversample 3x for validation
                seed = rng.randint(0, 2**31)
                scenario = templates[i % len(templates)].generate(difficulty, seed)
                
                # Validate
                if validate_scenario(scenario, task_type):
                    scenarios.append(scenario)
                    if len([s for s in scenarios if s["task_type"] == task_type 
                            and s["difficulty"] == difficulty]) >= count:
                        break
    
    return scenarios
```

### Step 2.3: Generate representations

For each scenario, produce JSON, ASCII, and natural language versions:

```python
def generate_all_formats(scenario):
    return {
        "json": format_as_json(scenario),
        "ascii": format_as_ascii(scenario),
        "natural_language": format_as_nl(scenario),
    }
```

### Step 2.4: Build evaluation DataFrames

```python
# For single-turn tasks (1, 2, 3)
df_trajectory = pd.DataFrame([
    {
        "scenario_id": s["id"],
        "difficulty": s["difficulty"],
        "representation": fmt,
        "prompt": build_prompt(s, fmt, "trajectory"),
        "ground_truth": json.dumps(simulate_ground_truth(s)),
    }
    for s in trajectory_scenarios
    for fmt in ["json", "ascii", "natural_language"]
])
```

---

## Phase 3: Kaggle Benchmark Tasks (Day 3-5)

### Step 3.1: Define single-turn tasks

```python
@kbench.task(name="physiq_trajectory")
def trajectory_task(llm, prompt, ground_truth) -> float:
    """Predict where an object ends up in a 2D physics simulation."""
    response = llm.prompt(prompt)
    
    predicted = parse_coordinates(response)
    actual = json.loads(ground_truth)
    
    if predicted is None:
        return 0.0
    
    return score_trajectory(predicted, actual["final_position"], 
                           actual["world_diagonal"])

# Evaluate across dataset
trajectory_results = trajectory_task.evaluate(
    llm=[kbench.llm],
    evaluation_data=df_trajectory
)
```

### Step 3.2: Define multi-turn tasks

```python
@kbench.task(name="physiq_tool_use")
def tool_use_task(llm, scenario_json) -> float:
    """Multi-turn: model proposes actions, simulation executes them."""
    scenario = json.loads(scenario_json)
    world = PhysIQWorld(scenario)
    
    # Initial prompt
    state_desc = world.get_state_description(scenario["format"])
    goal_desc = scenario["goal_description"]
    
    prompt = f"""You are in a 2D physics world. Your goal: {goal_desc}

Current state:
{state_desc}

Available actions:
- PLACE <object> AT <x> <y> ANGLE <degrees>
- PUSH <object> FORCE <newtons> DIRECTION <degrees>  
- REMOVE <object>

Propose your next action. Format: ACTION: <your action>
Explain your reasoning first."""
    
    turns_used = 0
    max_turns = 10
    
    for turn in range(max_turns):
        response = llm.prompt(prompt)
        turns_used += 1
        
        action = parse_action(response)
        if action is None:
            prompt = "I couldn't parse your action. Please try again with format: ACTION: <action>"
            continue
        
        # Execute action in simulation
        result = world.execute_action(action)
        new_state = world.get_state_description(scenario["format"])
        
        # Check goal
        if world.check_goal(scenario["goal"]):
            return score_tool_use(True, turns_used, max_turns, True)
        
        prompt = f"""Action result: {result}

Updated state:
{new_state}

Goal not yet achieved. Propose your next action."""
    
    # Max turns exceeded
    progress = world.measure_progress(scenario["goal"])
    return score_tool_use(False, max_turns, max_turns, True) * progress
```

### Step 3.3: Define adaptive replanning task

Same as tool_use_task but injects a perturbation after the first successful action:

```python
@kbench.task(name="physiq_replan")
def replan_task(llm, scenario_json) -> float:
    """Multi-turn with forced failure after first action."""
    scenario = json.loads(scenario_json)
    world = PhysIQWorld(scenario)
    perturbation = scenario["perturbation"]
    perturbation_injected = False
    
    # ... similar loop to tool_use_task, but:
    # After first successful action execution:
    if not perturbation_injected and action_succeeded:
        world.apply_perturbation(perturbation)
        perturbation_injected = True
        prompt = f"""Something unexpected happened! {perturbation['description']}

Updated state:
{world.get_state_description(scenario['format'])}

Your previous plan may no longer work. How do you adapt?"""
    
    # ... score with replan scoring function
```

### Step 3.4: Create the benchmark

```python
# In final cells, select the main task for leaderboard
# Each task is a separate notebook → separate Kaggle Task
# Then group into one Benchmark on the Kaggle UI

%choose physiq_trajectory  # for the trajectory task notebook
```

Actually, per the SDK docs, each task is one notebook. So we need **5 task notebooks** + **1 benchmark** that groups them.

---

## Phase 4: Run & Analyze (Day 5-7)

### Step 4.1: Run against available models

Use the Kaggle-provided model access:

```python
models = [
    kbench.llms["google/gemini-2.5-flash"],
    kbench.llms["google/gemini-2.5-pro"],
    # Add other available models
]

results = trajectory_task.evaluate(
    llm=models,
    evaluation_data=df_trajectory
)
```

### Step 4.2: Generate analysis

- Per-model scores across tasks
- Per-difficulty breakdown
- Format robustness analysis
- Failure mode categorization
- Statistical significance tests between models

### Step 4.3: Create visualizations

Radar charts, heatmaps, difficulty curves — all generated in the notebook for the writeup.

---

## Notebook Structure (for each task)

```
Cell 1:  Setup & imports
Cell 2:  Physics engine implementation
Cell 3:  Scenario generation for this task
Cell 4:  Format serialization
Cell 5:  Scoring functions
Cell 6:  Task definition (@kbench.task)
Cell 7:  Dataset creation (DataFrame)
Cell 8:  Evaluation run
Cell 9:  Results analysis & visualization
Cell 10: %choose task_name
```

---

## Timeline

| Day | Milestone |
|---|---|
| 1 | Physics engine core + validation |
| 2 | Scenario templates + generation pipeline |
| 3 | All 3 formats working, dataset generated |
| 4 | Single-turn tasks (1-3) as kbench tasks, tested |
| 5 | Multi-turn tasks (4-5), full pipeline working |
| 6 | Run against models, collect results |
| 7 | Analysis, visualizations, writeup |
| 8-10 | Polish, edge cases, submission |

---

## Risk Mitigation

| Risk | Mitigation |
|---|---|
| pymunk not available on Kaggle | pymunk is pure Python via cffi — pip install works. Fallback: implement basic Euler integrator for simple scenarios |
| Models can't parse structured output | Multiple fallback regex patterns; worst case, unparseable = 0 score |
| Multi-turn quota exhaustion | Budget: 10 turns × 40 scenarios × 3 formats = 1200 prompts per model for Task 4. Within $50/day quota. |
| Scenarios too easy or too hard | Difficulty calibration in Phase 2 — run against one model first, adjust parameters |
| Borderline ground truths | Validation pipeline rejects ambiguous scenarios |
