# Scenario Generation Pipeline

## Philosophy

Every scenario in PhysIQ is **procedurally generated** from a template + randomized parameters. This serves two critical purposes:

1. **Anti-contamination:** No scenario exists in any training corpus. The model cannot have memorized the answer.
2. **Difficulty control:** By tuning parameters, we create a reliable gradient from trivially easy to genuinely challenging.

## Architecture

```
Template (defines structure)
    +
Parameter Ranges (defines difficulty)
    +
Random Seed (ensures reproducibility)
    ↓
Scenario Generator
    ↓
Raw Scenario (objects, positions, properties)
    ↓
Validation (simulation confirms scenario is solvable / has clear answer)
    ↓
Serialization (JSON + ASCII + natural language)
    ↓
Final Scenario with Ground Truth
```

## Template System

Each task type has a set of templates. A template defines the *structure* of a scenario but leaves specific values as parameters.

### Example: Trajectory Prediction Template — "Ramp Launch"

```python
class RampLaunchTemplate:
    """Ball rolls down a ramp and becomes a projectile."""
    
    name = "ramp_launch"
    difficulty_range = ["easy", "medium"]
    
    # Parameterized values with ranges per difficulty
    params = {
        "easy": {
            "ramp_angle": (15, 30),        # degrees
            "ramp_length": (2.0, 4.0),     # meters
            "ball_radius": (0.15, 0.25),   # meters
            "ball_start_height": (3.0, 5.0), # meters
            "friction": (0.1, 0.3),
            "elasticity": (0.3, 0.7),
            "floor_present": True,
            "obstacles": 0,
        },
        "medium": {
            "ramp_angle": (20, 50),
            "ramp_length": (2.0, 6.0),
            "ball_radius": (0.1, 0.3),
            "ball_start_height": (3.0, 7.0),
            "friction": (0.05, 0.5),
            "elasticity": (0.2, 0.9),
            "floor_present": True,
            "obstacles": (1, 2),           # 1-2 obstacles to bounce off
        }
    }
```

### Template Catalog

**Task 1 — Trajectory Prediction (10 templates):**
- `ramp_launch`: Ball rolls down ramp, predict landing
- `table_roll`: Object rolls off table edge
- `bounce_path`: Ball with known elasticity bouncing off walls
- `pendulum_release`: Pendulum swings and releases object
- `conveyor_drop`: Object on moving surface reaches edge
- `multi_bounce`: Ball in enclosed space with multiple bounces
- `cannon_arc`: Object launched at angle (classic projectile variant)
- `friction_slide`: Object decelerating across surfaces with varying friction
- `domino_momentum`: Object transfers momentum through a line of objects
- `pinball_course`: Complex path with bumpers and slopes

**Task 2 — Stability Judgment (8 templates):**
- `simple_stack`: N blocks stacked vertically
- `offset_stack`: Blocks stacked with horizontal offsets (overhang)
- `pyramid`: Triangular arrangement
- `arch`: Two leaning supports with capstone
- `cantilever`: Overhanging beam with counterweight
- `mixed_shapes`: Circles on rectangles, irregular combos
- `friction_dependent`: Stability depends on friction values
- `chain_collapse`: One unstable element triggers cascading failure

**Task 3 — Causal Chain (8 templates):**
- `domino_line`: Classic domino chain
- `seesaw_launch`: Weight falls on seesaw, launches other side
- `rube_goldberg_simple`: 3-step machine
- `rube_goldberg_complex`: 5-7 step machine
- `branching_chain`: One event triggers two parallel outcomes
- `timing_gate`: Outcome depends on relative timing of two events
- `near_miss`: A chain where one interaction barely succeeds/fails
- `conservation_chain`: Momentum/energy transfers through multiple objects

**Task 4 — Tool Use Planning (6 templates):**
- `bridge_gap`: Create a bridge across a gap
- `reach_height`: Get object to a high platform
- `redirect_ball`: Use tools to guide ball to target
- `clear_path`: Remove obstacles blocking a path
- `launch_and_catch`: Launch object to land in target zone
- `lever_advantage`: Use mechanical advantage to move heavy object

**Task 5 — Adaptive Replanning (5 templates):**
Same as Task 4 templates, but with perturbation injected:
- `material_change`: Object has different friction/weight than described
- `structural_failure`: A support breaks under load
- `position_drift`: Object slides or shifts from where placed
- `missing_tool`: One planned tool turns out to be unavailable
- `new_obstacle`: An obstacle appears after initial plan

## Parameter Sampling

```python
import numpy as np

def sample_params(template, difficulty, seed):
    rng = np.random.RandomState(seed)
    params = {}
    for key, value_range in template.params[difficulty].items():
        if isinstance(value_range, tuple) and len(value_range) == 2:
            if isinstance(value_range[0], int):
                params[key] = rng.randint(value_range[0], value_range[1] + 1)
            else:
                params[key] = rng.uniform(value_range[0], value_range[1])
        else:
            params[key] = value_range  # fixed value
    return params
```

## Validation Pipeline

Not every randomly generated scenario is good. We validate each one:

```python
def validate_scenario(scenario, task_type):
    """Ensure scenario is well-formed and produces a clear answer."""
    
    # 1. Build and simulate
    world = build_world(scenario)
    result = simulate(world, scenario["time_limit"])
    
    # 2. Task-specific validation
    if task_type == "trajectory":
        # Ball must not leave bounds, must come to rest
        assert result.final_position is not None
        assert 0 < result.final_position[0] < scenario["bounds"]["width"]
        
    elif task_type == "stability":
        # Must be clearly stable OR clearly unstable (not borderline)
        stability_score = measure_stability_margin(result)
        assert abs(stability_score) > 0.3  # not ambiguous
        
    elif task_type == "causal_chain":
        # Chain must complete (or clearly fail) — no stalls
        assert len(result.collision_events) >= scenario["expected_steps"]
        
    elif task_type == "tool_use":
        # Goal must be achievable with available tools
        assert verify_solvable(scenario)  # brute-force check
        
    return True
```

We generate 3x more scenarios than needed and keep only those that pass validation. This ensures high dataset quality.

## Seed Management

```
Master seed: 42 (hardcoded for reproducibility)
├── Task 1 seed: hash("trajectory" + str(42)) → seeds per scenario
├── Task 2 seed: hash("stability" + str(42)) → seeds per scenario  
├── Task 3 seed: hash("causal" + str(42)) → seeds per scenario
├── Task 4 seed: hash("tool_use" + str(42)) → seeds per scenario
└── Task 5 seed: hash("replan" + str(42)) → seeds per scenario
```

Every scenario has a unique reproducible seed. The entire dataset can be regenerated identically from scratch.

## Anti-Contamination Measures

1. **Procedural generation** — scenarios don't exist in any textbook or dataset
2. **Randomized values** — even if a model has seen "ball on ramp" problems, the specific numbers are novel
3. **Non-standard setups** — we deliberately include unusual configurations (e.g., upside-down gravity, extremely bouncy materials, unusual object shapes) that are unlikely to appear in training data
4. **Multi-format presentation** — testing with JSON/ASCII/NL ensures the model isn't just pattern-matching on a familiar input format
