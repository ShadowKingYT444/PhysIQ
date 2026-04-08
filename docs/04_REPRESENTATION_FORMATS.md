# Representation Formats

## Why Multiple Formats Matter

A model that can reason about physics should be able to do so regardless of how the information is presented. If a model scores 80% on JSON descriptions but 30% on natural language descriptions of the *exact same scenario*, that tells us something important: **it's exploiting the structure of the input, not reasoning about the physics.**

We use three formats, and comparing performance across them is itself a key insight of the benchmark.

## Format 1: Structured JSON

The most explicit, least ambiguous format. Every value is precisely stated.

```json
{
  "task_id": "traj_017",
  "task_type": "trajectory_prediction",
  "difficulty": "medium",
  "representation": "json",
  "world": {
    "gravity": [0.0, -9.81],
    "bounds": {"width": 12.0, "height": 10.0},
    "damping": 0.99
  },
  "objects": [
    {
      "id": "ball_1",
      "type": "circle",
      "radius": 0.2,
      "position": [1.0, 8.0],
      "velocity": [4.0, 0.0],
      "mass": 0.5,
      "material": {"friction": 0.3, "elasticity": 0.7}
    },
    {
      "id": "ramp_1",
      "type": "static_polygon",
      "vertices": [[3.0, 6.0], [7.0, 3.0], [7.0, 3.0]],
      "material": {"friction": 0.2, "elasticity": 0.5}
    },
    {
      "id": "wall_1",
      "type": "static_segment",
      "start": [9.0, 0.0],
      "end": [9.0, 5.0],
      "material": {"friction": 0.3, "elasticity": 0.6}
    },
    {
      "id": "floor",
      "type": "static_segment",
      "start": [0.0, 0.0],
      "end": [12.0, 0.0],
      "material": {"friction": 0.4, "elasticity": 0.3}
    }
  ],
  "simulation_time": 6.0,
  "question": "After 6 seconds of simulation, what are the approximate [x, y] coordinates of ball_1? Round to 1 decimal place.",
  "answer_format": "[x, y]"
}
```

**Advantages for models:** No parsing ambiguity, machine-readable, no interpretation needed.  
**What it tests:** Pure physical reasoning — if a model fails on JSON, it's not a reading comprehension problem.

## Format 2: ASCII Art + Annotations

A visual-spatial representation using text characters. Tests whether models can build spatial understanding from a 2D character map.

```
╔══════════════════════════════════════════════════╗
║  PhysIQ Scenario: traj_017 (medium)              ║
║  World: 12m × 10m | g = 9.81 m/s² ↓             ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  ○→(4m/s)                                        ║
║  ·                                               ║
║     ╲                                            ║
║      ╲  ramp (friction=0.2)                      ║
║       ╲                                          ║
║        ╲              │wall                      ║
║         ╲             │(bounce=0.6)              ║
║          ╲            │                          ║
║           ╲           │                          ║
║            ╲          │                          ║
╚═══════════════════════════════════════(floor)═════╝
                                  friction=0.4

Legend:
  ○ = ball_1 (r=0.2m, m=0.5kg, bounce=0.7, friction=0.3)
  ╲ = ramp from (3,6) to (7,3)
  │ = wall from (9,0) to (9,5)

Question: Where is the ball after 6 seconds? → [x, y]
```

**Advantages for models:** Spatial layout is immediately visible; position relationships are intuitive.  
**What it tests:** Whether the model can translate a spatial layout into physical reasoning.  
**Known challenge:** ASCII art is imprecise — positions are approximate. We include exact coordinates in annotations.

## Format 3: Natural Language

A paragraph-form description, as if a human were describing the setup to a friend. This is the hardest format for models because they must:
1. Extract spatial information from prose
2. Build a mental model from an implicit rather than explicit description
3. Handle the natural ambiguity of language

```
Picture a tall, narrow room — 12 meters wide and 10 meters tall. 
Gravity is normal, pulling things down at about 9.8 m/s².

Up near the top-left corner, at about 1 meter from the left wall and 
8 meters off the ground, there's a small bouncy ball. It weighs half 
a kilogram, has a radius of 20 centimeters, and it's moving to the 
right at 4 meters per second. It's fairly bouncy (elasticity around 
0.7) with moderate grip (friction 0.3).

Running from roughly the middle-left area down to the middle of the 
room, there's a smooth ramp — think of it as a slide going from 
(3, 6) down to (7, 3). It's pretty slick, with low friction.

On the right side of the room, about 3 meters from the right wall, 
there's a vertical wall segment that goes from the floor up to about 
5 meters. It's moderately bouncy — things that hit it will rebound 
somewhat.

The floor runs the full width of the room and has decent friction.

If you let this system play out for 6 seconds, where does the ball 
end up? Give your answer as approximate [x, y] coordinates, rounded 
to one decimal place.
```

**Advantages for models:** Most "natural" input, closest to how a human would pose the problem.  
**What it tests:** Full pipeline: language understanding → spatial model building → physical reasoning.

## Format Equivalence Guarantee

All three formats describe the **exact same physical scenario** with the **exact same ground truth answer**. The only difference is the encoding. We verify this programmatically:

```python
def verify_format_equivalence(scenario_json, scenario_ascii, scenario_nl):
    """All three formats must parse to identical pymunk worlds."""
    world_json = build_world_from_json(scenario_json)
    world_ascii = build_world_from_json(scenario_json)  # ASCII/NL use same JSON backend
    world_nl = build_world_from_json(scenario_json)      # only presentation differs
    
    result_json = simulate(world_json)
    result_ascii = simulate(world_ascii)
    result_nl = simulate(world_nl)
    
    assert result_json == result_ascii == result_nl
```

The JSON is the canonical representation. ASCII and NL are generated *from* the JSON — the simulation always runs on the JSON. The model just sees different presentations.

## Prompt Construction

Each scenario is wrapped in a task-specific prompt template:

```python
TRAJECTORY_PROMPT = """You are given a 2D physics scenario. Your task is to 
predict where a specific object ends up after the simulation runs.

Think step by step about the forces, collisions, and motion involved. 
Then give your final answer.

SCENARIO:
{scenario_description}

ANSWER FORMAT: Respond with your reasoning, then on the final line write 
exactly: ANSWER: [x, y]
"""
```

The explicit answer format makes automated scoring reliable while still allowing us to examine the reasoning.

## Cross-Format Analysis Metric

Beyond task scores, we compute a **Format Robustness Score**:

```
FRS = 1 - (max_format_score - min_format_score) / max_format_score
```

A model with FRS = 1.0 performs identically across all formats (ideal).  
A model with FRS = 0.3 has large performance gaps across formats (fragile).

This is itself a novel insight: it measures how much of a model's "reasoning" is actually format-dependent pattern matching.
