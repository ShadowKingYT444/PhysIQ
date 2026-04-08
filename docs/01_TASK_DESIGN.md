# Task Design: PhysIQ Benchmark

## Design Principles

Every task follows three rules:
1. **Verifiable ground truth** — the physics engine produces an unambiguous answer
2. **No equation recall needed** — the scenario is described situationally, not as a textbook problem
3. **Procedurally parameterized** — each scenario has randomized values to prevent memorization

---

## Task 1: Trajectory Prediction

**Cognitive faculty tested:** Mental simulation of motion under forces  
**Format:** Single-turn  
**Scenarios:** 60 (20 easy / 20 medium / 20 hard)

### What the model sees

A description of a 2D world with objects, surfaces, and an initial action (e.g., "a ball is launched"). The model must predict where an object ends up.

### Difficulty scaling

| Difficulty | Description | Example |
|---|---|---|
| Easy | Single object, uniform gravity, flat surface | "A ball rolls off a table 1m high at 2m/s. Where does it land?" |
| Medium | One bounce/collision, angled surfaces | "A ball rolls down a 30° ramp, hits a wall, and bounces. Where?" |
| Hard | Multiple bounces, friction variations, moving platforms | "A ball enters a pinball-like course with 3 bumpers and a slope" |

### Example scenario (JSON format)

```json
{
  "world": {
    "gravity": [0, -9.81],
    "bounds": {"width": 10, "height": 8}
  },
  "objects": [
    {
      "id": "ball",
      "type": "circle",
      "radius": 0.2,
      "mass": 0.5,
      "position": [1.0, 6.0],
      "velocity": [3.0, 0.0],
      "material": {"friction": 0.3, "elasticity": 0.7}
    },
    {
      "id": "ramp",
      "type": "static_segment",
      "start": [2.0, 5.0],
      "end": [5.0, 3.0],
      "material": {"friction": 0.2, "elasticity": 0.5}
    },
    {
      "id": "floor",
      "type": "static_segment",
      "start": [0.0, 0.0],
      "end": [10.0, 0.0],
      "material": {"friction": 0.4, "elasticity": 0.3}
    }
  ],
  "question": "After 5 seconds of simulation, what is the approximate position of the ball? Give your answer as [x, y] coordinates rounded to 1 decimal place.",
  "time_limit": 5.0
}
```

### Same scenario (ASCII format)

```
World: 10m × 8m | Gravity: 9.81 m/s² downward

    ○→(3m/s)
    |
    +-------\
             \  ramp (30°)
              \  friction=0.2
               \
                +
════════════════════════════════
         floor (friction=0.4)

○ = ball (r=0.2m, m=0.5kg, bounce=0.7)

Q: Where is the ball after 5 seconds? Answer as [x, y].
```

### Same scenario (natural language format)

> Imagine a rectangular room that is 10 meters wide and 8 meters tall. Gravity pulls everything downward at 9.81 m/s².
>
> A small rubber ball (radius 20cm, mass 0.5kg, fairly bouncy) sits at position (1, 6) — near the top-left of the room. It's moving to the right at 3 meters per second.
>
> There's a ramp starting at (2, 5) and going down to (5, 3) — a roughly 30-degree slope with low friction. Below everything is a floor running the full width of the room with moderate friction.
>
> After 5 seconds pass, approximately where is the ball? Give coordinates as [x, y] rounded to one decimal place.

### Scoring

- **Exact match (within 0.5m):** 1.0
- **Close (within 1.5m):** 0.7
- **Reasonable direction (within 3m):** 0.3
- **Wrong direction or wildly off:** 0.0

Distance calculated as Euclidean distance between predicted and simulated final position.

---

## Task 2: Stability Judgment

**Cognitive faculty tested:** Spatial reasoning about balance, support, and collapse  
**Format:** Single-turn  
**Scenarios:** 60 (20 easy / 20 medium / 20 hard)

### What the model sees

A stack/arrangement of objects under gravity. The model must predict: (a) Will it collapse? (b) If yes, which piece fails first and in what direction?

### Difficulty scaling

| Difficulty | Description |
|---|---|
| Easy | 2-3 blocks, clearly stable or clearly unstable |
| Medium | 4-6 blocks, borderline center-of-mass situations |
| Hard | 7-10 blocks with varied shapes, sizes, friction; some with subtle instability |

### Example scenario (natural language)

> Three wooden blocks sit on a table:
> - Block A: 2m × 1m, sitting flat on the table, left edge at x=0
> - Block B: 1m × 1m, sitting on top of Block A, centered at x=1.5
> - Block C: 1m × 2m (tall), standing upright on Block B, centered at x=1.5
>
> All blocks have density 500 kg/m³ and friction coefficient 0.4 between all surfaces.
>
> Will this arrangement remain stable under gravity? If not, describe what happens first.

### Scoring

- **Correct stability judgment (stable/unstable):** 0.5
- **Correct failure mode (which block, which direction):** +0.3
- **Correct final resting state description:** +0.2

---

## Task 3: Causal Chain Reasoning

**Cognitive faculty tested:** Forward simulation of multi-body interactions, cause-and-effect chains  
**Format:** Single-turn  
**Scenarios:** 60 (20 easy / 20 medium / 20 hard)

### What the model sees

A Rube Goldberg-style setup where an initial trigger causes a chain of physical events. The model must predict the final state.

### Difficulty scaling

| Difficulty | Chain length | Description |
|---|---|---|
| Easy | 2 steps | Push A → A hits B → B falls into bucket |
| Medium | 3-4 steps | Ball rolls → hits lever → lever launches weight → weight breaks support |
| Hard | 5-7 steps | Complex chains with branching, timing dependencies, near-miss interactions |

### Example (medium difficulty, ASCII format)

```
Setup:
   [ball]→(2m/s)
   ═══════╗
          ║  ← wall
          ║
    ┌─────╨─────┐
    │   seesaw   │
    │     △      │
    └─────┬──[W]─┘
          │
          ▼ (if W launches)
      ┌───────┐
      │ target│
      │  box  │
      └───────┘

ball (0.3kg) rolls right → hits left side of seesaw → 
seesaw pivots → weight W (0.5kg) on right side launches up

Q: Does weight W land in the target box? If not, where does it end up?
```

### Scoring

- **Correct final outcome:** 0.5
- **Correct intermediate steps (each step):** 0.1 each (up to 0.5)
- Model must describe the chain, not just the answer

---

## Task 4: Tool Use Planning

**Cognitive faculty tested:** Multi-step planning, means-end reasoning, creative problem solving  
**Format:** Multi-turn (model proposes action → simulation executes → model sees result → repeats)  
**Scenarios:** 40 (15 easy / 15 medium / 10 hard)

### What the model sees

A world with a goal state and available objects/tools. The model must propose a sequence of actions to achieve the goal.

### Multi-turn interaction loop

```
Turn 1: [System presents world state + goal]
Turn 2: [Model proposes action, e.g. "Place plank from ledge A to ledge B"]
Turn 3: [System simulates action, returns new world state]
Turn 4: [Model proposes next action OR declares goal achieved]
... (up to 10 turns maximum)
```

### Example scenario

**Goal:** Get the ball from the left platform to the right platform.

```
World state:
  ┌───┐              ┌───┐
  │ ○ │              │   │ ← goal: ball here
  │   │    gap=3m    │   │
  ══════              ══════
  
Available tools:
  - Plank (4m long, can be placed as a bridge)
  - Wedge (can create a ramp)
  - Spring pad (launches objects upward)
  
Actions you can take:
  - PLACE <object> AT <position> ANGLE <degrees>
  - PUSH <object> WITH_FORCE <newtons> DIRECTION <angle>
  - REMOVE <object>
```

### Scoring

- **Goal achieved:** 0.6
- **Efficiency bonus (fewer actions):** up to 0.2
- **Reasoning quality (valid physics in explanation):** 0.2
- **Max turns exceeded without success:** 0.0 for goal, but partial credit for progress

---

## Task 5: Adaptive Replanning

**Cognitive faculty tested:** Cognitive flexibility, error recovery, plan modification under failure  
**Format:** Multi-turn with forced failure  
**Scenarios:** 30 (10 easy / 10 medium / 10 hard)

### Design

This is Task 4 with a twist: the system **deliberately introduces a perturbation** after the model's first plan. Maybe the ramp is slipperier than expected, or a block turns out to be heavier, or an object breaks. The model must recognize the failure and adapt.

### Example

```
[Turn 1 - System]: Same bridge scenario as Task 4.
[Turn 2 - Model]: "Place plank from left platform to right platform as bridge."
[Turn 3 - System]: "The plank was placed. However, the plank is heavier than 
  expected (20kg instead of 5kg). When the ball rolled onto it, the plank 
  tipped and fell into the gap. The ball is now on the floor of the gap. 
  New state: [updated world description]. How do you proceed?"
[Turn 4 - Model]: "Place the wedge against the right wall to create a ramp 
  up. Push the ball toward the wedge ramp."
...
```

### Scoring

- **Correct identification of failure:** 0.2
- **Valid recovery plan:** 0.3
- **Goal eventually achieved:** 0.3
- **Efficiency of recovery:** 0.2

### Why this task matters for Executive Functions

This is the purest test of the target cognitive faculty. Executive functions include:
- **Planning:** Tasks 4 and 5
- **Inhibitory control:** Task 5 forces the model to abandon its initial (now-failed) plan
- **Cognitive flexibility:** Task 5 requires switching strategies under new constraints
- **Working memory:** All multi-turn tasks require maintaining world state across turns

---

## Dataset Size Justification

250 base scenarios × 3 representation formats = 750 evaluation instances.

Statistical power analysis: With 60 scenarios per single-turn task, a 10% difference in accuracy between two models is detectable at p<0.05 with >80% power. This exceeds the competition's requirement for "sufficient sample size to be statistically significant."
