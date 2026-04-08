# Physics Engine Backend

## Why pymunk

**pymunk** is a Python wrapper around the Chipmunk2D physics engine. We chose it over alternatives for these reasons:

| Engine | Pros | Cons | Verdict |
|---|---|---|---|
| **pymunk** | Pure pip install, deterministic, fast, well-documented | 2D only | **Best fit** — 2D is our design |
| Box2D (pybox2d) | Industry standard | Complex build, hard to pip install on Kaggle | Too fragile for notebook env |
| matter.js | Great for web | JavaScript, not Python | Wrong language |
| Custom Euler integrator | Full control | Inaccurate for collisions, bounces | Not reliable enough |

pymunk installs cleanly with `pip install pymunk` and runs without GPU. It handles collision detection, friction, elasticity, joints, and constraints out of the box.

## Simulation Architecture

```
┌──────────────────────────────────────────┐
│              Scenario JSON               │
│  (objects, materials, initial conditions) │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│           World Builder                  │
│  - Parses scenario JSON                  │
│  - Creates pymunk Space                  │
│  - Adds bodies, shapes, constraints      │
│  - Sets gravity, damping                 │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│           Simulation Runner              │
│  - Steps physics at 60Hz (dt=1/60)       │
│  - Records state at each timestep        │
│  - Detects collisions, contacts          │
│  - Runs for T seconds (scenario-defined) │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│           State Extractor                │
│  - Final positions of all objects        │
│  - Collision event log                   │
│  - Stability flag (objects settled?)     │
│  - Goal achievement check                │
└──────────────────────────────────────────┘
```

## Key Implementation Details

### Determinism

pymunk/Chipmunk2D is **deterministic** — given the same initial conditions and timestep, it produces identical results every time. This is critical: we need the ground truth to be reproducible across notebook runs.

We enforce this by:
- Fixed timestep: `dt = 1/60` (always, never variable)
- Fixed iteration counts: `space.iterations = 20`
- Seeded scenario generation (separate from simulation determinism)

### Simulation parameters

```python
import pymunk

space = pymunk.Space()
space.gravity = (0, -981)       # cm/s² (we work in centimeters internally)
space.damping = 0.99            # slight velocity damping for stability
space.iterations = 20           # collision solver iterations
space.collision_slop = 0.5      # penetration tolerance

DT = 1.0 / 60.0                # 60 Hz physics step
```

### Object types we support

| Type | pymunk implementation | Use case |
|---|---|---|
| Circle | `pymunk.Circle` | Balls, wheels, round weights |
| Box | `pymunk.Poly.create_box` | Blocks, planks, platforms |
| Polygon | `pymunk.Poly` | Wedges, ramps, irregular shapes |
| Static segment | `pymunk.Segment` (static body) | Floors, walls, fixed ramps |
| Pivot joint | `pymunk.PivotJoint` | Seesaws, levers, hinges |
| Spring | `pymunk.DampedSpring` | Launchers, catapults |

### Material system

Each object has a material with three properties:
- **friction** (0.0–1.0): How much objects resist sliding
- **elasticity** (0.0–1.0): How bouncy collisions are (0 = no bounce, 1 = perfect bounce)
- **density** (kg/m²): Determines mass from shape area

Predefined materials:
```python
MATERIALS = {
    "rubber":  {"friction": 0.8, "elasticity": 0.8, "density": 1.2},
    "wood":    {"friction": 0.4, "elasticity": 0.3, "density": 0.6},
    "steel":   {"friction": 0.3, "elasticity": 0.5, "density": 7.8},
    "ice":     {"friction": 0.05, "elasticity": 0.2, "density": 0.9},
    "sponge":  {"friction": 0.9, "elasticity": 0.1, "density": 0.1},
}
```

### Collision event logging

We register collision handlers to track the causal chain:

```python
events = []

def on_collision(arbiter, space, data):
    shapes = arbiter.shapes
    events.append({
        "time": current_time,
        "type": "collision",
        "objects": [shape.body.label for shape in shapes],
        "impulse": arbiter.total_impulse,
        "position": arbiter.contact_point_set.points[0].point_a
    })
    return True

handler = space.add_default_collision_handler()
handler.begin = on_collision
```

This event log is used for:
- Scoring causal chain tasks (did the model predict the right sequence of events?)
- Debugging scenario generation
- Providing feedback in multi-turn tasks

### Stability detection

For Task 2, we need to determine if an arrangement is "stable." We define stability as:

```python
def is_stable(space, objects, settle_time=3.0, velocity_threshold=0.1):
    """Run sim for settle_time. If all objects have velocity < threshold, stable."""
    for _ in range(int(settle_time / DT)):
        space.step(DT)
    
    for obj in objects:
        if obj.body.velocity.length > velocity_threshold:
            return False
        if obj.body.angular_velocity > velocity_threshold:
            return False
    return True
```

### Action execution (for multi-turn tasks)

In Tasks 4 and 5, the model outputs actions that we parse and execute:

```python
ACTION_SCHEMA = {
    "PLACE": {
        "params": ["object_id", "position_x", "position_y", "angle_degrees"],
        "execute": lambda space, obj, x, y, a: place_object(space, obj, x, y, a)
    },
    "PUSH": {
        "params": ["object_id", "force_newtons", "direction_degrees"],
        "execute": lambda space, obj, f, d: apply_force(space, obj, f, d)
    },
    "REMOVE": {
        "params": ["object_id"],
        "execute": lambda space, obj: remove_object(space, obj)
    }
}
```

We use structured output or regex parsing on the model's response to extract actions. If the model outputs an unparseable action, it counts as a wasted turn.

## Performance

A single scenario simulation (5 seconds of physics time) runs in ~50ms on CPU. The entire benchmark (750 instances) can be simulated in under a minute. The bottleneck is LLM inference, not physics.
