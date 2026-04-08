# PhysIQ: Can Language Models Think in Physics, or Just Talk About It?

## A Benchmark for Interactive Physical Reasoning via Simulated Worlds

**Competition Track:** Executive Functions  
**Cognitive Faculty Isolated:** Multi-step planning, causal reasoning, and mental simulation in novel physical environments  
**Tagline:** *"Every toddler knows a tall tower of blocks will fall. No frontier model can reliably figure this out without having memorized the answer."*

---

## 1. The Core Insight

Existing physics benchmarks (PhysReason, UGPhysics, PHYBench, ABench-Physics) all test the same thing: **can a model solve textbook physics problems using memorized equations?** A model that scores 95% on projectile motion problems may have simply memorized that `x = v₀t·cos(θ)` — it has never actually *reasoned* about what happens when you throw a ball.

PhysIQ tests something fundamentally different: **can a model predict what will happen in a novel physical scenario by mentally simulating it**, the way humans do effortlessly every day?

We present the model with a 2D physics world — described textually (no vision required) — and ask it to:
- Predict outcomes ("will this tower collapse?")
- Plan action sequences ("how do you get the ball into the box?")
- Reason about causal chains ("if you push block A, what happens to block C?")

We then **run the actual physics simulation** (via pymunk/Box2D) and compare the model's prediction to ground truth. No ambiguity. No subjective grading. Pure verifiable physical reasoning.

## 2. Why This Is Novel (Differentiation from Prior Work)

| Existing Benchmarks | PhysIQ |
|---|---|
| Static textbook problems | Dynamic simulated scenarios |
| Tests equation recall | Tests mental simulation |
| Single correct formula to apply | Multiple valid reasoning paths |
| Model outputs a number | Model outputs predictions, plans, action sequences |
| High contamination risk (problems from textbooks) | Zero contamination (procedurally generated worlds) |
| No interaction | Multi-turn: model acts → world responds → model adapts |

**Key novelty:** PhysIQ scenarios are **procedurally generated** with parameterized randomness. No two runs produce the same scenario. This makes memorization impossible and isolates genuine reasoning.

## 3. What This Reveals About Model Behavior

PhysIQ answers: **"Can this model build an internal physical world model, or does it only manipulate symbols?"**

Specifically, it exposes:
- **Simulation gaps:** Models that can write the equations of motion but can't predict where a ball lands when you describe a ramp setup
- **Compositionality failures:** Models that handle single physical interactions but break down when 3+ objects interact in a causal chain
- **Planning rigidity:** Models that produce a plan but can't adapt when step 2 fails (a direct test of executive function)
- **Format robustness:** Whether reasoning changes when the same scenario is presented as JSON vs. ASCII vs. natural language

## 4. Benchmark Architecture

```
PhysIQ Benchmark
├── Task 1: Trajectory Prediction (single-turn, 60 scenarios)
│   └── "Where does this object end up?"
├── Task 2: Stability Judgment (single-turn, 60 scenarios)
│   └── "Will this structure collapse? If so, how?"
├── Task 3: Causal Chain Reasoning (single-turn, 60 scenarios)
│   └── "If X happens, what is the final state?"
├── Task 4: Tool Use Planning (multi-turn, 40 scenarios)
│   └── "Use available objects to achieve goal G"
└── Task 5: Adaptive Replanning (multi-turn, 30 scenarios)
    └── "Your plan failed at step N. Now what?"
```

Total: **250 scenarios** across 5 tasks, with 3 representation formats each (JSON, ASCII, natural language) = **750 evaluation instances**.

## 5. Scoring

Each task produces a score from 0.0 to 1.0:
- **Tasks 1-3 (prediction):** Scored by comparing predicted outcome to simulated ground truth. Partial credit via distance metrics.
- **Task 4 (planning):** Scored by whether the model's proposed action sequence achieves the goal when simulated.
- **Task 5 (adaptation):** Scored on both recovery success AND efficiency (fewer re-plan steps = better).

**Aggregate PhysIQ Score** = weighted average across tasks, designed to produce a **gradient of performance** (not all-or-nothing).

## 6. Expected Performance Gradient

Based on preliminary reasoning about model capabilities:
- **Random baseline:** ~5-15% (some binary judgments will be correct by chance)
- **Weak models (small open-source):** ~15-30% (may get basic trajectory right, fail on chains)
- **Strong models (GPT-4o, Gemini Flash):** ~40-60% (good on simple scenarios, degrade on complexity)
- **Reasoning models (o3, R1, Gemini Thinking):** ~50-70% (better planning, still fail on novel compositions)
- **Human baseline (informal):** ~85-95% (humans have strong physical intuition)

This gradient is the key competitive strength of PhysIQ: it **discriminates between models** meaningfully.

## 7. File Index

| File | Contents |
|---|---|
| `01_TASK_DESIGN.md` | Detailed design of all 5 tasks with examples |
| `02_PHYSICS_ENGINE.md` | How the simulation backend works (pymunk) |
| `03_SCENARIO_GENERATION.md` | Procedural generation pipeline for scenarios |
| `04_REPRESENTATION_FORMATS.md` | How worlds are serialized to text (JSON, ASCII, NL) |
| `05_SCORING_SYSTEM.md` | Evaluation metrics and partial credit |
| `06_IMPLEMENTATION_PLAN.md` | Step-by-step plan for building the Kaggle notebook |
| `07_WRITEUP_DRAFT.md` | Draft of the 1500-word competition writeup |
