# Scoring System

## Design Goals

1. **Gradient, not binary:** Every task produces a continuous score from 0.0 to 1.0
2. **Partial credit:** Getting the direction right but the distance wrong earns something
3. **Verifiable:** All scores are computed algorithmically, no LLM-as-judge needed for core metrics
4. **Discriminating:** The scoring must separate weak models from strong ones with statistical significance

---

## Task-Level Scoring

### Task 1: Trajectory Prediction

**Metric:** Euclidean distance between predicted position and simulated ground truth.

```python
def score_trajectory(predicted: tuple, actual: tuple, world_diagonal: float) -> float:
    """
    Score trajectory prediction with graceful degradation.
    
    Args:
        predicted: (x, y) predicted by model
        actual: (x, y) from simulation
        world_diagonal: diagonal of world bounds (for normalization)
    """
    distance = euclidean_distance(predicted, actual)
    normalized = distance / world_diagonal  # 0 = perfect, 1 = worst possible
    
    if normalized < 0.03:    # within ~3% of world size
        return 1.0
    elif normalized < 0.10:  # within ~10%
        return 0.8 - (normalized - 0.03) * (0.8 / 0.07)  # linear decay 0.8→0.0... 
    elif normalized < 0.25:  # within ~25%  
        return 0.4 - (normalized - 0.10) * (0.4 / 0.15)
    elif normalized < 0.50:  # within ~50%
        return 0.15
    else:
        return 0.0

    # Simplified: smooth decay curve
    # score = max(0, 1.0 - (normalized / 0.25) ** 1.5)
```

**Why normalized distance?** A 1-meter error in a 5×5 world is much worse than a 1-meter error in a 50×50 world.

**Answer parsing:**
```python
import re

def parse_coordinates(response: str) -> tuple:
    """Extract [x, y] from model response."""
    # Look for ANSWER: [x, y] pattern
    match = re.search(r'ANSWER:\s*\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]?', response)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    
    # Fallback: look for any coordinate-like pair in last 3 lines
    lines = response.strip().split('\n')[-3:]
    for line in reversed(lines):
        match = re.search(r'\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]?', line)
        if match:
            return (float(match.group(1)), float(match.group(2)))
    
    return None  # unparseable → score 0
```

### Task 2: Stability Judgment

**Metric:** Composite of binary judgment + failure mode description.

```python
def score_stability(predicted_stable: bool, predicted_failure: str,
                    actual_stable: bool, actual_failure_events: list) -> float:
    score = 0.0
    
    # Binary stability judgment (50% of score)
    if predicted_stable == actual_stable:
        score += 0.5
    
    if not actual_stable:
        # Failure identification (30% of score)
        # Check if model identified the first object to move/fall
        first_failure = actual_failure_events[0]
        if first_failure["object_id"] in predicted_failure:
            score += 0.15
        # Check if direction of failure is correct
        if first_failure["direction"] in predicted_failure:  # "left", "right", "topples"
            score += 0.15
        
        # Final state description (20% of score)
        # Use keyword matching for final resting configuration
        score += score_final_state(predicted_failure, actual_failure_events) * 0.2
    else:
        # If correctly predicted stable, full remaining credit
        score += 0.5
    
    return score
```

### Task 3: Causal Chain Reasoning

**Metric:** Step-by-step chain accuracy + final outcome.

```python
def score_causal_chain(predicted_steps: list, predicted_outcome: str,
                       actual_events: list, actual_outcome: str) -> float:
    score = 0.0
    
    # Final outcome (50% of score)
    if outcome_matches(predicted_outcome, actual_outcome):
        score += 0.5
    
    # Intermediate steps (50% of score, divided equally)
    if len(actual_events) > 0:
        step_value = 0.5 / len(actual_events)
        for actual_event in actual_events:
            for predicted_step in predicted_steps:
                if event_matches(predicted_step, actual_event):
                    score += step_value
                    break
    
    return min(score, 1.0)
```

**Event matching** is done by checking if the model mentioned:
- The correct objects involved
- The correct type of interaction (collision, launch, fall, etc.)
- The approximately correct order

### Task 4: Tool Use Planning

**Metric:** Goal achievement + efficiency.

```python
def score_tool_use(goal_achieved: bool, turns_used: int, 
                   max_turns: int, reasoning_valid: bool) -> float:
    score = 0.0
    
    # Goal achieved (60%)
    if goal_achieved:
        score += 0.6
        
        # Efficiency bonus (20%) — fewer turns = better
        efficiency = 1.0 - (turns_used - 1) / (max_turns - 1)  # normalized
        score += 0.2 * max(0, efficiency)
    else:
        # Partial credit for progress
        score += 0.6 * measure_progress_toward_goal()
    
    # Reasoning validity (20%) — are the stated physics principles correct?
    # This uses keyword/heuristic checking, not LLM-as-judge
    if reasoning_valid:
        score += 0.2
    
    return score
```

### Task 5: Adaptive Replanning

**Metric:** Failure recognition + recovery quality.

```python
def score_replan(failure_recognized: bool, recovery_plan_valid: bool,
                 goal_achieved: bool, recovery_turns: int) -> float:
    score = 0.0
    
    # Failure recognition (20%)
    if failure_recognized:
        score += 0.2
    
    # Valid recovery plan (30%)
    if recovery_plan_valid:
        score += 0.3
    
    # Goal eventually achieved (30%)
    if goal_achieved:
        score += 0.3
    
    # Recovery efficiency (20%)
    if goal_achieved and recovery_turns <= 3:
        score += 0.2
    elif goal_achieved and recovery_turns <= 5:
        score += 0.1
    
    return score
```

---

## Aggregate Scoring

### PhysIQ Composite Score

```python
TASK_WEIGHTS = {
    "trajectory":   0.20,
    "stability":    0.20,
    "causal_chain": 0.20,
    "tool_use":     0.20,
    "replanning":   0.20,
}

def physiq_score(task_scores: dict) -> float:
    """Weighted average across all tasks."""
    return sum(
        TASK_WEIGHTS[task] * score 
        for task, score in task_scores.items()
    )
```

Equal weighting by default. Each task score is the mean across all scenarios in that task.

### Format Robustness Score (FRS)

```python
def format_robustness_score(scores_by_format: dict) -> float:
    """How consistent is performance across JSON, ASCII, NL formats?"""
    values = list(scores_by_format.values())
    if max(values) == 0:
        return 0.0
    return 1.0 - (max(values) - min(values)) / max(values)
```

### Difficulty Scaling Score (DSS)

```python
def difficulty_scaling_score(scores_by_difficulty: dict) -> float:
    """Does performance degrade gracefully with difficulty?
    
    Ideal: easy > medium > hard with smooth gradient.
    Bad: easy ≈ hard (model is guessing) or easy << hard (broken).
    """
    easy = scores_by_difficulty.get("easy", 0)
    medium = scores_by_difficulty.get("medium", 0)
    hard = scores_by_difficulty.get("hard", 0)
    
    # Check for expected ordering
    if easy >= medium >= hard and easy > hard:
        return 1.0  # perfect gradient
    elif easy > hard:
        return 0.7  # right direction, not monotone
    else:
        return 0.3  # no meaningful difficulty scaling
```

---

## Statistical Significance

For comparing two models, we use paired bootstrap resampling:

```python
def is_significantly_different(scores_a, scores_b, n_bootstrap=10000, alpha=0.05):
    """Test if model A significantly outperforms model B."""
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    
    combined = np.concatenate([scores_a, scores_b])
    n = len(scores_a)
    
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        perm = np.random.permutation(combined)
        bootstrap_diffs.append(np.mean(perm[:n]) - np.mean(perm[n:]))
    
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
    return p_value < alpha, p_value
```

With 60 scenarios per task and 3 formats, we have 180 data points per task — sufficient for robust statistical comparisons.
