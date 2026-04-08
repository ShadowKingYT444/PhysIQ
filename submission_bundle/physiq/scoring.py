"""PhysIQ scoring module.

All scores are continuous 0.0-1.0 with partial credit.
Implements the five task scorers, aggregate metrics, helpers, and
paired bootstrap significance testing.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any

import numpy as np

# ── Task weights (equal) ─────────────────────────────────────────────────────

TASK_WEIGHTS: dict[str, float] = {
    "trajectory": 0.20,
    "stability": 0.20,
    "causal_chain": 0.20,
    "tool_use": 0.20,
    "replanning": 0.20,
}

# ── Helpers ──────────────────────────────────────────────────────────────────

_INTERACTION_KEYWORDS = {
    "collision", "collide", "collides", "hit", "hits", "strike", "strikes",
    "impact", "impacts", "launch", "launches", "launched",
    "fall", "falls", "fell", "drop", "drops", "dropped",
    "push", "pushes", "pushed", "slide", "slides", "slid",
    "roll", "rolls", "rolled", "tip", "tips", "tipped",
    "topple", "topples", "toppled", "bounce", "bounces", "bounced",
    "knock", "knocks", "knocked", "trigger", "triggers", "triggered",
    "catapult", "catapults", "catapulted", "swing", "swings",
}

_FAILURE_KEYWORDS = {
    "left", "right", "topple", "topples", "fall", "falls",
    "slide", "slides", "collapse", "collapses", "tip", "tips",
    "rotate", "rotates", "overturn", "overturns", "lean", "leans",
    "shift", "shifts", "tumble", "tumbles", "drop", "drops",
}

_STABLE_REASONING_KEYWORDS = {
    "center of mass", "centre of mass", "centroid", "center-of-mass",
    "support", "supported", "well-supported", "well supported",
    "symmetrical", "symmetric", "symmetry",
    "balance", "balanced", "equilibrium",
    "wide base", "broad base", "stable base",
    "low center", "low centre", "low center of gravity",
    "friction", "high friction", "grips",
    "stable geometry", "squat", "geometry",
}


def _normalise(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def outcome_matches(predicted: str, actual: str) -> bool:
    """Fuzzy string match for final outcomes.

    Returns True if the predicted outcome is semantically close enough to the
    actual outcome.  Uses token overlap ratio plus SequenceMatcher as a
    fallback so that minor wording differences don't cause false negatives.
    """
    p = _normalise(predicted)
    a = _normalise(actual)

    if not p or not a:
        return False

    # Exact match after normalisation
    if p == a:
        return True

    # Token-level overlap (Jaccard)
    p_tokens = set(p.split())
    a_tokens = set(a.split())
    if not a_tokens:
        return False
    overlap = len(p_tokens & a_tokens) / len(a_tokens)
    if overlap >= 0.6:
        return True

    # Character-level similarity
    if SequenceMatcher(None, p, a).ratio() >= 0.65:
        return True

    return False


def event_matches(predicted_step: str, actual_event: dict) -> bool:
    """Check whether a predicted step describes the same physical event.

    ``actual_event`` is a dict with at least ``objects`` (list[str]) and
    ``interaction`` (str, e.g. "collision").  Optional ``order`` is ignored
    here; ordering is the caller's responsibility.

    Returns True when the predicted text mentions at least one involved object
    **and** a compatible interaction keyword.
    """
    p = _normalise(predicted_step)
    p_tokens = set(p.split())

    # Object mention check
    objects = actual_event.get("objects", [])
    object_mentioned = any(_normalise(obj) in p for obj in objects)
    if not object_mentioned:
        return False

    # Interaction type check
    interaction = _normalise(actual_event.get("interaction", ""))
    interaction_tokens = set(interaction.split())
    # Direct mention of the canonical interaction word
    if interaction_tokens & p_tokens:
        return True
    # Broader keyword match
    if p_tokens & _INTERACTION_KEYWORDS:
        return True

    return False


def score_final_state(predicted_failure: str, actual_failure_events: list) -> float:
    """Keyword matching score for stability final-state description.

    Returns a value in [0, 1] indicating how well the predicted failure
    description matches the actual failure events' final-state keywords.
    """
    if not actual_failure_events:
        return 0.0

    p = _normalise(predicted_failure)
    if not p:
        return 0.0

    p_tokens = set(p.split())

    # Collect all relevant keywords from actual events
    target_keywords: set[str] = set()
    for evt in actual_failure_events:
        for field in ("direction", "final_state", "description", "interaction"):
            val = evt.get(field, "")
            if val:
                for tok in _normalise(val).split():
                    if tok in _FAILURE_KEYWORDS:
                        target_keywords.add(tok)
        # Also accept object names as partial evidence
        for obj in evt.get("objects", []):
            target_keywords.add(_normalise(obj))

    if not target_keywords:
        return 0.0

    matched = len(p_tokens & target_keywords)
    return min(1.0, matched / len(target_keywords))


# ── Answer parsing ───────────────────────────────────────────────────────────

def parse_coordinates(response: str) -> tuple[float, float] | None:
    """Extract [x, y] from a model response.

    Looks for ``ANSWER: [x, y]`` first; falls back to any coordinate-like
    pair in the last three lines.  Returns ``None`` if unparseable.
    """
    match = re.search(
        r"ANSWER:\s*\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]?", response
    )
    if match:
        return (float(match.group(1)), float(match.group(2)))

    lines = response.strip().split("\n")[-3:]
    for line in reversed(lines):
        match = re.search(r"\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]?", line)
        if match:
            return (float(match.group(1)), float(match.group(2)))

    return None


# ── Task 1: Trajectory Prediction ────────────────────────────────────────────

def score_trajectory(
    predicted: tuple[float, float],
    actual: tuple[float, float],
    world_diagonal: float,
) -> float:
    """Score trajectory prediction with graceful degradation.

    Args:
        predicted: (x, y) predicted by model.
        actual: (x, y) from simulation ground truth.
        world_diagonal: diagonal of world bounds for normalisation.

    Returns:
        Score in [0, 1].
    """
    if world_diagonal <= 0:
        return 0.0

    dx = predicted[0] - actual[0]
    dy = predicted[1] - actual[1]
    distance = math.sqrt(dx * dx + dy * dy)
    normalized = distance / world_diagonal

    if normalized < 0.03:
        return 1.0
    elif normalized < 0.10:
        # Linear decay from 0.8 at 0.03 down to ~0.0 at 0.10
        return 0.8 - (normalized - 0.03) * (0.8 / 0.07)
    elif normalized < 0.25:
        # Linear decay from 0.4 at 0.10 down to ~0.0 at 0.25
        return 0.4 - (normalized - 0.10) * (0.4 / 0.15)
    elif normalized < 0.50:
        return 0.15
    else:
        return 0.0


# ── Task 2: Stability Judgment ───────────────────────────────────────────────

def score_stability(
    predicted_stable: bool,
    predicted_failure: str,
    actual_stable: bool,
    actual_failure_events: list[dict[str, Any]],
) -> float:
    """Score stability judgment.

    Components for STABLE scenarios:
      - Correct binary judgment: 0.3
      - Reasoning quality (keywords explaining WHY it's stable): 0.2
      - Maximum: 0.5

    Components for UNSTABLE scenarios:
      - Correct binary judgment: 0.5
      - Failure object identification: 0.15
      - Failure direction identification: 0.15
      - Final state description: 0.2
      - Maximum: 1.0

    Returns:
        Score in [0, 1].
    """
    score = 0.0

    if actual_stable:
        # Stable scenario scoring
        if predicted_stable:
            score += 0.3  # binary judgment
            # Reasoning quality: does the model explain WHY it's stable?
            p_lower = (predicted_failure or "").lower()
            if any(kw in p_lower for kw in _STABLE_REASONING_KEYWORDS):
                score += 0.2
    else:
        # Unstable scenario scoring
        if not predicted_stable:
            score += 0.5  # binary judgment

        # Failure identification (30%): scored even if binary wrong, for partial credit
        if actual_failure_events:
            first_failure = actual_failure_events[0]
            object_id = first_failure.get("object_id", "")
            if object_id and object_id in predicted_failure:
                score += 0.15
            direction = first_failure.get("direction", "")
            if direction and direction in predicted_failure:
                score += 0.15

        # Final state description (20%)
        score += score_final_state(predicted_failure, actual_failure_events) * 0.2

    return min(score, 1.0)


# ── Task 3: Causal Chain Reasoning ───────────────────────────────────────────

def score_causal_chain(
    predicted_steps: list[str],
    predicted_outcome: str,
    actual_events: list[dict[str, Any]],
    actual_outcome: str,
    actual_target_position: tuple[float, float] | None = None,
    world_diagonal: float | None = None,
    predicted_position: tuple[float, float] | None = None,
) -> float:
    """Score causal chain reasoning.

    Components:
      - Final outcome (50%): quantitative position score if target_position provided,
        otherwise fuzzy text match fallback.
      - Correct event sequence (50%): events matched in timestamp order, not just any order.

    The quantitative path (preferred) scores the predicted final position of the target
    object using the same distance thresholds as trajectory scoring.  The fuzzy-text
    fallback is used only when ground truth positions are unavailable.

    Returns:
        Score in [0, 1].
    """
    score = 0.0

    # Final outcome (50%) — quantitative position preferred over text matching
    if actual_target_position is not None and world_diagonal and predicted_position is not None:
        score += score_trajectory(predicted_position, actual_target_position, world_diagonal) * 0.5
    elif outcome_matches(predicted_outcome, actual_outcome):
        score += 0.5

    # Intermediate steps (50%) — scored in correct temporal order using event timestamps
    if actual_events:
        # Sort actual events by timestamp if available
        sorted_events = sorted(actual_events, key=lambda e: e.get("time", 0))
        step_value = 0.5 / len(sorted_events)
        pred_cursor = 0  # pointer into predicted_steps — enforces ordering
        for actual_event in sorted_events:
            # Only try predicted steps that come after the last matched step
            for i in range(pred_cursor, len(predicted_steps)):
                if event_matches(predicted_steps[i], actual_event):
                    score += step_value
                    pred_cursor = i + 1  # advance cursor: enforces temporal ordering
                    break

    return min(score, 1.0)


# ── Task 4: Tool Use Planning ────────────────────────────────────────────────

def score_tool_use(
    goal_achieved: bool,
    turns_used: int,
    max_turns: int,
    reasoning_valid: bool,
    progress: float = 0.0,
) -> float:
    """Score tool-use planning.

    Components:
      - Goal achieved: 0.6
      - Efficiency (fewer turns): 0.2
      - Reasoning validity: 0.2
      - If not achieved: partial credit for measured progress.

    Args:
        goal_achieved: whether the goal was reached.
        turns_used: number of turns consumed.
        max_turns: maximum allowed turns.
        reasoning_valid: whether the stated physics reasoning is correct.
        progress: fraction [0, 1] of progress toward the goal when not
            achieved (default 0).

    Returns:
        Score in [0, 1].
    """
    score = 0.0

    if goal_achieved:
        score += 0.6
        # Efficiency bonus: 1 turn = best, max_turns = worst
        if max_turns > 1:
            efficiency = 1.0 - (turns_used - 1) / (max_turns - 1)
        else:
            efficiency = 1.0
        score += 0.2 * max(0.0, efficiency)
    else:
        # Partial credit for progress toward goal
        score += 0.6 * float(np.clip(progress, 0.0, 1.0))

    # Reasoning validity (20%)
    if reasoning_valid:
        score += 0.2

    return min(score, 1.0)


# ── Task 5: Adaptive Replanning ──────────────────────────────────────────────

def score_replan(
    failure_recognized: bool,
    recovery_plan_valid: bool,
    goal_achieved: bool,
    recovery_turns: int,
) -> float:
    """Score adaptive replanning.

    Components:
      - Failure recognition: 0.2
      - Valid recovery plan: 0.3
      - Goal achieved: 0.3
      - Recovery efficiency: 0.2

    Returns:
        Score in [0, 1].
    """
    score = 0.0

    if failure_recognized:
        score += 0.2

    if recovery_plan_valid:
        score += 0.3

    if goal_achieved:
        score += 0.3

    # Recovery efficiency (only if goal was achieved)
    if goal_achieved and recovery_turns <= 3:
        score += 0.2
    elif goal_achieved and recovery_turns <= 5:
        score += 0.1

    return score


# ── Aggregate scoring ────────────────────────────────────────────────────────

def physiq_score(task_scores: dict[str, float]) -> float:
    """Weighted average across all tasks.

    ``task_scores`` maps task name (e.g. "trajectory") to mean score.
    Missing tasks are silently skipped (weight redistributed among present
    tasks is *not* done -- missing tasks count as 0).
    """
    return sum(
        TASK_WEIGHTS.get(task, 0.0) * score
        for task, score in task_scores.items()
    )


def format_robustness_score(scores_by_format: dict[str, float]) -> float:
    """Format Robustness Score (FRS).

    ``scores_by_format`` maps format name ("json", "ascii", "nl") to score.

    FRS = 1 - (max - min) / max
    Perfect consistency across formats yields 1.0.
    """
    values = list(scores_by_format.values())
    if not values:
        return 0.0
    max_val = max(values)
    if max_val == 0:
        return 0.0
    return 1.0 - (max_val - min(values)) / max_val


def difficulty_scaling_score(scores_by_difficulty: dict[str, float]) -> float:
    """Difficulty Scaling Score (DSS).

    Expected: easy > medium > hard with a smooth gradient.

    Returns:
      - 1.0 if easy >= medium >= hard **and** easy > hard (perfect gradient).
      - 0.7 if easy > hard but not strictly monotone.
      - 0.3 otherwise (no meaningful difficulty scaling).
    """
    easy = scores_by_difficulty.get("easy", 0.0)
    medium = scores_by_difficulty.get("medium", 0.0)
    hard = scores_by_difficulty.get("hard", 0.0)

    if easy >= medium >= hard and easy > hard:
        return 1.0
    elif easy > hard:
        return 0.7
    else:
        return 0.3


# ── Statistical significance ─────────────────────────────────────────────────

def is_significantly_different(
    scores_a: np.ndarray | list[float],
    scores_b: np.ndarray | list[float],
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """Paired bootstrap resampling test.

    Tests whether model A significantly differs from model B using a
    permutation-based bootstrap.

    Args:
        scores_a: per-instance scores for model A.
        scores_b: per-instance scores for model B (same length).
        n_bootstrap: number of bootstrap iterations.
        alpha: significance level.

    Returns:
        (is_significant, p_value)
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    assert len(scores_a) == len(scores_b), (
        f"Score arrays must have equal length, got {len(scores_a)} vs {len(scores_b)}"
    )

    observed_diff = float(np.mean(scores_a) - np.mean(scores_b))
    combined = np.concatenate([scores_a, scores_b])
    n = len(scores_a)

    rng = np.random.default_rng(seed=42)
    bootstrap_diffs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        perm = rng.permutation(combined)
        bootstrap_diffs[i] = np.mean(perm[:n]) - np.mean(perm[n:])

    p_value = float(np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff)))
    return p_value < alpha, p_value
