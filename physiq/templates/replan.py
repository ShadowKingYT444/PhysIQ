"""Adaptive Replanning scenario templates (Task 5).

Five templates that reuse tool-use scenarios but inject a forced perturbation
after the first successful action, requiring the model to recognise failure
and adapt its plan.  Each template's ``generate()`` returns a scenario dict
that extends the tool-use format with a ``perturbation`` field.

Counts: easy=10, medium=10, hard=10 (total 30 scenarios).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from physiq.templates.tool_use import (
    BridgeGap,
    ClearPath,
    LaunchAndCatch,
    LeverAdvantage,
    ReachHeight,
    RedirectBall,
    _box,
    _circle,
    _static_seg,
    _tool_box,
    _tool_circle,
    _tool_polygon,
    _world,
)


# ── Base class ─────────────────────────────────────────────────────────────────

class ReplanTemplate(ABC):
    """Abstract base for an adaptive replanning scenario template."""

    name: str

    @abstractmethod
    def generate(self, difficulty: str, seed: int) -> dict:
        """Return a complete scenario dict with perturbation info."""
        ...


# ── 1. Material Change ────────────────────────────────────────────────────────

class MaterialChange(ReplanTemplate):
    """A key tool/object has unexpected material properties.

    Uses the BridgeGap base scenario -- the plank turns out to be much
    heavier or slipperier than expected, requiring the model to compensate
    (e.g., add support, change placement angle, use a second plank).
    """

    name = "material_change"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        base = BridgeGap().generate(difficulty, seed + 7000)

        # Override task metadata
        base["id"] = f"replan_material_change_{seed}"
        base["task_type"] = "replan"
        base["difficulty"] = difficulty

        # Perturbation: the first plank is much heavier / slippery
        if difficulty == "easy":
            new_mass = rng.uniform(8.0, 12.0)
            new_friction = 0.3
            desc = (
                f"The plank is heavier than expected ({new_mass:.1f} kg "
                f"instead of ~2 kg). It sags more under load."
            )
        elif difficulty == "medium":
            new_mass = rng.uniform(15.0, 25.0)
            new_friction = rng.uniform(0.05, 0.15)
            desc = (
                f"The plank weighs {new_mass:.1f} kg and is coated in ice "
                f"(friction={new_friction:.2f}). The ball slides off easily."
            )
        else:
            new_mass = rng.uniform(30.0, 50.0)
            new_friction = rng.uniform(0.02, 0.08)
            desc = (
                f"The plank is extremely heavy ({new_mass:.1f} kg) and icy "
                f"(friction={new_friction:.2f}). It cannot support itself "
                f"without additional bracing and the ball slides on it."
            )

        base["perturbation"] = {
            "type": "material_change",
            "description": desc,
            "object_id": "plank_1",
            "new_material": {
                "friction": round(new_friction, 3),
                "elasticity": 0.1,
            },
            "new_mass": round(new_mass, 1),
        }

        return base


# ── 2. Structural Failure ─────────────────────────────────────────────────────

class StructuralFailure(ReplanTemplate):
    """A support or surface breaks under load.

    Uses the ReachHeight base scenario -- a support block cracks and
    disappears after the model places weight on it.
    """

    name = "structural_failure"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        base = ReachHeight().generate(difficulty, seed + 8000)

        base["id"] = f"replan_structural_failure_{seed}"
        base["task_type"] = "replan"
        base["difficulty"] = difficulty

        # The first block the model places will break
        # (engine removes it via apply_perturbation)
        fail_obj = "block_1"

        if difficulty == "easy":
            desc = (
                "The support block cracks under the ball's weight and "
                "collapses. You need an alternative support."
            )
        elif difficulty == "medium":
            desc = (
                "The support block shatters when the ramp is placed on it. "
                "The ramp falls to the floor. You must find another way up."
            )
        else:
            desc = (
                "The support block disintegrates immediately. All objects "
                "resting on it fall. Reconstruct a path to the shelf."
            )

        base["perturbation"] = {
            "type": "structural_failure",
            "description": desc,
            "object_id": fail_obj,
        }

        return base


# ── 3. Position Drift ─────────────────────────────────────────────────────────

class PositionDrift(ReplanTemplate):
    """An object slides or shifts from where it was placed.

    Uses the RedirectBall base scenario -- after placement a deflector
    wall drifts on a sloped surface, misaligning the redirect path.
    """

    name = "position_drift"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        base = RedirectBall().generate(difficulty, seed + 9000)

        base["id"] = f"replan_position_drift_{seed}"
        base["task_type"] = "replan"
        base["difficulty"] = difficulty

        # The first wall placed drifts sideways/downward
        dx = {"easy": rng.uniform(0.5, 1.0),
              "medium": rng.uniform(1.0, 2.0),
              "hard": rng.uniform(2.0, 3.5)}[difficulty]
        dy = {"easy": rng.uniform(-0.3, 0.0),
              "medium": rng.uniform(-0.8, -0.3),
              "hard": rng.uniform(-1.5, -0.5)}[difficulty]

        desc = (
            f"The deflector wall slides {abs(dx):.1f}m to the right "
            f"and {abs(dy):.1f}m downward on the slippery surface. "
            f"The ball no longer hits it at the intended angle."
        )

        base["perturbation"] = {
            "type": "position_drift",
            "description": desc,
            "object_id": "wall_1",
            "dx": round(float(dx), 2),
            "dy": round(float(dy), 2),
        }

        return base


# ── 4. Missing Tool ───────────────────────────────────────────────────────────

class MissingTool(ReplanTemplate):
    """One planned tool becomes unavailable after the first action.

    Uses the LeverAdvantage base scenario -- the lever beam breaks when
    first used, leaving fewer tools for the task.
    """

    name = "missing_tool"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        base = LeverAdvantage().generate(difficulty, seed + 10000)

        base["id"] = f"replan_missing_tool_{seed}"
        base["task_type"] = "replan"
        base["difficulty"] = difficulty

        # Add extra replacement tools so the task remains solvable
        if difficulty == "easy":
            base["available_tools"].append(
                _tool_box("lever_2", 4.5, 0.15, "wood")
            )
            desc = (
                "The steel lever snaps when loaded. A shorter wooden "
                "lever is still available."
            )
        elif difficulty == "medium":
            base["available_tools"].append(
                _tool_box("plank_a", 3.0, 0.15, "wood")
            )
            base["available_tools"].append(
                _tool_box("plank_b", 3.0, 0.15, "wood")
            )
            desc = (
                "The lever breaks under the heavy crate. You have two "
                "shorter planks remaining -- can you combine them?"
            )
        else:
            base["available_tools"].append(
                _tool_box("plank_a", 2.5, 0.15, "wood")
            )
            desc = (
                "The lever shatters. Only a single short plank remains. "
                "You must find a creative alternative approach."
            )

        base["perturbation"] = {
            "type": "missing_tool",
            "description": desc,
            "tool_id": "lever_1",
        }

        return base


# ── 5. New Obstacle ───────────────────────────────────────────────────────────

class NewObstacle(ReplanTemplate):
    """An unexpected obstacle appears after the first action.

    Uses the LaunchAndCatch base scenario -- a wall or barrier drops
    into the projectile's flight path after the first move.
    """

    name = "new_obstacle"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        base = LaunchAndCatch().generate(difficulty, seed + 11000)

        base["id"] = f"replan_new_obstacle_{seed}"
        base["task_type"] = "replan"
        base["difficulty"] = difficulty

        # New obstacle appears between launch and target
        target_x = base["goal"]["target_position"][0]
        ball_x = 2.0  # approximate launch x

        obs_x = rng.uniform(ball_x + 2.0, target_x - 1.5)
        obs_h = {"easy": rng.uniform(2.0, 3.0),
                 "medium": rng.uniform(3.5, 5.0),
                 "hard": rng.uniform(5.5, 7.5)}[difficulty]

        obstacle_def = _static_seg(
            "surprise_wall",
            [round(float(obs_x), 2), 0.0],
            [round(float(obs_x), 2), round(float(obs_h), 2)],
        )

        desc_height = f"{obs_h:.1f}"
        if difficulty == "easy":
            desc = (
                f"A {desc_height}m wall drops down at x={obs_x:.1f}, "
                f"partially blocking the trajectory. You can lob over it."
            )
        elif difficulty == "medium":
            desc = (
                f"A {desc_height}m wall appears at x={obs_x:.1f}, "
                f"blocking the direct path. You need a higher arc or "
                f"a deflection strategy."
            )
        else:
            desc = (
                f"A tall {desc_height}m wall materializes at x={obs_x:.1f}, "
                f"completely blocking the flight path. Rethink the "
                f"trajectory or find a way around."
            )

        base["perturbation"] = {
            "type": "new_obstacle",
            "description": desc,
            "obstacle": obstacle_def,
        }

        return base


# ── Template registry ─────────────────────────────────────────────────────────

REPLAN_TEMPLATES: list[ReplanTemplate] = [
    MaterialChange(),
    StructuralFailure(),
    PositionDrift(),
    MissingTool(),
    NewObstacle(),
]
