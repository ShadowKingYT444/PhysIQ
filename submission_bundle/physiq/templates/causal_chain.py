"""Causal chain reasoning scenario templates (Task 3).

Eight templates that generate Rube Goldberg-style setups where an initial
trigger causes a chain of physical events.  The model must predict
intermediate steps and the final outcome.

Templates:
  1. DominoLine       -- classic domino chain (tall thin boxes toppling)
  2. SeesawLaunch     -- weight falls on seesaw, launches object
  3. RubeGoldbergSimple -- 3-step machine
  4. RubeGoldbergComplex -- 5-7 step machine
  5. BranchingChain   -- one event triggers two parallel outcomes
  6. TimingGate       -- outcome depends on relative timing of two events
  7. NearMiss         -- chain that barely succeeds or barely fails
  8. ConservationChain -- Newton's-cradle-style momentum transfers
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ── Helpers ─────────────────────────────────────────────────────────────────

MATERIALS = {
    "rubber": {"friction": 0.8, "elasticity": 0.8, "density": 1.2},
    "wood":   {"friction": 0.4, "elasticity": 0.3, "density": 0.6},
    "steel":  {"friction": 0.3, "elasticity": 0.5, "density": 7.8},
    "ice":    {"friction": 0.05, "elasticity": 0.2, "density": 0.9},
}


def _mat(name: str) -> dict:
    return MATERIALS[name].copy()


def _circle(oid: str, radius: float, pos: list, material: str,
            velocity: list | None = None, mass: float | None = None) -> dict:
    obj: dict[str, Any] = {
        "id": oid, "type": "circle", "radius": radius,
        "position": pos, "velocity": velocity or [0, 0],
        "material": _mat(material),
    }
    if mass is not None:
        obj["mass"] = mass
    return obj


def _box(oid: str, w: float, h: float, pos: list, material: str,
         angle: float = 0, velocity: list | None = None,
         mass: float | None = None) -> dict:
    obj: dict[str, Any] = {
        "id": oid, "type": "box", "width": w, "height": h,
        "position": pos, "velocity": velocity or [0, 0],
        "angle": angle, "material": _mat(material),
    }
    if mass is not None:
        obj["mass"] = mass
    return obj


def _segment(oid: str, start: list, end: list, material: str = "steel") -> dict:
    return {
        "id": oid, "type": "static_segment",
        "start": start, "end": end, "material": _mat(material),
    }


def _static_poly(oid: str, vertices: list, material: str = "steel") -> dict:
    return {
        "id": oid, "type": "static_polygon",
        "vertices": vertices, "material": _mat(material),
    }


def _world(width: float = 10.0, height: float = 8.0) -> dict:
    return {
        "gravity": [0, -9.81],
        "bounds": {"width": width, "height": height},
        "damping": 0.99,
    }


# ── Base class ──────────────────────────────────────────────────────────────

class CausalChainTemplate(ABC):
    """Abstract base for causal-chain scenario generators."""

    name: str
    difficulties: list[str]  # which difficulties this template covers

    @abstractmethod
    def generate(self, difficulty: str, seed: int) -> dict:
        ...

    def _scenario(self, seed: int, difficulty: str, objects: list,
                  trigger: dict, sim_time: float, expected_steps: int,
                  expected_outcome: str, question: str | None = None,
                  world: dict | None = None,
                  target_object: str | None = None) -> dict:
        return {
            "id": f"causal_{self.name}_{seed}",
            "task_type": "causal_chain",
            "difficulty": difficulty,
            "world": world or _world(),
            "objects": objects,
            "simulation_time": sim_time,
            "trigger": trigger,
            "question": question or (
                "Describe step by step what happens in this chain reaction, "
                "then predict the final state."
            ),
            "answer_format": (
                f"Step-by-step chain description, then ANSWER: <final outcome>. "
                f"Position of {target_object}: POSITION: [x, y]"
                if target_object else
                "Step-by-step chain description, then ANSWER: <final outcome>"
            ),
            "expected_steps": expected_steps,
            "expected_outcome": expected_outcome,
            "target_object": target_object,
        }


# ── 1. DominoLine ──────────────────────────────────────────────────────────

class DominoLine(CausalChainTemplate):
    """Classic domino chain: tall thin boxes topple into each other."""

    name = "domino_line"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        n_dominos = {"easy": 3, "medium": 5, "hard": 8}[difficulty]
        spacing_base = {"easy": 0.55, "medium": 0.50, "hard": 0.45}[difficulty]
        dom_w = 0.15
        dom_h = 0.80

        # Floor
        objects: list[dict] = [_segment("floor", [0, 0], [12, 0])]

        # Dominos placed left to right
        x_start = 1.5
        for i in range(n_dominos):
            spacing = spacing_base + rng.uniform(-0.03, 0.03)
            x = x_start + i * spacing
            objects.append(
                _box(f"domino_{i}", dom_w, dom_h, [x, dom_h / 2], "wood")
            )

        # Trigger ball
        ball_x = x_start - 0.5
        ball_y = dom_h / 2
        ball_r = 0.12 + rng.uniform(0, 0.03)
        objects.append(
            _circle("trigger_ball", ball_r, [ball_x, ball_y], "steel")
        )

        # Bucket at the end
        last_x = x_start + (n_dominos - 1) * spacing_base + 0.8
        objects.append(_segment("bucket_left",  [last_x, 0], [last_x, 0.4]))
        objects.append(_segment("bucket_right", [last_x + 0.6, 0], [last_x + 0.6, 0.4]))

        trigger_vx = 2.0 + rng.uniform(0, 0.5)
        trigger = {
            "type": "initial_velocity",
            "object": "trigger_ball",
            "velocity": [trigger_vx, 0],
        }

        outcome = (
            f"All {n_dominos} dominos topple sequentially from left to right. "
            f"The last domino falls into the bucket zone."
        )

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=2.0 + n_dominos * 0.3,
            expected_steps=n_dominos + 1,
            expected_outcome=outcome,
            target_object=f"domino_{n_dominos - 1}",
        )


# ── 2. SeesawLaunch ────────────────────────────────────────────────────────

class SeesawLaunch(CausalChainTemplate):
    """A weight falls onto one end of a seesaw, launching an object on the
    other side upward."""

    name = "seesaw_launch"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        plank_w = {"easy": 3.0, "medium": 2.5, "hard": 2.0}[difficulty]
        drop_h = {"easy": 2.0, "medium": 3.0, "hard": 4.0}[difficulty]

        pivot_x = 5.0
        pivot_y = 0.3

        objects: list[dict] = [
            # Floor
            _segment("floor", [0, 0], [12, 0]),
            # Pivot (small static triangle approximated as a box)
            _static_poly("pivot", [
                [pivot_x - 0.15, 0],
                [pivot_x + 0.15, 0],
                [pivot_x, pivot_y],
            ], "steel"),
            # Plank (resting on pivot)
            _box("plank", plank_w, 0.08,
                 [pivot_x, pivot_y + 0.04], "wood", mass=2.0),
            # Projectile sitting on the right end of the plank
            _circle("projectile", 0.10,
                    [pivot_x + plank_w / 2 - 0.2, pivot_y + 0.14], "rubber"),
            # Heavy weight that will drop onto the left end
            _circle("weight", 0.15 + rng.uniform(0, 0.05),
                    [pivot_x - plank_w / 2 + 0.2, pivot_y + 0.04 + drop_h],
                    "steel", mass=5.0 + rng.uniform(0, 2)),
        ]

        # Target zone above
        target_y = pivot_y + drop_h + 1.0
        objects.append(
            _segment("target_platform",
                     [pivot_x + plank_w / 2 + 0.5, target_y],
                     [pivot_x + plank_w / 2 + 2.0, target_y])
        )

        trigger = {
            "type": "initial_velocity",
            "object": "weight",
            "velocity": [0, 0],  # just falls under gravity
        }

        steps = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
        outcome = (
            "Weight falls on left side of seesaw, seesaw tilts, "
            "projectile is launched upward from the right side."
        )

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=3.0 + drop_h * 0.3,
            expected_steps=steps,
            expected_outcome=outcome,
            target_object="projectile",
        )


# ── 3. RubeGoldbergSimple ──────────────────────────────────────────────────

class RubeGoldbergSimple(CausalChainTemplate):
    """3-step machine: ball rolls down ramp -> hits block -> block falls
    into bucket."""

    name = "rube_goldberg_simple"
    difficulties = ["easy", "medium"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        ramp_angle = {"easy": 25, "medium": 20}[difficulty]
        ramp_len = 3.0 + rng.uniform(0, 0.5)

        # Ramp top-left
        ramp_x0 = 1.0
        ramp_y0 = 4.0
        ramp_x1 = ramp_x0 + ramp_len * math.cos(math.radians(ramp_angle))
        ramp_y1 = ramp_y0 - ramp_len * math.sin(math.radians(ramp_angle))

        # Block on a ledge at the bottom of the ramp
        block_x = ramp_x1 + 0.3
        block_y = ramp_y1 + 0.15
        ledge_y = ramp_y1

        # Bucket below the ledge
        bucket_x = block_x + 0.8
        bucket_y = 0.0

        objects: list[dict] = [
            _segment("floor", [0, 0], [12, 0]),
            _segment("ramp", [ramp_x0, ramp_y0], [ramp_x1, ramp_y1]),
            _segment("ledge", [ramp_x1, ledge_y], [block_x + 0.5, ledge_y]),
            _circle("ball", 0.12 + rng.uniform(0, 0.03),
                    [ramp_x0 + 0.2, ramp_y0 + 0.15], "steel"),
            _box("block", 0.3, 0.3, [block_x, block_y + 0.15], "wood"),
            # Bucket walls
            _segment("bucket_left",  [bucket_x, 0], [bucket_x, 0.5]),
            _segment("bucket_right", [bucket_x + 0.8, 0], [bucket_x + 0.8, 0.5]),
        ]

        trigger = {
            "type": "initial_velocity",
            "object": "ball",
            "velocity": [0.5, 0],
        }

        outcome = (
            "Ball rolls down ramp, hits the wooden block at the bottom, "
            "block slides off ledge and falls into the bucket."
        )

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=4.0,
            expected_steps=3,
            expected_outcome=outcome,
            target_object="block",
        )


# ── 4. RubeGoldbergComplex ─────────────────────────────────────────────────

class RubeGoldbergComplex(CausalChainTemplate):
    """5-7 step machine: ball -> ramp -> hits pendulum weight -> pendulum
    swings into block -> block slides onto seesaw -> seesaw launches
    marble -> marble lands in target."""

    name = "rube_goldberg_complex"
    difficulties = ["medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        n_extra = {"medium": 0, "hard": 2}[difficulty]

        objects: list[dict] = [
            _segment("floor", [0, 0], [14, 0]),
        ]

        # Step 1: Ball on upper ramp
        objects.append(
            _segment("ramp_1", [1.0, 5.0], [3.5, 3.5])
        )
        objects.append(
            _circle("ball_1", 0.12, [1.2, 5.15], "steel")
        )

        # Step 2: Ball hits a block on a shelf
        shelf_y = 3.5
        objects.append(
            _segment("shelf_1", [3.5, shelf_y], [5.5, shelf_y])
        )
        objects.append(
            _box("block_1", 0.25, 0.25, [4.0, shelf_y + 0.125], "wood")
        )

        # Step 3: Block slides off shelf, falls onto another ramp
        objects.append(
            _segment("ramp_2", [5.0, 3.0], [7.0, 1.5])
        )

        # Step 4: Block slides down ramp_2 and hits a line of small blocks
        platform_y = 1.5
        objects.append(
            _segment("platform_1", [7.0, platform_y], [10.0, platform_y])
        )

        block_positions = [7.5, 8.0, 8.5]
        if n_extra > 0:
            block_positions.extend([9.0, 9.5][:n_extra])

        for idx, bx in enumerate(block_positions):
            objects.append(
                _box(f"block_chain_{idx}", 0.15, 0.40,
                     [bx, platform_y + 0.20], "wood")
            )

        # Step 5: Last block in chain knocks ball into bucket
        final_ball_x = block_positions[-1] + 0.5
        objects.append(
            _circle("final_ball", 0.10,
                    [final_ball_x, platform_y + 0.10], "rubber")
        )

        # Bucket
        bucket_x = final_ball_x + 0.8
        objects.append(_segment("bucket_left",  [bucket_x, platform_y], [bucket_x, platform_y + 0.5]))
        objects.append(_segment("bucket_right", [bucket_x + 0.6, platform_y], [bucket_x + 0.6, platform_y + 0.5]))

        trigger = {
            "type": "initial_velocity",
            "object": "ball_1",
            "velocity": [0.5 + rng.uniform(0, 0.3), 0],
        }

        total_steps = 5 + len(block_positions)
        outcome = (
            "Ball rolls down ramp, hits block on shelf. "
            "Block slides off and falls onto second ramp, slides down and "
            f"topples a chain of {len(block_positions)} small blocks. "
            "Last block knocks the final ball into the bucket."
        )

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=5.0 + n_extra * 0.5,
            expected_steps=total_steps,
            expected_outcome=outcome,
            world=_world(width=14, height=8),
            target_object="final_ball",
        )


# ── 5. BranchingChain ──────────────────────────────────────────────────────

class BranchingChain(CausalChainTemplate):
    """One trigger event causes two parallel outcome branches."""

    name = "branching_chain"
    difficulties = ["medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        objects: list[dict] = [
            _segment("floor", [0, 0], [12, 0]),
        ]

        # Central ramp — ball rolls down and hits a wide block
        objects.append(_segment("ramp", [3.0, 4.0], [5.5, 2.0]))
        objects.append(_circle("trigger_ball", 0.15, [3.2, 4.2], "steel"))

        # Wide block sitting on a shelf — when hit it splits the force
        shelf_y = 2.0
        objects.append(_segment("shelf", [5.5, shelf_y], [8.5, shelf_y]))
        objects.append(
            _box("splitter_block", 0.5, 0.3, [6.0, shelf_y + 0.15], "wood")
        )

        # Branch A (left): a ball on the left side of the shelf, slightly
        # elevated, gets pushed off the ledge by the splitter
        objects.append(
            _circle("branch_a_ball", 0.10,
                    [5.7, shelf_y + 0.10], "rubber")
        )
        # Left bucket
        objects.append(_segment("bucket_a_left",  [4.5, 0], [4.5, 0.5]))
        objects.append(_segment("bucket_a_right", [5.3, 0], [5.3, 0.5]))

        # Branch B (right): dominos on the shelf, toppled by the splitter
        n_dominos = {"medium": 2, "hard": 4}[difficulty]
        for i in range(n_dominos):
            dx = 6.8 + i * 0.45
            objects.append(
                _box(f"domino_b_{i}", 0.12, 0.50,
                     [dx, shelf_y + 0.25], "wood")
            )

        # Right bucket
        right_bucket_x = 6.8 + n_dominos * 0.45 + 0.5
        objects.append(
            _segment("bucket_b_left",  [right_bucket_x, shelf_y],
                     [right_bucket_x, shelf_y + 0.5])
        )
        objects.append(
            _segment("bucket_b_right", [right_bucket_x + 0.6, shelf_y],
                     [right_bucket_x + 0.6, shelf_y + 0.5])
        )

        trigger = {
            "type": "initial_velocity",
            "object": "trigger_ball",
            "velocity": [1.0 + rng.uniform(0, 0.5), 0],
        }

        steps = 3 + n_dominos  # ramp + splitter impact + 2 branches
        outcome = (
            "Ball rolls down ramp and hits splitter block. "
            "Branch A: ball on the left is pushed off the shelf into "
            "the left bucket. "
            f"Branch B: splitter topples {n_dominos} dominos on the right."
        )

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=4.0 + n_dominos * 0.3,
            expected_steps=steps,
            expected_outcome=outcome,
            target_object="branch_a_ball",
        )


# ── 6. TimingGate ──────────────────────────────────────────────────────────

class TimingGate(CausalChainTemplate):
    """Outcome depends on relative timing of two converging events. A ball
    rolls down a ramp while a gate (sliding block) is in motion — the ball
    either passes through or is blocked depending on timing."""

    name = "timing_gate"
    difficulties = ["medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        # Gate speed determines timing window
        gate_speed = {"medium": 1.2, "hard": 0.8}[difficulty]
        ramp_angle = {"medium": 30, "hard": 25}[difficulty]

        ramp_len = 3.0
        ramp_x0 = 1.0
        ramp_y0 = 5.0
        ramp_x1 = ramp_x0 + ramp_len * math.cos(math.radians(ramp_angle))
        ramp_y1 = ramp_y0 - ramp_len * math.sin(math.radians(ramp_angle))

        # The gate is a block sliding horizontally on a platform
        gate_platform_y = ramp_y1 - 0.1
        gap_x = ramp_x1 + 1.0  # gap position in the platform

        objects: list[dict] = [
            _segment("floor", [0, 0], [12, 0]),
            # Main ramp
            _segment("ramp", [ramp_x0, ramp_y0], [ramp_x1, ramp_y1]),
            # Platform with a gap
            _segment("platform_left", [ramp_x1, gate_platform_y],
                     [gap_x, gate_platform_y]),
            _segment("platform_right", [gap_x + 0.5, gate_platform_y],
                     [gap_x + 3.0, gate_platform_y]),
            # Main ball
            _circle("main_ball", 0.12, [ramp_x0 + 0.2, ramp_y0 + 0.15],
                    "steel"),
            # Gate block that slides to cover or uncover the gap
            _box("gate_block", 0.5, 0.2,
                 [gap_x - 1.0, gate_platform_y + 0.1], "steel",
                 mass=3.0),
            # A trigger ball that pushes the gate
            _circle("gate_trigger", 0.10,
                    [gap_x - 2.5, gate_platform_y + 0.1], "steel"),
        ]

        # Second ramp for the ball after passing through the gap
        catch_y = gate_platform_y - 1.5
        objects.append(
            _segment("catch_ramp", [gap_x, catch_y], [gap_x + 2.0, catch_y])
        )

        # Bucket
        bucket_x = gap_x + 2.2
        objects.append(_segment("bucket_left",  [bucket_x, 0], [bucket_x, 0.5]))
        objects.append(_segment("bucket_right", [bucket_x + 0.6, 0],
                                [bucket_x + 0.6, 0.5]))

        trigger = {
            "type": "initial_velocity",
            "object": "main_ball",
            "velocity": [0.5, 0],
        }

        # Also set gate trigger in motion
        objects[-5]["velocity"] = [gate_speed, 0]  # gate_trigger

        steps = {"medium": 4, "hard": 5}[difficulty]
        outcome = (
            "Main ball rolls down the ramp onto the platform. "
            "Meanwhile the gate trigger pushes the gate block. "
            "Depending on timing, the ball either falls through the gap "
            "onto the catch ramp and into the bucket, or is blocked by the gate."
        )

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=5.0,
            expected_steps=steps,
            expected_outcome=outcome,
            target_object="main_ball",
        )


# ── 7. NearMiss ────────────────────────────────────────────────────────────

class NearMiss(CausalChainTemplate):
    """A chain where one interaction barely succeeds or barely fails.
    Hard difficulty only — at easy/medium the outcome would be ambiguous."""

    name = "near_miss"
    difficulties = ["hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        # The setup: a ball rolls down a ramp and must jump across a gap
        # to hit a target block. The gap width is tuned so it barely makes
        # it (or barely misses).
        succeeds = rng.choice([True, False])

        ramp_len = 2.5
        ramp_angle = 20
        ramp_x0 = 1.0
        ramp_y0 = 4.5
        ramp_x1 = ramp_x0 + ramp_len * math.cos(math.radians(ramp_angle))
        ramp_y1 = ramp_y0 - ramp_len * math.sin(math.radians(ramp_angle))

        # Small lip at end of ramp to give the ball a slight upward trajectory
        lip_angle = 10
        lip_len = 0.3
        lip_x = ramp_x1 + lip_len * math.cos(math.radians(lip_angle))
        lip_y = ramp_y1 + lip_len * math.sin(math.radians(lip_angle))

        # Gap width — tuned to barely succeed or fail
        gap = 1.2 if succeeds else 1.6

        # Target block on a platform across the gap
        target_x = lip_x + gap
        target_platform_y = ramp_y1 - 0.1

        objects: list[dict] = [
            _segment("floor", [0, 0], [12, 0]),
            _segment("ramp", [ramp_x0, ramp_y0], [ramp_x1, ramp_y1]),
            _segment("lip", [ramp_x1, ramp_y1], [lip_x, lip_y]),
            _segment("target_platform", [target_x - 0.2, target_platform_y],
                     [target_x + 2.0, target_platform_y]),
            _circle("ball", 0.12, [ramp_x0 + 0.2, ramp_y0 + 0.15], "steel"),
            _box("target_block", 0.3, 0.3,
                 [target_x + 0.3, target_platform_y + 0.15], "wood"),
        ]

        # If ball hits the target, the target falls into a bucket
        bucket_x = target_x + 1.5
        objects.append(_segment("bucket_left",  [bucket_x, 0], [bucket_x, 0.4]))
        objects.append(_segment("bucket_right", [bucket_x + 0.6, 0],
                                [bucket_x + 0.6, 0.4]))

        trigger_vx = 1.5 + rng.uniform(0, 0.3)
        trigger = {
            "type": "initial_velocity",
            "object": "ball",
            "velocity": [trigger_vx, 0],
        }

        if succeeds:
            outcome = (
                "Ball rolls down ramp, launches off the lip, barely clears "
                "the gap, lands on the target platform and hits the target "
                "block, which slides into the bucket."
            )
            steps = 4
        else:
            outcome = (
                "Ball rolls down ramp, launches off the lip, but falls short "
                "of the target platform and drops to the floor. The target "
                "block remains untouched."
            )
            steps = 2

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=4.0,
            expected_steps=steps,
            expected_outcome=outcome,
            target_object="ball",
        )


# ── 8. ConservationChain ───────────────────────────────────────────────────

class ConservationChain(CausalChainTemplate):
    """Newton's cradle-style momentum/energy transfer through a line of
    objects. A fast object strikes a row of stationary objects; the last
    one in line receives the transferred momentum."""

    name = "conservation_chain"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        n_balls = {"easy": 3, "medium": 5, "hard": 7}[difficulty]
        ball_r = 0.15
        ball_material = "steel"

        # All cradle balls are touching in a line on a flat platform
        platform_y = 1.0
        x_start = 3.0
        spacing = ball_r * 2.0  # exactly touching

        objects: list[dict] = [
            _segment("floor", [0, 0], [14, 0]),
            _segment("platform", [x_start - 1.0, platform_y],
                     [x_start + n_balls * spacing + 2.0, platform_y]),
        ]

        # Stationary cradle balls
        for i in range(n_balls):
            x = x_start + i * spacing
            objects.append(
                _circle(f"cradle_{i}", ball_r,
                        [x, platform_y + ball_r], ball_material)
            )

        # Striker ball coming in from the left
        striker_x = x_start - 1.5
        striker_v = 3.0 + rng.uniform(0, 1.0)
        objects.append(
            _circle("striker", ball_r,
                    [striker_x, platform_y + ball_r], ball_material)
        )

        # Target zone for the last ball
        last_x = x_start + (n_balls - 1) * spacing + 1.5
        objects.append(
            _segment("target_wall", [last_x, platform_y],
                     [last_x, platform_y + 1.0])
        )

        trigger = {
            "type": "initial_velocity",
            "object": "striker",
            "velocity": [striker_v, 0],
        }

        # With steel (elasticity=0.5), momentum partially transfers through
        # each ball.  At easy difficulty (3 balls) the end ball clearly moves;
        # at hard difficulty (7 balls) much energy is lost to intermediate
        # collisions.
        steps = n_balls + 1  # striker hit + n transfers
        if difficulty == "easy":
            outcome = (
                f"Striker hits the first of {n_balls} balls. Momentum "
                "transfers through the line and the last ball is knocked "
                "forward toward the target wall."
            )
        elif difficulty == "medium":
            outcome = (
                f"Striker collides with {n_balls}-ball chain. Each collision "
                "transfers momentum with some energy loss. The last ball "
                "moves forward with reduced speed."
            )
        else:
            outcome = (
                f"Striker collides with a chain of {n_balls} steel balls. "
                "Momentum progressively decreases through each collision. "
                "The last ball barely moves; most energy dissipated in "
                "intermediate impacts."
            )

        return self._scenario(
            seed, difficulty, objects, trigger,
            sim_time=3.0 + n_balls * 0.2,
            expected_steps=steps,
            expected_outcome=outcome,
            world=_world(width=14, height=6),
            target_object=f"cradle_{n_balls - 1}",
        )


# ── Template registry ───────────────────────────────────────────────────────

CAUSAL_CHAIN_TEMPLATES: list[CausalChainTemplate] = [
    DominoLine(),
    SeesawLaunch(),
    RubeGoldbergSimple(),
    RubeGoldbergComplex(),
    BranchingChain(),
    TimingGate(),
    NearMiss(),
    ConservationChain(),
]
