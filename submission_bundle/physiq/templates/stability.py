"""Stability judgment scenario templates (Task 2).

8 templates that generate scenarios where an LLM must predict whether
a 2-D arrangement of objects remains stable under gravity.

Design principle
----------------
Stability is determined by GEOMETRY (center of mass, base width, overhang,
slope angle), NOT by material name.  All scenarios use friction in the
range 0.3–0.8 so that the material name is never a direct proxy for the
stability outcome.

The PhysIQ engine's ``is_stable()`` runs 3 s of simulation and checks that
ALL dynamic bodies have velocity < 0.1 m/s and angular_velocity < 0.1 rad/s.
Objects that topple and come to rest still count as stable.  For UNSTABLE,
objects must still be MOVING after 3 s (slope keeps accelerating them, or
high KE + elasticity keeps them bouncing).

Strategies for instability without low friction
-----------------------------------------------
* Slope instability: friction_coeff < tan(slope_angle) → continuous sliding.
* Overhang collapse: CoM outside support edge → tips, falls, bounces (e=0.5).
* Asymmetric geometry: pyramid missing base blocks, narrow pillar, etc.
* Arch splay: pillars lean outward → capstone falls through (geometry).
* Tall-narrow column on slope: slides AND tips.

Template 7 (FrictionDependent) is the ONE template where friction IS the
discriminator — but the angles are specific enough that you must reason about
friction×cos(θ) vs sin(θ), not just "grippy vs slippery".
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Literal

import numpy as np

# ── Materials ────────────────────────────────────────────────────────────────

MATERIAL_DB = {
    "rubber": {"friction": 0.8, "elasticity": 0.8, "density": 1.2},
    "wood": {"friction": 0.4, "elasticity": 0.3, "density": 0.6},
    "steel": {"friction": 0.3, "elasticity": 0.5, "density": 7.8},
    "ice": {"friction": 0.05, "elasticity": 0.2, "density": 0.9},
    "sponge": {"friction": 0.9, "elasticity": 0.1, "density": 0.1},
}

Difficulty = Literal["easy", "medium", "hard"]

QUESTION = (
    "Will this arrangement remain stable under gravity? "
    "If not, describe what happens first and which object moves."
)

QUESTION_HARD = (
    "Will this arrangement remain stable under gravity? "
    "If unstable, which specific object moves first, in which direction?"
)


def _param_mat(friction: float, elasticity: float = 0.35, density: float = 1.2) -> dict:
    return {
        "friction": round(friction, 2),
        "elasticity": round(elasticity, 2),
        "density": round(density, 2),
    }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _mat(name: str) -> dict:
    """Return a copy of a named material dict."""
    return MATERIAL_DB[name].copy()


def _custom_mat(d: dict) -> dict:
    return dict(d)


def _ground(
    oid: str = "ground",
    width: float = 20.0,
    y: float = 0.0,
    material: str | dict = "steel",
) -> dict:
    """Standard ground segment.  Wide to catch sliding objects."""
    hw = width / 2.0
    mat = _mat(material) if isinstance(material, str) else _custom_mat(material)
    return {
        "id": oid,
        "type": "static_segment",
        "start": [-hw, y],
        "end": [hw, y],
        "material": mat,
    }


def _box(
    oid: str,
    width: float,
    height: float,
    x: float,
    y: float,
    material: str | dict = "wood",
    angle: float = 0.0,
) -> dict:
    mat = _mat(material) if isinstance(material, str) else _custom_mat(material)
    return {
        "id": oid,
        "type": "box",
        "width": width,
        "height": height,
        "position": [x, y],
        "angle": angle,
        "material": mat,
    }


def _circle(
    oid: str,
    radius: float,
    x: float,
    y: float,
    material: str | dict = "wood",
) -> dict:
    mat = _mat(material) if isinstance(material, str) else _custom_mat(material)
    return {
        "id": oid,
        "type": "circle",
        "radius": radius,
        "position": [x, y],
        "material": mat,
    }


def _scenario(
    template_name: str,
    seed: int,
    difficulty: Difficulty,
    objects: list[dict],
    expected_stable: bool,
    world_width: float = 20.0,
    world_height: float = 10.0,
    question: str | None = None,
) -> dict:
    q = question if question is not None else QUESTION
    return {
        "id": f"stab_{template_name}_{seed}",
        "task_type": "stability",
        "difficulty": difficulty,
        "world": {
            "gravity": [0, -9.81],
            "bounds": {"width": world_width, "height": world_height},
            "damping": 0.99,
        },
        "objects": objects,
        "simulation_time": 3.0,
        "question": q,
        "answer_format": "STABLE or UNSTABLE with explanation",
        "expected_stable": expected_stable,
    }


# ── Abstract base ────────────────────────────────────────────────────────────


class StabilityTemplate(ABC):
    """All stability templates share this interface."""

    name: str

    @abstractmethod
    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        ...


# ── 1. Simple Stack ──────────────────────────────────────────────────────────


class SimpleStack(StabilityTemplate):
    """N blocks stacked vertically.

    Stable: wide, squat blocks (w/h > 2) with friction 0.3–0.6 on flat ground.
            Geometry ensures CoM stays centred; damping kills any wobble.

    Unstable (geometry-driven, two sub-strategies):
      A) Tall narrow column (h/w > 4) with a heavily tilted top block (30–50°)
         on flat ground.  The column tips, falls from height, and with
         elasticity=0.5 keeps bouncing past 3 s.
      B) Tall narrow column on a slope where friction < tan(slope_angle).
         Uses steel-like friction (0.3) on 20–28° slopes → slides continuously.
    """

    name = "simple_stack"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        n_blocks = {"easy": rng.randint(2, 4), "medium": rng.randint(4, 6),
                    "hard": rng.randint(6, 9)}[difficulty]
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        objects: list[dict] = []

        if make_stable:
            # Wide squat blocks, varied friction 0.3–0.6; geometry is safe
            friction = round(rng.uniform(0.30, 0.60), 2)
            mat = _param_mat(friction, elasticity=0.30, density=1.2)
            objects.append(_ground(material=_param_mat(friction, 0.30, 1.5)))
            block_w = rng.uniform(0.6, 1.0)
            block_h = rng.uniform(0.15, 0.28)  # squat: w/h > 2
            y = block_h / 2.0
            for i in range(n_blocks):
                w = max(block_w - i * 0.02, 0.45)
                objects.append(_box(f"block_{i}", w, block_h, 0.0, y, mat))
                y += block_h
        else:
            # Unstable: tall narrow column, tilted top block, bouncy fall
            # Use moderate friction (0.4–0.6) — instability is geometric
            friction = round(rng.uniform(0.40, 0.60), 2)
            mat_base = _param_mat(friction, elasticity=0.30, density=1.0)
            mat_top = _param_mat(friction, elasticity=0.50, density=1.2)
            objects.append(_ground(material=_param_mat(friction, 0.35, 1.5)))

            # Strategy: sub-strategy A (tipping) or B (slope slide)
            strategy = rng.randint(0, 2)

            if strategy == 0:
                # A: tall narrow column on flat ground, tilted top block
                block_w = rng.uniform(0.12, 0.20)
                block_h = rng.uniform(0.30, 0.50)
                y = block_h / 2.0
                for i in range(n_blocks):
                    if i < n_blocks - 1:
                        objects.append(
                            _box(f"block_{i}", block_w, block_h, 0.0, y, mat_base)
                        )
                    else:
                        # Top block: wide + heavily tilted so it levers the column
                        top_w = block_w * rng.uniform(3.0, 5.0)
                        tilt = rng.uniform(30, 50) * rng.choice([-1.0, 1.0])
                        objects.append(
                            _box(f"block_{i}", top_w, block_h, 0.0, y, mat_top,
                                 angle=tilt)
                        )
                    y += block_h
            else:
                # B: place narrow column on a slope where friction < tan(angle)
                # steel friction=0.3 → tan(16.7°)=0.3, use 20–28° so it slides
                slope_angle = rng.uniform(20, 28)
                slope_fric = 0.30
                angle_rad = math.radians(slope_angle)
                platform_len = rng.uniform(2.0, 3.0)

                px_start = -platform_len / 2.0 * math.cos(angle_rad)
                py_start = 0.5
                px_end = platform_len / 2.0 * math.cos(angle_rad)
                py_end = py_start + platform_len * math.sin(angle_rad)

                objects[0] = _ground(
                    material=_param_mat(slope_fric, 0.35, 1.5)
                )
                objects.append({
                    "id": "slope",
                    "type": "static_segment",
                    "start": [px_start, py_start],
                    "end": [px_end, py_end],
                    "material": _param_mat(slope_fric, 0.30, 1.5),
                })

                block_w = rng.uniform(0.14, 0.22)
                block_h = rng.uniform(0.30, 0.50)
                mat_slide = _param_mat(slope_fric, elasticity=0.50, density=1.2)

                for i in range(n_blocks):
                    frac = (i + 0.5) / n_blocks
                    cx = px_start + frac * (px_end - px_start)
                    cy = py_start + frac * (py_end - py_start)
                    cx -= (block_h / 2.0) * math.sin(angle_rad)
                    cy += (block_h / 2.0) * math.cos(angle_rad)
                    objects.append(
                        _box(f"block_{i}", block_w, block_h, cx, cy, mat_slide,
                             angle=slope_angle)
                    )

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 2. Offset Stack ─────────────────────────────────────────────────────────


class OffsetStack(StabilityTemplate):
    """Blocks with horizontal offsets (overhang).

    Stable: small offsets < 25% of block_w; CoM stays over support.
            Friction 0.35–0.55 — geometry is what keeps it safe.

    Unstable: large offsets 55–75% of block_w; CoM is OUTSIDE the support
              edge.  The stack tips due to geometry, falls, and bounces
              (elasticity=0.5).  Friction is 0.4–0.6 — NOT low friction.
    """

    name = "offset_stack"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        n_blocks = {"easy": rng.randint(2, 4), "medium": rng.randint(4, 6),
                    "hard": rng.randint(6, 9)}[difficulty]
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        block_w = rng.uniform(0.5, 0.9)
        block_h = rng.uniform(0.18, 0.28)
        objects: list[dict] = []

        if make_stable:
            # Varied friction 0.35–0.55; small offsets keep CoM safe
            friction = round(rng.uniform(0.35, 0.55), 2)
            mat = _param_mat(friction, elasticity=0.30, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.30, 1.5)))
            y = block_h / 2.0
            x = 0.0
            sign = float(rng.choice([-1, 1]))
            for i in range(n_blocks):
                dx = sign * rng.uniform(0.02, block_w * 0.22) if i > 0 else 0.0
                sign *= -1
                x += dx
                objects.append(_box(f"block_{i}", block_w, block_h, x, y, mat))
                y += block_h
        else:
            # Large overhang: CoM falls outside support → geometry instability
            # Use moderate friction 0.4–0.6 (NOT low friction)
            friction = round(rng.uniform(0.40, 0.60), 2)
            mat = _param_mat(friction, elasticity=0.50, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.35, 1.5)))
            y = block_h / 2.0
            x = 0.0
            direction = float(rng.choice([-1.0, 1.0]))
            for i in range(n_blocks):
                if i == 0:
                    dx = 0.0
                else:
                    # 55–75% overhang: centre of upper block past edge of lower
                    dx = direction * block_w * rng.uniform(0.55, 0.75)
                x += dx
                # Slight tilt in overhang direction on the top block
                angle = direction * rng.uniform(3, 8) if i == n_blocks - 1 else 0.0
                objects.append(
                    _box(f"block_{i}", block_w, block_h, x, y, mat, angle=angle)
                )
                y += block_h

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 3. Pyramid ───────────────────────────────────────────────────────────────


class Pyramid(StabilityTemplate):
    """Triangular arrangement of blocks.

    Stable: symmetric pyramid, friction 0.3–0.5; wide base ensures stability.

    Unstable: asymmetric — 1–2 base blocks removed from one side so the
              structure lacks support and collapses.  Friction 0.4–0.65 so
              instability is purely geometric.  Elasticity=0.5 keeps pieces
              moving after the fall.
    """

    name = "pyramid"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        base_count = {"easy": rng.randint(2, 4), "medium": rng.randint(3, 5),
                      "hard": rng.randint(4, 7)}[difficulty]
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        block_w = rng.uniform(0.35, 0.60)
        block_h = rng.uniform(0.18, 0.28)
        objects: list[dict] = []
        idx = 0

        if make_stable:
            # Symmetric pyramid; friction 0.3–0.5 (moderate, not rubber)
            friction = round(rng.uniform(0.30, 0.50), 2)
            mat = _param_mat(friction, elasticity=0.30, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.30, 1.5)))
            for row in range(base_count):
                count = base_count - row
                sx = -(count - 1) * block_w / 2.0
                y = block_h / 2.0 + row * block_h
                for col in range(count):
                    x = sx + col * block_w
                    objects.append(
                        _box(f"pyr_{idx}", block_w, block_h, x, y, mat)
                    )
                    idx += 1
        else:
            # Asymmetric: remove base blocks → geometry fails
            # Friction 0.4–0.65; bouncy so pieces stay moving after collapse
            friction = round(rng.uniform(0.40, 0.65), 2)
            mat = _param_mat(friction, elasticity=0.50, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.40, 1.5)))

            all_blocks: list[dict | None] = []
            base_indices: list[int] = []
            for row in range(base_count):
                count = base_count - row
                sx = -(count - 1) * block_w / 2.0
                y = block_h / 2.0 + row * block_h
                for col in range(count):
                    x = sx + col * block_w
                    if row == 0:
                        base_indices.append(len(all_blocks))
                    all_blocks.append(
                        _box(f"pyr_{idx}", block_w, block_h, x, y, mat)
                    )
                    idx += 1

            # Remove 1–2 base blocks from one side (left side = first indices)
            n_remove = min(len(base_indices), max(1, base_count // 2))
            for ri in range(n_remove):
                all_blocks[base_indices[ri]] = None

            objects.extend(b for b in all_blocks if b is not None)

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 4. Arch ──────────────────────────────────────────────────────────────────


class Arch(StabilityTemplate):
    """Two pillars with capstone on top.

    Stable: vertical pillars (lean 0–3°), capstone wider than gap+pillars,
            friction 0.3–0.5.  Geometry ensures load transfers properly.

    Unstable: pillars lean outward 20–35° on flat ground, capstone too narrow
              (0.5–0.8× gap width), friction 0.35–0.6.  Pillars splay apart
              under capstone weight — geometry failure, not friction failure.
    """

    name = "arch"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        pillar_w = rng.uniform(0.20, 0.40)
        pillar_h_map = {
            "easy": rng.uniform(0.5, 0.8),
            "medium": rng.uniform(0.7, 1.2),
            "hard": rng.uniform(1.0, 1.6),
        }
        pillar_h = pillar_h_map[difficulty]
        cap_h = rng.uniform(0.12, 0.25)
        gap = rng.uniform(0.30, 0.60)
        objects: list[dict] = []

        if make_stable:
            # Friction 0.3–0.5; vertical pillars; wide capstone
            friction = round(rng.uniform(0.30, 0.50), 2)
            mat = _param_mat(friction, elasticity=0.30, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.30, 1.5)))

            cap_w = gap + 2 * pillar_w + rng.uniform(0.10, 0.30)
            lean = rng.uniform(-2, 2)  # near-vertical
            left_x = -(gap / 2.0 + pillar_w / 2.0)
            right_x = gap / 2.0 + pillar_w / 2.0
            pil_y = pillar_h / 2.0
            cap_y = pillar_h + cap_h / 2.0

            objects.extend([
                _box("left_pillar", pillar_w, pillar_h, left_x, pil_y, mat,
                     angle=-lean),
                _box("right_pillar", pillar_w, pillar_h, right_x, pil_y, mat,
                     angle=lean),
                _box("capstone", cap_w, cap_h, 0.0, cap_y, mat),
            ])
        else:
            # Pillars lean outward 20–35°; narrow capstone; geometry splay
            # Use friction 0.35–0.6 — instability is NOT friction-driven
            friction = round(rng.uniform(0.35, 0.60), 2)
            mat = _param_mat(friction, elasticity=0.50, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.40, 1.5)))

            lean = rng.uniform(20, 35)
            cap_w = gap * rng.uniform(0.50, 0.80)  # narrower than gap

            left_x = -(gap / 2.0 + pillar_w / 2.0)
            right_x = gap / 2.0 + pillar_w / 2.0
            pil_y = pillar_h / 2.0
            cap_y = pillar_h + cap_h / 2.0
            cap_x = rng.uniform(0.1, 0.3) * rng.choice([-1.0, 1.0])

            objects.extend([
                _box("left_pillar", pillar_w, pillar_h, left_x, pil_y, mat,
                     angle=lean),      # leans outward left
                _box("right_pillar", pillar_w, pillar_h, right_x, pil_y, mat,
                     angle=-lean),     # leans outward right
                _box("capstone", cap_w, cap_h, cap_x, cap_y, mat),
            ])

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 5. Cantilever ────────────────────────────────────────────────────────────


class Cantilever(StabilityTemplate):
    """Overhanging beam on a support pedestal.

    Stable: 20–35% overhang past support edge; heavy steel counterweight on
            short side keeps CoM over support.  Friction 0.3–0.5.

    Unstable: 75–90% overhang past support edge; heavy steel tip weight on
              overhanging end — CoM far outside support.  Beam tips due to
              geometry.  Friction 0.35–0.55; elasticity=0.5 so fallen beam
              keeps bouncing past 3 s.
    """

    name = "cantilever"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        support_w = rng.uniform(0.40, 0.70)
        support_h = rng.uniform(0.30, 0.60)
        beam_w_map = {
            "easy": rng.uniform(1.0, 1.5),
            "medium": rng.uniform(1.5, 2.5),
            "hard": rng.uniform(2.0, 3.0),
        }
        beam_w = beam_w_map[difficulty]
        beam_h = rng.uniform(0.10, 0.18)
        beam_y = support_h + beam_h / 2.0
        objects: list[dict] = []

        if make_stable:
            friction = round(rng.uniform(0.30, 0.50), 2)
            mat = _param_mat(friction, elasticity=0.30, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.30, 1.5)))
            objects.append(
                _box("support", support_w, support_h, 0.0,
                     support_h / 2.0, mat)
            )
            # 20–35% overhang; CoM stays over support
            overhang_frac = rng.uniform(0.20, 0.35)
            beam_cx = beam_w * (overhang_frac - 0.5)
            objects.append(_box("beam", beam_w, beam_h, beam_cx, beam_y, mat))
            # Heavy steel counterweight on short side
            cw = rng.uniform(0.20, 0.35)
            cw_x = beam_cx - beam_w / 2.0 + cw / 2.0
            cw_y = beam_y + beam_h / 2.0 + cw / 2.0
            objects.append(_box("counterweight", cw, cw, cw_x, cw_y, "steel"))
        else:
            # 75–90% overhang + heavy tip weight → CoM far outside support
            friction = round(rng.uniform(0.35, 0.55), 2)
            mat = _param_mat(friction, elasticity=0.50, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.40, 1.5)))
            objects.append(
                _box("support", support_w, support_h, 0.0,
                     support_h / 2.0, mat)
            )
            overhang_frac = rng.uniform(0.75, 0.90)
            beam_cx = beam_w * (overhang_frac - 0.5)
            objects.append(_box("beam", beam_w, beam_h, beam_cx, beam_y, mat))
            # Heavy steel tip weight on overhanging end
            tip = rng.uniform(0.25, 0.40)
            tip_x = beam_cx + beam_w / 2.0 - tip / 2.0
            tip_y = beam_y + beam_h / 2.0 + tip / 2.0
            tip_mat = _param_mat(friction, elasticity=0.50, density=7.8)
            objects.append(_box("tip_weight", tip, tip, tip_x, tip_y, tip_mat))

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 6. Mixed Shapes ─────────────────────────────────────────────────────────


class MixedShapes(StabilityTemplate):
    """Circles on rectangles, irregular combinations.

    Stable: wide flat base (w=1.2–2.0), ball centred on base, friction 0.35–0.6.
            Wide base ensures CoM stays over support regardless of friction.

    Unstable: very narrow pedestal (w=0.08–0.15), heavy ball with large lateral
              offset (0.3–0.6× radius).  Geometry causes the ball to roll off
              and topple the pedestal.  Friction 0.4–0.65; elasticity=0.5.
    """

    name = "mixed_shapes"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        n_extra = {"easy": 0, "medium": rng.randint(1, 3),
                   "hard": rng.randint(2, 5)}[difficulty]
        objects: list[dict] = []

        if make_stable:
            friction = round(rng.uniform(0.35, 0.60), 2)
            mat = _param_mat(friction, elasticity=0.30, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.30, 1.5)))
            # Wide flat base: geometry keeps everything centred
            base_w = rng.uniform(1.2, 2.0)
            base_h = rng.uniform(0.15, 0.25)
            objects.append(
                _box("base", base_w, base_h, 0.0, base_h / 2.0, mat)
            )
            radius = rng.uniform(0.08, min(0.15, base_w / 4.0))
            objects.append(
                _circle("ball_0", radius, 0.0, base_h + radius, mat)
            )
            y_cursor = base_h + 2 * radius
            for i in range(n_extra):
                bw = rng.uniform(0.3, base_w * 0.5)
                bh = rng.uniform(0.10, 0.18)
                objects.append(
                    _box(f"extra_{i}", bw, bh, 0.0, y_cursor + bh / 2.0, mat)
                )
                y_cursor += bh
        else:
            # Narrow pedestal + offset heavy ball → geometry instability
            friction = round(rng.uniform(0.40, 0.65), 2)
            mat = _param_mat(friction, elasticity=0.50, density=1.0)
            ball_mat = _param_mat(friction, elasticity=0.50, density=5.0)
            objects.append(_ground(material=_param_mat(friction, 0.40, 1.5),
                                   width=30.0))
            ped_w = rng.uniform(0.08, 0.15)
            ped_h = rng.uniform(0.5, 1.0)
            objects.append(
                _box("pedestal", ped_w, ped_h, 0.0, ped_h / 2.0, mat)
            )
            radius = rng.uniform(0.25, 0.45)
            # Offset beyond pedestal edge → ball rolls off immediately
            offset = rng.uniform(radius * 0.30, radius * 0.60) * rng.choice([-1, 1])
            objects.append(
                _circle("ball_0", radius, offset, ped_h + radius, ball_mat)
            )
            y_cursor = ped_h + 2 * radius
            for i in range(n_extra):
                r = rng.uniform(0.10, 0.25)
                ex_mat = _param_mat(friction, elasticity=0.50, density=1.0)
                objects.append(
                    _circle(f"ball_{i + 1}", r,
                             rng.uniform(-0.10, 0.10), y_cursor + r, ex_mat)
                )
                y_cursor += 2 * r

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 7. Friction Dependent ────────────────────────────────────────────────────


class FrictionDependent(StabilityTemplate):
    """Stability depends on friction vs slope angle (the ONE allowed exception).

    Stable: friction 0.5–0.7 on slope 15–22°.  friction > tan(angle) so
            static friction force exceeds gravitational component → held.
            tan(15°)≈0.27, tan(22°)≈0.40 → friction=0.5–0.7 safely exceeds.

    Unstable: friction 0.25–0.40 on slope 20–28° where friction < tan(angle).
              tan(20°)≈0.36, tan(28°)≈0.53 → chosen so friction is definitively
              less than tan(angle) for the specific pair, causing continuous
              sliding.  This requires reasoning about f×cos(θ) vs sin(θ),
              not just "grippy vs slippery".
    """

    name = "friction_dependent"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        n_blocks = {"easy": rng.randint(2, 4), "medium": rng.randint(3, 6),
                    "hard": rng.randint(5, 9)}[difficulty]

        if make_stable:
            # friction > tan(angle) → static friction holds the block
            ramp_angle = rng.uniform(15, 22)
            # Ensure friction > tan(angle) with margin
            min_friction = math.tan(math.radians(ramp_angle)) + 0.10
            friction = round(rng.uniform(max(min_friction, 0.50), 0.70), 2)
        else:
            # friction < tan(angle) → slides continuously
            ramp_angle = rng.uniform(20, 28)
            # Ensure friction < tan(angle) with margin
            max_friction = math.tan(math.radians(ramp_angle)) - 0.05
            friction = round(rng.uniform(0.25, min(max_friction, 0.40)), 2)

        mat = _param_mat(friction, elasticity=0.35, density=1.0)
        angle_rad = math.radians(ramp_angle)
        platform_len = rng.uniform(2.0, 3.5)

        px_start = -platform_len / 2.0 * math.cos(angle_rad)
        py_start = 0.5
        px_end = platform_len / 2.0 * math.cos(angle_rad)
        py_end = py_start + platform_len * math.sin(angle_rad)

        objects: list[dict] = [
            _ground(material=_param_mat(friction, 0.35, 1.5), width=20.0),
            {
                "id": "ramp",
                "type": "static_segment",
                "start": [px_start, py_start],
                "end": [px_end, py_end],
                "material": mat,
            },
        ]

        for i in range(n_blocks):
            frac = (i + 0.5) / n_blocks
            cx = px_start + frac * (px_end - px_start)
            cy = py_start + frac * (py_end - py_start)
            block_w = rng.uniform(0.20, 0.40)
            block_h = rng.uniform(0.15, 0.25)
            cx -= (block_h / 2.0) * math.sin(angle_rad)
            cy += (block_h / 2.0) * math.cos(angle_rad)
            objects.append(
                _box(f"slider_{i}", block_w, block_h, cx, cy, mat,
                     angle=ramp_angle)
            )

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 8. Chain Collapse ────────────────────────────────────────────────────────


class ChainCollapse(StabilityTemplate):
    """Domino-like chain: one piece tips and knocks the rest.

    Stable: well-spaced dominoes (spacing > 2× height), minimal initial tilt
            (±2°), friction 0.3–0.5.  Wide spacing means no chain reaction.

    Unstable: tight spacing (0.3–0.45× height), first domino tilted 35–55°.
              The geometric cascade falls regardless of friction.
              Friction 0.3–0.5 (same range as stable); elasticity=0.5 keeps
              dominoes moving after collapse.  Instability is spacing/geometry.
    """

    name = "chain_collapse"

    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        make_stable = bool(rng.randint(0, 2))
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        n_dominoes = {"easy": rng.randint(3, 5), "medium": rng.randint(5, 8),
                      "hard": rng.randint(7, 11)}[difficulty]
        domino_w = rng.uniform(0.06, 0.12)
        domino_h = rng.uniform(0.40, 0.70)
        objects: list[dict] = []

        if make_stable:
            # Same friction range as unstable — only spacing differs
            friction = round(rng.uniform(0.30, 0.50), 2)
            mat = _param_mat(friction, elasticity=0.30, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.30, 1.5)))
            spacing = domino_h * rng.uniform(2.0, 3.0)  # > 2× height
            for i in range(n_dominoes):
                x = -((n_dominoes - 1) * spacing) / 2.0 + i * spacing
                y = domino_h / 2.0
                angle = rng.uniform(-1.5, 1.5)
                objects.append(
                    _box(f"domino_{i}", domino_w, domino_h, x, y, mat,
                         angle=angle)
                )
        else:
            # Tight spacing (0.3–0.45× height); first domino pre-tilted 35–55°
            # Same friction range — geometry drives the cascade
            friction = round(rng.uniform(0.30, 0.50), 2)
            mat = _param_mat(friction, elasticity=0.50, density=1.0)
            objects.append(_ground(material=_param_mat(friction, 0.35, 1.5),
                                   width=30.0))
            spacing = domino_h * rng.uniform(0.30, 0.45)
            for i in range(n_dominoes):
                x = -((n_dominoes - 1) * spacing) / 2.0 + i * spacing
                y = domino_h / 2.0
                if i == 0:
                    angle = rng.uniform(35, 55)
                else:
                    angle = rng.uniform(-2, 2)
                objects.append(
                    _box(f"domino_{i}", domino_w, domino_h, x, y, mat,
                         angle=angle)
                )

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── 9. Adversarial Stability ─────────────────────────────────────────────────


class AdversarialStability(StabilityTemplate):
    """Adversarial scenarios where the naive pattern-matching answer is WRONG.

    Counterintuitive stable cases:
      0. TallHeavyBase — tall narrow tower with heavy steel base (low CoM) on
         rubber; LOOKS unstable, IS stable.
      1. BalancedLean — box leaning against a wall, counterweight block on the
         opposite side locks it in equilibrium; LOOKS precarious, IS stable.
      2. BallInGroove — two slightly inward-leaning pillars with a ball wedged
         in the gap between their tops; the ball prevents splay; IS stable.

    Counterintuitive unstable cases:
      3. AsymmetricMass — wide row of blocks that looks symmetric, but one end
         is steel (density 7.8) vs wood (density 0.6); off-centre CoM tips it.
      4. JustTooNarrowArch — arch with two solid pillars; capstone is 2% too
         narrow to bridge the gap and falls through; LOOKS like a solid arch.
      5. HiddenTopSlope — symmetric-looking stack on a steep slope (22–28°)
         with friction 0.30–0.38 (< tan(angle)) → continuous sliding.
    """

    name = "adversarial_stability"

    # ------------------------------------------------------------------
    # Sub-type 0: tall, narrow upper stack on a wide, dense base
    # ------------------------------------------------------------------
    def _sub0_tall_heavy_base(self, rng: np.random.RandomState) -> tuple[list[dict], bool]:
        n_blocks = rng.randint(4, 7)
        ground_friction = round(rng.uniform(0.5, 0.7), 2)
        objects: list[dict] = [_ground(material=_param_mat(ground_friction, 0.35, 1.5))]

        # Bottom 1-2 heavy steel-like blocks: wide and dense
        n_heavy = rng.randint(1, 3)
        base_w = round(rng.uniform(0.8, 1.2), 3)
        base_h = round(rng.uniform(0.18, 0.28), 3)
        base_density = round(rng.uniform(5.0, 7.0), 2)
        base_friction = round(rng.uniform(0.5, 0.7), 2)
        base_mat = _param_mat(base_friction, elasticity=0.30, density=base_density)

        y = base_h / 2.0
        for i in range(n_heavy):
            objects.append(_box(f"base_{i}", base_w, base_h, 0.0, y, base_mat))
            y += base_h

        # Upper blocks: narrow and light
        upper_w = round(rng.uniform(0.2, 0.4), 3)
        upper_h = round(rng.uniform(0.22, 0.35), 3)
        upper_density = round(rng.uniform(0.3, 0.5), 2)
        upper_friction = round(rng.uniform(0.4, 0.6), 2)
        upper_mat = _param_mat(upper_friction, elasticity=0.30, density=upper_density)

        for i in range(n_blocks - n_heavy):
            objects.append(_box(f"upper_{i}", upper_w, upper_h, 0.0, y, upper_mat))
            y += upper_h

        return objects, True  # stable: heavy base gives very low CoM

    # ------------------------------------------------------------------
    # Sub-type 1: leaning box with a steel counterweight on the other side
    # ------------------------------------------------------------------
    def _sub1_balanced_lean(self, rng: np.random.RandomState) -> tuple[list[dict], bool]:
        friction = round(rng.uniform(0.4, 0.6), 2)
        objects: list[dict] = [_ground(material=_param_mat(friction, 0.35, 1.5))]

        # Wide base block sitting flat on the ground (anchor for the lean)
        base_w = round(rng.uniform(1.2, 1.8), 3)
        base_h = round(rng.uniform(0.18, 0.28), 3)
        base_mat = _param_mat(friction, elasticity=0.30, density=0.6)
        base_y = base_h / 2.0
        objects.append(_box("base", base_w, base_h, 0.0, base_y, base_mat))

        # A second block leaning against the left wall (tilted 15-25°)
        lean_w = round(rng.uniform(0.3, 0.5), 3)
        lean_h = round(rng.uniform(0.6, 1.0), 3)
        lean_angle = round(rng.uniform(15.0, 25.0), 1)  # degrees, leans right
        lean_mat = _param_mat(friction, elasticity=0.30, density=0.7)
        # Place the leaning block on the left of the base, tilted
        lean_x = -base_w / 2.0 + lean_w / 2.0 + 0.05
        lean_y = base_h + lean_h / 2.0
        objects.append(_box("lean_block", lean_w, lean_h, lean_x, lean_y, lean_mat,
                            angle=lean_angle))

        # Counterweight block: steel, on the right side of the base block top
        cw_w = round(rng.uniform(0.2, 0.35), 3)
        cw_h = round(rng.uniform(0.18, 0.28), 3)
        cw_density = round(rng.uniform(5.5, 7.8), 2)
        cw_mat = _param_mat(friction, elasticity=0.30, density=cw_density)
        cw_x = base_w / 2.0 - cw_w / 2.0 - 0.05
        cw_y = base_h + cw_h / 2.0
        objects.append(_box("counterweight", cw_w, cw_h, cw_x, cw_y, cw_mat))

        return objects, True  # stable: counterweight balances lean

    # ------------------------------------------------------------------
    # Sub-type 2: two inward-leaning pillars with ball wedged in the gap
    # ------------------------------------------------------------------
    def _sub2_ball_in_groove(self, rng: np.random.RandomState) -> tuple[list[dict], bool]:
        friction = round(rng.uniform(0.4, 0.6), 2)
        objects: list[dict] = [_ground(material=_param_mat(friction, 0.35, 1.5))]

        ball_r = round(rng.uniform(0.10, 0.16), 3)
        # Gap between pillar inner faces: 0.6-1.0× ball diameter
        gap = ball_r * 2.0 * round(rng.uniform(0.60, 1.00), 2)
        pillar_w = round(rng.uniform(0.15, 0.25), 3)
        pillar_h = round(rng.uniform(0.8, 1.4), 3)

        # Pillar centres at ±(gap/2 + pillar_w/2)
        px = gap / 2.0 + pillar_w / 2.0
        py = pillar_h / 2.0
        # Slight inward tilt (2-4°) so ball is actively wedged
        tilt = round(rng.uniform(2.0, 4.0), 1)
        pillar_mat = _param_mat(friction, elasticity=0.30, density=0.8)

        objects.append(_box("pillar_l", pillar_w, pillar_h, -px, py, pillar_mat,
                            angle=tilt))
        objects.append(_box("pillar_r", pillar_w, pillar_h,  px, py, pillar_mat,
                            angle=-tilt))

        # Ball sits in the groove near the top of the pillars
        ball_y = pillar_h - ball_r * 0.5
        ball_mat = _param_mat(friction, elasticity=0.40, density=1.0)
        objects.append(_circle("ball", ball_r, 0.0, ball_y, ball_mat))

        return objects, True  # stable: ball locks pillar tops together

    # ------------------------------------------------------------------
    # Sub-type 3: wide row of blocks — asymmetric mass (steel vs wood)
    # ------------------------------------------------------------------
    def _sub3_asymmetric_mass(self, rng: np.random.RandomState) -> tuple[list[dict], bool]:
        n_blocks = rng.randint(3, 6)
        friction = round(rng.uniform(0.4, 0.6), 2)
        # High elasticity so it keeps moving after tipping
        objects: list[dict] = [_ground(material=_param_mat(friction, 0.50, 1.5))]

        block_w = round(rng.uniform(0.4, 0.6), 3)
        block_h = round(rng.uniform(0.25, 0.40), 3)

        # Steel block index: always at one end (0 or last)
        steel_idx = rng.randint(0, 2) * (n_blocks - 1)  # 0 or n_blocks-1

        total_w = n_blocks * block_w
        x_start = -total_w / 2.0 + block_w / 2.0
        y = block_h / 2.0

        for i in range(n_blocks):
            x = x_start + i * block_w
            if i == steel_idx:
                mat = _param_mat(0.30, elasticity=0.50, density=7.8)
            else:
                mat = _param_mat(friction, elasticity=0.30, density=0.6)
            objects.append(_box(f"block_{i}", block_w, block_h, x, y, mat))

        return objects, False  # unstable: off-centre CoM tips toward steel end

    # ------------------------------------------------------------------
    # Sub-type 4: arch with capstone that is just too narrow to bridge
    # ------------------------------------------------------------------
    def _sub4_just_too_narrow_arch(self, rng: np.random.RandomState) -> tuple[list[dict], bool]:
        friction = round(rng.uniform(0.5, 0.7), 2)
        objects: list[dict] = [_ground(material=_param_mat(friction, 0.35, 1.5))]

        pillar_w = round(rng.uniform(0.3, 0.5), 3)
        pillar_h = round(rng.uniform(0.8, 1.4), 3)
        # Gap between pillar tops
        gap = round(rng.uniform(0.5, 0.9), 3)

        px = gap / 2.0 + pillar_w / 2.0
        py = pillar_h / 2.0
        pillar_mat = _param_mat(friction, elasticity=0.30, density=0.8)

        objects.append(_box("pillar_l", pillar_w, pillar_h, -px, py, pillar_mat))
        objects.append(_box("pillar_r", pillar_w, pillar_h,  px, py, pillar_mat))

        # Capstone: 2% too narrow to fully bridge the gap (gap × 0.48)
        capstone_w = round(gap * 0.48, 3)
        capstone_h = round(rng.uniform(0.18, 0.30), 3)
        capstone_y = pillar_h + capstone_h / 2.0
        capstone_mat = _param_mat(0.50, elasticity=0.50, density=1.0)
        objects.append(_box("capstone", capstone_w, capstone_h, 0.0, capstone_y,
                            capstone_mat))

        return objects, False  # unstable: capstone falls through the gap

    # ------------------------------------------------------------------
    # Sub-type 5: symmetric stack on a hidden steep slope
    # ------------------------------------------------------------------
    def _sub5_hidden_top_slope(self, rng: np.random.RandomState) -> tuple[list[dict], bool]:
        # Slope steep enough that friction < tan(angle) → blocks slide
        slope_angle = round(rng.uniform(22.0, 28.0), 1)
        friction = round(rng.uniform(0.30, 0.38), 2)  # tan(22°)=0.404 > 0.38
        angle_rad = math.radians(slope_angle)

        # Flat catch ground first
        objects: list[dict] = [_ground(material=_param_mat(friction, 0.35, 1.5))]

        # Slope segment
        platform_len = round(rng.uniform(2.5, 3.5), 2)
        slope_cx = 0.0
        slope_y_lo = 0.50
        px_start = slope_cx - platform_len / 2.0 * math.cos(angle_rad)
        py_start = slope_y_lo
        px_end = slope_cx + platform_len / 2.0 * math.cos(angle_rad)
        py_end = slope_y_lo + platform_len * math.sin(angle_rad)

        slope_mat = _param_mat(friction, elasticity=0.30, density=1.5)
        objects.append({
            "id": "slope",
            "type": "static_segment",
            "start": [round(px_start, 3), round(py_start, 3)],
            "end":   [round(px_end,   3), round(py_end,   3)],
            "material": slope_mat,
        })

        # Symmetric-looking stack of 2-4 blocks sitting on the slope
        n_blocks = rng.randint(2, 5)
        block_w = round(rng.uniform(0.25, 0.40), 3)
        block_h = round(rng.uniform(0.18, 0.30), 3)
        block_mat = _param_mat(friction, elasticity=0.50, density=1.0)

        # Stack centre at the midpoint of the slope, offset perpendicular to surface
        mid_frac = 0.5
        cx = px_start + mid_frac * (px_end - px_start)
        cy = py_start + mid_frac * (py_end - py_start)
        for i in range(n_blocks):
            perp_offset = block_h * (i + 0.5)
            bx = cx - perp_offset * math.sin(angle_rad)
            by = cy + perp_offset * math.cos(angle_rad)
            objects.append(_box(f"block_{i}", block_w, block_h,
                                round(bx, 3), round(by, 3),
                                block_mat, angle=slope_angle))

        return objects, False  # unstable: friction < tan(slope) → slides

    # ------------------------------------------------------------------
    # Main generate method
    # ------------------------------------------------------------------
    def generate(self, difficulty: Difficulty, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        sub_type = rng.randint(0, 6)
        q = QUESTION_HARD if difficulty == "hard" else QUESTION

        dispatch = {
            0: self._sub0_tall_heavy_base,
            1: self._sub1_balanced_lean,
            2: self._sub2_ball_in_groove,
            3: self._sub3_asymmetric_mass,
            4: self._sub4_just_too_narrow_arch,
            5: self._sub5_hidden_top_slope,
        }
        objects, make_stable = dispatch[sub_type](rng)

        return _scenario(self.name, seed, difficulty, objects, make_stable,
                         question=q)


# ── Registry ─────────────────────────────────────────────────────────────────

STABILITY_TEMPLATES: list[StabilityTemplate] = [
    SimpleStack(),
    OffsetStack(),
    Pyramid(),
    Arch(),
    Cantilever(),
    MixedShapes(),
    FrictionDependent(),
    ChainCollapse(),
    AdversarialStability(),
]
