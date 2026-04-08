"""Task 1: Trajectory Prediction scenario templates.

10 templates that generate scenarios where the model must predict
the final [x, y] position of a target object after simulation.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(rng: np.random.RandomState, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _randint(rng: np.random.RandomState, lo: int, hi: int) -> int:
    return int(rng.randint(lo, hi + 1))


def _choice(rng: np.random.RandomState, seq):
    return seq[rng.randint(0, len(seq))]


def _make_world(width: float, height: float, damping: float = 0.99) -> dict:
    return {
        "gravity": [0, -9.81],
        "bounds": {"width": width, "height": height},
        "damping": damping,
    }


def _floor(width: float, friction: float = 0.4, elasticity: float = 0.3,
           y: float = 0.0, label: str = "floor") -> dict:
    return {
        "id": label,
        "type": "static_segment",
        "start": [-0.5, y],
        "end": [width + 0.5, y],
        "radius": 0.1,
        "material": {"friction": friction, "elasticity": elasticity, "density": 1.0},
    }


def _wall(x: float, height: float, friction: float = 0.3,
          elasticity: float = 0.5, label: str = "wall") -> dict:
    return {
        "id": label,
        "type": "static_segment",
        "start": [x, -0.5],
        "end": [x, height + 0.5],
        "radius": 0.1,
        "material": {"friction": friction, "elasticity": elasticity, "density": 1.0},
    }


def _ceiling(width: float, height: float, friction: float = 0.3,
             elasticity: float = 0.5) -> dict:
    return {
        "id": "ceiling",
        "type": "static_segment",
        "start": [-0.5, height],
        "end": [width + 0.5, height],
        "radius": 0.1,
        "material": {"friction": friction, "elasticity": elasticity, "density": 1.0},
    }


def _segment(start: list, end: list, friction: float, elasticity: float,
             label: str) -> dict:
    return {
        "id": label,
        "type": "static_segment",
        "start": start,
        "end": end,
        "radius": 0.05,
        "material": {"friction": friction, "elasticity": elasticity, "density": 1.0},
    }


def _circle(cid: str, pos: list, radius: float, material: str | dict,
            velocity: list | None = None) -> dict:
    obj = {
        "id": cid,
        "type": "circle",
        "position": pos,
        "radius": radius,
        "material": material,
    }
    if velocity:
        obj["velocity"] = velocity
    return obj


def _box(bid: str, pos: list, w: float, h: float, material: str | dict,
         velocity: list | None = None, angle: float = 0) -> dict:
    obj = {
        "id": bid,
        "type": "box",
        "position": pos,
        "width": w,
        "height": h,
        "material": material,
        "angle": angle,
    }
    if velocity:
        obj["velocity"] = velocity
    return obj


def _scenario(sid: str, difficulty: str, world: dict, objects: list,
              sim_time: float, target: str, question: str | None = None) -> dict:
    if question is None:
        question = (
            f"After {sim_time:.1f} seconds of simulation, what are the "
            f"approximate [x, y] coordinates of '{target}'?"
        )
    return {
        "id": sid,
        "task_type": "trajectory",
        "difficulty": difficulty,
        "world": world,
        "objects": objects,
        "simulation_time": sim_time,
        "question": question,
        "answer_format": "[x, y]",
        "target_object": target,
    }


# ---------------------------------------------------------------------------
# 1. RampLaunch
# ---------------------------------------------------------------------------

class RampLaunch:
    """Ball rolls down a ramp and becomes a projectile. Predict landing."""

    name = "ramp_launch"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 10, 14)
        H = _uniform(rng, 8, 12)

        if difficulty == "easy":
            ramp_angle = _uniform(rng, 20, 30)
            ramp_len = _uniform(rng, 2.0, 3.5)
            ball_r = _uniform(rng, 0.15, 0.25)
            ramp_fric = _uniform(rng, 0.15, 0.3)
            ramp_elast = _uniform(rng, 0.2, 0.4)
            sim_time = 4.0
        elif difficulty == "medium":
            ramp_angle = _uniform(rng, 25, 45)
            ramp_len = _uniform(rng, 2.5, 5.0)
            ball_r = _uniform(rng, 0.1, 0.3)
            ramp_fric = _uniform(rng, 0.05, 0.4)
            ramp_elast = _uniform(rng, 0.2, 0.6)
            sim_time = 5.0
        else:
            ramp_angle = _uniform(rng, 30, 55)
            ramp_len = _uniform(rng, 3.0, 6.0)
            ball_r = _uniform(rng, 0.08, 0.25)
            ramp_fric = _uniform(rng, 0.02, 0.5)
            ramp_elast = _uniform(rng, 0.1, 0.8)
            sim_time = 6.0

        rad = math.radians(ramp_angle)

        # Ramp goes from upper-left (top) to lower-right (base) so ball
        # rolls rightward and launches off the end as a projectile.
        ramp_top_x = _uniform(rng, 1.0, 2.5)
        ramp_top_y = ramp_len * math.sin(rad) + _uniform(rng, 0.5, 1.5)
        ramp_base_x = ramp_top_x + ramp_len * math.cos(rad)
        ramp_base_y = ramp_top_y - ramp_len * math.sin(rad)

        # Ball starts near the top of the ramp
        bx = ramp_top_x + 0.3 * math.cos(rad)
        by = ramp_top_y - 0.3 * math.sin(rad) + ball_r

        mat_name = _choice(rng, ["rubber", "steel", "wood"])

        objects = [
            _floor(W, friction=0.4, elasticity=ramp_elast),
            _wall(0, H, friction=0.3, elasticity=ramp_elast, label="wall_left"),
            _wall(W, H, friction=0.3, elasticity=ramp_elast, label="wall_right"),
            _segment([ramp_top_x, ramp_top_y], [ramp_base_x, ramp_base_y],
                     friction=ramp_fric, elasticity=ramp_elast, label="ramp"),
            _circle("ball", [bx, by], ball_r, mat_name),
        ]

        # Hard: add a ledge the ball may land on
        if difficulty == "hard":
            ledge_x = _uniform(rng, ramp_base_x + 1.0, min(ramp_base_x + 4.0, W - 2.0))
            ledge_y = _uniform(rng, 0.5, 2.5)
            ledge_len = _uniform(rng, 1.5, 3.0)
            objects.append(
                _segment([ledge_x, ledge_y],
                         [ledge_x + ledge_len, ledge_y],
                         friction=0.4, elasticity=ramp_elast, label="ledge")
            )

        return _scenario(
            f"traj_ramp_launch_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 2. TableRoll
# ---------------------------------------------------------------------------

class TableRoll:
    """Object rolls off table edge, predict where it lands."""

    name = "table_roll"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 9, 13)
        H = _uniform(rng, 7, 11)

        if difficulty == "easy":
            table_h = _uniform(rng, 2.0, 3.5)
            table_w = _uniform(rng, 3.0, 5.0)
            ball_r = _uniform(rng, 0.15, 0.25)
            init_vx = _uniform(rng, 0.5, 2.0)
            sim_time = 4.0
        elif difficulty == "medium":
            table_h = _uniform(rng, 2.5, 5.0)
            table_w = _uniform(rng, 2.5, 5.0)
            ball_r = _uniform(rng, 0.1, 0.25)
            init_vx = _uniform(rng, 1.0, 4.0)
            sim_time = 5.0
        else:
            table_h = _uniform(rng, 3.0, 6.0)
            table_w = _uniform(rng, 2.0, 5.0)
            ball_r = _uniform(rng, 0.08, 0.2)
            init_vx = _uniform(rng, 1.5, 5.0)
            sim_time = 6.0

        table_x = _uniform(rng, 0.5, 2.0)
        mat_name = _choice(rng, ["rubber", "wood", "steel"])
        table_fric = _uniform(rng, 0.2, 0.6)

        # Table top (horizontal surface)
        objects = [
            _floor(W),
            _wall(0, H, label="wall_left"),
            _wall(W, H, label="wall_right"),
            _segment([table_x, table_h],
                     [table_x + table_w, table_h],
                     friction=table_fric, elasticity=0.2, label="table_top"),
            # Table left leg
            _segment([table_x, 0], [table_x, table_h],
                     friction=0.3, elasticity=0.2, label="table_leg_l"),
            # Table right leg
            _segment([table_x + table_w, 0], [table_x + table_w, table_h],
                     friction=0.3, elasticity=0.2, label="table_leg_r"),
            _circle("ball", [table_x + 0.5, table_h + ball_r + 0.05],
                     ball_r, mat_name, velocity=[init_vx, 0]),
        ]

        # Medium/hard: add lower shelf the ball might hit
        if difficulty in ("medium", "hard"):
            shelf_y = _uniform(rng, 0.5, table_h * 0.6)
            shelf_x = table_x + table_w + _uniform(rng, 0.5, 2.0)
            shelf_len = _uniform(rng, 1.0, 2.5)
            objects.append(
                _segment([shelf_x, shelf_y], [shelf_x + shelf_len, shelf_y],
                         friction=0.4, elasticity=0.3, label="shelf")
            )

        # Hard: add a second ball on the shelf that gets knocked
        if difficulty == "hard":
            ball2_r = _uniform(rng, 0.1, 0.2)
            objects.append(
                _circle("ball2", [shelf_x + shelf_len * 0.5, shelf_y + ball2_r + 0.05],
                         ball2_r, _choice(rng, ["rubber", "wood"]))
            )

        return _scenario(
            f"traj_table_roll_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 3. BouncePath
# ---------------------------------------------------------------------------

class BouncePath:
    """Ball launched with initial velocity, bounces off walls / floor."""

    name = "bounce_path"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 8, 12)
        H = _uniform(rng, 7, 10)

        ball_r = _uniform(rng, 0.12, 0.22)
        elast = _uniform(rng, 0.6, 0.95)  # bouncy

        if difficulty == "easy":
            vx = _uniform(rng, 1.0, 3.0)
            vy = _uniform(rng, 2.0, 5.0)
            start_x = _uniform(rng, 1.0, 3.0)
            start_y = _uniform(rng, 1.0, 2.5)
            sim_time = 4.0
        elif difficulty == "medium":
            vx = _uniform(rng, 2.0, 5.0)
            vy = _uniform(rng, 3.0, 7.0)
            start_x = _uniform(rng, 0.5, 2.5)
            start_y = _uniform(rng, 0.5, 2.0)
            sim_time = 5.0
        else:
            vx = _uniform(rng, 3.0, 7.0)
            vy = _uniform(rng, 4.0, 9.0)
            start_x = _uniform(rng, 0.5, 2.0)
            start_y = _uniform(rng, 0.5, 1.5)
            sim_time = 7.0

        wall_elast = elast
        wall_fric = _uniform(rng, 0.05, 0.2)

        objects = [
            _floor(W, friction=wall_fric, elasticity=wall_elast),
            _wall(0, H, friction=wall_fric, elasticity=wall_elast, label="wall_left"),
            _wall(W, H, friction=wall_fric, elasticity=wall_elast, label="wall_right"),
            _circle("ball", [start_x, start_y], ball_r,
                    {"friction": 0.1, "elasticity": elast, "density": 1.0},
                    velocity=[vx, vy]),
        ]

        # Medium: add an angled deflector
        if difficulty == "medium":
            dx = _uniform(rng, W * 0.3, W * 0.6)
            dy = _uniform(rng, H * 0.3, H * 0.6)
            dlen = _uniform(rng, 1.0, 2.0)
            dang = _uniform(rng, -40, 40)
            rad = math.radians(dang)
            objects.append(
                _segment([dx, dy],
                         [dx + dlen * math.cos(rad), dy + dlen * math.sin(rad)],
                         friction=wall_fric, elasticity=wall_elast,
                         label="deflector")
            )

        # Hard: add ceiling + two deflectors
        if difficulty == "hard":
            objects.append(_ceiling(W, H, friction=wall_fric, elasticity=wall_elast))
            for i in range(2):
                dx = _uniform(rng, W * 0.2, W * 0.8)
                dy = _uniform(rng, H * 0.2, H * 0.7)
                dlen = _uniform(rng, 1.0, 2.5)
                dang = _uniform(rng, -50, 50)
                rad = math.radians(dang)
                objects.append(
                    _segment([dx, dy],
                             [dx + dlen * math.cos(rad),
                              dy + dlen * math.sin(rad)],
                             friction=wall_fric, elasticity=wall_elast,
                             label=f"deflector_{i}")
                )

        return _scenario(
            f"traj_bounce_path_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 4. PendulumRelease
# ---------------------------------------------------------------------------

class PendulumRelease:
    """Simplified pendulum: ball launched at angle from height,
    simulating the release arc. Predict landing position."""

    name = "pendulum_release"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 10, 14)
        H = _uniform(rng, 8, 12)

        if difficulty == "easy":
            launch_h = _uniform(rng, 3.0, 5.0)
            launch_speed = _uniform(rng, 2.0, 4.0)
            launch_angle = _uniform(rng, 20, 50)
            ball_r = _uniform(rng, 0.15, 0.25)
            sim_time = 4.0
        elif difficulty == "medium":
            launch_h = _uniform(rng, 4.0, 7.0)
            launch_speed = _uniform(rng, 3.0, 6.0)
            launch_angle = _uniform(rng, 15, 65)
            ball_r = _uniform(rng, 0.1, 0.25)
            sim_time = 5.0
        else:
            launch_h = _uniform(rng, 5.0, 9.0)
            launch_speed = _uniform(rng, 4.0, 8.0)
            launch_angle = _uniform(rng, 10, 75)
            ball_r = _uniform(rng, 0.08, 0.2)
            sim_time = 6.0

        launch_x = _uniform(rng, 2.0, 4.0)
        rad = math.radians(launch_angle)
        vx = launch_speed * math.cos(rad)
        vy = launch_speed * math.sin(rad)
        mat_name = _choice(rng, ["rubber", "steel", "wood"])

        # Anchor post (visual context)
        post_x = launch_x
        post_top = launch_h + 1.5

        objects = [
            _floor(W),
            _wall(0, H, label="wall_left"),
            _wall(W, H, label="wall_right"),
            _segment([post_x, 0], [post_x, post_top],
                     friction=0.3, elasticity=0.2, label="post"),
            _circle("ball", [launch_x, launch_h], ball_r, mat_name,
                    velocity=[vx, vy]),
        ]

        # Medium: add a platform to land on
        if difficulty in ("medium", "hard"):
            plat_x = launch_x + _uniform(rng, 3.0, 6.0)
            plat_y = _uniform(rng, 1.0, 3.0)
            plat_len = _uniform(rng, 1.5, 3.0)
            objects.append(
                _segment([plat_x, plat_y], [plat_x + plat_len, plat_y],
                         friction=0.4, elasticity=0.3, label="platform")
            )

        # Hard: add a second bounce wall
        if difficulty == "hard":
            wall_x = launch_x + _uniform(rng, 5.0, W - 2.0)
            objects.append(
                _wall(wall_x, H, friction=0.2, elasticity=0.6, label="bounce_wall")
            )

        return _scenario(
            f"traj_pendulum_release_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 5. ConveyorDrop
# ---------------------------------------------------------------------------

class ConveyorDrop:
    """Object moving horizontally (initial velocity) on a platform
    reaches the edge and falls. Predict landing spot."""

    name = "conveyor_drop"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 10, 14)
        H = _uniform(rng, 8, 11)

        if difficulty == "easy":
            plat_h = _uniform(rng, 3.0, 5.0)
            plat_w = _uniform(rng, 3.0, 5.0)
            obj_vx = _uniform(rng, 1.0, 3.0)
            ball_r = _uniform(rng, 0.15, 0.25)
            plat_fric = _uniform(rng, 0.05, 0.15)
            sim_time = 4.0
        elif difficulty == "medium":
            plat_h = _uniform(rng, 4.0, 6.0)
            plat_w = _uniform(rng, 2.5, 5.0)
            obj_vx = _uniform(rng, 2.0, 5.0)
            ball_r = _uniform(rng, 0.1, 0.22)
            plat_fric = _uniform(rng, 0.02, 0.1)
            sim_time = 5.0
        else:
            plat_h = _uniform(rng, 5.0, 8.0)
            plat_w = _uniform(rng, 2.0, 4.5)
            obj_vx = _uniform(rng, 3.0, 7.0)
            ball_r = _uniform(rng, 0.08, 0.18)
            plat_fric = _uniform(rng, 0.01, 0.08)
            sim_time = 6.0

        plat_x = _uniform(rng, 1.0, 3.0)
        mat_name = _choice(rng, ["steel", "wood", "rubber"])

        objects = [
            _floor(W),
            _wall(0, H, label="wall_left"),
            _wall(W, H, label="wall_right"),
            _segment([plat_x, plat_h], [plat_x + plat_w, plat_h],
                     friction=plat_fric, elasticity=0.2, label="conveyor"),
            # Lip wall at start to prevent backward roll
            _segment([plat_x, plat_h], [plat_x, plat_h + 0.3],
                     friction=0.3, elasticity=0.2, label="lip"),
            _circle("ball", [plat_x + 0.5, plat_h + ball_r + 0.05],
                     ball_r, mat_name, velocity=[obj_vx, 0]),
        ]

        # Medium: add a sloped landing ramp
        if difficulty in ("medium", "hard"):
            ramp_x = plat_x + plat_w + _uniform(rng, 0.5, 2.0)
            ramp_top_y = _uniform(rng, 1.0, plat_h * 0.6)
            ramp_len = _uniform(rng, 2.0, 4.0)
            objects.append(
                _segment([ramp_x, ramp_top_y],
                         [ramp_x + ramp_len, 0],
                         friction=0.3, elasticity=0.3, label="landing_ramp")
            )

        # Hard: add a second conveyor at a lower level
        if difficulty == "hard":
            c2_h = _uniform(rng, 1.0, 2.5)
            c2_x = plat_x + plat_w + _uniform(rng, 2.0, 4.0)
            c2_w = _uniform(rng, 2.0, 3.5)
            objects.append(
                _segment([c2_x, c2_h], [c2_x + c2_w, c2_h],
                         friction=plat_fric, elasticity=0.2, label="conveyor_2")
            )

        return _scenario(
            f"traj_conveyor_drop_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 6. MultiBounce
# ---------------------------------------------------------------------------

class MultiBounce:
    """Ball in a partially enclosed space bouncing multiple times.
    Walls on left/right/bottom; predict final resting position."""

    name = "multi_bounce"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 8, 11)
        H = _uniform(rng, 7, 10)

        ball_r = _uniform(rng, 0.12, 0.22)
        elast = _uniform(rng, 0.7, 0.95)

        if difficulty == "easy":
            vx = _uniform(rng, 2.0, 4.0)
            vy = _uniform(rng, 3.0, 5.0)
            sim_time = 5.0
        elif difficulty == "medium":
            vx = _uniform(rng, 3.0, 6.0)
            vy = _uniform(rng, 4.0, 7.0)
            sim_time = 7.0
        else:
            vx = _uniform(rng, 4.0, 8.0)
            vy = _uniform(rng, 5.0, 9.0)
            sim_time = 10.0

        # Randomize start corner
        sx = _uniform(rng, ball_r + 0.2, W * 0.3)
        sy = _uniform(rng, ball_r + 0.2, 2.0)

        w_fric = _uniform(rng, 0.02, 0.1)

        objects = [
            _floor(W, friction=w_fric, elasticity=elast),
            _wall(0, H, friction=w_fric, elasticity=elast, label="wall_left"),
            _wall(W, H, friction=w_fric, elasticity=elast, label="wall_right"),
            _ceiling(W, H, friction=w_fric, elasticity=elast),
            _circle("ball", [sx, sy], ball_r,
                    {"friction": 0.05, "elasticity": elast, "density": 1.0},
                    velocity=[vx, vy]),
        ]

        # Medium: add one interior obstacle
        if difficulty == "medium":
            ox = _uniform(rng, W * 0.3, W * 0.7)
            oy = _uniform(rng, H * 0.3, H * 0.7)
            olen = _uniform(rng, 1.0, 2.0)
            oang = _uniform(rng, -45, 45)
            rad = math.radians(oang)
            objects.append(
                _segment([ox, oy],
                         [ox + olen * math.cos(rad), oy + olen * math.sin(rad)],
                         friction=w_fric, elasticity=elast, label="obstacle_0")
            )

        # Hard: add 3 interior obstacles
        if difficulty == "hard":
            for i in range(3):
                ox = _uniform(rng, W * 0.15, W * 0.85)
                oy = _uniform(rng, H * 0.15, H * 0.85)
                olen = _uniform(rng, 0.8, 2.2)
                oang = _uniform(rng, -60, 60)
                rad = math.radians(oang)
                objects.append(
                    _segment([ox, oy],
                             [ox + olen * math.cos(rad),
                              oy + olen * math.sin(rad)],
                             friction=w_fric, elasticity=elast,
                             label=f"obstacle_{i}")
                )

        return _scenario(
            f"traj_multi_bounce_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 7. CannonArc
# ---------------------------------------------------------------------------

class CannonArc:
    """Classic projectile motion: object launched at an angle from a
    cannon platform. Predict landing coordinates."""

    name = "cannon_arc"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 12, 15)
        H = _uniform(rng, 8, 12)

        if difficulty == "easy":
            cannon_h = _uniform(rng, 0.5, 2.0)
            launch_angle = _uniform(rng, 30, 60)
            launch_speed = _uniform(rng, 4.0, 7.0)
            ball_r = _uniform(rng, 0.15, 0.25)
            sim_time = 4.0
        elif difficulty == "medium":
            cannon_h = _uniform(rng, 1.0, 4.0)
            launch_angle = _uniform(rng, 15, 70)
            launch_speed = _uniform(rng, 5.0, 10.0)
            ball_r = _uniform(rng, 0.1, 0.22)
            sim_time = 5.0
        else:
            cannon_h = _uniform(rng, 2.0, 6.0)
            launch_angle = _uniform(rng, 10, 80)
            launch_speed = _uniform(rng, 6.0, 12.0)
            ball_r = _uniform(rng, 0.08, 0.18)
            sim_time = 6.0

        cannon_x = _uniform(rng, 1.0, 3.0)
        rad = math.radians(launch_angle)
        vx = launch_speed * math.cos(rad)
        vy = launch_speed * math.sin(rad)
        mat_name = _choice(rng, ["steel", "wood", "rubber"])

        objects = [
            _floor(W),
            _wall(0, H, label="wall_left"),
            _wall(W, H, label="wall_right"),
            # Cannon platform
            _segment([cannon_x - 0.5, cannon_h],
                     [cannon_x + 0.5, cannon_h],
                     friction=0.4, elasticity=0.3, label="cannon_base"),
            _circle("ball", [cannon_x, cannon_h + ball_r + 0.05],
                     ball_r, mat_name, velocity=[vx, vy]),
        ]

        # Medium: add a wall to bounce off
        if difficulty == "medium":
            wall_x = _uniform(rng, W * 0.5, W * 0.75)
            wall_top = _uniform(rng, 2.0, 5.0)
            objects.append(
                _segment([wall_x, 0], [wall_x, wall_top],
                         friction=0.2, elasticity=0.5, label="mid_wall")
            )

        # Hard: add wind (slight horizontal bias via tilted gravity),
        # plus an elevated landing zone and a wall
        if difficulty == "hard":
            wind_x = _uniform(rng, -1.0, 1.0)
            # Encode as a slight gravity tilt
            objects[0]  # floor already exists; modify world later
            wall_x = _uniform(rng, W * 0.4, W * 0.65)
            wall_top = _uniform(rng, 3.0, 7.0)
            objects.append(
                _segment([wall_x, 0], [wall_x, wall_top],
                         friction=0.2, elasticity=0.5, label="mid_wall")
            )
            # Elevated landing platform
            lp_x = _uniform(rng, wall_x + 1.0, W - 2.0)
            lp_y = _uniform(rng, 0.5, 2.5)
            lp_len = _uniform(rng, 2.0, 4.0)
            objects.append(
                _segment([lp_x, lp_y], [lp_x + lp_len, lp_y],
                         friction=0.4, elasticity=0.3, label="landing_platform")
            )
            world = _make_world(W, H)
            world["gravity"] = [wind_x, -9.81]
            return _scenario(
                f"traj_cannon_arc_{seed}", difficulty,
                world, objects, sim_time, "ball",
            )

        return _scenario(
            f"traj_cannon_arc_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 8. FrictionSlide
# ---------------------------------------------------------------------------

class FrictionSlide:
    """Object slides across surfaces with different friction coefficients.
    Predict where it stops."""

    name = "friction_slide"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 10, 15)
        H = _uniform(rng, 6, 9)

        if difficulty == "easy":
            n_zones = 2
            init_vx = _uniform(rng, 3.0, 6.0)
            ball_r = _uniform(rng, 0.15, 0.25)
            sim_time = 6.0
        elif difficulty == "medium":
            n_zones = 3
            init_vx = _uniform(rng, 4.0, 8.0)
            ball_r = _uniform(rng, 0.12, 0.22)
            sim_time = 8.0
        else:
            n_zones = 5
            init_vx = _uniform(rng, 5.0, 10.0)
            ball_r = _uniform(rng, 0.08, 0.18)
            sim_time = 10.0

        mat_name = _choice(rng, ["steel", "wood", "rubber"])

        # Divide floor into friction zones
        zone_width = W / n_zones
        objects = [
            _wall(0, H, label="wall_left"),
            _wall(W, H, label="wall_right"),
        ]
        friction_materials = ["ice", "wood", "rubber", "steel", "sponge"]
        for i in range(n_zones):
            fric_mat = _choice(rng, friction_materials)
            from physiq.materials import MATERIALS
            fric_val = MATERIALS[fric_mat]["friction"]
            elast_val = MATERIALS[fric_mat]["elasticity"]
            x0 = i * zone_width
            x1 = (i + 1) * zone_width
            objects.append(
                _segment([x0, 0], [x1, 0],
                         friction=fric_val, elasticity=elast_val,
                         label=f"zone_{fric_mat}_{i}")
            )

        objects.append(
            _circle("ball", [0.5, ball_r + 0.05], ball_r, mat_name,
                    velocity=[init_vx, 0])
        )

        # Hard: add a slight downhill slope in one zone and an uphill in another
        if difficulty == "hard":
            slope_zone = _randint(rng, 1, n_zones - 2)
            x0 = slope_zone * zone_width
            x1 = (slope_zone + 1) * zone_width
            drop = _uniform(rng, 0.2, 0.6)
            objects.append(
                _segment([x0, drop], [x1, 0],
                         friction=0.1, elasticity=0.2, label="downslope")
            )
            up_zone = min(slope_zone + 2, n_zones - 1)
            ux0 = up_zone * zone_width
            ux1 = (up_zone + 1) * zone_width
            rise = _uniform(rng, 0.15, 0.4)
            objects.append(
                _segment([ux0, 0], [ux1, rise],
                         friction=0.3, elasticity=0.2, label="upslope")
            )

        return _scenario(
            f"traj_friction_slide_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 9. DominoMomentum
# ---------------------------------------------------------------------------

class DominoMomentum:
    """A ball strikes a line of domino-like boxes, transferring momentum.
    Predict where the last domino (or the ball) ends up."""

    name = "domino_momentum"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 12, 15)
        H = _uniform(rng, 7, 10)

        if difficulty == "easy":
            n_dominoes = 3
            ball_vx = _uniform(rng, 3.0, 5.0)
            ball_r = _uniform(rng, 0.2, 0.3)
            dom_w = _uniform(rng, 0.15, 0.25)
            dom_h = _uniform(rng, 0.6, 0.9)
            spacing = _uniform(rng, 0.5, 0.8)
            sim_time = 5.0
        elif difficulty == "medium":
            n_dominoes = 5
            ball_vx = _uniform(rng, 4.0, 7.0)
            ball_r = _uniform(rng, 0.15, 0.25)
            dom_w = _uniform(rng, 0.1, 0.2)
            dom_h = _uniform(rng, 0.5, 1.0)
            spacing = _uniform(rng, 0.4, 0.7)
            sim_time = 6.0
        else:
            n_dominoes = 8
            ball_vx = _uniform(rng, 5.0, 9.0)
            ball_r = _uniform(rng, 0.1, 0.2)
            dom_w = _uniform(rng, 0.08, 0.18)
            dom_h = _uniform(rng, 0.4, 1.2)
            spacing = _uniform(rng, 0.3, 0.6)
            sim_time = 8.0

        start_x = _uniform(rng, 1.0, 2.5)
        dom_mat = _choice(rng, ["wood", "steel"])

        objects = [
            _floor(W),
            _wall(0, H, label="wall_left"),
            _wall(W, H, label="wall_right"),
        ]

        # Ball
        objects.append(
            _circle("ball", [start_x, ball_r + 0.05], ball_r,
                    _choice(rng, ["steel", "rubber"]),
                    velocity=[ball_vx, 0])
        )

        # Dominoes (tall thin boxes standing upright)
        first_dom_x = start_x + _uniform(rng, 1.0, 2.0)
        target_id = f"domino_{n_dominoes - 1}"
        for i in range(n_dominoes):
            dx = first_dom_x + i * (dom_w + spacing)
            objects.append(
                _box(f"domino_{i}", [dx, dom_h / 2 + 0.05],
                     dom_w, dom_h, dom_mat)
            )

        # Hard: add varying domino sizes (increasing mass cascade)
        if difficulty == "hard":
            # Make dominoes progressively larger.
            # Dominoes are the last n_dominoes entries in the objects list.
            domino_start_idx = len(objects) - n_dominoes
            for i in range(n_dominoes):
                scale = 1.0 + 0.15 * i
                obj = objects[domino_start_idx + i]
                obj["width"] = dom_w * scale
                obj["height"] = dom_h * scale
                obj["position"] = [obj["position"][0],
                                   (dom_h * scale) / 2 + 0.05]

        return _scenario(
            f"traj_domino_momentum_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, target_id,
            question=(
                f"After {sim_time:.1f} seconds of simulation, what are the "
                f"approximate [x, y] coordinates of '{target_id}'?"
            ),
        )


# ---------------------------------------------------------------------------
# 10. PinballCourse
# ---------------------------------------------------------------------------

class PinballCourse:
    """Complex course with bumpers (circles), slopes, and walls.
    Ball rolls through; predict exit position."""

    name = "pinball_course"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 10, 14)
        H = _uniform(rng, 9, 12)

        ball_r = _uniform(rng, 0.12, 0.2)
        elast = _uniform(rng, 0.65, 0.9)
        ball_mat = {"friction": 0.15, "elasticity": elast, "density": 1.2}

        if difficulty == "easy":
            n_bumpers = 2
            n_slopes = 1
            sim_time = 5.0
            init_vy = _uniform(rng, -1.0, -3.0)
        elif difficulty == "medium":
            n_bumpers = 4
            n_slopes = 2
            sim_time = 7.0
            init_vy = _uniform(rng, -2.0, -4.0)
        else:
            n_bumpers = 7
            n_slopes = 3
            sim_time = 10.0
            init_vy = _uniform(rng, -3.0, -5.0)

        # Ball drops from near the top
        start_x = _uniform(rng, W * 0.3, W * 0.7)
        start_y = H - _uniform(rng, 0.5, 1.5)
        init_vx = _uniform(rng, -1.5, 1.5)

        objects = [
            _floor(W, friction=0.3, elasticity=0.3),
            _wall(0, H, friction=0.2, elasticity=elast, label="wall_left"),
            _wall(W, H, friction=0.2, elasticity=elast, label="wall_right"),
            _circle("ball", [start_x, start_y], ball_r, ball_mat,
                    velocity=[init_vx, init_vy]),
        ]

        # Place bumpers (static circles — implemented as small static_polygons
        # approximated by short segments forming a cross)
        # Actually, use static segments arranged as short horizontal platforms
        # that act as bumpers. Each bumper is a short angled segment.
        for i in range(n_bumpers):
            bx = _uniform(rng, W * 0.1, W * 0.9)
            by = _uniform(rng, H * 0.15, H * 0.8)
            blen = _uniform(rng, 0.5, 1.2)
            bang = _uniform(rng, -30, 30)
            brad = math.radians(bang)
            objects.append(
                _segment([bx, by],
                         [bx + blen * math.cos(brad),
                          by + blen * math.sin(brad)],
                         friction=0.1, elasticity=elast,
                         label=f"bumper_{i}")
            )

        # Place slopes (longer angled segments)
        for i in range(n_slopes):
            sx = _uniform(rng, W * 0.05, W * 0.6)
            sy = _uniform(rng, H * 0.1, H * 0.65)
            slen = _uniform(rng, 1.5, 3.5)
            sang = _uniform(rng, -25, -5)  # slightly downward to the right
            srad = math.radians(sang)
            objects.append(
                _segment([sx, sy],
                         [sx + slen * math.cos(srad),
                          sy + slen * math.sin(srad)],
                         friction=0.2, elasticity=0.4,
                         label=f"slope_{i}")
            )

        # Hard: add a funnel at the bottom to constrain exit
        if difficulty == "hard":
            funnel_cx = W / 2
            funnel_gap = _uniform(rng, 0.8, 1.5)
            funnel_h = _uniform(rng, 1.5, 3.0)
            objects.append(
                _segment([funnel_cx - 3.0, funnel_h],
                         [funnel_cx - funnel_gap / 2, 0.3],
                         friction=0.2, elasticity=0.4, label="funnel_left")
            )
            objects.append(
                _segment([funnel_cx + 3.0, funnel_h],
                         [funnel_cx + funnel_gap / 2, 0.3],
                         friction=0.2, elasticity=0.4, label="funnel_right")
            )

        return _scenario(
            f"traj_pinball_course_{seed}", difficulty,
            _make_world(W, H), objects, sim_time, "ball",
        )


# ---------------------------------------------------------------------------
# 11. ReverseTrajectory
# ---------------------------------------------------------------------------

class ReverseTrajectory:
    """Ball launched rightward bounces off the right wall and ends up LEFT
    of its starting x-position. Tests whether models assume 'ball goes right'
    without simulating the bounce-back."""

    name = "reverse_trajectory"
    difficulties = ["medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        # Deliberately narrow world so ball reaches right wall quickly.
        W = _uniform(rng, 6, 8)
        H = _uniform(rng, 5, 8)

        if difficulty == "medium":
            ball_x = _uniform(rng, 0.8, 1.5)
            ball_y = _uniform(rng, 0.3, 0.8)
            vx = _uniform(rng, 4, 6)
            wall_elast = _uniform(rng, 0.7, 0.85)
            floor_fric = _uniform(rng, 0.6, 0.75)
            sim_time = _uniform(rng, 3.0, 3.5)
            objects = [
                _floor(W, friction=floor_fric, elasticity=0.1),
                _wall(W, H, friction=0.1, elasticity=wall_elast, label="right_wall"),
                _circle("ball", [ball_x, ball_y + 0.2], 0.2,
                        {"friction": 0.5, "elasticity": 0.4, "density": 1.0},
                        velocity=[vx, 0.0]),
            ]
        else:  # hard — add an angled deflector after the right-wall bounce
            ball_x = _uniform(rng, 0.8, 1.5)
            ball_y = _uniform(rng, 0.3, 0.8)
            vx = _uniform(rng, 6, 8)
            wall_elast = _uniform(rng, 0.8, 0.9)
            floor_fric = _uniform(rng, 0.6, 0.8)
            sim_time = _uniform(rng, 3.5, 4.0)
            # Angled deflector on the left side redirects the returning ball further left.
            defl_x = _uniform(rng, 1.5, 2.5)
            defl_y = _uniform(rng, 0.6, 1.4)
            defl_len = _uniform(rng, 0.8, 1.4)
            defl_angle = _uniform(rng, 30, 50)
            rad = math.radians(defl_angle)
            objects = [
                _floor(W, friction=floor_fric, elasticity=0.1),
                _wall(W, H, friction=0.1, elasticity=wall_elast, label="right_wall"),
                _segment(
                    [defl_x, defl_y],
                    [defl_x + defl_len * math.cos(rad),
                     defl_y + defl_len * math.sin(rad)],
                    friction=0.2, elasticity=0.6, label="deflector",
                ),
                _circle("ball", [ball_x, ball_y + 0.2], 0.2,
                        {"friction": 0.5, "elasticity": 0.4, "density": 1.0},
                        velocity=[vx, 0.0]),
            ]

        question = (
            f"After {sim_time:.1f} seconds, what are the approximate [x, y] "
            f"coordinates of 'ball'? Note: the ball may bounce off walls and "
            f"end up in a surprising location."
        )
        world = _make_world(W, H)
        return _scenario(
            f"traj_reverse_trajectory_{seed}", difficulty,
            world, objects, sim_time, "ball", question,
        )


# ---------------------------------------------------------------------------
# 12. FrictionStop
# ---------------------------------------------------------------------------

class FrictionStop:
    """Ball on a steep slope with very high friction decelerates and stops
    BEFORE reaching the bottom. Tests whether models assume objects always
    reach the bottom of ramps."""

    name = "friction_stop"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 8, 12)
        H = _uniform(rng, 6, 10)

        if difficulty == "easy":
            slope_angle = _uniform(rng, 10, 15)
            slope_fric = _uniform(rng, 0.7, 0.9)
            slope_elast = _uniform(rng, 0.1, 0.2)
            init_speed = _uniform(rng, 0.5, 1.5)
            sim_time = 3.0
        elif difficulty == "medium":
            slope_angle = _uniform(rng, 20, 25)
            slope_fric = _uniform(rng, 0.6, 0.8)
            slope_elast = _uniform(rng, 0.1, 0.2)
            init_speed = _uniform(rng, 1.0, 2.0)
            sim_time = 3.5
        else:  # hard
            slope_angle = _uniform(rng, 25, 32)
            slope_fric = _uniform(rng, 0.5, 0.65)
            slope_elast = _uniform(rng, 0.1, 0.15)
            init_speed = _uniform(rng, 1.5, 2.5)
            sim_time = 4.0

        rad = math.radians(slope_angle)
        slope_len = _uniform(rng, 3.5, 5.5)

        # Slope: top-left to bottom-right, anchored so the foot sits above the floor.
        foot_x = _uniform(rng, 1.5, W * 0.45)
        foot_y = _uniform(rng, 0.6, 1.5)
        top_x = foot_x - slope_len * math.cos(rad)
        top_y = foot_y + slope_len * math.sin(rad)

        # Ball starts 85% of the way up the slope (near top).
        t_ball = 0.85
        ball_x = foot_x - t_ball * slope_len * math.cos(rad)
        ball_y = foot_y + t_ball * slope_len * math.sin(rad) + 0.2

        # Initial velocity directed down the slope.
        vel_x = init_speed * math.cos(rad)
        vel_y = -init_speed * math.sin(rad)

        objects = [
            _floor(W, friction=0.3, elasticity=0.1),
            _segment(
                [top_x, top_y], [foot_x, foot_y],
                friction=slope_fric, elasticity=slope_elast,
                label="ramp",
            ),
            _circle("ball", [ball_x, ball_y], 0.18,
                    {"friction": slope_fric, "elasticity": slope_elast, "density": 1.2},
                    velocity=[vel_x, vel_y]),
        ]

        question = (
            f"After {sim_time:.1f} seconds, what are the approximate [x, y] "
            f"coordinates of 'ball'? Note: the ball starts moving but may not "
            f"reach the bottom of the ramp due to friction."
        )
        world = _make_world(W, H)
        return _scenario(
            f"traj_friction_stop_{seed}", difficulty,
            world, objects, sim_time, "ball", question,
        )


# ---------------------------------------------------------------------------
# 13. LateralGravity
# ---------------------------------------------------------------------------

class LateralGravity:
    """Trajectory under non-standard gravity direction (lateral or upward-biased).
    Tests whether models rely on 'gravity = downward' priors."""

    name = "lateral_gravity"
    difficulties = ["easy", "medium", "hard"]

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        W = _uniform(rng, 7, 10)
        H = _uniform(rng, 7, 10)
        ball_r = 0.2

        if difficulty == "easy":
            # Diagonal gravity: mostly leftward with partial downward pull.
            gx, gy = -9.81, -4.0
            # Ball starts near right side, moving leftward.
            ball_x = _uniform(rng, W - 1.5, W - 0.8)
            ball_y = _uniform(rng, H * 0.3, H * 0.6)
            vx = _uniform(rng, -2.0, -0.5)
            vy = _uniform(rng, -1.0, 1.0)
            sim_time = _uniform(rng, 2.5, 3.5)
            objects = [
                _floor(W, friction=0.3, elasticity=0.3),
                _wall(0.0, H, friction=0.4, elasticity=0.3, label="left_wall"),
                _wall(W, H, friction=0.3, elasticity=0.3, label="right_wall"),
                _ceiling(W, H, friction=0.3, elasticity=0.3),
                _circle("ball", [ball_x, ball_y], ball_r,
                        {"friction": 0.4, "elasticity": 0.3, "density": 1.0},
                        velocity=[vx, vy]),
            ]

        elif difficulty == "medium":
            # Inverted gravity: ball falls upward onto the ceiling.
            gx, gy = 0.0, 9.81
            ball_x = _uniform(rng, W * 0.3, W * 0.7)
            ball_y = _uniform(rng, 0.5, H * 0.35)
            vx = _uniform(rng, -1.5, 1.5)
            vy = _uniform(rng, 0.0, 2.0)
            sim_time = _uniform(rng, 2.5, 3.5)
            objects = [
                _floor(W, friction=0.3, elasticity=0.3),
                _wall(0.0, H, friction=0.3, elasticity=0.3, label="left_wall"),
                _wall(W, H, friction=0.3, elasticity=0.3, label="right_wall"),
                _ceiling(W, H, friction=0.5, elasticity=0.3),
                _circle("ball", [ball_x, ball_y], ball_r,
                        {"friction": 0.4, "elasticity": 0.3, "density": 1.0},
                        velocity=[vx, vy]),
            ]

        else:  # hard — 45° diagonal gravity, same magnitude as standard
            gx, gy = -6.94, -6.94
            ball_x = _uniform(rng, W * 0.4, W * 0.8)
            ball_y = _uniform(rng, H * 0.3, H * 0.7)
            angle = _uniform(rng, 0, 360)
            speed = _uniform(rng, 2.0, 5.0)
            vx = speed * math.cos(math.radians(angle))
            vy = speed * math.sin(math.radians(angle))
            sim_time = _uniform(rng, 3.0, 4.5)
            objects = [
                _floor(W, friction=0.4, elasticity=0.3),
                _wall(0.0, H, friction=0.4, elasticity=0.3, label="left_wall"),
                _wall(W, H, friction=0.3, elasticity=0.3, label="right_wall"),
                _ceiling(W, H, friction=0.4, elasticity=0.3),
                _circle("ball", [ball_x, ball_y], ball_r,
                        {"friction": 0.4, "elasticity": 0.3, "density": 1.0},
                        velocity=[vx, vy]),
            ]

        world = _make_world(W, H)
        world["gravity"] = [gx, gy]

        question = (
            f"After {sim_time:.1f} seconds, what are the approximate [x, y] "
            f"coordinates of 'ball'? Note: Gravity in this simulation is "
            f"[{gx:.2f}, {gy:.2f}] m/s²."
        )
        return _scenario(
            f"traj_lateral_gravity_{seed}", difficulty,
            world, objects, sim_time, "ball", question,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TRAJECTORY_TEMPLATES: list = [
    RampLaunch(),
    TableRoll(),
    BouncePath(),
    PendulumRelease(),
    ConveyorDrop(),
    MultiBounce(),
    CannonArc(),
    FrictionSlide(),
    DominoMomentum(),
    PinballCourse(),
    ReverseTrajectory(),
    FrictionStop(),
    LateralGravity(),
]
