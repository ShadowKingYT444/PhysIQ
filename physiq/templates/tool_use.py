"""Tool Use Planning scenario templates (Task 4).

Six templates that require the model to propose multi-turn PLACE/PUSH/REMOVE
actions to achieve a physical objective.  Each template's ``generate()``
returns a complete scenario dict compatible with :class:`PhysIQWorld`.

Counts: easy=15, medium=15, hard=10 (total 40 scenarios).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────────────

def _static_seg(sid: str, start: list, end: list, mat: str = "steel") -> dict:
    return {
        "id": sid,
        "type": "static_segment",
        "start": start,
        "end": end,
        "radius": 0.02,
        "material": mat,
    }


def _static_poly(sid: str, verts: list, mat: str = "steel") -> dict:
    return {
        "id": sid,
        "type": "static_polygon",
        "vertices": verts,
        "material": mat,
    }


def _box(bid: str, w: float, h: float, pos: list, mat: str = "wood",
         angle: float = 0, mass: float | None = None) -> dict:
    d: dict[str, Any] = {
        "id": bid,
        "type": "box",
        "width": w,
        "height": h,
        "position": pos,
        "angle": angle,
        "material": mat,
    }
    if mass is not None:
        d["mass"] = mass
    return d


def _circle(cid: str, r: float, pos: list, mat: str = "rubber",
            vel: list | None = None, mass: float | None = None) -> dict:
    d: dict[str, Any] = {
        "id": cid,
        "type": "circle",
        "radius": r,
        "position": pos,
        "material": mat,
    }
    if vel is not None:
        d["velocity"] = vel
    if mass is not None:
        d["mass"] = mass
    return d


def _tool_box(tid: str, w: float, h: float, mat: str = "wood") -> dict:
    return {"id": tid, "type": "box", "width": w, "height": h, "material": mat}


def _tool_circle(tid: str, r: float, mat: str = "rubber") -> dict:
    return {"id": tid, "type": "circle", "radius": r, "material": mat}


def _tool_polygon(tid: str, verts: list, mat: str = "wood") -> dict:
    return {"id": tid, "type": "polygon", "vertices": verts, "material": mat}


def _world(width: float = 12.0, height: float = 10.0) -> dict:
    return {
        "gravity": [0, -9.81],
        "bounds": {"width": width, "height": height},
        "damping": 0.99,
    }


# ── Base class ─────────────────────────────────────────────────────────────────

class ToolUseTemplate(ABC):
    """Abstract base for a tool-use scenario template."""

    name: str

    @abstractmethod
    def generate(self, difficulty: str, seed: int) -> dict:
        """Return a complete scenario dict."""
        ...


# ── 1. Bridge Gap ──────────────────────────────────────────────────────────────

class BridgeGap(ToolUseTemplate):
    """Create a bridge across a gap to get a ball from left to right platform."""

    name = "bridge_gap"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        gap_w = {"easy": 2.0, "medium": 3.0, "hard": 4.5}[difficulty]
        plat_h = rng.uniform(2.0, 3.0)
        plat_w = rng.uniform(2.5, 3.5)
        world_w = plat_w * 2 + gap_w + 2.0
        world_h = plat_h + 5.0

        left_x = 1.0
        right_x = left_x + plat_w + gap_w

        objects = [
            _static_poly("left_platform",
                         [[left_x, 0], [left_x + plat_w, 0],
                          [left_x + plat_w, plat_h], [left_x, plat_h]]),
            _static_poly("right_platform",
                         [[right_x, 0], [right_x + plat_w, 0],
                          [right_x + plat_w, plat_h], [right_x, plat_h]]),
            _circle("ball", 0.2,
                    [left_x + plat_w / 2, plat_h + 0.3], "rubber"),
        ]

        # Available planks -- easy gets one long plank, harder gets shorter ones
        plank_len = {"easy": gap_w + 1.5, "medium": gap_w + 0.8, "hard": gap_w * 0.65}[difficulty]
        tools = [_tool_box("plank_1", plank_len, 0.2, "wood")]
        if difficulty in ("medium", "hard"):
            tools.append(_tool_box("plank_2", plank_len, 0.2, "wood"))
        if difficulty == "hard":
            tools.append(_tool_box("support_1", 0.4, gap_w * 0.4, "steel"))

        target_x = right_x + plat_w / 2
        target_y = plat_h + 0.3
        ball_pos = objects[-1]["position"]
        initial_dist = math.hypot(target_x - ball_pos[0], target_y - ball_pos[1])

        return {
            "id": f"tool_bridge_gap_{seed}",
            "task_type": "tool_use",
            "difficulty": difficulty,
            "world": _world(world_w, world_h),
            "objects": objects,
            "available_tools": tools,
            "goal": {
                "type": "position",
                "object_id": "ball",
                "target_position": [round(target_x, 2), round(target_y, 2)],
                "tolerance": 0.5,
                "initial_distance": round(initial_dist, 2),
            },
            "goal_description": (
                "Get the ball from the left platform across the gap "
                "to the right platform."
            ),
            "simulation_time": 5.0,
        }


# ── 2. Reach Height ───────────────────────────────────────────────────────────

class ReachHeight(ToolUseTemplate):
    """Get an object to a high platform using ramps or stacked supports."""

    name = "reach_height"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        target_h = {"easy": 3.0, "medium": 5.0, "hard": 7.0}[difficulty]
        shelf_w = rng.uniform(2.0, 3.0)
        world_w = 12.0
        world_h = target_h + 3.0

        objects = [
            # Floor
            _static_seg("floor", [0, 0], [world_w, 0]),
            # High shelf on the right
            _static_poly("shelf",
                         [[world_w - shelf_w - 0.5, target_h],
                          [world_w - 0.5, target_h],
                          [world_w - 0.5, target_h + 0.3],
                          [world_w - shelf_w - 0.5, target_h + 0.3]]),
            # Support pillar for shelf
            _static_poly("pillar",
                         [[world_w - 1.0, 0],
                          [world_w - 0.5, 0],
                          [world_w - 0.5, target_h],
                          [world_w - 1.0, target_h]]),
            # Ball on the floor
            _circle("ball", 0.25, [1.5, 0.3], "rubber"),
        ]

        tools = []
        if difficulty == "easy":
            # One ramp is enough
            ramp_l = rng.uniform(3.5, 4.5)
            tools.append(_tool_box("ramp_1", ramp_l, 0.15, "wood"))
            tools.append(_tool_box("block_1", 1.0, target_h * 0.7, "steel"))
        elif difficulty == "medium":
            tools.append(_tool_box("ramp_1", 3.0, 0.15, "wood"))
            tools.append(_tool_box("ramp_2", 3.0, 0.15, "wood"))
            tools.append(_tool_box("block_1", 1.0, target_h * 0.45, "steel"))
            tools.append(_tool_box("block_2", 1.0, target_h * 0.45, "steel"))
        else:
            tools.append(_tool_box("ramp_1", 2.5, 0.15, "wood"))
            tools.append(_tool_box("ramp_2", 2.5, 0.15, "wood"))
            tools.append(_tool_box("block_1", 0.8, target_h * 0.35, "steel"))
            tools.append(_tool_box("block_2", 0.8, target_h * 0.35, "steel"))
            tools.append(_tool_box("block_3", 0.8, target_h * 0.35, "steel"))

        target_x = world_w - shelf_w / 2 - 0.5
        target_y = target_h + 0.55
        ball_pos = objects[-1]["position"]
        initial_dist = math.hypot(target_x - ball_pos[0], target_y - ball_pos[1])

        return {
            "id": f"tool_reach_height_{seed}",
            "task_type": "tool_use",
            "difficulty": difficulty,
            "world": _world(world_w, world_h),
            "objects": objects,
            "available_tools": tools,
            "goal": {
                "type": "position",
                "object_id": "ball",
                "target_position": [round(target_x, 2), round(target_y, 2)],
                "tolerance": 0.6,
                "initial_distance": round(initial_dist, 2),
            },
            "goal_description": (
                f"Get the ball onto the shelf at height {target_h:.1f}m."
            ),
            "simulation_time": 6.0,
        }


# ── 3. Redirect Ball ──────────────────────────────────────────────────────────

class RedirectBall(ToolUseTemplate):
    """Use walls/wedges to redirect a rolling ball into a target zone."""

    name = "redirect_ball"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        world_w = 14.0
        world_h = 10.0

        # Ball starts on a ramp, rolls right
        ramp_angle = rng.uniform(15, 30)
        ball_start = [1.5, 5.0]
        ball_speed = {"easy": 2.0, "medium": 3.5, "hard": 5.0}[difficulty]
        vx = ball_speed * math.cos(math.radians(-ramp_angle))
        vy = ball_speed * math.sin(math.radians(-ramp_angle))

        # Target zone varies by difficulty
        if difficulty == "easy":
            target = [rng.uniform(8.0, 10.0), rng.uniform(0.5, 1.5)]
        elif difficulty == "medium":
            target = [rng.uniform(9.0, 11.0), rng.uniform(3.0, 5.0)]
        else:
            target = [rng.uniform(10.0, 12.0), rng.uniform(5.0, 7.0)]

        objects = [
            _static_seg("floor", [0, 0], [world_w, 0]),
            _static_seg("left_wall", [0, 0], [0, world_h]),
            _static_seg("right_wall", [world_w, 0], [world_w, world_h]),
            # Starting ramp
            _static_seg("start_ramp", [0.5, 5.5], [3.5, 4.0]),
            _circle("ball", 0.2, ball_start, "rubber", vel=[vx, vy]),
        ]

        # Deflector walls/wedges the model can place
        n_deflectors = {"easy": 2, "medium": 3, "hard": 4}[difficulty]
        tools = []
        for i in range(n_deflectors):
            w = rng.uniform(1.5, 3.0)
            tools.append(_tool_box(f"wall_{i+1}", w, 0.2, "steel"))
        # Add a wedge for harder levels
        if difficulty in ("medium", "hard"):
            tools.append(_tool_polygon(
                "wedge_1",
                [[0, 0], [1.5, 0], [0, 1.0]],
                "wood",
            ))

        target = [round(t, 2) for t in target]
        initial_dist = math.hypot(target[0] - ball_start[0],
                                  target[1] - ball_start[1])

        return {
            "id": f"tool_redirect_ball_{seed}",
            "task_type": "tool_use",
            "difficulty": difficulty,
            "world": _world(world_w, world_h),
            "objects": objects,
            "available_tools": tools,
            "goal": {
                "type": "position",
                "object_id": "ball",
                "target_position": target,
                "tolerance": {"easy": 0.8, "medium": 0.6, "hard": 0.4}[difficulty],
                "initial_distance": round(initial_dist, 2),
            },
            "goal_description": (
                f"Redirect the rolling ball to reach ({target[0]}, {target[1]})."
            ),
            "simulation_time": 6.0,
        }


# ── 4. Clear Path ─────────────────────────────────────────────────────────────

class ClearPath(ToolUseTemplate):
    """Remove obstacles blocking a ball from reaching a target zone."""

    name = "clear_path"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        world_w = 12.0
        world_h = 8.0

        n_obstacles = {"easy": 2, "medium": 3, "hard": 5}[difficulty]
        path_y = rng.uniform(2.0, 3.5)

        objects = [
            _static_seg("floor", [0, 0], [world_w, 0]),
            _static_seg("ceiling", [0, world_h], [world_w, world_h]),
            # Ramp to give ball initial velocity
            _static_seg("ramp", [0.5, path_y + 1.0], [2.5, path_y]),
            _circle("ball", 0.2, [1.0, path_y + 1.3], "rubber"),
        ]

        # Generate obstacle positions spread across the path
        obstacle_xs = np.linspace(3.5, world_w - 2.0, n_obstacles)
        for i, ox in enumerate(obstacle_xs):
            obs_w = rng.uniform(0.3, 0.6)
            obs_h = rng.uniform(1.5, 3.0)
            objects.append(_box(
                f"obstacle_{i+1}", obs_w, obs_h,
                [round(float(ox), 2), path_y + obs_h / 2],
                mat="steel",
                mass=rng.uniform(5.0, 15.0),
            ))

        # Available removal / pushing tools
        tools = []
        # Can remove obstacles directly on easy, need to push on harder
        if difficulty == "easy":
            pass  # model can just REMOVE obstacles
        else:
            tools.append(_tool_box("pusher_1", 0.5, 0.5, "steel"))
        if difficulty == "hard":
            tools.append(_tool_box("pusher_2", 0.5, 0.5, "steel"))
            tools.append(_tool_circle("roller_1", 0.3, "steel"))

        # Target: clear zone at the end
        zone_x_min = world_w - 2.0
        zone_x_max = world_w
        zone_y_min = 0.0
        zone_y_max = path_y + 2.0

        target_pos = [round(world_w - 1.0, 2), round(path_y, 2)]
        ball_pos = objects[3]["position"]
        initial_dist = math.hypot(target_pos[0] - ball_pos[0],
                                  target_pos[1] - ball_pos[1])

        return {
            "id": f"tool_clear_path_{seed}",
            "task_type": "tool_use",
            "difficulty": difficulty,
            "world": _world(world_w, world_h),
            "objects": objects,
            "available_tools": tools,
            "goal": {
                "type": "position",
                "object_id": "ball",
                "target_position": target_pos,
                "tolerance": 0.6,
                "initial_distance": round(initial_dist, 2),
            },
            "goal_description": (
                "Clear the obstacles blocking the ball's path and get it "
                "to the far right side."
            ),
            "simulation_time": 5.0,
        }


# ── 5. Launch and Catch ───────────────────────────────────────────────────────

class LaunchAndCatch(ToolUseTemplate):
    """Launch a projectile to land in a target zone (basin/cup)."""

    name = "launch_and_catch"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        world_w = 14.0
        world_h = 12.0

        launch_x = rng.uniform(1.0, 2.5)
        launch_h = rng.uniform(0.5, 1.5)

        # Target basin location
        target_x = {"easy": rng.uniform(6.0, 8.0),
                     "medium": rng.uniform(8.0, 10.0),
                     "hard": rng.uniform(10.0, 12.0)}[difficulty]
        target_y = rng.uniform(0.5, 2.0)
        basin_w = {"easy": 2.0, "medium": 1.5, "hard": 1.0}[difficulty]

        objects = [
            _static_seg("floor", [0, 0], [world_w, 0]),
            # Launch platform
            _static_poly("launch_pad",
                         [[launch_x - 0.5, 0], [launch_x + 0.5, 0],
                          [launch_x + 0.5, launch_h],
                          [launch_x - 0.5, launch_h]]),
            # Catch basin (U shape)
            _static_seg("basin_left",
                        [target_x - basin_w / 2, target_y],
                        [target_x - basin_w / 2, target_y + 1.5]),
            _static_seg("basin_bottom",
                        [target_x - basin_w / 2, target_y],
                        [target_x + basin_w / 2, target_y]),
            _static_seg("basin_right",
                        [target_x + basin_w / 2, target_y],
                        [target_x + basin_w / 2, target_y + 1.5]),
            # Projectile ball
            _circle("ball", 0.2, [launch_x, launch_h + 0.3], "rubber"),
        ]

        # Add walls/obstacles on harder levels
        if difficulty in ("medium", "hard"):
            wall_x = rng.uniform(target_x - 3.0, target_x - 1.5)
            wall_h = rng.uniform(3.0, 5.0)
            objects.append(_static_seg(
                "wall_obstacle", [wall_x, 0], [wall_x, wall_h],
            ))
        if difficulty == "hard":
            overhang_y = rng.uniform(4.0, 6.0)
            objects.append(_static_seg(
                "overhang",
                [target_x - 2.0, overhang_y],
                [target_x + 2.0, overhang_y],
            ))

        # Tools: ramps and pushers to launch the ball
        tools = [
            _tool_box("ramp_1", 3.0, 0.15, "wood"),
        ]
        if difficulty in ("medium", "hard"):
            tools.append(_tool_box("ramp_2", 2.5, 0.15, "wood"))
        if difficulty == "hard":
            tools.append(_tool_polygon(
                "launcher_1",
                [[0, 0], [1.0, 0], [0.5, 0.8]],
                "steel",
            ))

        target_pos = [round(target_x, 2), round(target_y + 0.3, 2)]
        ball_pos = objects[-1]["position"] if objects[-1]["id"] == "ball" else objects[5]["position"]
        initial_dist = math.hypot(target_pos[0] - ball_pos[0],
                                  target_pos[1] - ball_pos[1])

        return {
            "id": f"tool_launch_and_catch_{seed}",
            "task_type": "tool_use",
            "difficulty": difficulty,
            "world": _world(world_w, world_h),
            "objects": objects,
            "available_tools": tools,
            "goal": {
                "type": "position",
                "object_id": "ball",
                "target_position": target_pos,
                "tolerance": {"easy": 0.6, "medium": 0.5, "hard": 0.3}[difficulty],
                "initial_distance": round(initial_dist, 2),
            },
            "goal_description": (
                f"Launch the ball from the platform into the basin "
                f"at ({target_pos[0]}, {target_pos[1]})."
            ),
            "simulation_time": 6.0,
        }


# ── 6. Lever Advantage ────────────────────────────────────────────────────────

class LeverAdvantage(ToolUseTemplate):
    """Use mechanical advantage (lever/fulcrum) to move a heavy object."""

    name = "lever_advantage"

    def generate(self, difficulty: str, seed: int) -> dict:
        rng = np.random.RandomState(seed)

        world_w = 12.0
        world_h = 8.0

        heavy_mass = {"easy": 10.0, "medium": 25.0, "hard": 50.0}[difficulty]
        heavy_x = rng.uniform(3.0, 5.0)
        heavy_y = 0.4  # sitting on the floor

        target_x = rng.uniform(8.0, 10.0)
        target_y = rng.uniform(2.0, 4.0)

        objects = [
            _static_seg("floor", [0, 0], [world_w, 0]),
            _static_seg("left_wall", [0, 0], [0, world_h]),
            _static_seg("right_wall", [world_w, 0], [world_w, world_h]),
            # Heavy crate to move
            _box("heavy_crate", 0.8, 0.8,
                 [round(heavy_x, 2), heavy_y], "steel", mass=heavy_mass),
            # Target shelf
            _static_poly("target_shelf",
                         [[target_x - 1.5, target_y],
                          [target_x + 1.5, target_y],
                          [target_x + 1.5, target_y + 0.2],
                          [target_x - 1.5, target_y + 0.2]]),
            _static_poly("shelf_support",
                         [[target_x + 1.0, 0],
                          [target_x + 1.5, 0],
                          [target_x + 1.5, target_y],
                          [target_x + 1.0, target_y]]),
        ]

        # Lever beam and fulcrum
        lever_len = {"easy": 5.0, "medium": 4.0, "hard": 3.5}[difficulty]
        tools = [
            _tool_box("lever_1", lever_len, 0.15, "steel"),
            # Fulcrum (triangle)
            _tool_polygon("fulcrum_1",
                          [[0, 0], [0.8, 0], [0.4, 0.6]],
                          "steel"),
        ]
        if difficulty in ("medium", "hard"):
            tools.append(_tool_box("ramp_1", 3.0, 0.15, "wood"))
        if difficulty == "hard":
            # Counterweight for the lever
            tools.append(_tool_circle("weight_1", 0.3, "steel"))

        target_pos = [round(target_x, 2), round(target_y + 0.6, 2)]
        initial_dist = math.hypot(target_pos[0] - heavy_x,
                                  target_pos[1] - heavy_y)

        return {
            "id": f"tool_lever_advantage_{seed}",
            "task_type": "tool_use",
            "difficulty": difficulty,
            "world": _world(world_w, world_h),
            "objects": objects,
            "available_tools": tools,
            "goal": {
                "type": "position",
                "object_id": "heavy_crate",
                "target_position": target_pos,
                "tolerance": 0.6,
                "initial_distance": round(initial_dist, 2),
            },
            "goal_description": (
                f"Use the lever to move the heavy crate ({heavy_mass:.0f} kg) "
                f"onto the target shelf."
            ),
            "simulation_time": 6.0,
        }


# ── Template registry ─────────────────────────────────────────────────────────

TOOL_USE_TEMPLATES: list[ToolUseTemplate] = [
    BridgeGap(),
    ReachHeight(),
    RedirectBall(),
    ClearPath(),
    LaunchAndCatch(),
    LeverAdvantage(),
]
