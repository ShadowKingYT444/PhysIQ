"""PhysIQ physics engine — pymunk wrapper for deterministic 2D simulation."""

from __future__ import annotations

import math
import re
from typing import Any

import pymunk

from physiq.materials import resolve_material

# ── Constants (from CLAUDE.md) ──────────────────────────────────────────────
DT = 1.0 / 60.0          # 60 Hz timestep
GRAVITY = (0, -981)       # cm/s² internally
ITERATIONS = 20           # collision solver iterations
DAMPING = 0.99
COLLISION_SLOP = 0.5
SETTLE_TIME = 3.0         # seconds for stability check
VELOCITY_THRESHOLD = 0.1  # stability velocity threshold
MAX_TURNS = 10            # multi-turn cap

# ── Scale helpers (user-facing m → internal cm) ─────────────────────────────
SCALE = 100.0  # 1 m = 100 cm

def m2cm(v):
    if isinstance(v, (list, tuple)):
        return tuple(x * SCALE for x in v)
    return v * SCALE

def cm2m(v):
    if isinstance(v, (list, tuple)):
        return tuple(x / SCALE for x in v)
    return v / SCALE


class PhysIQWorld:
    """Build and run a pymunk simulation from a scenario dict."""

    def __init__(self, scenario: dict):
        self.scenario = scenario
        self.space = pymunk.Space()

        # Gravity: scenario stores m/s², engine uses cm/s²
        grav = scenario.get("world", {}).get("gravity", [0, -9.81])
        self.space.gravity = (grav[0] * SCALE, grav[1] * SCALE)

        self.space.iterations = ITERATIONS
        self.space.damping = scenario.get("world", {}).get("damping", DAMPING)
        self.space.collision_slop = COLLISION_SLOP

        self.objects: dict[str, dict] = {}   # id → {body, shapes, def}
        self.events: list[dict] = []
        self._current_time = 0.0
        self._goal_def = scenario.get("goal")

        self._setup_collision_handler()
        self._build(scenario.get("objects", []))

    # ── World construction ──────────────────────────────────────────────────

    def _setup_collision_handler(self):
        self.space.on_collision(
            collision_type_a=None,
            collision_type_b=None,
            begin=self._on_collision,
        )

    def _on_collision(self, arbiter, space, data):
        shapes = arbiter.shapes
        labels = []
        for s in shapes:
            label = getattr(s, "label", None) or getattr(s.body, "label", "?")
            labels.append(label)
        pts = arbiter.contact_point_set.points
        pos = (pts[0].point_a.x / SCALE, pts[0].point_a.y / SCALE) if pts else (0, 0)
        self.events.append({
            "time": round(self._current_time, 4),
            "type": "collision",
            "objects": labels,
            "impulse": (arbiter.total_impulse.x, arbiter.total_impulse.y),
            "position": pos,
        })

    def _build(self, object_defs: list[dict]):
        for odef in object_defs:
            oid = odef["id"]
            otype = odef["type"]
            mat = resolve_material(odef)

            if otype == "circle":
                entry = self._add_circle(odef, mat)
            elif otype == "box":
                entry = self._add_box(odef, mat)
            elif otype == "polygon":
                entry = self._add_polygon(odef, mat)
            elif otype in ("static_segment", "segment"):
                entry = self._add_static_segment(odef, mat)
            elif otype == "static_polygon":
                entry = self._add_static_polygon(odef, mat)
            elif otype == "pivot_joint":
                entry = self._add_pivot_joint(odef)
                continue
            elif otype == "spring":
                entry = self._add_spring(odef)
                continue
            else:
                raise ValueError(f"Unknown object type: {otype}")

            self.objects[oid] = entry

    # ── Object builders ─────────────────────────────────────────────────────

    def _make_dynamic_body(self, odef: dict, mass: float, moment: float):
        body = pymunk.Body(mass, moment)
        pos = odef.get("position", [0, 0])
        body.position = m2cm(pos)
        vel = odef.get("velocity", [0, 0])
        body.velocity = m2cm(vel)
        angle_deg = odef.get("angle", 0)
        body.angle = math.radians(angle_deg)
        body.label = odef["id"]
        return body

    def _add_circle(self, odef: dict, mat: dict) -> dict:
        r_cm = odef["radius"] * SCALE
        mass = odef.get("mass")
        if mass is None:
            area = math.pi * (odef["radius"] ** 2)
            mass = mat["density"] * area
        moment = pymunk.moment_for_circle(mass, 0, r_cm)
        body = self._make_dynamic_body(odef, mass, moment)
        shape = pymunk.Circle(body, r_cm)
        shape.friction = mat["friction"]
        shape.elasticity = mat["elasticity"]
        shape.label = odef["id"]
        self.space.add(body, shape)
        return {"body": body, "shapes": [shape], "def": odef}

    def _add_box(self, odef: dict, mat: dict) -> dict:
        w_cm = odef["width"] * SCALE
        h_cm = odef["height"] * SCALE
        mass = odef.get("mass")
        if mass is None:
            area = odef["width"] * odef["height"]
            mass = mat["density"] * area
        moment = pymunk.moment_for_box(mass, (w_cm, h_cm))
        body = self._make_dynamic_body(odef, mass, moment)
        shape = pymunk.Poly.create_box(body, (w_cm, h_cm))
        shape.friction = mat["friction"]
        shape.elasticity = mat["elasticity"]
        shape.label = odef["id"]
        self.space.add(body, shape)
        return {"body": body, "shapes": [shape], "def": odef}

    def _add_polygon(self, odef: dict, mat: dict) -> dict:
        verts_cm = [m2cm(v) for v in odef["vertices"]]
        mass = odef.get("mass")
        if mass is None:
            # Approximate area via shoelace
            area = _polygon_area(odef["vertices"])
            mass = mat["density"] * area
        moment = pymunk.moment_for_poly(mass, verts_cm)
        body = self._make_dynamic_body(odef, mass, moment)
        shape = pymunk.Poly(body, verts_cm)
        shape.friction = mat["friction"]
        shape.elasticity = mat["elasticity"]
        shape.label = odef["id"]
        self.space.add(body, shape)
        return {"body": body, "shapes": [shape], "def": odef}

    def _add_static_segment(self, odef: dict, mat: dict) -> dict:
        body = self.space.static_body
        a = m2cm(odef["start"])
        b = m2cm(odef["end"])
        radius = odef.get("radius", 0.01) * SCALE
        shape = pymunk.Segment(body, a, b, radius)
        shape.friction = mat["friction"]
        shape.elasticity = mat["elasticity"]
        shape.label = odef["id"]
        self.space.add(shape)
        return {"body": body, "shapes": [shape], "def": odef, "static": True}

    def _add_static_polygon(self, odef: dict, mat: dict) -> dict:
        body = self.space.static_body
        verts_cm = [m2cm(v) for v in odef["vertices"]]
        shape = pymunk.Poly(body, verts_cm)
        shape.friction = mat["friction"]
        shape.elasticity = mat["elasticity"]
        shape.label = odef["id"]
        self.space.add(shape)
        return {"body": body, "shapes": [shape], "def": odef, "static": True}

    def _add_pivot_joint(self, odef: dict) -> dict:
        a_id = odef["body_a"]
        b_id = odef["body_b"]
        body_a = self.objects[a_id]["body"]
        body_b = self.objects[b_id]["body"]
        anchor = m2cm(odef.get("anchor", [0, 0]))
        joint = pymunk.PivotJoint(body_a, body_b, anchor)
        self.space.add(joint)
        return {"joint": joint, "def": odef}

    def _add_spring(self, odef: dict) -> dict:
        a_id = odef["body_a"]
        b_id = odef["body_b"]
        body_a = self.objects[a_id]["body"]
        body_b = self.objects[b_id]["body"]
        anchor_a = m2cm(odef.get("anchor_a", [0, 0]))
        anchor_b = m2cm(odef.get("anchor_b", [0, 0]))
        rest_length = odef.get("rest_length", 1.0) * SCALE
        stiffness = odef.get("stiffness", 100.0)
        damping = odef.get("damping", 5.0)
        spring = pymunk.DampedSpring(
            body_a, body_b, anchor_a, anchor_b,
            rest_length, stiffness, damping,
        )
        self.space.add(spring)
        return {"spring": spring, "def": odef}

    # ── Simulation ──────────────────────────────────────────────────────────

    def simulate(self, duration: float, dt: float = DT) -> dict:
        """Run simulation for `duration` seconds, return final state + events."""
        steps = int(duration / dt)
        for _ in range(steps):
            self.space.step(dt)
            self._current_time += dt

        return {
            "final_state": self.get_state(),
            "events": list(self.events),
            "duration": duration,
        }

    def step(self, dt: float = DT):
        """Single physics step."""
        self.space.step(dt)
        self._current_time += dt

    def get_state(self) -> dict:
        """Current state of all dynamic objects in user-facing meters."""
        state = {}
        for oid, entry in self.objects.items():
            if entry.get("static"):
                continue
            body = entry["body"]
            if body is self.space.static_body:
                continue
            state[oid] = {
                "position": [round(body.position.x / SCALE, 4),
                             round(body.position.y / SCALE, 4)],
                "velocity": [round(body.velocity.x / SCALE, 4),
                             round(body.velocity.y / SCALE, 4)],
                "angle": round(math.degrees(body.angle), 2),
                "angular_velocity": round(body.angular_velocity, 4),
            }
        return state

    # ── Stability check ─────────────────────────────────────────────────────

    def is_stable(self, settle_time: float = SETTLE_TIME,
                  velocity_threshold: float = VELOCITY_THRESHOLD) -> tuple[bool, list]:
        """Run settle_time seconds and check if all objects are at rest.
        Returns (is_stable, failure_events)."""
        initial_positions = {}
        for oid, entry in self.objects.items():
            if entry.get("static"):
                continue
            body = entry["body"]
            if body is not self.space.static_body:
                initial_positions[oid] = (body.position.x, body.position.y)

        events_before = len(self.events)
        steps = int(settle_time / DT)
        for _ in range(steps):
            self.space.step(DT)
            self._current_time += DT

        failure_events = self.events[events_before:]
        for oid, entry in self.objects.items():
            if entry.get("static"):
                continue
            body = entry["body"]
            if body is self.space.static_body:
                continue
            if body.velocity.length / SCALE > velocity_threshold:
                return False, failure_events
            if abs(body.angular_velocity) > velocity_threshold:
                return False, failure_events

        return True, failure_events

    # ── Action execution (multi-turn tasks) ─────────────────────────────────

    def execute_action(self, action: dict) -> str:
        """Execute a parsed action dict. Returns description of result."""
        atype = action["type"]

        if atype == "PLACE":
            return self._exec_place(action)
        elif atype == "PUSH":
            return self._exec_push(action)
        elif atype == "REMOVE":
            return self._exec_remove(action)
        else:
            return f"Unknown action type: {atype}"

    def _exec_place(self, action: dict) -> str:
        obj_id = action["object_id"]
        x, y = action["x"], action["y"]
        angle = action.get("angle", 0)

        # Find object definition in scenario's available_tools
        tools = self.scenario.get("available_tools", [])
        tool_def = None
        for t in tools:
            if t["id"] == obj_id:
                tool_def = t
                break
        if tool_def is None:
            return f"Object '{obj_id}' not found in available tools."

        # Build the object into the world
        tool_def = dict(tool_def)
        tool_def["position"] = [x, y]
        tool_def["angle"] = angle
        tool_def["velocity"] = [0, 0]
        mat = resolve_material(tool_def)

        otype = tool_def["type"]
        if otype == "circle":
            entry = self._add_circle(tool_def, mat)
        elif otype == "box":
            entry = self._add_box(tool_def, mat)
        elif otype == "polygon":
            entry = self._add_polygon(tool_def, mat)
        elif otype in ("static_segment", "segment"):
            entry = self._add_static_segment(tool_def, mat)
        else:
            return f"Cannot place object of type '{otype}'."

        self.objects[obj_id] = entry

        # Settle briefly
        for _ in range(int(0.5 / DT)):
            self.space.step(DT)
            self._current_time += DT

        return f"Placed {obj_id} at ({x}, {y}) angle {angle}°."

    def _exec_push(self, action: dict) -> str:
        obj_id = action["object_id"]
        if obj_id not in self.objects:
            return f"Object '{obj_id}' not found."

        entry = self.objects[obj_id]
        if entry.get("static"):
            return f"Cannot push static object '{obj_id}'."

        body = entry["body"]
        force_n = action["force"]
        direction_deg = action["direction"]
        fx = force_n * math.cos(math.radians(direction_deg)) * SCALE
        fy = force_n * math.sin(math.radians(direction_deg)) * SCALE
        body.apply_impulse_at_local_point((fx, fy))

        # Simulate 1 second to see result
        for _ in range(int(1.0 / DT)):
            self.space.step(DT)
            self._current_time += DT

        return f"Pushed {obj_id} with {force_n}N at {direction_deg}°."

    def _exec_remove(self, action: dict) -> str:
        obj_id = action["object_id"]
        if obj_id not in self.objects:
            return f"Object '{obj_id}' not found."

        entry = self.objects[obj_id]
        for shape in entry["shapes"]:
            self.space.remove(shape)
        body = entry["body"]
        if body is not self.space.static_body:
            self.space.remove(body)
        del self.objects[obj_id]
        return f"Removed {obj_id}."

    # ── Goal checking ───────────────────────────────────────────────────────

    def check_goal(self, goal: dict | None = None) -> bool:
        """Check if the scenario goal is achieved."""
        goal = goal or self._goal_def
        if goal is None:
            return False

        gtype = goal["type"]
        if gtype == "position":
            target_id = goal["object_id"]
            target_pos = goal["target_position"]
            tolerance = goal.get("tolerance", 0.5)
            if target_id not in self.objects:
                return False
            body = self.objects[target_id]["body"]
            pos = (body.position.x / SCALE, body.position.y / SCALE)
            dist = math.hypot(pos[0] - target_pos[0], pos[1] - target_pos[1])
            return dist <= tolerance

        elif gtype == "contact":
            obj_a = goal["object_a"]
            obj_b = goal["object_b"]
            for ev in self.events:
                if ev["type"] == "collision":
                    if obj_a in ev["objects"] and obj_b in ev["objects"]:
                        return True
            return False

        elif gtype == "cleared":
            blocked_zone = goal["zone"]
            for oid, entry in self.objects.items():
                if entry.get("static"):
                    continue
                body = entry["body"]
                if body is self.space.static_body:
                    continue
                px = body.position.x / SCALE
                py = body.position.y / SCALE
                if (blocked_zone["x_min"] <= px <= blocked_zone["x_max"]
                        and blocked_zone["y_min"] <= py <= blocked_zone["y_max"]):
                    return False
            return True

        return False

    def measure_progress(self, goal: dict | None = None) -> float:
        """Return 0..1 indicating how close we are to the goal."""
        goal = goal or self._goal_def
        if goal is None:
            return 0.0

        gtype = goal["type"]
        if gtype == "position":
            target_id = goal["object_id"]
            target_pos = goal["target_position"]
            if target_id not in self.objects:
                return 0.0
            body = self.objects[target_id]["body"]
            pos = (body.position.x / SCALE, body.position.y / SCALE)
            dist = math.hypot(pos[0] - target_pos[0], pos[1] - target_pos[1])
            initial_dist = goal.get("initial_distance", 10.0)
            return max(0.0, 1.0 - dist / initial_dist)

        return 0.0

    # ── Perturbation (Task 5) ───────────────────────────────────────────────

    def apply_perturbation(self, perturbation: dict):
        """Apply a perturbation to the world for replanning tasks."""
        ptype = perturbation["type"]

        if ptype == "material_change":
            obj_id = perturbation["object_id"]
            if obj_id in self.objects:
                for shape in self.objects[obj_id]["shapes"]:
                    new_mat = perturbation.get("new_material", {})
                    if "friction" in new_mat:
                        shape.friction = new_mat["friction"]
                    if "elasticity" in new_mat:
                        shape.elasticity = new_mat["elasticity"]
                if "new_mass" in perturbation:
                    body = self.objects[obj_id]["body"]
                    body.mass = perturbation["new_mass"]

        elif ptype == "structural_failure":
            obj_id = perturbation["object_id"]
            if obj_id in self.objects:
                self._exec_remove({"object_id": obj_id})

        elif ptype == "position_drift":
            obj_id = perturbation["object_id"]
            if obj_id in self.objects:
                body = self.objects[obj_id]["body"]
                dx = perturbation.get("dx", 0) * SCALE
                dy = perturbation.get("dy", 0) * SCALE
                body.position = (body.position.x + dx, body.position.y + dy)

        elif ptype == "missing_tool":
            tool_id = perturbation["tool_id"]
            tools = self.scenario.get("available_tools", [])
            self.scenario["available_tools"] = [
                t for t in tools if t["id"] != tool_id
            ]

        elif ptype == "new_obstacle":
            obs_def = perturbation["obstacle"]
            mat = resolve_material(obs_def)
            otype = obs_def["type"]
            if otype in ("static_segment", "segment"):
                entry = self._add_static_segment(obs_def, mat)
            elif otype == "box":
                entry = self._add_box(obs_def, mat)
            else:
                entry = self._add_static_segment(obs_def, mat)
            self.objects[obs_def["id"]] = entry

    # ── State description for multi-turn prompts ────────────────────────────

    def get_state_description(self, fmt: str = "json") -> str:
        """Serialize current state in requested format (json/ascii/nl)."""
        state = self.get_state()
        # Also include static objects for context
        from physiq.formats import format_as_json, format_as_ascii, format_as_nl
        scenario_snapshot = self._build_snapshot(state)
        if fmt == "json":
            return format_as_json(scenario_snapshot)
        elif fmt == "ascii":
            return format_as_ascii(scenario_snapshot)
        else:
            return format_as_nl(scenario_snapshot)

    def _build_snapshot(self, dynamic_state: dict) -> dict:
        """Build a scenario-like dict from current world state."""
        objects = []
        for oid, entry in self.objects.items():
            odef = dict(entry["def"])
            if oid in dynamic_state:
                odef["position"] = dynamic_state[oid]["position"]
                odef["velocity"] = dynamic_state[oid]["velocity"]
                odef["angle"] = dynamic_state[oid]["angle"]
            objects.append(odef)
        return {
            "world": self.scenario.get("world", {}),
            "objects": objects,
        }


# ── Action parsing ──────────────────────────────────────────────────────────

def parse_action(response: str) -> dict | None:
    """Parse an action from model response text."""
    # Look for ACTION: line
    match = re.search(
        r'ACTION:\s*(PLACE|PUSH|REMOVE)\s+(.*)',
        response, re.IGNORECASE,
    )
    if not match:
        # Fallback: search last 5 lines
        lines = response.strip().split('\n')[-5:]
        for line in reversed(lines):
            match = re.search(
                r'(PLACE|PUSH|REMOVE)\s+(.*)',
                line, re.IGNORECASE,
            )
            if match:
                break
        if not match:
            return None

    action_type = match.group(1).upper()
    rest = match.group(2).strip()

    if action_type == "PLACE":
        # PLACE <obj> AT <x> <y> ANGLE <deg>
        m = re.match(
            r'(\S+)\s+AT\s+(-?[\d.]+)\s+(-?[\d.]+)(?:\s+ANGLE\s+(-?[\d.]+))?',
            rest, re.IGNORECASE,
        )
        if not m:
            return None
        return {
            "type": "PLACE",
            "object_id": m.group(1),
            "x": float(m.group(2)),
            "y": float(m.group(3)),
            "angle": float(m.group(4)) if m.group(4) else 0,
        }

    elif action_type == "PUSH":
        # PUSH <obj> WITH_FORCE <n> DIRECTION <deg>
        # Also accept: PUSH <obj> FORCE <n> DIRECTION <deg>
        m = re.match(
            r'(\S+)\s+(?:WITH_FORCE|FORCE)\s+(-?[\d.]+)\s+DIRECTION\s+(-?[\d.]+)',
            rest, re.IGNORECASE,
        )
        if not m:
            return None
        return {
            "type": "PUSH",
            "object_id": m.group(1),
            "force": float(m.group(2)),
            "direction": float(m.group(3)),
        }

    elif action_type == "REMOVE":
        # REMOVE <obj>
        m = re.match(r'(\S+)', rest)
        if not m:
            return None
        return {"type": "REMOVE", "object_id": m.group(1)}

    return None


def parse_coordinates(response: str) -> tuple | None:
    """Extract [x, y] from model response."""
    # ANSWER: [x, y]
    match = re.search(
        r'ANSWER:\s*\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]?',
        response,
    )
    if match:
        return (float(match.group(1)), float(match.group(2)))
    # Fallback: last 3 lines
    lines = response.strip().split('\n')[-3:]
    for line in reversed(lines):
        match = re.search(r'\[?\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]?', line)
        if match:
            return (float(match.group(1)), float(match.group(2)))
    return None


# ── Utility ─────────────────────────────────────────────────────────────────

def _polygon_area(vertices: list) -> float:
    """Shoelace formula for polygon area in m²."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2.0
