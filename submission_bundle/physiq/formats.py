"""PhysIQ format serializers — JSON, ASCII art, and natural language.

Converts scenario dicts into three text representations and wraps them
in task-specific prompt templates for LLM evaluation.
"""

from __future__ import annotations

import json
import math
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

ASCII_COLS = 50
ASCII_ROWS = 25

# Spatial terms for natural language generation
_HORIZ = {
    (0.0, 0.2): "at the far left",
    (0.2, 0.4): "on the left side",
    (0.4, 0.6): "near the center",
    (0.6, 0.8): "on the right side",
    (0.8, 1.0): "at the far right",
}

_VERT = {
    (0.0, 0.2): "near the floor",
    (0.2, 0.4): "in the lower portion",
    (0.4, 0.6): "at mid-height",
    (0.6, 0.8): "toward the top",
    (0.8, 1.0): "near the ceiling",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _spatial_phrase(x: float, y: float, width: float, height: float) -> str:
    """Return a relative spatial description for a position."""
    rx = x / width if width > 0 else 0.5
    ry = y / height if height > 0 else 0.5
    h = "near the center"
    for (lo, hi), phrase in _HORIZ.items():
        if lo <= rx < hi:
            h = phrase
            break
    v = "at mid-height"
    for (lo, hi), phrase in _VERT.items():
        if lo <= ry < hi:
            v = phrase
            break
    return f"{v}, {h}"


def _describe_mass(mass: float) -> str:
    """Human-readable mass description."""
    if mass < 0.01:
        return f"{mass * 1000:.0f} grams"
    if mass < 1.0:
        grams = mass * 1000
        if abs(grams - round(grams)) < 0.1:
            return f"{round(grams)} grams"
        return f"{mass:.2f} kg"
    if abs(mass - round(mass)) < 0.05:
        return f"{round(mass)} kilogram{'s' if round(mass) != 1 else ''}"
    if abs(mass - 0.5) < 0.05:
        return "half a kilogram"
    return f"{mass:.1f} kg"


def _describe_size_circle(radius: float) -> str:
    """Human-readable circle size."""
    cm = radius * 100
    if cm < 100:
        return f"a radius of {cm:.0f} centimeters"
    return f"a radius of {radius:.1f} meters"


def _velocity_arrow(vx: float, vy: float) -> str:
    """Return an ASCII arrow character for a velocity vector."""
    if abs(vx) < 0.01 and abs(vy) < 0.01:
        return ""
    angle = math.atan2(vy, vx)
    deg = math.degrees(angle)
    if -22.5 <= deg < 22.5:
        return ">"
    if 22.5 <= deg < 67.5:
        return "/"
    if 67.5 <= deg < 112.5:
        return "^"
    if 112.5 <= deg < 157.5:
        return "\\"
    if deg >= 157.5 or deg < -157.5:
        return "<"
    if -157.5 <= deg < -112.5:
        return "\\"
    if -112.5 <= deg < -67.5:
        return "v"
    if -67.5 <= deg < -22.5:
        return "/"
    return ">"


def _velocity_description(vx: float, vy: float) -> str:
    """Human-readable velocity description."""
    speed = math.hypot(vx, vy)
    if speed < 0.01:
        return "at rest"
    parts = []
    if abs(vx) >= 0.01:
        direction = "to the right" if vx > 0 else "to the left"
        parts.append(f"{abs(vx):.1f} m/s {direction}")
    if abs(vy) >= 0.01:
        direction = "upward" if vy > 0 else "downward"
        parts.append(f"{abs(vy):.1f} m/s {direction}")
    return "moving " + " and ".join(parts)


def _material_adjective(mat: dict) -> str:
    """Infer a descriptive adjective from material properties."""
    f = mat.get("friction", 0.4)
    e = mat.get("elasticity", 0.3)
    if e >= 0.7:
        return "bouncy"
    if f >= 0.7:
        return "grippy"
    if f <= 0.1:
        return "slippery"
    if e <= 0.15:
        return "soft"
    return "smooth"


def _guess_material_name(mat: dict) -> str | None:
    """Try to match material props to a known material name."""
    from physiq.materials import MATERIALS
    for name, props in MATERIALS.items():
        if (abs(mat.get("friction", -1) - props["friction"]) < 0.01
                and abs(mat.get("elasticity", -1) - props["elasticity"]) < 0.01):
            return name
    return None


def _object_type_name(obj: dict) -> str:
    """Human-friendly type name for an object."""
    otype = obj.get("type", "")
    if otype == "circle":
        return "ball"
    if otype == "box":
        return "block"
    if otype in ("polygon", "static_polygon"):
        return "shape"
    if otype in ("static_segment", "segment"):
        return "wall"
    return otype.replace("_", " ")


# ── Format: JSON ──────────────────────────────────────────────────────────────

# Keys that must NEVER appear in a prompt — they directly encode the answer.
_ANSWER_LEAKING_KEYS = frozenset({
    "_ground_truth",   # computed ground truth (underscore prefix convention)
    "ground_truth",    # same key without underscore (legacy / serialised output)
    "expected_stable", # stability answer
    "expected_outcome",# causal chain answer text
    "expected_steps",  # causal chain step count answer
    "answer",          # any generic answer key
})


def format_as_json(scenario: dict) -> str:
    """Pretty-print scenario as a JSON code block (canonical format).

    Uses a strict whitelist of presentable fields so that answer-leaking
    keys (_ground_truth, ground_truth, expected_stable, expected_outcome,
    expected_steps, answer) are never included even if present in the dict.
    """
    visible = {}

    if "task_id" in scenario:
        visible["task_id"] = scenario["task_id"]
    if "id" in scenario:
        visible["id"] = scenario["id"]
    if "task_type" in scenario:
        visible["task_type"] = scenario["task_type"]
    if "difficulty" in scenario:
        visible["difficulty"] = scenario["difficulty"]

    if "world" in scenario:
        visible["world"] = scenario["world"]

    if "objects" in scenario:
        clean_objects = []
        for obj in scenario["objects"]:
            # Strip per-object answer/internal keys
            clean = {k: v for k, v in obj.items()
                     if k not in _ANSWER_LEAKING_KEYS and k != "_internal"}
            clean_objects.append(clean)
        visible["objects"] = clean_objects

    if "simulation_time" in scenario:
        visible["simulation_time"] = scenario["simulation_time"]
    if "question" in scenario:
        visible["question"] = scenario["question"]
    if "answer_format" in scenario:
        visible["answer_format"] = scenario["answer_format"]

    # Causal chain: include the trigger so the model knows what initiates the chain
    if "trigger" in scenario:
        visible["trigger"] = scenario["trigger"]

    # Multi-turn fields
    if "goal_description" in scenario:
        visible["goal_description"] = scenario["goal_description"]
    if "available_tools" in scenario:
        visible["available_tools"] = scenario["available_tools"]

    # Safety assertion — verify no answer keys leaked through
    assert not any(k in visible for k in _ANSWER_LEAKING_KEYS), (
        f"Answer-leaking key found in visible JSON: "
        f"{[k for k in _ANSWER_LEAKING_KEYS if k in visible]}"
    )

    return "```json\n" + json.dumps(visible, indent=2) + "\n```"


# ── Format: ASCII Art ─────────────────────────────────────────────────────────

def format_as_ascii(scenario: dict) -> str:
    """Generate an ASCII art visualization of the 2D physics world.

    Produces a bordered character grid with objects placed at scaled
    positions, a legend with exact numeric properties, and the question.
    """
    world = scenario.get("world", {})
    bounds = world.get("bounds", {"width": 12.0, "height": 10.0})
    w = bounds.get("width", 12.0)
    h = bounds.get("height", 10.0)
    grav = world.get("gravity", [0.0, -9.81])
    objects = scenario.get("objects", [])

    # Grid dimensions (interior)
    cols = ASCII_COLS
    rows = ASCII_ROWS

    # Initialize grid with spaces
    grid = [[" "] * cols for _ in range(rows)]

    def world_to_grid(wx: float, wy: float) -> tuple[int, int]:
        """Convert world coords to grid (col, row). Row 0 = top."""
        gc = int((wx / w) * (cols - 1)) if w > 0 else 0
        gr = int(((h - wy) / h) * (rows - 1)) if h > 0 else 0
        gc = max(0, min(cols - 1, gc))
        gr = max(0, min(rows - 1, gr))
        return gc, gr

    def safe_set(r: int, c: int, ch: str):
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = ch

    legend_entries: list[str] = []
    has_floor = False

    for obj in objects:
        otype = obj.get("type", "")
        oid = obj.get("id", "?")
        mat = obj.get("material", {})
        if isinstance(mat, str):
            from physiq.materials import MATERIALS
            mat = MATERIALS.get(mat, {})

        if otype == "circle":
            pos = obj.get("position", [0, 0])
            vel = obj.get("velocity", [0, 0])
            gc, gr = world_to_grid(pos[0], pos[1])
            safe_set(gr, gc, "\u25cb")  # ○
            arrow = _velocity_arrow(vel[0], vel[1])
            if arrow:
                safe_set(gr, gc + 1, arrow)
            mass = obj.get("mass", "?")
            radius = obj.get("radius", "?")
            vel_str = ""
            if abs(vel[0]) > 0.01 or abs(vel[1]) > 0.01:
                vel_str = f", v=({vel[0]},{vel[1]}) m/s"
            legend_entries.append(
                f"  \u25cb = {oid} (r={radius}m, m={mass}kg, "
                f"bounce={mat.get('elasticity', '?')}, "
                f"friction={mat.get('friction', '?')}{vel_str})"
            )

        elif otype == "box":
            pos = obj.get("position", [0, 0])
            bw = obj.get("width", 0.5)
            bh = obj.get("height", 0.5)
            gc, gr = world_to_grid(pos[0], pos[1])
            # Draw box as a small block
            half_w = max(1, int((bw / w) * cols / 2))
            half_h = max(1, int((bh / h) * rows / 2))
            for dr in range(-half_h, half_h + 1):
                for dc in range(-half_w, half_w + 1):
                    safe_set(gr + dr, gc + dc, "#")
            mass = obj.get("mass", "?")
            legend_entries.append(
                f"  # = {oid} ({bw}m x {bh}m, m={mass}kg, "
                f"bounce={mat.get('elasticity', '?')}, "
                f"friction={mat.get('friction', '?')})"
            )

        elif otype in ("polygon", "static_polygon"):
            verts = obj.get("vertices", [])
            if verts:
                # Draw vertices and edges
                prev_gc, prev_gr = None, None
                for v in verts:
                    gc, gr = world_to_grid(v[0], v[1])
                    if prev_gc is not None:
                        _draw_line(grid, prev_gr, prev_gc, gr, gc, rows, cols)
                    prev_gc, prev_gr = gc, gr
                # Close polygon
                if len(verts) > 2:
                    gc0, gr0 = world_to_grid(verts[0][0], verts[0][1])
                    _draw_line(grid, prev_gr, prev_gc, gr0, gc0, rows, cols)

            fric = mat.get("friction", "?")
            label = f"  \u2572 = {oid}"
            if verts:
                coords = " ".join(f"({v[0]},{v[1]})" for v in verts)
                label += f" [{coords}]"
            label += f" (friction={fric})"
            legend_entries.append(label)

        elif otype in ("static_segment", "segment"):
            start = obj.get("start", [0, 0])
            end = obj.get("end", [0, 0])
            sc, sr = world_to_grid(start[0], start[1])
            ec, er = world_to_grid(end[0], end[1])

            is_floor = (abs(start[1]) < 0.01 and abs(end[1]) < 0.01
                        and abs(end[0] - start[0]) > w * 0.8)

            if is_floor:
                has_floor = True
                # Floor drawn as bottom border enhancement
                for c in range(sc, min(ec + 1, cols)):
                    safe_set(rows - 1, c, "\u2550")  # ═
            elif abs(start[0] - end[0]) < 0.01:
                # Vertical
                for r in range(min(sr, er), max(sr, er) + 1):
                    safe_set(r, sc, "\u2502")  # │
            elif abs(start[1] - end[1]) < 0.01:
                # Horizontal
                for c in range(min(sc, ec), max(sc, ec) + 1):
                    safe_set(sr, c, "\u2550")  # ═
            else:
                # Diagonal
                _draw_line(grid, sr, sc, er, ec, rows, cols)

            fric = mat.get("friction", "?")
            elast = mat.get("elasticity", "?")
            if is_floor:
                legend_entries.append(
                    f"  \u2550 = {oid} (floor, friction={fric})"
                )
            else:
                legend_entries.append(
                    f"  \u2502/\u2572 = {oid} from ({start[0]},{start[1]}) "
                    f"to ({end[0]},{end[1]}) "
                    f"(bounce={elast}, friction={fric})"
                )

    # Build header
    task_id = scenario.get("task_id", "")
    difficulty = scenario.get("difficulty", "")
    header_label = f"PhysIQ Scenario: {task_id}"
    if difficulty:
        header_label += f" ({difficulty})"
    grav_str = f"g = {abs(grav[1]):.2f} m/s\u00b2 \u2193"
    world_line = f"World: {w}m \u00d7 {h}m | {grav_str}"
    damping = world.get("damping")
    if damping is not None and damping != 0.99:
        world_line += f" | damping={damping}"

    inner_width = cols + 2  # +2 for side borders
    border_h = "\u2550" * inner_width
    top_border = f"\u2554{border_h}\u2557"
    mid_border = f"\u2560{border_h}\u2563"
    bot_border = f"\u255a{border_h}\u255d"

    def pad_line(text: str) -> str:
        stripped = text[:inner_width]
        return f"\u2551 {stripped:<{cols}} \u2551"

    lines: list[str] = []
    lines.append(top_border)
    lines.append(pad_line(header_label))
    lines.append(pad_line(world_line))
    lines.append(mid_border)

    for row in grid:
        lines.append(pad_line("".join(row)))

    if has_floor:
        floor_label = "(floor)"
        floor_line = "\u2550" * (cols - len(floor_label)) + floor_label
        lines.append(f"\u255a\u2550{floor_line}\u2550\u255d")
    else:
        lines.append(bot_border)

    lines.append("")
    lines.append("Legend:")
    for entry in legend_entries:
        lines.append(entry)

    # Question
    question = scenario.get("question")
    if question:
        lines.append("")
        ans_fmt = scenario.get("answer_format", "")
        if ans_fmt:
            lines.append(f"Question: {question} \u2192 {ans_fmt}")
        else:
            lines.append(f"Question: {question}")

    return "\n".join(lines)


def _draw_line(grid: list[list[str]], r0: int, c0: int, r1: int, c1: int,
               rows: int, cols: int):
    """Bresenham line drawing on the character grid."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    # Choose character based on dominant direction
    if dc > dr * 2:
        ch = "\u2550"   # ═ mostly horizontal
    elif dr > dc * 2:
        ch = "\u2502"   # │ mostly vertical
    elif (sr > 0) == (sc > 0):
        ch = "\u2572"   # ╲ down-right or up-left
    else:
        ch = "\u2571"   # ╱ down-left or up-right

    r, c = r0, c0
    while True:
        if 0 <= r < rows and 0 <= c < cols:
            grid[r][c] = ch
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


# ── Format: Natural Language ──────────────────────────────────────────────────

def format_as_nl(scenario: dict) -> str:
    """Generate a natural language paragraph description of the scenario.

    Describes the world, each object conversationally with all numeric
    values woven into the prose, and ends with the question.
    """
    world = scenario.get("world", {})
    bounds = world.get("bounds", {"width": 12.0, "height": 10.0})
    w = bounds.get("width", 12.0)
    h = bounds.get("height", 10.0)
    grav = world.get("gravity", [0.0, -9.81])
    objects = scenario.get("objects", [])
    sim_time = scenario.get("simulation_time")

    paragraphs: list[str] = []

    # World description
    aspect = ""
    if w > h * 1.5:
        aspect = "wide, low-ceilinged"
    elif h > w * 1.5:
        aspect = "tall, narrow"
    elif abs(w - h) < 1.0:
        aspect = "roughly square"
    else:
        aspect = "rectangular"

    world_desc = (
        f"Imagine a {aspect} two-dimensional space, "
        f"{w:.0f} meters wide and {h:.0f} meters tall. "
        f"Gravity acts downward at {abs(grav[1]):.2f} m/s\u00b2"
    )
    damping = world.get("damping")
    if damping is not None and damping < 0.98:
        world_desc += f", with noticeable air resistance (damping factor {damping})"
    world_desc += "."
    paragraphs.append(world_desc)

    # Describe objects
    for obj in objects:
        paragraphs.append(_describe_object_nl(obj, w, h))

    # Question
    question = scenario.get("question")
    if question:
        # Only add temporal prefix if the question doesn't already mention the time
        time_mentioned = sim_time is not None and (
            f"{sim_time:.0f} second" in question
            or f"{sim_time} second" in question
        )
        if sim_time is not None and not time_mentioned:
            q_lower = question[0].lower() + question[1:]
            paragraphs.append(
                f"If this system runs for {sim_time:.0f} seconds, {q_lower}"
            )
        else:
            paragraphs.append(question)

    return "\n\n".join(paragraphs)


def _describe_object_nl(obj: dict, world_w: float, world_h: float) -> str:
    """Generate a natural language description of a single object."""
    otype = obj.get("type", "")
    oid = obj.get("id", "unknown")
    mat = obj.get("material", {})
    if isinstance(mat, str):
        from physiq.materials import MATERIALS
        mat = MATERIALS.get(mat, {})
    mat_name = _guess_material_name(mat)
    adj = _material_adjective(mat)

    if otype == "circle":
        return _describe_circle_nl(obj, oid, mat, mat_name, adj, world_w, world_h)
    elif otype == "box":
        return _describe_box_nl(obj, oid, mat, mat_name, adj, world_w, world_h)
    elif otype in ("polygon", "static_polygon"):
        return _describe_polygon_nl(obj, oid, mat, adj)
    elif otype in ("static_segment", "segment"):
        return _describe_segment_nl(obj, oid, mat, world_w)
    else:
        return f"There is an object called {oid} of type {otype}."


def _describe_circle_nl(obj: dict, oid: str, mat: dict,
                         mat_name: str | None, adj: str,
                         w: float, h: float) -> str:
    pos = obj.get("position", [0, 0])
    vel = obj.get("velocity", [0, 0])
    mass = obj.get("mass")
    radius = obj.get("radius", 0.2)
    spatial = _spatial_phrase(pos[0], pos[1], w, h)

    # Build description
    size_word = "small" if radius < 0.3 else "medium-sized" if radius < 0.8 else "large"
    material_phrase = f"{adj} {mat_name}" if mat_name else adj
    parts = [
        f"There is a {size_word}, {material_phrase} ball ({oid}) "
        f"positioned {spatial}, at coordinates ({pos[0]:.1f}, {pos[1]:.1f})."
    ]

    details = []
    if mass is not None:
        details.append(f"weighing {_describe_mass(mass)}")
    details.append(f"with {_describe_size_circle(radius)}")
    if details:
        parts.append("It is " + ", ".join(details) + ".")

    vel_desc = _velocity_description(vel[0], vel[1])
    if vel_desc != "at rest":
        parts.append(f"It is {vel_desc}.")

    fric = mat.get("friction")
    elast = mat.get("elasticity")
    prop_parts = []
    if elast is not None:
        if elast >= 0.7:
            prop_parts.append(f"quite bouncy (elasticity {elast:.1f})")
        elif elast >= 0.4:
            prop_parts.append(f"moderately bouncy (elasticity {elast:.1f})")
        else:
            prop_parts.append(f"not very bouncy (elasticity {elast:.1f})")
    if fric is not None:
        if fric >= 0.6:
            prop_parts.append(f"high grip (friction {fric:.1f})")
        elif fric >= 0.3:
            prop_parts.append(f"moderate grip (friction {fric:.1f})")
        else:
            prop_parts.append(f"low grip (friction {fric:.2f})")
    if prop_parts:
        parts.append("It has " + " and ".join(prop_parts) + ".")

    return " ".join(parts)


def _describe_box_nl(obj: dict, oid: str, mat: dict,
                      mat_name: str | None, adj: str,
                      w: float, h: float) -> str:
    pos = obj.get("position", [0, 0])
    vel = obj.get("velocity", [0, 0])
    mass = obj.get("mass")
    bw = obj.get("width", 1.0)
    bh = obj.get("height", 1.0)
    spatial = _spatial_phrase(pos[0], pos[1], w, h)

    material_phrase = f"{adj} {mat_name}" if mat_name else adj
    parts = [
        f"A {material_phrase} rectangular block ({oid}) sits "
        f"{spatial}, centered at ({pos[0]:.1f}, {pos[1]:.1f}). "
        f"It measures {bw:.1f} meters wide by {bh:.1f} meters tall"
    ]

    if mass is not None:
        parts[0] += f" and weighs {_describe_mass(mass)}"
    parts[0] += "."

    vel_desc = _velocity_description(vel[0], vel[1])
    if vel_desc != "at rest":
        parts.append(f"It is {vel_desc}.")

    angle = obj.get("angle", 0)
    if abs(angle) > 0.5:
        parts.append(f"It is rotated {angle:.0f} degrees from horizontal.")

    return " ".join(parts)


def _describe_polygon_nl(obj: dict, oid: str, mat: dict, adj: str) -> str:
    verts = obj.get("vertices", [])
    is_static = obj.get("type", "").startswith("static")
    fric = mat.get("friction", 0.4)

    if not verts:
        return f"There is a {'fixed' if is_static else ''} polygon ({oid})."

    n = len(verts)
    # Detect ramp-like shapes (triangular with one roughly horizontal edge)
    is_ramp = n == 3 or (n > 2 and _looks_like_ramp(verts))

    if is_ramp and n >= 2:
        top = max(verts, key=lambda v: v[1])
        bottom = min(verts, key=lambda v: v[1])
        if fric < 0.15:
            surface = "slick"
        elif fric < 0.3:
            surface = "fairly smooth"
        else:
            surface = "rough"
        coords = ", ".join(f"({v[0]:.1f}, {v[1]:.1f})" for v in verts)
        desc = (
            f"There is a {surface} {'fixed ' if is_static else ''}ramp ({oid}) "
            f"with vertices at {coords}. It slopes from around "
            f"({top[0]:.1f}, {top[1]:.1f}) down to ({bottom[0]:.1f}, {bottom[1]:.1f})"
        )
        if fric < 0.2:
            desc += ", with very little friction"
        desc += f" (friction {fric:.2f})."
        return desc
    else:
        coords = ", ".join(f"({v[0]:.1f}, {v[1]:.1f})" for v in verts)
        kind = "fixed" if is_static else "free-moving"
        return (
            f"A {kind} {adj} polygon ({oid}) has vertices at {coords} "
            f"(friction {fric:.2f}, elasticity {mat.get('elasticity', '?')})."
        )


def _looks_like_ramp(verts: list) -> bool:
    """Heuristic: a polygon is ramp-like if its height span > width span * 0.3."""
    if len(verts) < 3:
        return False
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    return dx > 0.1 and dy > 0.1 and dy > dx * 0.3


def _describe_segment_nl(obj: dict, oid: str, mat: dict,
                          world_w: float) -> str:
    start = obj.get("start", [0, 0])
    end = obj.get("end", [0, 0])
    fric = mat.get("friction", 0.4)
    elast = mat.get("elasticity", 0.3)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)

    # Detect floor
    if (abs(start[1]) < 0.01 and abs(end[1]) < 0.01
            and abs(dx) > world_w * 0.8):
        fric_desc = "decent" if fric >= 0.3 else "low"
        return (
            f"The floor runs the full width of the room with "
            f"{fric_desc} friction ({fric:.2f})."
        )

    # Detect vertical wall
    if abs(dx) < 0.01:
        bounce_desc = ""
        if elast >= 0.5:
            bounce_desc = "moderately bouncy"
        elif elast >= 0.7:
            bounce_desc = "very bouncy"
        else:
            bounce_desc = "not very bouncy"
        return (
            f"A vertical wall segment ({oid}) extends from "
            f"({start[0]:.1f}, {start[1]:.1f}) up to "
            f"({end[0]:.1f}, {end[1]:.1f}), about {length:.1f} meters tall. "
            f"It is {bounce_desc} (elasticity {elast:.2f}, friction {fric:.2f})."
        )

    # Detect horizontal surface
    if abs(dy) < 0.01:
        return (
            f"A horizontal surface ({oid}) runs from "
            f"({start[0]:.1f}, {start[1]:.1f}) to ({end[0]:.1f}, {end[1]:.1f}), "
            f"about {length:.1f} meters long "
            f"(friction {fric:.2f}, elasticity {elast:.2f})."
        )

    # General diagonal
    return (
        f"A diagonal segment ({oid}) stretches from "
        f"({start[0]:.1f}, {start[1]:.1f}) to ({end[0]:.1f}, {end[1]:.1f}), "
        f"spanning about {length:.1f} meters "
        f"(friction {fric:.2f}, elasticity {elast:.2f})."
    )


# ── Prompt Templates ──────────────────────────────────────────────────────────

_TRAJECTORY_PROMPT = """\
You are given a 2D physics scenario. Your task is to mentally simulate the \
physics and predict the final position of the specified object after the \
simulation runs.

Think step by step about the forces acting on the object, any collisions \
it will undergo, and how its trajectory will evolve over time. Consider \
gravity, friction, elasticity, and any ramps or walls it may interact with.

SCENARIO:
{scenario}

ANSWER FORMAT: First explain your reasoning step by step. Then on the \
final line write exactly:
ANSWER: {answer_format}"""

_STABILITY_PROMPT = """\
You are given a 2D arrangement of objects under gravity. Your task is to \
determine whether this arrangement will remain stable (all objects at rest \
or nearly at rest) after 3 seconds, or whether something will fall, slide, \
topple, or collapse.

Analyze the support structure, center of mass positions, friction, and \
contact geometry. Consider whether each object is adequately supported.

SCENARIO:
{scenario}

ANSWER FORMAT: First explain your reasoning about the stability of each \
object and the overall structure. Then on the final line write exactly:
ANSWER: STABLE or ANSWER: UNSTABLE
If unstable, on the next line write: FAILURE: <brief description of what \
fails first and why>"""

_CAUSAL_CHAIN_PROMPT = """\
You are given a chain-reaction setup in 2D physics. Multiple objects are \
arranged so that motion of one triggers interactions with others in sequence. \
Your task is to predict what happens step by step and determine the final \
outcome.

Trace the causal chain carefully: what moves first, what it hits, what \
that causes, and so on. Pay attention to masses, velocities, elasticity, \
and friction at each step.

SCENARIO:
{scenario}

ANSWER FORMAT: First describe each step of the chain reaction in order. \
Then on the final line write exactly:
ANSWER: {answer_format}"""

_TOOL_USE_PROMPT = """\
You are in a 2D physics world and must achieve a specific goal by placing, \
pushing, or removing objects. You have a limited number of turns (max 10) \
to reach the goal.

GOAL: {goal}

AVAILABLE ACTIONS (one per turn):
  PLACE <object_id> AT <x> <y> ANGLE <degrees>
  PUSH <object_id> WITH_FORCE <newtons> DIRECTION <angle_degrees>
  REMOVE <object_id>

AVAILABLE TOOLS/OBJECTS:
{tools}

CURRENT STATE:
{scenario}

Think about what physical interactions are needed to achieve the goal. \
Plan your actions efficiently — fewer turns scores higher.

ACTION FORMAT: First explain your reasoning, then on the final line write \
exactly:
ACTION: <your action>"""

_REPLAN_PROMPT = """\
You are in a 2D physics world and must achieve a specific goal by placing, \
pushing, or removing objects. You have a limited number of turns (max 10) \
to reach the goal.

IMPORTANT: The world may change unexpectedly between turns. A surface might \
become slippery, an object might break, a tool might become unavailable, or \
a new obstacle might appear. When this happens, you must recognize the change, \
understand why your previous plan no longer works, and adapt your strategy.

GOAL: {goal}

AVAILABLE ACTIONS (one per turn):
  PLACE <object_id> AT <x> <y> ANGLE <degrees>
  PUSH <object_id> WITH_FORCE <newtons> DIRECTION <angle_degrees>
  REMOVE <object_id>

AVAILABLE TOOLS/OBJECTS:
{tools}

CURRENT STATE:
{scenario}

Pay attention to any changes from the previous state. If something has \
changed, explain what happened and how you are adapting your plan.

ACTION FORMAT: First explain your reasoning, then on the final line write \
exactly:
ACTION: <your action>"""


def _format_tools_list(tools: list[dict]) -> str:
    """Format available tools for multi-turn prompts."""
    if not tools:
        return "  (none)"
    lines = []
    for t in tools:
        tid = t.get("id", "?")
        ttype = t.get("type", "?")
        parts = [f"  - {tid} ({ttype})"]
        if "radius" in t:
            parts.append(f"r={t['radius']}m")
        if "width" in t and "height" in t:
            parts.append(f"{t['width']}m x {t['height']}m")
        if "mass" in t:
            parts.append(f"m={t['mass']}kg")
        mat = t.get("material", {})
        if isinstance(mat, str):
            parts.append(f"material={mat}")
        elif mat:
            if "friction" in mat:
                parts.append(f"friction={mat['friction']}")
            if "elasticity" in mat:
                parts.append(f"bounce={mat['elasticity']}")
        lines.append(", ".join(parts))
    return "\n".join(lines)


# ── Public API: build_prompt ──────────────────────────────────────────────────

_FORMATTERS = {
    "json": format_as_json,
    "ascii": format_as_ascii,
    "nl": format_as_nl,
}


def build_prompt(scenario: dict, fmt: str, task_type: str) -> str:
    """Build a complete prompt for a given scenario, format, and task type.

    Parameters
    ----------
    scenario : dict
        Full scenario dictionary.
    fmt : str
        One of "json", "ascii", "nl".
    task_type : str
        One of "trajectory_prediction", "stability", "causal_chain",
        "tool_use", "replan".

    Returns
    -------
    str
        The fully assembled prompt ready to send to an LLM.
    """
    formatter = _FORMATTERS.get(fmt)
    if formatter is None:
        raise ValueError(f"Unknown format {fmt!r}. Use 'json', 'ascii', or 'nl'.")

    scenario_text = formatter(scenario)
    answer_format = scenario.get("answer_format", "[x, y]")

    # Normalize task_type aliases
    tt = task_type.lower().replace("-", "_")
    if tt in ("trajectory", "trajectory_prediction"):
        return _TRAJECTORY_PROMPT.format(
            scenario=scenario_text,
            answer_format=answer_format,
        )
    elif tt in ("stability", "stability_judgment"):
        return _STABILITY_PROMPT.format(scenario=scenario_text)
    elif tt in ("causal", "causal_chain", "causal_chain_reasoning"):
        return _CAUSAL_CHAIN_PROMPT.format(
            scenario=scenario_text,
            answer_format=answer_format,
        )
    elif tt in ("tool", "tool_use", "tool_use_planning"):
        goal = scenario.get("goal_description", scenario.get("question", ""))
        tools = scenario.get("available_tools", [])
        return _TOOL_USE_PROMPT.format(
            goal=goal,
            tools=_format_tools_list(tools),
            scenario=scenario_text,
        )
    elif tt in ("replan", "replanning", "adaptive_replanning"):
        goal = scenario.get("goal_description", scenario.get("question", ""))
        tools = scenario.get("available_tools", [])
        return _REPLAN_PROMPT.format(
            goal=goal,
            tools=_format_tools_list(tools),
            scenario=scenario_text,
        )
    else:
        raise ValueError(
            f"Unknown task type {task_type!r}. Use 'trajectory_prediction', "
            f"'stability', 'causal_chain', 'tool_use', or 'replan'."
        )
