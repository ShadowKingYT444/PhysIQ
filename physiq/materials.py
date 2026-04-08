"""Predefined material properties for PhysIQ objects."""

MATERIALS = {
    "rubber": {"friction": 0.8, "elasticity": 0.8, "density": 1.2},
    "wood": {"friction": 0.4, "elasticity": 0.3, "density": 0.6},
    "steel": {"friction": 0.3, "elasticity": 0.5, "density": 7.8},
    "ice": {"friction": 0.05, "elasticity": 0.2, "density": 0.9},
    "sponge": {"friction": 0.9, "elasticity": 0.1, "density": 0.1},
}


def resolve_material(obj_def: dict) -> dict:
    """Resolve material from name or inline dict."""
    mat = obj_def.get("material", {})
    if isinstance(mat, str):
        return MATERIALS[mat].copy()
    # Inline material — fill defaults from wood
    defaults = MATERIALS["wood"]
    return {
        "friction": mat.get("friction", defaults["friction"]),
        "elasticity": mat.get("elasticity", defaults["elasticity"]),
        "density": mat.get("density", defaults["density"]),
    }
