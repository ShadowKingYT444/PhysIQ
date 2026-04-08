"""Scenario template registry for all PhysIQ tasks."""


def _load_registry():
    from physiq.templates.trajectory import TRAJECTORY_TEMPLATES
    from physiq.templates.stability import STABILITY_TEMPLATES
    from physiq.templates.causal_chain import CAUSAL_CHAIN_TEMPLATES
    from physiq.templates.tool_use import TOOL_USE_TEMPLATES
    from physiq.templates.replan import REPLAN_TEMPLATES

    return {
        "trajectory": TRAJECTORY_TEMPLATES,
        "stability": STABILITY_TEMPLATES,
        "causal_chain": CAUSAL_CHAIN_TEMPLATES,
        "tool_use": TOOL_USE_TEMPLATES,
        "replan": REPLAN_TEMPLATES,
    }


SCENARIO_COUNTS = {
    "trajectory": {"easy": 20, "medium": 20, "hard": 20},
    "stability": {"easy": 20, "medium": 20, "hard": 20},
    "causal_chain": {"easy": 20, "medium": 20, "hard": 20},
    "tool_use": {"easy": 15, "medium": 15, "hard": 10},
    "replan": {"easy": 10, "medium": 10, "hard": 10},
}

# Lazy-loaded
_registry = None


def get_registry():
    global _registry
    if _registry is None:
        _registry = _load_registry()
    return _registry


# Backwards compat for `from physiq.templates import TEMPLATE_REGISTRY`
def __getattr__(name):
    if name == "TEMPLATE_REGISTRY":
        return get_registry()
    raise AttributeError(name)
