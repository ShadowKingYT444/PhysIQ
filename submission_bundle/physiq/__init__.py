"""PhysIQ: Physics Reasoning Benchmark for LLMs.

Lazy imports to allow incremental development.
"""


def __getattr__(name):
    if name == "PhysIQWorld":
        from physiq.engine import PhysIQWorld
        return PhysIQWorld
    if name == "MATERIALS":
        from physiq.materials import MATERIALS
        return MATERIALS
    if name in (
        "score_trajectory", "score_stability", "score_causal_chain",
        "score_tool_use", "score_replan", "physiq_score", "format_robustness_score",
    ):
        import physiq.scoring as _s
        return getattr(_s, name)
    if name in ("format_as_json", "format_as_ascii", "format_as_nl", "build_prompt"):
        import physiq.formats as _f
        return getattr(_f, name)
    if name in ("generate_dataset", "build_evaluation_dataframes"):
        import physiq.generation as _g
        return getattr(_g, name)
    raise AttributeError(f"module 'physiq' has no attribute {name!r}")
