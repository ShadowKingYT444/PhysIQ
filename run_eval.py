#!/usr/bin/env python3
"""PhysIQ LLM Evaluation Pipeline

Runs frontier models against the PhysIQ benchmark and reports per-model
scores broken down by task type, difficulty, and representation format.

QUICK START
-----------
# Dry run (no API calls, tests the pipeline):
    python run_eval.py --dry-run --model gemini-2.0-flash

# Single model:
    python run_eval.py --model gemini-2.0-flash

# Multiple models:
    python run_eval.py --model gemini-2.0-flash --model gpt-4o --model claude-sonnet

# All models with configured API keys:
    python run_eval.py --models-all

# Limit scenarios for quick smoke test:
    python run_eval.py --model gemini-2.0-flash --max-scenarios 5

# See which models have API keys configured:
    python run_eval.py --list-models

SUPPORTED MODELS
----------------
Google (GEMINI_API_KEY or GOOGLE_API_KEY):
    gemini-2.5-pro, gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash

OpenAI (OPENAI_API_KEY):
    gpt-4o, gpt-4o-mini, o1, o1-mini, o3-mini

Anthropic (ANTHROPIC_API_KEY):
    claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001

DeepSeek (DEEPSEEK_API_KEY):
    deepseek-r1, deepseek-chat

CONVENIENT ALIASES
------------------
    gemini-flash -> gemini-2.0-flash
    gemini-pro   -> gemini-1.5-pro
    claude-sonnet -> claude-sonnet-4-6
    claude-opus   -> claude-opus-4-6
    claude-haiku  -> claude-haiku-4-5-20251001
    deepseek      -> deepseek-chat

OUTPUTS (saved to outputs/ by default)
---------------------------------------
    eval_<model>.csv        — per-scenario scores for that model
    eval_comparison.csv     — cross-model summary (with multiple models)

ADDING API KEYS
---------------
Add keys to the .env file in this directory:
    GEMINI_API_KEY=...
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...
    DEEPSEEK_API_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from physiq.formats import build_prompt
from physiq.scoring import (
    parse_coordinates,
    score_causal_chain,
    score_replan,
    score_stability,
    score_tool_use,
    score_trajectory,
)

# ── Provider / model registry ─────────────────────────────────────────────────

PROVIDER_CONFIG: dict[str, dict] = {
    "google": {
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "models": [
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "o1", "o1-mini", "o3-mini"],
    },
    "anthropic": {
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ],
    },
    "deepseek": {
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-r1", "deepseek-chat"],
    },
}

MODEL_ALIASES: dict[str, str] = {
    "gemini-flash": "gemini-2.0-flash",
    "gemini-pro": "gemini-1.5-pro",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-opus": "claude-opus-4-6",
    "claude-haiku": "claude-haiku-4-5-20251001",
    "gpt4o": "gpt-4o",
    "deepseek": "deepseek-chat",
}

FORMATS = ["json", "ascii", "nl"]


def _provider_for(model_id: str) -> Optional[str]:
    for provider, cfg in PROVIDER_CONFIG.items():
        if model_id in cfg["models"]:
            return provider
    return None


# ── LLM clients ───────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def complete(self, messages: list[dict]) -> str:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OpenAIClient(LLMClient):
    def __init__(self, model_id: str, api_key: str, base_url: Optional[str] = None) -> None:
        super().__init__(model_id)
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package missing — run: pip install openai")
        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def complete(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


class AnthropicClient(LLMClient):
    def __init__(self, model_id: str, api_key: str) -> None:
        super().__init__(model_id)
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("anthropic package missing — run: pip install anthropic")
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, messages: list[dict]) -> str:
        system: Optional[str] = None
        chat_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat_msgs.append(m)
        kwargs: dict = dict(model=self.model_id, max_tokens=1024, messages=chat_msgs)
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text if resp.content else ""


class GoogleClient(LLMClient):
    def __init__(self, model_id: str, api_key: str) -> None:
        super().__init__(model_id)
        try:
            from google import genai
            self._genai = genai
            self._client = genai.Client(api_key=api_key)
        except ImportError:
            raise RuntimeError(
                "google-genai package missing -- run: pip install google-genai"
            )

    def complete(self, messages: list[dict]) -> str:
        import time as _time
        from google.genai import types as genai_types

        if len(messages) == 1:
            contents = messages[0]["content"]
        else:
            contents = []
            for m in messages[:-1]:
                role = "model" if m["role"] == "assistant" else "user"
                contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=m["content"])]))
            contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=messages[-1]["content"])]))

        for attempt in range(5):
            try:
                resp = self._client.models.generate_content(
                    model=self.model_id,
                    contents=contents,
                )
                return resp.text or ""
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    wait = 2 ** (attempt + 2)  # 4, 8, 16, 32, 64 s
                    _time.sleep(wait)
                    continue
                raise
        raise RuntimeError(f"Rate limit persisted after 5 retries for model {self.model_id}")


class DryRunClient(LLMClient):
    """Returns plausible dummy responses without making API calls."""

    _RESPONSES: dict[str, str] = {
        "trajectory": (
            "Analyzing projectile motion with initial velocity and gravity:\n"
            "ANSWER: [3.2, 0.1]"
        ),
        "stability": "The arrangement is STABLE. All objects are well-supported and balanced.",
        "causal_chain": (
            "1. trigger_ball rolls rightward and strikes domino_0\n"
            "2. domino_0 topples and hits domino_1\n"
            "3. domino_1 falls onto domino_2\n"
            "OUTCOME: All dominoes fall in sequence ending at rest on the floor."
        ),
        "tool_use": "PLACE plank_1 AT (5.15, 2.78) ANGLE 0",
        "replan": (
            "I notice the plank is behaving unexpectedly — it seems heavier than anticipated.\n"
            "Adjusting strategy:\n"
            "PUSH ball WITH_FORCE 50 DIRECTION 0"
        ),
    }

    def complete(self, messages: list[dict]) -> str:
        content = (messages[-1]["content"] if messages else "").lower()
        for key, resp in self._RESPONSES.items():
            if key in content:
                return resp
        return "ANSWER: [1.0, 1.0]"


def build_client(model_id: str, dry_run: bool = False) -> LLMClient:
    if dry_run:
        return DryRunClient(model_id)

    provider = _provider_for(model_id)
    if provider is None:
        raise ValueError(
            f"Unknown model '{model_id}'. Run --list-models to see all options."
        )

    cfg = PROVIDER_CONFIG[provider]

    if provider == "google":
        api_key = next((os.getenv(k) for k in cfg["env_keys"] if os.getenv(k)), None)
    else:
        api_key = os.getenv(cfg.get("env_key", ""))

    if not api_key:
        env = cfg.get("env_key") or " or ".join(cfg.get("env_keys", []))
        raise ValueError(f"No API key for {provider}. Set {env} in .env.")

    if provider == "google":
        return GoogleClient(model_id, api_key)
    elif provider == "openai":
        return OpenAIClient(model_id, api_key)
    elif provider == "anthropic":
        return AnthropicClient(model_id, api_key)
    elif provider == "deepseek":
        return OpenAIClient(model_id, api_key, base_url=cfg["base_url"])
    raise AssertionError(f"Unhandled provider: {provider}")


# ── Response parsers ──────────────────────────────────────────────────────────

def parse_trajectory(response: str) -> Optional[tuple[float, float]]:
    return parse_coordinates(response)


def parse_stability(response: str) -> tuple[bool, str]:
    upper = response.upper()
    # Check UNSTABLE before STABLE so "UNSTABLE" is not caught as "STABLE"
    stable = "UNSTABLE" not in upper
    failure = ""
    for line in response.splitlines():
        if any(kw in line.lower() for kw in ("fail", "toppl", "fall", "collaps", "slide", "tip")):
            failure = line.strip()
            break
    return stable, failure


def parse_causal_chain(response: str) -> tuple[list[str], str]:
    steps: list[str] = []
    outcome = ""
    for line in response.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^\d+[.)]\s", s) or re.match(r"^step\s+\d+", s, re.IGNORECASE):
            steps.append(s)
        low = s.lower()
        if any(low.startswith(p) for p in ("outcome:", "final:", "answer:", "result:", "conclusion:")):
            outcome = s.split(":", 1)[-1].strip()
    if not outcome:
        lines = [l.strip() for l in response.splitlines() if l.strip()]
        outcome = lines[-1] if lines else ""
    return steps, outcome


def parse_action(response: str) -> Optional[str]:
    """Extract one PLACE/PUSH/REMOVE action from a response.

    Uses case-sensitive match first (UPPERCASE keywords as instructed in prompts),
    falling back to case-insensitive only on lines that contain AT/WITH_FORCE/
    to avoid matching prose like 'place the plank so it spans…'.
    """
    # 1) Strict: uppercase keyword + structural tokens (AT, WITH_FORCE, etc.)
    m = re.search(
        r"((?:PLACE)\s+[\w_]+\s+AT\s+.+?)(?:\n|$)", response
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"((?:PUSH)\s+[\w_]+\s+WITH_FORCE\s+.+?)(?:\n|$)", response
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"((?:REMOVE)\s+[\w_]+)(?:\s|$)", response
    )
    if m:
        return m.group(1).strip()

    # 2) Fallback: case-insensitive but require structural tokens so prose doesn't match
    m = re.search(
        r"((?:PLACE|place)\s+[\w_]+\s+(?:AT|at)\s+.+?)(?:\n|$)", response, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()
    m = re.search(
        r"((?:PUSH|push)\s+[\w_]+\s+(?:WITH_FORCE|with_force)\s+.+?)(?:\n|$)", response, re.IGNORECASE
    )
    if m:
        return m.group(1).strip()

    return None


# ── Evaluators ────────────────────────────────────────────────────────────────

def eval_single_turn(
    scenario: dict, fmt: str, client: LLMClient
) -> tuple[float, str, str]:
    """Returns (score, response, error_message)."""
    task_type = scenario["task_type"]
    gt = scenario.get("ground_truth", {})

    prompt = build_prompt(scenario, fmt, task_type)
    response = client.complete([{"role": "user", "content": prompt}])

    try:
        if task_type == "trajectory":
            coords = parse_trajectory(response)
            if coords is None:
                return 0.0, response, "unparseable coordinates"
            score = score_trajectory(
                coords,
                tuple(gt["final_position"]),
                float(gt["world_diagonal"]),
            )

        elif task_type == "stability":
            pred_stable, pred_failure = parse_stability(response)
            score = score_stability(
                pred_stable,
                pred_failure,
                gt.get("stable", True),
                gt.get("failure_events", []),
            )

        elif task_type == "causal_chain":
            steps, outcome = parse_causal_chain(response)
            score = score_causal_chain(
                steps,
                outcome,
                gt.get("events", []),
                gt.get("outcome", ""),
            )

        else:
            return 0.0, response, f"unexpected task type for single-turn: {task_type}"

    except Exception as exc:
        return 0.0, response, str(exc)

    return score, response, ""


def eval_multiturn(
    scenario: dict, fmt: str, client: LLMClient, dry_run: bool = False
) -> tuple[float, str, str]:
    """Run tool_use or replan evaluation with live physics simulation."""
    task_type = scenario["task_type"]
    gt = scenario.get("ground_truth", {})
    # goal is in both scenario["goal"] and scenario["ground_truth"]["goal"]
    goal = gt.get("goal", scenario.get("goal", {}))
    max_turns = 10

    # In dry-run mode, skip simulation entirely
    if dry_run:
        prompt = build_prompt(scenario, fmt, task_type)
        response = client.complete([{"role": "user", "content": prompt}])
        action = parse_action(response)
        score = 0.4 if action else 0.2  # synthetic score: parseable action → better
        return score, response, ""

    try:
        from physiq.engine import PhysIQWorld
        from physiq.engine import parse_action as engine_parse_action

        world = PhysIQWorld(scenario)
        if task_type == "replan":
            perturbation = scenario.get("perturbation")
            if perturbation:
                try:
                    world.apply_perturbation(perturbation)
                except Exception:
                    pass
    except Exception as exc:
        return 0.0, "", f"World init failed: {exc}"

    prompt = build_prompt(scenario, fmt, task_type)
    messages: list[dict] = [{"role": "user", "content": prompt}]
    log: list[str] = []

    turns_used = 0
    goal_achieved = False
    reasoning_valid = True
    failure_recognized = False

    for turn in range(max_turns):
        try:
            response = client.complete(messages)
        except Exception as exc:
            return 0.0, "\n".join(log), f"API error on turn {turn + 1}: {exc}"

        log.append(f"[Turn {turn + 1}]\n{response[:300]}")
        messages.append({"role": "assistant", "content": response})

        # Replan: check if model noticed the perturbation
        if task_type == "replan" and not failure_recognized:
            low = response.lower()
            if any(w in low for w in ("unexpected", "changed", "heavier", "different", "perturb", "adjust", "notice")):
                failure_recognized = True

        action_dict = engine_parse_action(response)
        if not action_dict:
            reasoning_valid = False
            messages.append({"role": "user", "content": (
                f"[System] Turn {turn + 1}: No valid action found. "
                "Use: PLACE <obj> AT (<x>, <y>) ANGLE <deg>  |  "
                "PUSH <obj> WITH_FORCE <N> DIRECTION <deg>  |  REMOVE <obj>"
            )})
            continue

        try:
            result = world.execute_action(action_dict)
            turns_used += 1
        except Exception as exc:
            result = f"Action failed: {exc}"
            reasoning_valid = False

        if world.check_goal(goal):
            goal_achieved = True
            break

        progress = world.measure_progress(goal)
        try:
            state_desc = world.get_state_description("nl")
        except Exception:
            state_desc = "(unavailable)"

        messages.append({"role": "user", "content": (
            f"[System] Turn {turn + 1}: {result}\n"
            f"Progress: {progress:.0%} | Turns remaining: {max_turns - turn - 1}\n"
            f"Current state: {state_desc}"
        )})

    final_progress = world.measure_progress(goal)

    if task_type == "tool_use":
        score = score_tool_use(
            goal_achieved, turns_used, max_turns, reasoning_valid, final_progress
        )
    else:  # replan
        recovery_valid = turns_used > 0 and reasoning_valid
        score = score_replan(failure_recognized, recovery_valid, goal_achieved, turns_used)

    return score, "\n".join(log), ""


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    scenario_id: str
    task_type: str
    difficulty: str
    fmt: str
    model: str
    score: float
    error: str = ""
    latency_s: float = 0.0
    response_snippet: str = ""


def evaluate(
    scenario: dict,
    fmt: str,
    client: LLMClient,
    dry_run: bool = False,
    retries: int = 2,
) -> EvalResult:
    task = scenario["task_type"]
    sid = scenario.get("id", "?")

    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            if task in ("tool_use", "replan"):
                score, response, error = eval_multiturn(scenario, fmt, client, dry_run)
            else:
                score, response, error = eval_single_turn(scenario, fmt, client)

            return EvalResult(
                scenario_id=sid,
                task_type=task,
                difficulty=scenario.get("difficulty", "?"),
                fmt=fmt,
                model=client.model_id,
                score=score,
                error=error,
                latency_s=round(time.time() - t0, 2),
                response_snippet=response[:300],
            )
        except Exception as exc:
            if attempt == retries:
                return EvalResult(
                    scenario_id=sid,
                    task_type=task,
                    difficulty=scenario.get("difficulty", "?"),
                    fmt=fmt,
                    model=client.model_id,
                    score=0.0,
                    error=str(exc),
                    latency_s=round(time.time() - t0, 2),
                )
            time.sleep(2**attempt)

    # unreachable
    raise RuntimeError("evaluate loop exhausted")


# ── Output helpers ────────────────────────────────────────────────────────────

def save_results(results: list[EvalResult], out_dir: Path, model_id: str) -> Path:
    safe = re.sub(r"[^\w.-]", "_", model_id)
    path = out_dir / f"eval_{safe}.csv"
    pd.DataFrame([vars(r) for r in results]).to_csv(path, index=False)
    return path


def print_model_summary(results: list[EvalResult], model_id: str) -> None:
    df = pd.DataFrame([vars(r) for r in results])
    overall = df["score"].mean()

    print(f"\n{'='*60}")
    print(f"  {model_id}")
    print(f"  Overall PhysIQ score: {overall:.3f}  (n={len(df)} eval instances)")
    print("=" * 60)

    print("\nBy task:")
    for task, g in df.groupby("task_type"):
        print(f"  {task:<22} {g['score'].mean():.3f}  (n={len(g)})")

    print("\nBy difficulty:")
    for diff, g in df.groupby("difficulty"):
        print(f"  {diff:<12} {g['score'].mean():.3f}")

    print("\nBy format:")
    fmt_means = df.groupby("fmt")["score"].mean()
    for fmt, s in fmt_means.items():
        print(f"  {fmt:<8} {s:.3f}")

    if len(fmt_means) >= 2 and fmt_means.max() > 0:
        frs = 1.0 - (fmt_means.max() - fmt_means.min()) / fmt_means.max()
        print(f"\n  Format Robustness Score (FRS): {frs:.3f}  (1.0 = perfectly stable)")

    n_err = (df["error"] != "").sum()
    if n_err:
        print(f"\n  Errors: {n_err}/{len(df)} — run with higher verbosity or check CSV for details")


def print_cross_model_summary(
    all_results: dict[str, list[EvalResult]]
) -> pd.DataFrame:
    rows = []
    for model, results in all_results.items():
        df = pd.DataFrame([vars(r) for r in results])
        row: dict = {"model": model, "overall": round(df["score"].mean(), 3)}
        for t in sorted(df["task_type"].unique()):
            row[t] = round(df[df["task_type"] == t]["score"].mean(), 3)
        for f in sorted(df["fmt"].unique()):
            row[f"fmt_{f}"] = round(df[df["fmt"] == f]["score"].mean(), 3)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("model")
    print(f"\n{'='*60}")
    print("  Cross-Model Comparison")
    print(f"{'='*60}")
    print(summary.to_string())
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def detect_available_models() -> list[str]:
    available = []
    for provider, cfg in PROVIDER_CONFIG.items():
        if provider == "google":
            has_key = any(os.getenv(k) for k in cfg["env_keys"])
        else:
            has_key = bool(os.getenv(cfg.get("env_key", "")))
        if has_key:
            available.extend(cfg["models"])
    return available


def resolve_model(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        metavar="MODEL",
        help="Model to evaluate (repeatable). E.g. --model gemini-2.0-flash --model gpt-4o",
    )
    parser.add_argument(
        "--models-all",
        action="store_true",
        help="Run all models for which API keys are configured",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test the pipeline without real API calls (uses dummy responses)",
    )
    parser.add_argument(
        "--scenarios",
        default="outputs/scenarios.json",
        help="Path to scenarios JSON (default: outputs/scenarios.json)",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        metavar="N",
        help="Limit to first N scenarios (useful for quick tests)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=FORMATS,
        default=FORMATS,
        help="Representation formats to test (default: json ascii nl)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to write CSV results (default: outputs/)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Show all supported models and their API key status, then exit",
    )

    args = parser.parse_args()

    if args.list_models:
        configured = set(detect_available_models())
        for provider, cfg in PROVIDER_CONFIG.items():
            env = cfg.get("env_key") or " / ".join(cfg.get("env_keys", []))
            has = any(m in configured for m in cfg["models"])
            status = "key found" if has else f"missing {env}"
            print(f"\n{provider.upper()} -- {status}")
            for m in cfg["models"]:
                mark = "[ok]" if m in configured else "[ ]"
                print(f"  {mark} {m}")
        print("\nAliases:", ", ".join(f"{k}={v}" for k, v in MODEL_ALIASES.items()))
        return

    # Resolve model list
    if args.models_all:
        model_ids = detect_available_models()
        if not model_ids and not args.dry_run:
            print("No API keys configured. Add keys to .env or use --dry-run.")
            sys.exit(1)
        if args.dry_run and not model_ids:
            model_ids = ["dry-run-placeholder"]
    elif args.models:
        model_ids = [resolve_model(m) for m in args.models]
    else:
        parser.print_help()
        print("\nError: specify --model MODEL, --models-all, or --dry-run --model MODEL.")
        sys.exit(1)

    # Load scenarios
    sp = Path(args.scenarios)
    if not sp.exists():
        print(f"Scenarios file not found: {sp}")
        print("Run 'python run_benchmark.py' first to generate scenarios.")
        sys.exit(1)

    with open(sp) as f:
        scenarios = json.load(f)

    if args.max_scenarios:
        scenarios = scenarios[: args.max_scenarios]

    total_evals = len(scenarios) * len(args.formats)
    print(
        f"Loaded {len(scenarios)} scenarios | "
        f"formats: {args.formats} | "
        f"models: {model_ids}"
    )
    print(f"Total eval calls per model: {total_evals}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    all_results: dict[str, list[EvalResult]] = {}

    for model_id in model_ids:
        print(f"\n{'-'*60}")
        print(f"Model: {model_id}  ({'dry run' if args.dry_run else 'live'})")
        print(f"{'-'*60}")

        try:
            client = build_client(model_id, dry_run=args.dry_run)
        except (ValueError, RuntimeError) as exc:
            print(f"  SKIP: {exc}")
            continue

        results: list[EvalResult] = []
        done = 0

        for scenario in scenarios:
            for fmt in args.formats:
                done += 1
                sid = scenario.get("id", "?")[:30]
                task = scenario.get("task_type", "?")
                diff = scenario.get("difficulty", "?")
                print(
                    f"  [{done:3d}/{total_evals}] {sid:<30} {task:<14} {diff:<8} {fmt:<5}",
                    end="",
                    flush=True,
                )
                r = evaluate(scenario, fmt, client, dry_run=args.dry_run)
                results.append(r)
                status = f"  {r.score:.3f}"
                if r.error:
                    status += f"  ERR: {r.error[:40]}"
                print(status)

        client.close()
        all_results[model_id] = results

        csv_path = save_results(results, out_dir, model_id)
        print(f"\nSaved: {csv_path}")
        print_model_summary(results, model_id)

    if not all_results:
        print("\nNo models evaluated.")
        return

    comp_path = out_dir / "eval_comparison.csv"
    if len(all_results) > 1:
        summary = print_cross_model_summary(all_results)
        summary.to_csv(comp_path)
        print(f"\nComparison table saved: {comp_path}")
    else:
        # Single model — dump full results as the comparison file too
        model_id = next(iter(all_results))
        pd.DataFrame([vars(r) for r in all_results[model_id]]).to_csv(comp_path, index=False)
        print(f"\nFull results also saved: {comp_path}")


if __name__ == "__main__":
    main()
