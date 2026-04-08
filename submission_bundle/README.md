# PhysIQ

A benchmark for testing whether LLMs can actually reason about 2D physics, not just recall equations.

The setup: 250 procedurally generated scenarios across 5 tasks, each presented in three formats (JSON, ASCII art, natural language) for 750 total evaluation instances. Ground truth comes from a real physics simulator (pymunk, deterministic at 60 Hz), so scoring is completely objective. Scenarios are generated from parameterized templates with a fixed seed so everything reproduces identically and there's no training data contamination risk.

## Tasks

| # | Task | Scenarios | Type |
|---|------|-----------|------|
| 1 | Trajectory Prediction | 60 | Single-turn |
| 2 | Stability Judgment | 60 | Single-turn |
| 3 | Causal Chain Reasoning | 60 | Single-turn |
| 4 | Tool Use Planning | 40 | Multi-turn (≤10 turns) |
| 5 | Adaptive Replanning | 30 | Multi-turn + forced mid-run perturbation |

Tasks 1–3 are single-turn: given a scenario, predict what happens. Tasks 4–5 are interactive: the model issues PLACE/PUSH/REMOVE actions against a live simulation to achieve a goal. Task 5 injects a surprise perturbation (material change, broken support, missing tool, etc.) after the first action to test error recovery.

## Results

Tested three Gemini models. All three converge around 0.31 regardless of capability tier.

| Model | PhysIQ Score | Best Task | Worst Task |
|-------|:-----------:|-----------|------------|
| Gemini 2.0 Flash | 0.318 | Replanning (0.456) | Tool Use (0.206) |
| Gemini 2.5 Flash | 0.300 | Replanning (0.478) | Causal Chain (0.142) |
| Gemini 2.5 Pro | 0.318 | Replanning (0.500) | Causal Chain (0.132) |

Causal chain reasoning — predicting sequential multi-body interactions — is where all three fall apart. The 22× latency gap between 2.0 Flash and 2.5 Pro buys essentially nothing on non-standard physics scenarios.

## Quick start

```bash
pip install -e .

# generate 50 scenarios + verify ground truth
python run_benchmark.py --scenarios 50

# run against a model
python run_eval.py --model gemini-2.0-flash --max-scenarios 15

# list all supported models
python run_eval.py --list-models
```

Copy `.env.example` to `.env` and fill in your API keys.

## Layout

```
physiq/           physics engine, formatters, scoring, 35 scenario templates
notebooks/        physiq_benchmark.ipynb — all 5 tasks in one notebook
results/          pre-generated eval results for 3 Gemini models
run_benchmark.py  generate scenarios + verify ground truth
run_eval.py       run any supported LLM against the benchmark
writeup.md        competition writeup
```

Built for the Kaggle AGI Benchmarks competition, Executive Functions track.
