# PhysIQ — Physics Reasoning Benchmark for AGI Evaluation

**Track:** Executive Functions  
**Scope:** 250 scenarios × 3 formats = 750 evaluation instances

---

##  Quick Start

Get up and running with PhysIQ in minutes.

```bash
# 1. Install dependencies
pip install -e .

# 2. Generate scenarios and perform ground-truth self-checks
python run_benchmark.py --scenarios 50

# 3. Run evaluation against a model
python run_eval.py --model gemini-2.0-flash --max-scenarios 15

# 4. List all supported models
python run_eval.py --list-models
```

---

##What this measures

PhysIQ tests whether LLMs can **mentally simulate 2D physics** — not just recall textbook equations. It features five tasks of increasing cognitive complexity:

| # | Task | Type | Focus |
|---|------|------|-------|
| 1 | **Trajectory Prediction** | Single-turn | Mental simulation under forces |
| 2 | **Stability Judgment** | Single-turn | Spatial reasoning about balance and support |
| 3 | **Causal Chain Reasoning** | Single-turn | Multi-body cause-and-effect chains |
| 4 | **Tool Use Planning** | Multi-turn (10) | Means-end reasoning, creative problem solving |
| 5 | **Adaptive Replanning** | Multi-turn (10) | Cognitive flexibility, error recovery |

---

## 📁 Project Structure

```text
physiq/                  # Core Python package
├── engine.py            # pymunk physics world & simulation
├── formats.py           # JSON, ASCII art, and natural language formatters
├── generation.py        # Scenario generation & validation
├── materials.py         # Material property definitions
├── scoring.py           # Task scorers & aggregate metrics
└── templates/           # 35 scenario templates across 5 tasks

notebooks/               # Kaggle Benchmark notebooks
├── 00_physiq_benchmark.ipynb # Aggregate analysis
└── 01-05_*.ipynb        # Individual per-task notebooks

results/                 # Evaluation results & reports
├── eval_*.csv           # Full evaluation results per model
└── results_analysis.md  # Detailed analysis of findings

docs/                    # Design documents (00-06)
run_benchmark.py         # CLI for generating scenarios
run_eval.py              # CLI for LLM evaluation pipeline
writeup.md               # Competition writeup
```

---

## 📊 Key Results

| Model | PhysIQ Score | Best Task | Worst Task | Format Robustness |
|-------|:------------:|-----------|------------|:-----------------:|
| **Gemini 2.0 Flash** | 0.318 | Replanning (0.456) | Tool Use (0.206) | 0.841 |
| **Gemini 2.5 Flash** | 0.300 | Replanning (0.478) | Causal Chain (0.142) | 0.717 |
| **Gemini 2.5 Pro** | 0.318 | Replanning (0.500) | Causal Chain (0.132) | 0.704 |


---

## Supported Models

Run evaluations across multiple providers. Set the respective environment variable for your chosen API:

| Provider | Environment Variable | Models |
|----------|----------------------|--------|
| Google | `GEMINI_API_KEY` | Gemini 2.0 Flash, Gemini 2.5 Flash, Gemini 2.5 Pro |
| OpenAI | `OPENAI_API_KEY` | GPT-4o, o1, o3 |
| Anthropic | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet, Haiku |
| DeepSeek | `DEEPSEEK_API_KEY` | DeepSeek Chat, Reasoner |

Run `python run_eval.py --list-models` to see the exact model identifiers used by the evaluator.
