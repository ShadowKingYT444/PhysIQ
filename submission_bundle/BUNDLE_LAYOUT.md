## Bundle Layout

This folder contains everything needed to understand, reproduce, and verify the PhysIQ benchmark. The `physiq/` package is the core library (physics engine, scenario generation, formatters, scoring, and 35 scenario templates). `run_benchmark.py` generates the 750-instance dataset with deterministic pymunk ground truth; `run_eval.py` runs any supported LLM against it. The `notebooks/` directory holds the six Kaggle-format task notebooks (one aggregate + one per task). Pre-generated results for three Gemini models live in `results/`.

```
submission_bundle/
├── writeup.md               # Competition writeup (< 1500 words)
├── README.md                # Quick-start, key results, supported models
├── pyproject.toml           # Dependencies (Python ≥ 3.11, pymunk, etc.)
├── .env.example             # Required API key names
├── run_benchmark.py         # Generate scenarios + ground-truth validation
├── run_eval.py              # Multi-model LLM evaluation pipeline
├── physiq/                  # Core Python package
│   ├── engine.py            # pymunk physics world + simulation
│   ├── formats.py           # JSON / ASCII art / natural language formatters
│   ├── generation.py        # Scenario generation + validation
│   ├── materials.py         # Material property definitions
│   ├── scoring.py           # All 5 task scorers + aggregate metrics
│   └── templates/           # 35 scenario templates (5 task files)
├── notebooks/               # Kaggle Benchmark notebooks
│   ├── 00_physiq_benchmark.ipynb    # Aggregate analysis
│   ├── 01_trajectory_prediction.ipynb
│   ├── 02_stability_judgment.ipynb
│   ├── 03_causal_chain_reasoning.ipynb
│   ├── 04_tool_use_planning.ipynb
│   └── 05_adaptive_replanning.ipynb
└── results/                 # Pre-generated evaluation results
    ├── scenarios.json               # 750 evaluation instances
    ├── ground_truths.json           # pymunk ground-truth answers
    ├── scores.csv                   # Per-instance scores (all models)
    ├── eval_comparison.csv          # Side-by-side model comparison
    ├── eval_gemini-2.0-flash.csv
    ├── eval_gemini-2.5-flash.csv
    ├── eval_gemini-2.5-pro.csv
    ├── results_analysis.md          # Detailed statistical analysis
    └── summary_report.txt           # One-page summary
```
