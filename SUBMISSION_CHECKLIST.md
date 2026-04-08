# PhysIQ Submission Checklist

## Before Submitting on Kaggle

### 1. Create Kaggle Benchmark
- [ ] Upload all 6 notebooks to Kaggle as a Benchmark
  - `00_physiq_benchmark.ipynb` (aggregate analysis)
  - `01_trajectory_prediction.ipynb`
  - `02_stability_judgment.ipynb`
  - `03_causal_chain_reasoning.ipynb`
  - `04_tool_use_planning.ipynb`
  - `05_adaptive_replanning.ipynb`
- [ ] Upload `physiq/` package as a Kaggle Dataset (for notebook imports)
- [ ] Set benchmark to **private** (auto-publishes after deadline)

### 2. Create Kaggle Writeup
- [ ] Click "New Writeup" at the competition page
- [ ] Select track: **Executive Functions**
- [ ] Copy content from `writeup.md` into the writeup editor
- [ ] Add a **cover image** (required for Media Gallery)
- [ ] Attach the Benchmark as a **project link** ("Add a link" under Attachments)
- [ ] Optionally attach a public notebook

### 3. Final Checks
- [ ] Writeup is under 1,500 words (currently ~1,157 words)
- [ ] Benchmark link is attached as project link
- [ ] Cover image is uploaded
- [ ] Click **Submit** (top right corner)

## File Overview

| File | Purpose |
|------|---------|
| `writeup.md` | Competition writeup text (copy into Kaggle) |
| `README.md` | Project documentation |
| `notebooks/` | 6 Kaggle Benchmark notebooks |
| `physiq/` | Core Python package (engine, formats, scoring, templates) |
| `run_benchmark.py` | Scenario generation + ground-truth validation |
| `run_eval.py` | Multi-model LLM evaluation pipeline |
| `results/` | Evaluation CSVs + analysis |
| `docs/` | Design documents (00-06) |
| `pyproject.toml` | Python dependencies |

## Key Results to Highlight (3 Models, Diverse Templates)

- **PhysIQ Scores:** 2.0 Flash=0.318, 2.5 Flash=0.300, 2.5 Pro=0.318
- **Key Finding:** All 3 models converge to ~0.31 on diverse non-standard physics — no model can brute-force genuine reasoning
- **Difficulty Gradient:** All 3 models degrade easy > medium > hard
- **Format Robustness:** FRS decreases with capability: 0.841 → 0.717 → 0.704 (stronger models more format-sensitive on non-standard scenarios)
- **Headline:** Diverse physics templates equalize model performance, confirming the benchmark measures genuine spatial reasoning not equation recall
