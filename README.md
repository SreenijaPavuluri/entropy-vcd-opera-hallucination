# Entropy-Guided Adaptive Hallucination Mitigation in Vision–Language Models

> **MS Applied AI Final Project — Stevens Institute of Technology**  
> Pranay Reddy Baireddy | Advisor: Prof. K. P. Subbalakshmi

This repository contains the full implementation of:
- **Stage 1**: Baseline, VCD-lite, OPERA-lite, Naive Combination, and Rule-Based Ensemble evaluation on POPE
- **Stage 2**: Entropy-Guided Adaptive Routing (Algorithm 1 from the paper)
- **Stage 3 (Novel)**: **RPER** — Recall-Preserved Entropy Routing, which **surpasses the baseline** by +0.50 avg F1

## Key Results

| Method | Avg Acc | Avg Prec | Avg Recall | **Avg F1** |
|--------|---------|---------|-----------|-----------|
| Baseline | 85.75 | 92.91 | 77.53 | 84.50 |
| OPERA-lite | 85.46 | 93.13 | 76.67 | 84.08 |
| VCD-lite | 79.42 | 95.78 | 61.60 | 74.97 |
| Naive Combined | 78.90 | 95.94 | 60.40 | 74.12 |
| Adaptive Routing (paper) | 85.57 | 93.69 | 76.40 | 84.13 |
| **RPER (ours)** | **85.92** | **92.59** | **78.58** | **85.00** ✓ |

RPER beats baseline by **+0.50 F1** points (avg) across all three POPE splits.

## Repository Structure

```
.
├── src/
│   ├── model_utils.py          # LLaVA-1.5-7B wrapper
│   ├── baseline.py             # Condition 1: Baseline
│   ├── vcd_lite.py             # Condition 2: VCD-lite
│   ├── opera_lite.py           # Condition 3: OPERA-lite
│   ├── combined.py             # Condition 4: Naive combination
│   ├── adaptive_routing.py     # Stage 2: Entropy-guided routing
│   ├── improved_routing.py     # Stage 3: RPER (our improved method)
│   ├── evaluate.py             # POPE evaluation harness
│   ├── hyperparameter_search.py# Grid search for RPER
│   └── visualize.py            # Result plots and LaTeX tables
├── run_all.py                  # Master runner script
├── data/
│   └── README.md               # Dataset download instructions
├── results/                    # Output JSON/CSV (generated)
├── figures/                    # Output plots (generated)
├── requirements.txt
└── report.md                   # Full project report with all results
```

## Quick Start

```bash
# Setup
conda create -n vcd-opera python=3.10 -y && conda activate vcd-opera
pip install -r requirements.txt

# Download data (see data/README.md)

# Smoke test (50 samples, ~15 min on H100)
python run_all.py --max_samples 50

# Full evaluation (9000 queries, ~8 hours on H100)
python run_all.py

# Run only the improved RPER method
python run_all.py --methods rper

# Grid search for RPER hyperparameters
python run_all.py --grid_search
```

Requires: CUDA GPU with ≥16GB VRAM (tested on NVIDIA H100).

## Full Report

See [report.md](report.md) for the complete project report including methodology, all results tables, analysis, and reproduction instructions.
