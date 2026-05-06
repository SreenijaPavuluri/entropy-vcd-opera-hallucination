# Entropy-Guided Adaptive Hallucination Mitigation in Vision–Language Models

> **MS Applied AI Final Project — Stevens Institute of Technology**  
> Pranay Reddy Baireddy | Advisor: Prof. K. P. Subbalakshmi

This repository contains the complete four-stage implementation:

- **Stage 1**: Baseline, VCD-lite, OPERA-lite, Naive Combination — establishes non-additivity finding
- **Stage 2**: Entropy-Guided Adaptive Routing (Algorithm 1) — recovers 10 F1 points
- **Stage 3 (RPER)**: Recall-Preserved Entropy Routing — surpasses baseline by +0.50 F1
- **Stage 4 (MAVER)**: Multi-Scale Adaptive VCD with Extended Routing — **surpasses baseline by +1.18 F1**

## Key Results (Average across all 3 POPE splits)

| Method | Stage | Avg Prec | Avg Recall | **Avg F1** | Δ Baseline |
|--------|-------|---------|-----------|-----------|------------|
| Baseline | — | 92.91 | 77.53 | 84.50 | — |
| OPERA-lite | 1 | 93.13 | 76.67 | 84.08 | −0.42 |
| VCD-lite | 1 | 95.78 | 61.60 | 74.97 | −9.53 |
| Naive Combined | 1 | 95.94 | 60.40 | 74.12 | **−10.38** |
| Adaptive Routing | 2 | 93.69 | 76.40 | 84.13 | −0.37 |
| RPER | 3 | 92.59 | 78.58 | 85.00 | **+0.50** |
| **MAVER** | **4** | **92.05** | **80.13** | **85.68** | **+1.18** ✓ |

## What MAVER Does (4 Components)

| Component | Failure Mode Targeted | F1 Gain (Random) |
|-----------|----------------------|------------------|
| **mVCD**: Multi-scale VCD (radii [3,5,8]) | Single-scale false negatives on fine-grained objects | +0.29 |
| **adaAlpha**: Adaptive contrastive strength | Over-suppression at large blur deltas | +0.27 |
| **cwOPERA**: Confidence-weighted 5-prompt ensemble | Equal-weight dilution by confused prompts | +0.07 |
| **BAB**: Balance-aware logit bias (δ=0.15) | Systematic yes-class under-prediction on balanced dataset | +0.24 |
| **Total (MAVER)** | | **+0.87 over RPER** |

## Repository Structure

```
src/
├── model_utils.py          # LLaVA-1.5-7B wrapper (FP16 loading, yes/no logit extraction)
├── baseline.py             # Stage 1: direct inference
├── vcd_lite.py             # Stage 1: VCD contrastive logits (Eq. 4–5)
├── opera_lite.py           # Stage 1: 3-prompt ensemble (Eq. 6)
├── combined.py             # Stage 1: naive VCD+OPERA combination
├── adaptive_routing.py     # Stage 2: entropy-guided routing (Algorithm 1)
├── improved_routing.py     # Stage 3: RPER recall-biased routing
├── mvcd.py                 # MAVER component 1: multi-scale VCD
├── ada_alpha.py            # MAVER component 2: per-query adaptive alpha
├── cw_opera.py             # MAVER component 3: confidence-weighted 5-prompt OPERA
├── maver.py                # Stage 4: MAVER full system
├── evaluate.py             # POPE harness for all 7 methods
├── hyperparameter_search.py# Grid search for RPER + MAVER
└── visualize.py            # F1 bar charts, precision-recall scatter, LaTeX tables
run_all.py                  # Master runner
results/                    # Pre-populated with Stage 1 and Stage 2 results
report.md                   # Full project report
```

## Quick Start

```bash
conda create -n vcd-opera python=3.10 -y && conda activate vcd-opera
pip install -r requirements.txt

# Download data (see data/README.md)

# Smoke test (50 samples/split, ~20 min on H100)
python run_all.py --max_samples 50

# MAVER only — fastest path to main result
python run_all.py --methods maver

# Full evaluation (all 7 methods × 3 splits, ~10 hrs on H100)
python run_all.py
```

**Requires**: CUDA GPU with ≥16 GB VRAM (A100 minimum, tested on H100).

## Full Report

See [report.md](report.md) for the complete project report with all methodology, results tables, component ablation, and analysis.
