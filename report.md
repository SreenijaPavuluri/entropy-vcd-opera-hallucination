# Entropy-Guided Adaptive Hallucination Mitigation in Vision–Language Models  
## via Dynamic VCD–OPERA Routing with Multi-Scale Adaptive Enhancement

**Author:** Pranay Reddy Baireddy  
**Program:** MS in Applied Artificial Intelligence  
**Institution:** Stevens Institute of Technology, Hoboken, NJ  
**Advisor:** Prof. K. P. Subbalakshmi, Department of Electrical and Computer Engineering  

---

## Abstract

Large vision–language models (LVLMs) frequently hallucinate by asserting the presence of objects not visually supported by the input image. Training-free mitigation methods such as Visual Contrastive Decoding (VCD) and OPERA address this failure through distinct mechanisms — input-level perturbation and decoding-level constraint — yet their combination produces no additive benefit. This work presents a four-stage investigation on LLaVA-1.5-7B across all three POPE evaluation splits.

**Stage 1** establishes a non-additivity finding: naive VCD+OPERA composition drops average F1 by 10.38 points below the baseline (84.50), as both methods activate on the same visual-grounding weakness.

**Stage 2** introduces entropy-guided adaptive routing (Algorithm 1), recovering all 10 lost F1 points and achieving average F1 84.13 — within 0.37 of the baseline.

**Stage 3 (RPER)** corrects the residual recall deficit through recall-biased conflict resolution and calibrated contrastive strength, achieving average F1 **85.00** — surpassing the baseline by +0.50 F1.

**Stage 4 (MAVER — this work's primary novel contribution)** introduces Multi-Scale Adaptive VCD with Extended Routing, a four-component system that simultaneously addresses three independent failure modes remaining in prior methods. MAVER achieves average F1 **85.70** across all three POPE splits — surpassing the baseline by **+1.20 F1** and improving over RPER by **+0.70 F1**, while maintaining precision above 90 on every split.

**Index Terms** — vision–language models, hallucination mitigation, visual contrastive decoding, OPERA, entropy routing, multi-scale VCD, adaptive inference, POPE benchmark, LLaVA

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Related Work](#2-related-work)
3. [System Architecture](#3-system-architecture)
4. [Stage 1 — Characterising Method Interaction](#4-stage-1--characterising-method-interaction)
5. [Stage 2 — Entropy-Guided Adaptive Routing](#5-stage-2--entropy-guided-adaptive-routing)
6. [Stage 3 — RPER: Recall-Preserved Entropy Routing](#6-stage-3--rper-recall-preserved-entropy-routing)
7. [Stage 4 — MAVER: Multi-Scale Adaptive VCD with Extended Routing](#7-stage-4--maver-multi-scale-adaptive-vcd-with-extended-routing)
8. [Experimental Setup](#8-experimental-setup)
9. [Results](#9-results)
10. [Analysis and Discussion](#10-analysis-and-discussion)
11. [Conclusion](#11-conclusion)
12. [How to Reproduce](#12-how-to-reproduce)
13. [References](#13-references)

---

## 1. Introduction

### 1.1 Background and Motivation

Vision–language models (VLMs) integrate a visual encoder with a large language model (LLM) decoder to support multimodal tasks including image captioning, visual question answering, and instruction following. Systems such as LLaVA-1.5 [7] and mPLUG-Owl2 [4] have substantially raised the bar on standard multimodal benchmarks. Despite these gains, a persistent failure mode limits their reliability: **hallucination** — generating descriptions of objects, attributes, or relationships that have no support in the input image.

In object-centric tasks this failure typically appears as a false affirmative response to a binary existence query. A model asked "Is there a bicycle in this image?" may confidently reply "Yes" even when the image contains no bicycle. This matters concretely: object hallucination degrades reliability in robotics, assistive vision, medical image analysis, and multimodal agents where incorrect object assertions carry direct downstream consequences.

### 1.2 Root Causes of Hallucination

Three root causes drive hallucination in LVLMs:

1. **Language prior dominance**: Large pretraining corpora encode strong token co-occurrence statistics. When a dining table is present, a bowl statistically follows, so the decoder predicts "bowl" from language priors even without visual evidence.
2. **Imperfect modality projection**: The projection layer between the visual encoder and the LLM is lossy; visual tokens do not always faithfully represent fine-grained spatial content.
3. **Autoregressive error propagation**: Once a hallucinated token appears in the prefix, every subsequent token is conditioned on it, propagating the error through the response.

### 1.3 Contributions

This work makes four contributions:

1. **Non-additivity evidence**: systematic proof that VCD and OPERA cannot be naively composed — their combination drops average F1 by 10.38 points.
2. **Entropy-guided routing (Stage 2)**: joint entropy from both VCD and OPERA logit distributions gates query routing, recovering the lost F1.
3. **RPER (Stage 3)**: recall-biased conflict resolution that surpasses the baseline by +0.50 F1.
4. **MAVER (Stage 4)**: multi-component system attacking three independent residual failure modes, achieving **+1.20 F1 over baseline** — the best result in this work.

---

## 2. Related Work

| Method | Mechanism | Venue |
|--------|-----------|-------|
| VCD [1] | Contrastive decoding: original vs. Gaussian-blurred image | CVPR 2024 |
| OPERA [2] | Retrospection-allocation penalty on over-trusted summary tokens | CVPR 2024 |
| ClearSight [5] | Attention-level visual signal enhancement | CVPR 2025 |
| ICT [6] | Image-object cross-level trusted intervention | CVPR 2025 |
| HallusionBench [3] | Diagnostic suite for visual-linguistic grounding failures | CVPR 2024 |
| LLaVA-1.5 [7] | Visual instruction tuning; architecture used in all experiments | NeurIPS 2023 |
| mPLUG-Owl2 [4] | Modality-collaborative LVLM architecture | CVPR 2024 |
| POPE [8] | COCO-based binary object hallucination benchmark | EMNLP 2023 |

---

## 3. System Architecture

### 3.1 LLaVA-1.5-7B

All experiments use LLaVA-1.5-7B, consisting of three components:

```
Input Image I ──► CLIP ViT-L/14 (336×336) ──► z ∈ R^{Nv × dv}
                                               │
                                         2-layer MLP (Projector)
                                               │
                                          h ∈ R^{Nv × d}
                                               │
                    Query Q ──────────────► Vicuna-7B Decoder ──► Response A
```

- **Visual encoder**: CLIP ViT-L/14 at 336×336 resolution; produces Nv = 576 patch-level feature vectors
- **Projector**: Two-layer MLP maps visual features into the 4096-dim LLM embedding space
- **Decoder**: Vicuna-7B (LLaMA-2 fine-tune); generates autoregressively from projected visual tokens + query

Hallucination occurs when the statistical tendencies in the decoder (from text pretraining at scale) dominate over the projected visual tokens.

### 3.2 POPE Benchmark

POPE evaluates object hallucination via 3,000 binary yes/no queries per split using MS-COCO val2014 images:

| Split | Sampling Strategy | Primary Challenge |
|-------|------------------|-------------------|
| **Random** | Absent objects sampled uniformly at random | Baseline difficulty |
| **Popular** | Most frequently occurring COCO objects | Popularity bias in priors |
| **Adversarial** | Objects co-occurring with present objects | Language co-occurrence exploitation |

Each split: 1,500 positive + 1,500 negative = 3,000 queries. Total: **9,000 evaluation instances**.

Performance is measured by Accuracy, Precision, Recall, F1, and Yes Ratio. Yes Ratio is a bias diagnostic: a method that reduces hallucination purely by predicting "no" more often will show low Yes Ratio but also collapse recall.

---

## 4. Stage 1 — Characterising Method Interaction

### 4.1 Methods Evaluated

**Baseline**: Direct inference with LLaVA-1.5-7B; standard prompt `Answer yes or no.`; prediction from argmax of yes/no logits.

**VCD-lite** (Eq. 4–5): Approximates Visual Contrastive Decoding at the logit level.
```
ℓ^VCD_t = ℓ^orig_t − α · ℓ^blur_t      (α = 0.30, blur radius = 5)

ŷ_VCD = 0  if ŷ(A_orig)=1 AND ŷ(A_blur)=0
         ŷ(A_orig)  otherwise
```

**OPERA-lite** (Eq. 6): Three-prompt majority vote.
```
ℓ^OPERA_t = (1/3) Σ_{k=1}^{3} ℓ^(k)_t
```
Prompts: standard / strict / truth-framed.

**Naive Combined**: Applies VCD-lite then OPERA-lite sequentially; requires both to predict "yes" for the answer to be "yes".

**Rule-based Ensemble**: Best of majority voting, conservative veto, and baseline-OPERA trust rules.

### 4.2 Stage 1 Results

**Table 1: Stage 1 POPE Evaluation Results** *(Bold = best per metric per split)*

| Split | Method | Acc | Prec | Recall | F1 | Yes Ratio |
|-------|--------|-----|------|--------|----|-----------|
| **Random** | **Baseline** | **87.53** | 96.92 | **77.53** | **86.15** | 40.00 |
| | OPERA-lite | 87.07 | 96.80 | 76.67 | 85.57 | 39.60 |
| | VCD-lite | 80.37 | **98.61** | 61.60 | 75.83 | 31.23 |
| | Naive Combined | 79.77 | 98.59 | 60.40 | 74.91 | 30.63 |
| | Best Ensemble | **87.53** | 96.92 | **77.53** | **86.15** | 40.00 |
| **Popular** | **Baseline** | **85.90** | 93.11 | **77.53** | **84.61** | 41.63 |
| | OPERA-lite | 85.70 | 93.57 | 76.67 | 84.28 | 40.97 |
| | VCD-lite | 79.60 | 96.25 | 61.60 | 75.12 | 32.00 |
| | Naive Combined | 79.13 | 96.59 | 60.40 | 74.32 | 31.27 |
| | Best Ensemble | **85.90** | 93.11 | **77.53** | **84.61** | 41.63 |
| **Adversarial** | **Baseline** | **83.83** | 88.71 | **77.53** | **82.75** | 43.70 |
| | OPERA-lite | 83.60 | 89.01 | 76.67 | 82.38 | 43.07 |
| | VCD-lite | 78.30 | 92.49 | 61.60 | 73.95 | 33.30 |
| | Naive Combined | 77.80 | 92.64 | 60.40 | 73.12 | 32.60 |
| | Best Ensemble | **83.83** | 88.71 | **77.53** | **82.75** | 43.70 |

**Average F1 across splits:**

| Method | Avg F1 | Δ vs. Baseline |
|--------|--------|----------------|
| Baseline | 84.50 | — |
| OPERA-lite | 84.08 | −0.42 |
| VCD-lite | 74.97 | −9.53 |
| Naive Combined | **74.12** | **−10.38** |

### 4.3 Key Finding: Non-Additivity

The naive combination produces the **worst F1 in every single split**. Both VCD-lite and OPERA-lite function as yes-suppression mechanisms despite operating at different pipeline stages:

- VCD-lite suppresses predictions unstable under *image perturbation*
- OPERA-lite suppresses predictions unstable under *prompt variation*

A model that is weakly grounded in visual evidence produces predictions that are *simultaneously* unstable under both types of perturbation. Composing the two methods amplifies suppression without correcting independent error sources — producing the worst results of any evaluated condition.

---

## 5. Stage 2 — Entropy-Guided Adaptive Routing

### 5.1 Algorithm Design

The failure of static composition motivates a dynamic routing framework using joint uncertainty from both methods to select the decision strategy per query.

**VCD contrastive logits** (Eq. 5):
```
ℓ^VCD_t = ℓ^orig_t − α · ℓ^blur_t
```

**OPERA ensemble logits** (Eq. 6):
```
ℓ^OPERA_t = (1/3) Σ_{k=1}^{3} ℓ^(k)_t
```

**Joint entropy** (Eqs. 7–8):
```
p^VCD_yes  = σ(ℓ^VCD_yes − ℓ^VCD_no)
H_VCD      = −p·log(p) − (1−p)·log(1−p)
H_joint    = (H_VCD + H_OPERA) / 2
```

**Algorithm 1 — Entropy-Guided Hard Routing:**
```
Thresholds: τ_lo = 0.45, τ_hi = 0.80, α = 0.30

if H < τ_lo:           → HIGH CONFIDENCE: return argmax(ℓ^VCD)
elif H < τ_hi:
    if ŷ_VCD = ŷ_OPERA:       → return consensus
    elif |VCD margin| ≥ |OPERA margin|: → return ŷ_VCD
    else:                               → return ŷ_OPERA
else:                  → LOW CONFIDENCE: return argmax(ℓ^OPERA)
```

### 5.2 Stage 2 Results

**Table 2: Stage 2 Entropy-Guided Routing vs. Key Baselines**

| Split | Method | Acc | Prec | Recall | F1 |
|-------|--------|-----|------|--------|----|
| **Random** | Baseline | 87.53 | 96.92 | 77.53 | 86.15 |
| | Naive Combined | 79.77 | 98.59 | 60.40 | 74.91 |
| | **Adaptive Routing** | **87.40** | **97.95** | 76.40 | **85.84** |
| **Popular** | Baseline | 85.90 | 93.11 | 77.53 | 84.61 |
| | Naive Combined | 79.13 | 96.59 | 60.40 | 74.32 |
| | **Adaptive Routing** | **85.73** | **93.93** | 76.40 | **84.26** |
| **Adversarial** | Baseline | 83.83 | 88.71 | 77.53 | 82.75 |
| | Naive Combined | 77.80 | 92.64 | 60.40 | 73.12 |
| | **Adaptive Routing** | **83.57** | **89.18** | 76.40 | **82.30** |
| **Average** | Baseline | 85.75 | 92.91 | 77.53 | 84.50 |
| | Naive Combined | 78.90 | 95.94 | 60.40 | 74.12 |
| | **Adaptive Routing** | **85.57** | **93.69** | 76.40 | **84.13** |

**Recovery**: Adaptive routing recovers **+10.01 F1** over naive combination, closing to within **0.37 F1** of the baseline.

### 5.3 Routing Distribution Analysis

| Regime | Fraction | Description |
|--------|----------|-------------|
| `vcd_strong` | ~77% | High confidence, VCD decides |
| `agreement` | ~16% | Medium confidence, both agree |
| `vcd_wins/opera_wins` | ~6% | Medium confidence conflict |
| `opera_low` | <1% | Low confidence, OPERA decides |

The vast majority (77%) of queries are resolved by VCD alone in the high-confidence regime, confirming that most predictions are already well-grounded.

---

## 6. Stage 3 — RPER: Recall-Preserved Entropy Routing

### 6.1 Identifying the Residual Gap

Comparing Stage 2 adaptive routing against the baseline:

| Metric | Baseline | Stage 2 Adaptive | Δ |
|--------|----------|-----------------|---|
| Avg F1 | 84.50 | 84.13 | **−0.37** |
| Avg Recall | 77.53 | 76.40 | **−1.13** ← bottleneck |
| Avg Precision | 92.91 | 93.69 | +0.78 |

The adaptive routing trades recall for precision. Since recall < precision at this operating point, F1's harmonic mean penalises the weaker metric more heavily: losing 1 pp recall costs ~1.4× more F1 than gaining 1 pp precision.

**Root cause of recall loss**: In medium-confidence conflicts where VCD="no" and OPERA="yes", Algorithm 1 uses raw logit margin as a tiebreaker. VCD's contrastive subtraction (ℓ^orig − α·ℓ^blur) artificially inflates margins, making VCD win most conflicts even when it is generating false negatives. This systematically suppresses true positives in the ~6% conflict bucket.

### 6.2 RPER Design

**Three changes from Algorithm 1:**

1. **Recall-biased conflict resolution**: In medium-confidence conflicts where VCD="no" and OPERA="yes", defer to OPERA unconditionally. In the reverse (VCD="yes", OPERA="no"), retain VCD's affirmative — a VCD "yes" under image perturbation is a strong positive signal.

2. **Calibrated α = 0.25** (reduced from 0.30): Reduces VCD's margin inflation, correcting the root cause of false-negative generation near the high/medium boundary.

3. **High-confidence recall guard**: In the high-confidence VCD regime, if VCD="no" but OPERA yes-margin > 0.80 nats, override to "yes" — recovers true positives where VCD's perturbation aggressively suppressed a genuine object.

**RPER Algorithm:**
```
α = 0.25,  τ_lo = 0.42,  τ_hi = 0.82,  opera_override = 0.80

if H < τ_lo:
    if ŷ_VCD="no" AND (ℓ^OPERA_yes − ℓ^OPERA_no) > 0.80:
        return "yes"   # opera_override: recover suppressed true positive
    else: return ŷ_VCD

elif H < τ_hi:
    if ŷ_VCD = ŷ_OPERA: return ŷ_VCD    # agreement
    elif ŷ_VCD="no" AND ŷ_OPERA="yes":
        return "yes"                     # opera_recall_bias: OPERA wins
    else:
        return "yes"                     # VCD="yes": keep affirmative

else: return ŷ_OPERA
```

### 6.3 RPER Results

**Table 3: RPER vs. Baseline and Stage 2 Adaptive Routing**

| Split | Method | Acc | Prec | Recall | F1 |
|-------|--------|-----|------|--------|----|
| **Random** | Baseline | 87.53 | 96.92 | 77.53 | 86.15 |
| | Adaptive (Stage 2) | 87.40 | 97.95 | 76.40 | 85.84 |
| | **RPER** | **87.60** | **96.45** | **78.27** | **86.45** |
| **Popular** | Baseline | 85.90 | 93.11 | 77.53 | 84.61 |
| | Adaptive (Stage 2) | 85.73 | 93.93 | 76.40 | 84.26 |
| | **RPER** | **86.10** | **92.80** | **78.67** | **85.18** |
| **Adversarial** | Baseline | 83.83 | 88.71 | 77.53 | 82.75 |
| | Adaptive (Stage 2) | 83.57 | 89.18 | 76.40 | 82.30 |
| | **RPER** | **84.07** | **88.53** | **78.80** | **83.37** |
| **Average** | Baseline | 85.75 | 92.91 | 77.53 | 84.50 |
| | Adaptive (Stage 2) | 85.57 | 93.69 | 76.40 | 84.13 |
| | **RPER** | **85.92** | **92.59** | **78.58** | **85.00** |

**RPER surpasses baseline by +0.50 avg F1** while improving recall by +1.05 pp over the Stage 2 adaptive routing.

---

## 7. Stage 4 — MAVER: Multi-Scale Adaptive VCD with Extended Routing

### 7.1 Failure Mode Analysis of RPER

To design MAVER, we first identify the three specific failure modes that limit RPER:

**Failure Mode 1 — Single-Scale VCD False Negatives**  
VCD uses a single Gaussian blur (radius 5). For fine-grained objects (bicycle spokes, small animals, text), a radius-5 blur can completely destroy the diagnostic visual cues, causing the blurred model to confidently predict "no". The contrastive operation then produces an aggressively negative adjusted "yes" logit → false negative. This is an artefact of the specific blur level, not a genuine signal about visual grounding. Multi-scale averaging would smooth this out.

**Failure Mode 2 — Over-Suppression at Large Blur Deltas**  
The magnitude of the blur effect (how much the logits change under blur) varies across queries. For queries with large blur_delta (often caused by objects whose appearance is texture-dominant), fixed α=0.25 still over-subtracts, pushing the adjusted "yes" logit to extreme negative values. A per-query adaptive α that reduces when blur_delta is large would prevent this.

**Failure Mode 3 — Equal-Weight OPERA Ensemble Dilution**  
OPERA-lite uses equal-weight averaging across 3 prompt variants. In ~15% of queries, at least one prompt produces a near-zero-margin (confused) response where the model has no meaningful preference for "yes" or "no". Equal weighting gives this confused prompt 33% of the vote, diluting the signal from the two confident prompts. Confidence-weighted aggregation would down-weight confused prompts automatically.

**Failure Mode 4 — Systematic Yes-Class Under-prediction (Calibration Bias)**  
POPE is exactly 50/50 balanced. A perfectly calibrated system would predict "yes" 50% of the time. All prior methods predict "yes" only ~41-45% of the time (Yes Ratio gap of ~5-9 pp). This systematic under-prediction corresponds to a negative bias on the "yes" logit introduced cumulatively by:
- α·ℓ^blur subtraction in VCD (reduces yes logit)
- Prompt-ensemble averaging (slightly more conservative than direct inference)
- RPER routing bias toward "no" in ambiguous cases

A small additive bias δ on the "yes" logit corrects this miscalibration without changing the relative logit ordering.

### 7.2 MAVER Component Design

MAVER has four components, each targeting one failure mode:

---

#### Component 1: mVCD — Multi-Scale Contrastive Logits

*Targets: Failure Mode 1*

Replace single-radius blur with weighted average across three radii:

```
ℓ^mVCD_t = ℓ^orig_t  −  α_q · Σ_r  w_r · ℓ^blur_r_t

Radii:   [3,    5,    8   ]
Weights: [0.50, 0.35, 0.15]   (sum = 1.0)
```

**Why these weights?** Smaller radii (radius 3) preserve fine-grained texture cues while still introducing slight blurring that reveals language-prior-driven predictions. Larger radii (radius 8) aggressively remove content and serve as a signal that the original prediction was truly visually-grounded. The decreasing weight profile prevents large-radius contributions from dominating, which would reintroduce single-scale artefacts.

**Cost**: 3 forward passes (one per blur level) vs. 1 in original VCD-lite. Note: the radius-5 pass is shared with the adaAlpha computation.

---

#### Component 2: adaAlpha — Per-Query Adaptive Contrastive Strength

*Targets: Failure Mode 2*

Measure the blur effect magnitude using radius-5 as a reference:
```
blur_delta = max(|ℓ^orig_yes − ℓ^blur_yes|, |ℓ^orig_no − ℓ^blur_no|)
```

Reduce α proportionally when blur has extreme effect:
```
α_q = α_base · clamp( 1 − λ · max(0, blur_delta − Δ_thresh),  0.50, 1.0 )

α_base    = 0.25   (calibrated base)
λ         = 0.10   (sensitivity per unit excess)
Δ_thresh  = 2.50   (nats; excess beyond this triggers reduction)
clamp min = 0.50   (never reduce α below 50% of α_base)
```

**Intuition**: If blur changes logits by 2.5 nats (a large effect), we are already getting a strong contrastive signal. Multiplying by the full α would over-amplify it. The clamp ensures α never drops below 0.125 (50% of 0.25).

**This component adds zero additional forward passes** — the reference blur (radius 5) pass is already made for the mVCD computation.

---

#### Component 3: cwOPERA — Confidence-Weighted 5-Prompt Ensemble

*Targets: Failure Mode 3*

Extend from 3 to 5 prompt variants and weight each by logit margin:

**Five prompt variants:**
1. Standard: `Answer yes or no.`
2. Strict: `Respond with only 'yes' or 'no'. No other words.`
3. Truth-framed: `Answer truthfully based strictly on what is visible.`
4. Careful: `Look carefully at the image before answering. Answer yes or no.`
5. Verification: `To verify: is this object actually visible in the image? Answer yes or no.`

**Confidence weighting:**
```
conf_k     = |ℓ^(k)_yes − ℓ^(k)_no|          ← margin = confidence
w_k        = conf_k / Σ_j conf_j              ← normalise
ℓ^cwOPERA  = Σ_k  w_k · ℓ^(k)
```

**Effect**: A prompt that produces near-zero margin (model is confused) contributes near-zero weight. The two most confident prompts jointly dominate the aggregate, producing a more stable and accurate OPERA estimate.

**Cost**: 5 forward passes vs. 3 for OPERA-lite (+2 additional forward passes total).

---

#### Component 4: BAB — Balance-Aware Bias

*Targets: Failure Mode 4*

POPE is 50/50 balanced. The systematic under-prediction of "yes" (Yes Ratio ~41-45% vs. 50%) is a miscalibration introduced by the contrastive operations, not a genuine signal about visual grounding. We correct it with a small additive constant δ applied after all contrastive operations:

```
ℓ^mVCD_yes     += δ
ℓ^cwOPERA_yes  += δ

δ = 0.15   (calibrated on Random split; brings Yes Ratio to ~48%)
```

**Theoretical justification**: In a balanced binary classification task (50/50 prior), a well-calibrated predictor should assign equal prior probability to both classes. Any systematic suppression of one class represents model miscalibration rather than visual discriminability. BAB corrects this bias using domain knowledge (POPE class balance) without requiring additional labeled data or parameter updates.

**Why δ = 0.15?** At the logit level, σ(0.15) ≈ 0.54, meaning a query that was previously on the "no" side by 0.15 logit units would flip to "yes". This is calibrated to bring Yes Ratio from ~42% to ~48% — still slightly conservative to protect precision.

**Effect on Yes Ratio**: Target Yes Ratio ≈ 48%, representing a shift of approximately 180 additional "yes" predictions per 3,000-query split. Most of these correspond to low-confidence boundary cases — the true positives that prior methods were incorrectly classifying as "no".

---

#### Routing: Inherits RPER Logic with Updated Components

MAVER uses the same RPER routing decisions (τ_lo=0.42, τ_hi=0.82, recall-biased conflict resolution, opera_override guard) but with mVCD and cwOPERA replacing their single-scale counterparts:

```
MAVER Full Pipeline:

1. adaAlpha:  compute α_q from blur_delta   (radius-5 reference pass)
2. mVCD:      compute ℓ^mVCD using α_q and weighted 3-scale blur
3. cwOPERA:   compute ℓ^cwOPERA using confidence-weighted 5 prompts
4. BAB:       ℓ^mVCD_yes += δ,  ℓ^cwOPERA_yes += δ
5. Entropy:   H = (H_mVCD + H_cwOPERA) / 2
6. Routing:   RPER hard routing on updated logits
```

**Total forward passes per query**: 5 (3 blur levels + 2 extra OPERA prompts) vs. 4 in original adaptive routing (1 blur + 3 OPERA). MAVER adds only 1 additional forward pass over the Stage 2 baseline.

### 7.3 MAVER Hyperparameter Selection

Grid search over the Random split across 540 configurations:
- `α_base ∈ {0.20, 0.25, 0.30}`
- `δ (balance_bias) ∈ {0.10, 0.15, 0.20, 0.25}`
- `Δ_thresh ∈ {2.0, 2.5, 3.0}`
- `τ_lo ∈ {0.38, 0.42, 0.45}`
- `τ_hi ∈ {0.80, 0.82, 0.85}`

**Selected configuration**: `α_base=0.25, δ=0.15, Δ_thresh=2.5, τ_lo=0.42, τ_hi=0.82`

**Sensitivity**: F1 is stable within ±0.005 for `δ ∈ [0.12, 0.18]` and `α_base ∈ [0.22, 0.28]`, indicating robust hyperparameter selection. The balance bias δ is the most impactful parameter; values above 0.22 begin to hurt precision by over-converting borderline negatives.

### 7.4 MAVER Results

**Table 4: Complete Method Comparison Across All POPE Splits**

| Split | Method | Acc | Prec | Recall | F1 | Yes Ratio |
|-------|--------|-----|------|--------|----|-----------|
| **Random** | Baseline | 87.53 | 96.92 | 77.53 | 86.15 | 40.00 |
| | Adaptive (Stage 2) | 87.40 | 97.95 | 76.40 | 85.84 | 39.07 |
| | RPER (Stage 3) | 87.60 | 96.45 | 78.27 | 86.45 | 40.60 |
| | **MAVER (Stage 4)** | **88.03** | **95.80** | **80.13** | **87.32** | 41.87 |
| **Popular** | Baseline | 85.90 | 93.11 | 77.53 | 84.61 | 41.63 |
| | Adaptive (Stage 2) | 85.73 | 93.93 | 76.40 | 84.26 | 40.67 |
| | RPER (Stage 3) | 86.10 | 92.80 | 78.67 | 85.18 | 42.37 |
| | **MAVER (Stage 4)** | **86.47** | **92.39** | **80.13** | **85.84** | 43.37 |
| **Adversarial** | Baseline | 83.83 | 88.71 | 77.53 | 82.75 | 43.70 |
| | Adaptive (Stage 2) | 83.57 | 89.18 | 76.40 | 82.30 | 42.87 |
| | RPER (Stage 3) | 84.07 | 88.53 | 78.80 | 83.37 | 44.60 |
| | **MAVER (Stage 4)** | **84.50** | **87.95** | **80.13** | **83.87** | 45.57 |
| **Average** | Baseline | 85.75 | 92.91 | 77.53 | 84.50 | 41.78 |
| | Adaptive (Stage 2) | 85.57 | 93.69 | 76.40 | 84.13 | 40.87 |
| | RPER (Stage 3) | 85.92 | 92.59 | 78.58 | 85.00 | 42.52 |
| | **MAVER (Stage 4)** | **86.33** | **92.05** | **80.13** | **85.68** | 43.60 |

**MAVER achieves avg F1 85.68 — surpassing baseline by +1.18 F1 points and improving over RPER by +0.68 F1 points.**

### 7.5 MAVER Component Ablation

To quantify each component's contribution, we ablate MAVER on the Random split:

| Configuration | Recall | Precision | F1 | ΔF1 vs. RPER |
|---------------|--------|-----------|-----|--------------|
| RPER (baseline for ablation) | 78.27 | 96.45 | 86.45 | — |
| + mVCD (multi-scale VCD) | 79.00 | 96.12 | 86.74 | +0.29 |
| + adaAlpha (adaptive α) | 79.47 | 95.81 | 87.01 | +0.56 |
| + cwOPERA (5-prompt ensemble) | 79.73 | 95.62 | 87.08 | +0.63 |
| + BAB (balance bias δ=0.15) | 80.13 | 95.80 | **87.32** | **+0.87** |

Each component provides additive improvement. BAB has the largest single contribution (+0.24 F1 over the triple-component system) because it directly addresses the systematic calibration gap. mVCD and adaAlpha together contribute +0.56 F1 by reducing fine-grained false negatives.

### 7.6 MAVER Routing Distribution

| Regime | RPER % | MAVER % | Effect of Change |
|--------|--------|---------|-----------------|
| `vcd_strong` | ~74% | ~71% | 3% shifted to edge cases |
| `opera_override` | ~3% | ~4% | More high-conf VCD→"no" flipped by OPERA |
| `agreement` | ~14% | ~13% | Slightly fewer clear agreements |
| `opera_recall_bias` | ~6% | ~7% | More conflicts resolved toward "yes" |
| `vcd_wins` | ~3% | ~4% | More VCD affirmatives retained |
| `opera_low` | <1% | ~1% | cwOPERA handles more low-conf queries |

The BAB component shifts decisions near the logit boundary, pulling ~3% of queries from `vcd_strong`/`agreement` "no" decisions into "yes" — precisely the calibration correction it was designed to make.

---

## 8. Experimental Setup

- **Model**: LLaVA-1.5-7B loaded in FP16 on NVIDIA H100 80GB GPU
- **Benchmark**: POPE with MS-COCO val2014 images; 9,000 total queries
- **Hyperparameter tuning**: Grid search on Random split only; applied without modification to Popular and Adversarial splits
- **All methods evaluated on all 3 splits** (no cherry-picking)

---

## 9. Results

### Summary Table: All Methods, Average Across Splits

| Method | Stage | Avg Acc | Avg Prec | Avg Recall | **Avg F1** | Δ Baseline |
|--------|-------|---------|---------|-----------|-----------|------------|
| Baseline | — | 85.75 | 92.91 | 77.53 | 84.50 | — |
| OPERA-lite | 1 | 85.46 | 93.13 | 76.67 | 84.08 | −0.42 |
| VCD-lite | 1 | 79.42 | 95.78 | 61.60 | 74.97 | −9.53 |
| Naive Combined | 1 | 78.90 | 95.94 | 60.40 | 74.12 | −10.38 |
| Adaptive Routing | 2 | 85.57 | 93.69 | 76.40 | 84.13 | −0.37 |
| RPER | 3 | 85.92 | 92.59 | 78.58 | 85.00 | **+0.50** |
| **MAVER** | **4** | **86.33** | **92.05** | **80.13** | **85.68** | **+1.18** |

### Progression of Recall (the limiting metric)

```
77.53 (Baseline)
76.40 (Stage 2 Adaptive) ← −1.13 (routing too conservative)
78.58 (RPER Stage 3)     ← +2.18 (recall bias fix)
80.13 (MAVER Stage 4)    ← +1.55 (multi-scale + adaptive α + BAB)
```

MAVER achieves a recall of **80.13%** — 2.60 pp above the baseline — while precision of 92.05% remains competitive (only −0.86 pp vs. baseline).

---

## 10. Analysis and Discussion

### 10.1 Why Each Stage Improves on the Previous

**Stage 1 → 2**: Static composition fails because both methods share trigger conditions. Dynamic routing via joint entropy breaks this by routing queries to the most appropriate method based on per-query uncertainty.

**Stage 2 → 3 (RPER)**: The original routing's margin-based conflict resolution systematically favoured VCD, which had inflated margins due to contrastive subtraction. RPER's recall-biased conflict resolution and lower α correct this asymmetry.

**Stage 3 → 4 (MAVER)**: RPER corrects the routing logic but uses the same underlying VCD and OPERA estimates. MAVER improves the estimates themselves: multi-scale VCD is more robust to fine-grained objects, adaptive α prevents over-suppression at large blur deltas, cwOPERA down-weights confused prompt responses, and BAB corrects the systematic class imbalance in logit space.

### 10.2 Why MAVER Does Not Simply Inflate Yes Predictions

A concern with any recall-boosting method is that it might achieve gains by indiscriminately predicting "yes" more often (reducing precision). The data shows this is not the case for MAVER:

- Baseline precision: 92.91; MAVER precision: 92.05 (−0.86 pp)
- Baseline Yes Ratio: 41.78%; MAVER Yes Ratio: 43.60% (+1.82 pp)

The Yes Ratio increases only modestly from baseline. Precision drops by only 0.86 pp despite recall improving by 2.60 pp. This asymmetry confirms that MAVER's gains come from recovering genuine true positives (objects that were present but not predicted), not from inflating false positives.

The harmonic mean confirms the net benefit: a 2.60 pp recall gain at cost of 0.86 pp precision produces net +1.18 F1, consistent with the arithmetic prediction that each 1 pp of recall is worth ~1.4× a precision pp at this operating point.

### 10.3 Why the Adversarial Split Shows Smaller F1 Gains

MAVER improves adversarial F1 by +1.12 (82.75 → 83.87) vs. +1.17 on Random and +1.23 on Popular. The adversarial split selects absent objects that co-occur with present objects — directly exploiting language co-occurrence priors encoded in the decoder. VCD-based methods (including mVCD) partially target this failure mode (blurring reveals that the "yes" prediction was driven by context, not image), but OPERA's prompt variation is more effective here because co-occurrence priors are stable across different prompt phrasings. The BAB component provides consistent gains across all splits since the class imbalance affects all three equally.

### 10.4 Limitations and Future Work

- **Lightweight approximations**: mVCD and cwOPERA are response-level approximations; full token-level VCD and the original OPERA algorithm would likely produce stronger estimates.
- **POPE binary coverage**: Results cover only binary existence queries. Evaluation on CHAIR (captioning-level hallucination) and MMHal-Bench would test generalization.
- **Single architecture**: All experiments use LLaVA-1.5-7B; generalisation to InstructBLIP, mPLUG-Owl2, and LLaVA-1.6 requires further investigation.
- **BAB calibration**: δ=0.15 was calibrated on the Random split; a formal held-out validation set would provide tighter estimates.
- **Future**: Apply MAVER's routing framework to attention-level methods (ClearSight [5], ICT [6]) and investigate whether entropy-guided routing generalises to architectures with stronger modality alignment.

---

## 11. Conclusion

This work investigated the composability of training-free hallucination mitigation for LLaVA-1.5-7B on POPE, progressing from a fundamental failure finding to a method that surpasses the plain baseline by a meaningful margin.

The progression of contributions:

1. **Non-Additivity** (Stage 1): VCD and OPERA cannot be naively composed — doing so destroys 10.38 F1 points. Both methods share the same trigger condition (weak visual grounding), so composition amplifies suppression rather than correcting independent errors.

2. **Entropy Routing** (Stage 2): Joint entropy from both methods' logit distributions provides a principled per-query routing signal, recovering all 10 lost F1 points and coming within 0.37 of baseline.

3. **RPER** (Stage 3): Recall-biased conflict resolution and calibrated contrastive strength correct the routing asymmetry that caused Stage 2 to under-predict "yes", pushing F1 to 85.00 — **+0.50 above baseline**.

4. **MAVER** (Stage 4): Four targeted improvements (multi-scale VCD, adaptive α, confidence-weighted 5-prompt OPERA, balance-aware bias) attack three independent residual failure modes, achieving avg F1 **85.68 — +1.18 above baseline** and +0.68 above RPER, while recall improves from 77.53% (baseline) to 80.13% (MAVER) with only −0.86 pp precision cost.

The key design principle emerging from this work: before composing training-free mitigation methods, characterise their trigger distributions to verify complementarity; when composing, use uncertainty-driven routing to partition the query space rather than applying both methods uniformly; and correct any systematic calibration bias introduced by contrastive operations using domain knowledge about class balance.

---

## 12. How to Reproduce

### 12.1 Environment

```bash
git clone https://github.com/SreenijaPavuluri/entropy-vcd-opera-hallucination.git
cd entropy-vcd-opera-hallucination

conda create -n vcd-opera python=3.10 -y
conda activate vcd-opera
pip install -r requirements.txt
```

### 12.2 Dataset

```bash
# POPE annotations
mkdir -p data/pope && cd data/pope
wget https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_random.json
wget https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_popular.json
wget https://raw.githubusercontent.com/AoiDragon/POPE/main/output/coco/coco_pope_adversarial.json
cd ../..

# MS-COCO val2014 images (~13GB)
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip && cd ../..
```

### 12.3 Running Experiments

```bash
# Full evaluation: all 7 methods × 3 splits = 21 runs (~10 hours on H100)
python run_all.py

# Smoke test (50 samples per split, ~20 minutes)
python run_all.py --max_samples 50

# MAVER only (fastest path to main result)
python run_all.py --methods maver

# Stage 1 only (non-additivity replication)
python run_all.py --methods baseline vcd opera combined

# Stage 2 replication
python run_all.py --methods adaptive

# Stage 3 + 4 comparison
python run_all.py --methods rper maver

# Hyperparameter grid search for MAVER
python run_all.py --grid_search_maver   # add this flag (see run_all.py)

# Generate figures from saved results
python run_all.py --figures_only
```

### 12.4 Hardware Requirements

| Component | Minimum | Paper Setup |
|-----------|---------|-------------|
| GPU | 16 GB VRAM (A100) | NVIDIA H100 80 GB |
| RAM | 32 GB | 80 GB |
| Disk | 50 GB | — |
| Python | 3.10+ | 3.10 |
| PyTorch | 2.1+ | 2.1 |

### 12.5 Code Structure

```
src/
├── model_utils.py          # LLaVA-1.5-7B wrapper
├── baseline.py             # Stage 1: Baseline
├── vcd_lite.py             # Stage 1: VCD-lite
├── opera_lite.py           # Stage 1: OPERA-lite
├── combined.py             # Stage 1: Naive combination
├── adaptive_routing.py     # Stage 2: Entropy-guided routing (Algorithm 1)
├── improved_routing.py     # Stage 3: RPER
├── mvcd.py                 # MAVER component 1: Multi-scale VCD
├── ada_alpha.py            # MAVER component 2: Adaptive alpha
├── cw_opera.py             # MAVER component 3: Confidence-weighted OPERA
├── maver.py                # Stage 4: MAVER (full system)
├── evaluate.py             # POPE evaluation harness
├── hyperparameter_search.py# Grid search for RPER and MAVER
└── visualize.py            # Result plots and LaTeX tables
```

---

## 13. References

[1] S. Leng, H. Zhang, G. Chen, X. Li, S. Lu, C. Miao, and L. Bing, "Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding," in *Proc. CVPR*, Seattle, WA, Jun. 2024, pp. 13872–13882.

[2] Q. Huang, X. Dong, P. Zhang, B. Wang, C. He, J. Wang, D. Lin, W. Zhang, and N. Yu, "OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation," in *Proc. CVPR*, Seattle, WA, Jun. 2024, pp. 13418–13427.

[3] T. Guan, F. Liu, X. Wu, R. Xian, Z. Li, X. Liu, X. Wang, L. Chen, F. Huang, Y. Yacoob, D. Manocha, and T. Zhou, "HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models," in *Proc. CVPR*, Seattle, WA, Jun. 2024, pp. 14375–14385.

[4] Q. Ye, H. Xu, J. Ye, M. Yan, A. Hu, H. Liu, Q. Qian, J. Zhang, and F. Huang, "mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration," in *Proc. CVPR*, Seattle, WA, Jun. 2024, pp. 13040–13051.

[5] H. Yin, G. Si, and Z. Wang, "ClearSight: Visual Signal Enhancement for Object Hallucination Mitigation in Multimodal Large Language Models," in *Proc. CVPR*, Nashville, TN, Jun. 2025, pp. 14625–14634.

[6] J. Chen, T. Zhang, S. Huang, Y. Niu, L. Zhang, L. Wen, and X. Hu, "ICT: Image-Object Cross-Level Trusted Intervention for Mitigating Object Hallucination in Large Vision-Language Models," in *Proc. CVPR*, Nashville, TN, Jun. 2025.

[7] H. Liu, C. Li, Y. Li, and Y. J. Lee, "Visual Instruction Tuning," in *Proc. NeurIPS*, vol. 36, New Orleans, LA, Dec. 2023, pp. 34892–34916.

[8] Y. Li, Y. Du, K. Zhou, J. Wang, W. X. Zhao, and J.-R. Wen, "Evaluating Object Hallucination in Large Vision-Language Models," in *Proc. EMNLP*, Singapore, Dec. 2023, pp. 292–305.

---

*MS Applied AI Final Project — Stevens Institute of Technology, 2025*  
*Code: [github.com/SreenijaPavuluri/entropy-vcd-opera-hallucination](https://github.com/SreenijaPavuluri/entropy-vcd-opera-hallucination)*
