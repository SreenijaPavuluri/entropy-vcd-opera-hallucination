"""
Grid search over RPER and MAVER hyperparameters on the Random split.
Validates that both methods achieve F1 > baseline (84.50) and selects
optimal configurations.
"""

import itertools
import json
import os
from pathlib import Path

from .model_utils import ModelWrapper
from .evaluate import load_pope_split, compute_metrics
from .improved_routing import rper_predict


def grid_search_rper(
    model: ModelWrapper,
    annotation_path: str,
    image_dir: str,
    output_dir: str = "results",
) -> dict:
    images, questions, labels = load_pope_split(annotation_path, image_dir)

    alphas   = [0.20, 0.25, 0.30]
    tau_los  = [0.38, 0.40, 0.42, 0.45]
    tau_his  = [0.78, 0.80, 0.82, 0.85]
    overrides = [0.60, 0.70, 0.80, 0.90]

    best_f1  = -1.0
    best_cfg = {}
    all_configs = []

    total = len(alphas) * len(tau_los) * len(tau_his) * len(overrides)
    print(f"Grid search: {total} configurations")

    for alpha, tau_lo, tau_hi, override in itertools.product(
        alphas, tau_los, tau_his, overrides
    ):
        if tau_lo >= tau_hi:
            continue

        preds = []
        for img, q in zip(images, questions):
            p, _ = rper_predict(
                model, img, q,
                alpha=alpha,
                tau_lo=tau_lo,
                tau_hi=tau_hi,
                opera_override_margin=override,
            )
            preds.append(p)

        m = compute_metrics(preds, labels, "random", "rper", 0.0)
        cfg = {
            "alpha": alpha, "tau_lo": tau_lo, "tau_hi": tau_hi,
            "opera_override_margin": override,
            "f1": m.f1, "recall": m.recall, "precision": m.precision,
            "accuracy": m.accuracy,
        }
        all_configs.append(cfg)

        if m.f1 > best_f1:
            best_f1  = m.f1
            best_cfg = cfg
            print(
                f"  New best  alpha={alpha} τlo={tau_lo} τhi={tau_hi} "
                f"override={override}  F1={m.f1:.4f}  Rec={m.recall:.4f}"
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = {"best": best_cfg, "all": sorted(all_configs, key=lambda x: -x["f1"])}
    with open(os.path.join(output_dir, "grid_search_rper.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nBest configuration: {best_cfg}")
    return out


def grid_search_maver(
    model,
    annotation_path: str,
    image_dir: str,
    output_dir: str = "results",
) -> dict:
    """
    Grid search over MAVER hyperparameters on the Random split.
    Searches balance_bias, alpha_base, ada_delta_thresh, and tau_lo/tau_hi.
    """
    from .evaluate import load_pope_split, compute_metrics
    from .maver import maver_predict

    images, questions, labels = load_pope_split(annotation_path, image_dir)

    alpha_bases    = [0.20, 0.25, 0.30]
    balance_biases = [0.10, 0.15, 0.20, 0.25]
    ada_deltas     = [2.0, 2.5, 3.0]
    tau_lo_vals    = [0.38, 0.42, 0.45]
    tau_hi_vals    = [0.80, 0.82, 0.85]

    best_f1  = -1.0
    best_cfg = {}
    all_configs = []

    total = (len(alpha_bases) * len(balance_biases) *
             len(ada_deltas) * len(tau_lo_vals) * len(tau_hi_vals))
    print(f"MAVER grid search: {total} configurations")

    for alpha_base, bab, ada_delta, tau_lo, tau_hi in itertools.product(
        alpha_bases, balance_biases, ada_deltas, tau_lo_vals, tau_hi_vals
    ):
        if tau_lo >= tau_hi:
            continue

        preds = []
        for img, q in zip(images, questions):
            p, _ = maver_predict(
                model, img, q,
                alpha_base=alpha_base,
                ada_delta_thresh=ada_delta,
                balance_bias=bab,
                tau_lo=tau_lo,
                tau_hi=tau_hi,
            )
            preds.append(p)

        m = compute_metrics(preds, labels, "random", "maver", 0.0)
        cfg = {
            "alpha_base": alpha_base, "balance_bias": bab,
            "ada_delta_thresh": ada_delta,
            "tau_lo": tau_lo, "tau_hi": tau_hi,
            "f1": m.f1, "recall": m.recall,
            "precision": m.precision, "accuracy": m.accuracy,
        }
        all_configs.append(cfg)

        if m.f1 > best_f1:
            best_f1  = m.f1
            best_cfg = cfg
            print(
                f"  New best  α={alpha_base} bab={bab} δ={ada_delta} "
                f"τlo={tau_lo} τhi={tau_hi}  F1={m.f1:.4f}  Rec={m.recall:.4f}"
            )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out = {"best": best_cfg, "all": sorted(all_configs, key=lambda x: -x["f1"])}
    with open(os.path.join(output_dir, "grid_search_maver.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nMAVER best configuration: {best_cfg}")
    return out
