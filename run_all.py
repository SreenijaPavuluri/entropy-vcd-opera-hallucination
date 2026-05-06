#!/usr/bin/env python3
"""
Master runner: reproduces all results from the paper and runs the
improved RPER method.

Usage
-----
# Full evaluation (all methods, all splits, 9000 queries — needs H100):
python run_all.py

# Quick smoke-test (50 samples per split):
python run_all.py --max_samples 50

# Only the improved method:
python run_all.py --methods rper

# Grid search for RPER hyperparameters on Random split:
python run_all.py --grid_search

# Generate figures from saved results:
python run_all.py --figures_only
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.model_utils import ModelWrapper
from src.evaluate import run_full_evaluation, evaluate_split
from src.hyperparameter_search import grid_search_rper
from src.visualize import generate_all


def parse_args():
    p = argparse.ArgumentParser(description="Entropy-VCD-OPERA evaluation")
    p.add_argument("--data_root",    default="data",    help="Root containing pope/ and coco/")
    p.add_argument("--output_dir",   default="results", help="Output directory for results")
    p.add_argument("--figures_dir",  default="figures", help="Output directory for figures")
    p.add_argument("--model_id",     default="llava-hf/llava-1.5-7b-hf")
    p.add_argument("--device",       default="cuda",    choices=["cuda", "cpu", "mps"])
    p.add_argument("--max_samples",  type=int, default=None,
                   help="Limit samples per split (None = full 3000)")
    p.add_argument("--methods",      nargs="+",
                   default=["baseline", "vcd", "opera", "combined", "adaptive", "rper", "maver"],
                   help="Methods to evaluate")
    p.add_argument("--splits",       nargs="+",
                   default=["random", "popular", "adversarial"])
    p.add_argument("--grid_search",  action="store_true",
                   help="Run RPER hyperparameter grid search on Random split")
    p.add_argument("--figures_only", action="store_true",
                   help="Skip evaluation, generate figures from saved results")
    return p.parse_args()


def main():
    args = parse_args()

    if args.figures_only:
        results_path = os.path.join(args.output_dir, "all_results.json")
        if not os.path.exists(results_path):
            print(f"Error: {results_path} not found. Run evaluation first.")
            sys.exit(1)
        generate_all(results_path, args.figures_dir)
        return

    print(f"Loading LLaVA-1.5-7B on {args.device} ...")
    model = ModelWrapper(model_id=args.model_id, device=args.device)
    print("Model loaded.\n")

    if args.grid_search:
        print("=== RPER Hyperparameter Grid Search ===")
        pope_dir = os.path.join(args.data_root, "pope")
        image_dir = os.path.join(args.data_root, "coco", "val2014")
        grid_search_rper(
            model,
            annotation_path=os.path.join(pope_dir, "coco_pope_random.json"),
            image_dir=image_dir,
            output_dir=args.output_dir,
        )
        return

    print("=== Running Full Evaluation ===")
    print(f"Methods : {args.methods}")
    print(f"Splits  : {args.splits}")
    print(f"Samples : {args.max_samples or 'all (3000 per split)'}\n")

    from src.evaluate import load_pope_split, compute_metrics, Metrics
    from dataclasses import asdict
    import json, csv, time
    from pathlib import Path
    from tqdm import tqdm

    splits_map = {
        "random":      os.path.join(args.data_root, "pope", "coco_pope_random.json"),
        "popular":     os.path.join(args.data_root, "pope", "coco_pope_popular.json"),
        "adversarial": os.path.join(args.data_root, "pope", "coco_pope_adversarial.json"),
    }
    image_dir = os.path.join(args.data_root, "coco", "val2014")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    from src.baseline       import baseline_predict
    from src.vcd_lite       import vcd_predict
    from src.opera_lite     import opera_predict
    from src.combined       import combined_predict
    from src.adaptive_routing import adaptive_predict
    from src.improved_routing import rper_predict
    from src.maver import maver_predict

    all_results = []

    for split_name in args.splits:
        ann_path = splits_map[split_name]
        images, questions, labels = load_pope_split(ann_path, image_dir)
        if args.max_samples:
            images    = images[:args.max_samples]
            questions = questions[:args.max_samples]
            labels    = labels[:args.max_samples]

        for method in args.methods:
            print(f"\n[{split_name.upper()}] {method}  ({len(images)} samples)")
            preds = []
            t0 = time.time()
            for img, q in tqdm(zip(images, questions), total=len(images), desc=method):
                if   method == "baseline":  p = baseline_predict(model, img, q)
                elif method == "vcd":       p = vcd_predict(model, img, q)
                elif method == "opera":     p = opera_predict(model, img, q)
                elif method == "combined":  p = combined_predict(model, img, q)
                elif method == "adaptive":  p, _ = adaptive_predict(model, img, q)
                elif method == "rper":      p, _ = rper_predict(model, img, q)
                elif method == "maver":     p, _ = maver_predict(model, img, q)
                else: raise ValueError(f"Unknown method: {method}")
                preds.append(p)

            elapsed = time.time() - t0
            m = compute_metrics(preds, labels, split_name, method, elapsed)
            all_results.append(asdict(m))
            print(
                f"  Acc={m.accuracy:.2f}  Prec={m.precision:.2f}  "
                f"Rec={m.recall:.2f}  F1={m.f1:.2f}  YesRatio={m.yes_ratio:.2f}  "
                f"({elapsed:.0f}s)"
            )

    # Save
    json_path = os.path.join(args.output_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    csv_path = os.path.join(args.output_dir, "metrics_summary.csv")
    if all_results:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"CSV saved      → {csv_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Split':<12} {'Method':<15} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'YesR':>7}")
    print("="*80)
    for r in all_results:
        print(
            f"{r['split']:<12} {r['method']:<15} "
            f"{r['accuracy']:>7.2f} {r['precision']:>7.2f} "
            f"{r['recall']:>7.2f} {r['f1']:>7.2f} {r['yes_ratio']:>7.2f}"
        )

    generate_all(json_path, args.figures_dir)


if __name__ == "__main__":
    main()
