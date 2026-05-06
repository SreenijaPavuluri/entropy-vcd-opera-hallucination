"""
POPE benchmark evaluation harness.

Loads any split (random / popular / adversarial), runs all methods:
  Stage 1 : baseline, vcd, opera, combined
  Stage 2 : adaptive (entropy-guided routing, paper Algorithm 1)
  Stage 3 : rper (Recall-Preserved Entropy Routing)
  Stage 4 : maver (Multi-Scale Adaptive VCD with Extended Routing)

and saves results to JSON + a metrics CSV.
"""

import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from tqdm import tqdm

from .model_utils import ModelWrapper
from .baseline import baseline_predict
from .vcd_lite import vcd_predict
from .opera_lite import opera_predict
from .combined import combined_predict
from .adaptive_routing import adaptive_predict
from .improved_routing import rper_predict
from .maver import maver_predict


@dataclass
class Metrics:
    split: str
    method: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    yes_ratio: float
    tp: int
    fp: int
    tn: int
    fn: int
    n_total: int
    elapsed_sec: float


def compute_metrics(
    preds: list[str],
    labels: list[int],
    split: str,
    method: str,
    elapsed: float,
) -> Metrics:
    tp = fp = tn = fn = 0
    for p, l in zip(preds, labels):
        yhat = 1 if p == "yes" else 0
        if   yhat == 1 and l == 1: tp += 1
        elif yhat == 1 and l == 0: fp += 1
        elif yhat == 0 and l == 0: tn += 1
        else:                       fn += 1

    n = len(preds)
    acc   = (tp + tn) / n
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    yrat  = sum(1 for p in preds if p == "yes") / n

    return Metrics(
        split=split, method=method,
        accuracy=round(acc  * 100, 2),
        precision=round(prec * 100, 2),
        recall=round(rec   * 100, 2),
        f1=round(f1        * 100, 2),
        yes_ratio=round(yrat * 100, 2),
        tp=tp, fp=fp, tn=tn, fn=fn,
        n_total=n, elapsed_sec=round(elapsed, 1),
    )


def load_pope_split(
    annotation_path: str,
    image_dir: str,
) -> tuple[list[Image.Image], list[str], list[int]]:
    with open(annotation_path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    images, questions, labels = [], [], []
    for item in data:
        fname = item["image"]
        img_path = os.path.join(image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        questions.append(item["text"])
        labels.append(1 if item["label"].lower() == "yes" else 0)

    return images, questions, labels


def evaluate_split(
    model: ModelWrapper,
    annotation_path: str,
    image_dir: str,
    split_name: str,
    methods: list[str] | None = None,
    max_samples: int | None = None,
) -> list[Metrics]:
    """
    Evaluate all requested methods on a single POPE split.

    methods: list from {"baseline","vcd","opera","combined","adaptive","rper","maver"}
             defaults to all.
    """
    if methods is None:
        methods = ["baseline", "vcd", "opera", "combined", "adaptive", "rper", "maver"]

    images, questions, labels = load_pope_split(annotation_path, image_dir)

    if max_samples is not None:
        images    = images[:max_samples]
        questions = questions[:max_samples]
        labels    = labels[:max_samples]

    all_metrics = []

    for method in methods:
        print(f"\n[{split_name}] Running method: {method} ({len(images)} samples)")
        preds = []
        t0 = time.time()

        for img, q in tqdm(zip(images, questions), total=len(images), desc=method):
            if method == "baseline":
                p = baseline_predict(model, img, q)
            elif method == "vcd":
                p = vcd_predict(model, img, q)
            elif method == "opera":
                p = opera_predict(model, img, q)
            elif method == "combined":
                p = combined_predict(model, img, q)
            elif method == "adaptive":
                p, _ = adaptive_predict(model, img, q)
            elif method == "rper":
                p, _ = rper_predict(model, img, q)
            elif method == "maver":
                p, _ = maver_predict(model, img, q)
            else:
                raise ValueError(f"Unknown method: {method}")

            preds.append(p)

        elapsed = time.time() - t0
        m = compute_metrics(preds, labels, split_name, method, elapsed)
        all_metrics.append(m)
        print(
            f"  Acc={m.accuracy:.2f}  Prec={m.precision:.2f}  "
            f"Rec={m.recall:.2f}  F1={m.f1:.2f}  YesRatio={m.yes_ratio:.2f}"
        )

    return all_metrics


def run_full_evaluation(
    model: ModelWrapper,
    data_root: str = "data",
    output_dir: str = "results",
    max_samples: int | None = None,
) -> dict:
    """
    Run all methods on all three POPE splits.
    Saves results/all_results.json and results/metrics_summary.csv.
    """
    pope_dir  = os.path.join(data_root, "pope")
    image_dir = os.path.join(data_root, "coco", "val2014")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    splits = {
        "random":     os.path.join(pope_dir, "coco_pope_random.json"),
        "popular":    os.path.join(pope_dir, "coco_pope_popular.json"),
        "adversarial": os.path.join(pope_dir, "coco_pope_adversarial.json"),
    }

    all_results: list[Metrics] = []

    for split_name, ann_path in splits.items():
        results = evaluate_split(
            model, ann_path, image_dir, split_name, max_samples=max_samples
        )
        all_results.extend(results)

    # Save JSON
    json_path = os.path.join(output_dir, "all_results.json")
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"\nResults saved to {json_path}")

    # Save CSV
    import csv
    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    fields = list(asdict(all_results[0]).keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            w.writerow(asdict(r))
    print(f"CSV saved to {csv_path}")

    return {"results": [asdict(r) for r in all_results]}
