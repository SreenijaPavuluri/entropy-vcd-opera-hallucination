"""
Multi-Scale VCD (mVCD)

Problem with single-scale VCD
------------------------------
Standard VCD-lite blurs with a fixed radius-5 Gaussian and subtracts the
resulting logits with a constant α=0.30.  Two failure modes emerge:

  FN failure: for a query where the object is genuinely present, a radius-5
  blur can occasionally collapse texture cues that are diagnostic for that
  specific object (e.g., thin structures like bicycle spokes, fine-grained
  text).  At that single scale the blurred model confidently says "no", so
  the contrastive operation produces a strongly negative adjusted logit for
  "yes" → false negative.

  FP failure (less common): for a query driven by language priors, some blur
  levels do not fully suppress the prior-driven "yes" because the prior
  emerges from context tokens, not image tokens.  A single blur level may
  not expose this reliably.

Multi-scale approach
---------------------
Compute contrastive logits under N blur radii and take a weighted average of
the blur contributions.  Smaller radii preserve more texture (reducing FN
failures on fine-grained objects); larger radii aggressively remove visual
content (better at exposing language-prior-only predictions).

    ℓ^mVCD_t = ℓ^orig_t − α · Σ_r  w_r · ℓ^blur_r_t

Weights w_r (sum to 1) emphasise smaller perturbations:
    radii = [3, 5, 8],  w = [0.50, 0.35, 0.15]

Using 3 forward passes (instead of 1 for single-scale VCD) keeps the
overhead manageable.  The weighted average blur logit is more stable than
any individual scale, reducing per-query variance in the contrastive signal.

This directly reduces the FN rate: fewer genuine objects are accidentally
classified as "no" because an extreme single-scale blur collapsed their cue.
"""

import numpy as np
from PIL import Image
from .model_utils import ModelWrapper

BLUR_RADII   = [3, 5, 8]
BLUR_WEIGHTS = [0.50, 0.35, 0.15]   # emphasise subtle perturbation


def mvcd_logits(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
    alpha: float = 0.25,
    radii: list[int]   = None,
    weights: list[float] = None,
) -> tuple[float, float]:
    """
    Multi-scale VCD contrastive logits.

        ℓ^mVCD_t = ℓ^orig_t  −  α · Σ_r w_r · ℓ^blur_r_t

    Returns (l_yes_mvcd, l_no_mvcd).

    Parameters
    ----------
    alpha   : contrastive strength (per-query adaptive version in maver.py)
    radii   : blur radii list; defaults to BLUR_RADII = [3, 5, 8]
    weights : per-radius weights; defaults to BLUR_WEIGHTS = [0.50, 0.35, 0.15]
    """
    if radii   is None: radii   = BLUR_RADII
    if weights is None: weights = BLUR_WEIGHTS

    assert len(radii) == len(weights), "radii and weights must have the same length"
    w_total = sum(weights)
    w_norm  = [w / w_total for w in weights]

    l_yes_orig, l_no_orig = model.get_yn_logits(image, question, "standard")

    avg_blur_yes = 0.0
    avg_blur_no  = 0.0
    for r, w in zip(radii, w_norm):
        blurred = ModelWrapper.blur_image(image, radius=r)
        l_y, l_n = model.get_yn_logits(blurred, question, "standard")
        avg_blur_yes += w * l_y
        avg_blur_no  += w * l_n

    l_yes_mvcd = l_yes_orig - alpha * avg_blur_yes
    l_no_mvcd  = l_no_orig  - alpha * avg_blur_no
    return l_yes_mvcd, l_no_mvcd


def mvcd_predict(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
    alpha: float = 0.25,
) -> str:
    l_yes, l_no = mvcd_logits(model, image, question, alpha)
    return "yes" if l_yes >= l_no else "no"
