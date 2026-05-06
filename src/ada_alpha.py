"""
Blur-Effect Adaptive Alpha (adaAlpha)

Problem with fixed α
---------------------
Standard VCD uses a fixed contrastive strength α for all queries.  But the
magnitude of the blur effect — how much blurring changes the model's logits —
varies substantially across images and objects:

  • Fine-grained objects (bicycle spokes, small animals, text on signs):
    blur radius 5 dramatically changes logits → large blur_delta.
    With fixed α=0.25, the contrastive subtraction produces a very negative
    adjusted "yes" logit → false negative even when the object is present.

  • Coarse-grained objects (sky, sofa, large vehicle):
    blur barely changes logits → small blur_delta.
    Standard α works correctly here.

  • Language-prior-driven hallucinations:
    blurring doesn't help much at any scale (the prior is in the text stream).
    These are better handled by OPERA than by VCD.

Adaptive fix
------------
Measure blur_delta = max(|ℓ_yes_orig − ℓ_yes_blur|, |ℓ_no_orig − ℓ_no_blur|)
using the radius-5 blur as a reference.

Reduce α when blur_delta is large, preventing the contrastive operation from
over-amplifying already-large differences:

    α_q = α_base · clamp( 1 − λ · max(0, blur_delta − Δ_thresh) , 0.5, 1.0 )

Parameters:
    α_base    = 0.25   (calibrated base; lower than paper's 0.30)
    λ         = 0.10   (sensitivity of reduction per unit of excess delta)
    Δ_thresh  = 2.50   (nats; excess beyond this triggers reduction)
    clamp min = 0.50   (never reduce α by more than 50%)

Effect: queries where blur has extreme effect (fine-grained objects) get a
gentler contrastive operation.  Queries where blur has small effect
(language-prior cases) keep full α.  This selectively reduces false negatives
without introducing false positives.

This function is designed to be used *before* calling mvcd_logits() so that
the per-query α is substituted in.
"""

import numpy as np
from PIL import Image
from .model_utils import ModelWrapper


def compute_adaptive_alpha(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
    alpha_base: float = 0.25,
    sensitivity: float = 0.10,
    delta_thresh: float = 2.50,
    min_alpha_fraction: float = 0.50,
    reference_radius: int = 5,
) -> tuple[float, float]:
    """
    Compute per-query adaptive contrastive strength α_q.

    Returns
    -------
    (alpha_q, blur_delta)
        alpha_q   : adaptive α to use for this query
        blur_delta: the measured blur effect magnitude (for diagnostics)
    """
    l_yes_orig, l_no_orig = model.get_yn_logits(image, question, "standard")
    blurred = ModelWrapper.blur_image(image, radius=reference_radius)
    l_yes_blur, l_no_blur = model.get_yn_logits(blurred, question, "standard")

    blur_delta = max(
        abs(l_yes_orig - l_yes_blur),
        abs(l_no_orig  - l_no_blur),
    )

    excess = max(0.0, blur_delta - delta_thresh)
    reduction = sensitivity * excess
    scale = max(min_alpha_fraction, 1.0 - reduction)

    alpha_q = alpha_base * scale
    return alpha_q, blur_delta
