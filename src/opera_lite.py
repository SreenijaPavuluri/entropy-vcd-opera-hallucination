"""
Stage 1 – Condition 3: OPERA-lite
Approximation of OPERA (Huang et al., CVPR 2024) via three-prompt ensemble.

OPERA addresses hallucination tied to over-trust in summary tokens by applying
a retrospection-allocation penalty. OPERA-lite approximates this through majority
vote over three prompt variants, targeting decoding instability without
attention-level access. Implements Eq. (6).
"""

import numpy as np
from PIL import Image
from .model_utils import ModelWrapper


PROMPT_STYLES = ("standard", "strict", "truth")


def opera_logits(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
) -> tuple[float, float]:
    """
    OPERA ensemble logits as mean over three prompt variants (Eq. 6):

        ℓ^OPERA_t = (1/3) Σ_k ℓ^(k)_t

    Returns (l_yes_opera, l_no_opera).
    """
    yes_sum = 0.0
    no_sum  = 0.0
    for style in PROMPT_STYLES:
        l_yes, l_no = model.get_yn_logits(image, question, style)
        yes_sum += l_yes
        no_sum  += l_no
    return yes_sum / 3.0, no_sum / 3.0


def opera_predict(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
) -> str:
    """
    Majority vote over three prompt variants.
    Ties default to "yes" (recall-preserving).
    """
    votes = [model.predict(image, question, s) for s in PROMPT_STYLES]
    yes_count = votes.count("yes")
    return "yes" if yes_count >= 2 else "no"
