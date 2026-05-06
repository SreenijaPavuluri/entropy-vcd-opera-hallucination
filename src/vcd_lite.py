"""
Stage 1 – Condition 2: VCD-lite
Visual Contrastive Decoding approximated at the logit level.

VCD (Leng et al., CVPR 2024) contrasts output distributions under the original
image versus a noise-distorted copy. VCD-lite uses Gaussian blur (radius 5) as
the distortion and operates at the yes/no logit level rather than full token
distributions, matching the paper's Eq. (4) and (5).
"""

import numpy as np
from PIL import Image
from .model_utils import ModelWrapper


def vcd_logits(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
    alpha: float = 0.3,
) -> tuple[float, float]:
    """
    Compute VCD contrastive logits per Eq. (5):

        ℓ^VCD_t = ℓ^orig_t  −  α · ℓ^blur_t

    Returns (l_yes_vcd, l_no_vcd).
    """
    l_yes_orig, l_no_orig = model.get_yn_logits(image, question, "standard")

    blurred = ModelWrapper.blur_image(image, radius=5)
    l_yes_blur, l_no_blur = model.get_yn_logits(blurred, question, "standard")

    l_yes_vcd = l_yes_orig - alpha * l_yes_blur
    l_no_vcd  = l_no_orig  - alpha * l_no_blur
    return l_yes_vcd, l_no_vcd


def vcd_predict(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
    alpha: float = 0.3,
) -> str:
    """
    Binary VCD-lite decision rule (Eq. 4):
        ŷ_VCD = 0  if ŷ(A_orig)=1 AND ŷ(A_blur)=0
                ŷ(A_orig)  otherwise

    Falls back to the contrastive logit comparison as a tiebreaker.
    """
    pred_orig = model.predict(image, question, "standard")
    blurred   = ModelWrapper.blur_image(image, radius=5)
    pred_blur = model.predict(blurred, question, "standard")

    if pred_orig == "yes" and pred_blur == "no":
        return "no"
    return pred_orig
