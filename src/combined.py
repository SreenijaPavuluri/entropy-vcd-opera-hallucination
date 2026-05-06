"""
Stage 1 – Condition 4: Naive combination of VCD-lite and OPERA-lite.
Applies both methods sequentially; confirms the non-additivity finding.
"""

from PIL import Image
from .model_utils import ModelWrapper
from .vcd_lite import vcd_predict
from .opera_lite import opera_predict


def combined_predict(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
    alpha: float = 0.3,
) -> str:
    """
    Sequential application: VCD-lite first, then OPERA majority vote.
    Both methods must agree on "yes" for the answer to be "yes".
    Confirmed to degrade F1 by ~10 points vs. baseline (paper Table I).
    """
    vcd_ans   = vcd_predict(model, image, question, alpha)
    opera_ans = opera_predict(model, image, question)

    if vcd_ans == "yes" and opera_ans == "yes":
        return "yes"
    return "no"
