"""
Stage 1 – Condition 1: Baseline
Direct inference with LLaVA-1.5-7B using the standard prompt.
"""

from PIL import Image
from .model_utils import ModelWrapper


def baseline_predict(model: ModelWrapper, image: Image.Image, question: str) -> str:
    """
    Plain argmax over yes/no logits under the standard prompt.
    Returns "yes" or "no".
    """
    return model.predict(image, question, prompt_style="standard")
