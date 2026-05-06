"""
Confidence-Weighted OPERA (cwOPERA)

Problem with equal-weight 3-prompt ensemble
---------------------------------------------
OPERA-lite averages logits across three prompt variants with equal weight:

    ℓ^OPERA_t = (1/3) Σ_k ℓ^(k)_t

Equal weighting treats all prompt variants identically, but in practice some
prompts produce near-random logits for certain query types (the phrasing
confuses the model for that specific question).  A prompt that produces a
near-zero margin (logit_yes ≈ logit_no) carries no useful signal but still
contributes 1/3 of the ensemble weight, diluting the confident prompts.

Two additional problems with only 3 prompts:
  1. Small sample: three predictions can only produce majority outcomes
     (2-1 or 3-0); there is no mechanism to weigh partial confidence.
  2. The three standard prompts cover only command-style variation
     ("answer yes or no", "only yes/no", "truthfully").  Adding
     instruction-following and verification styles broadens the coverage
     of phrasing space.

Confidence-weighted extension
-------------------------------
Use 5 prompt variants and weight each by its logit margin:

    conf_k = |ℓ^(k)_yes − ℓ^(k)_no|        (0 = random, large = confident)
    w_k    = conf_k / Σ_j conf_j            (normalised)
    ℓ^cwOPERA_t = Σ_k  w_k · ℓ^(k)_t

Effect:
  • Prompts where the model is confident contribute more to the ensemble.
  • Near-zero-margin (confused) prompts are automatically down-weighted.
  • Two new prompts add "visual inspection" and "careful observation" styles
    that capture different aspects of instruction-following behaviour.

The result is a more stable OPERA logit estimate, especially for the
medium-confidence band where ensemble quality most affects routing decisions.

Five prompt variants
---------------------
1. Standard      : "Answer yes or no."
2. Strict        : "Respond with only 'yes' or 'no'. No other words."
3. Truth-framed  : "Answer truthfully ... based strictly on what is visible."
4. Careful       : "Look carefully at the image. Answer yes or no."
5. Verification  : "To verify: is this object actually present? Answer yes or no."
"""

import numpy as np
from PIL import Image
from .model_utils import ModelWrapper


_PROMPT_STYLES_5 = ("standard", "strict", "truth", "careful", "verification")


class ModelWrapperWithExtraPrompts(ModelWrapper):
    """
    Adds two additional prompt styles used by cwOPERA.
    Extends ModelWrapper.get_yn_logits() with "careful" and "verification".
    """

    def _build_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"

    def _build_prompt_careful(self, question: str) -> str:
        return (
            f"USER: <image>\n{question}\n"
            "Look carefully at the image before answering. Answer yes or no.\nASSISTANT:"
        )

    def _build_prompt_verification(self, question: str) -> str:
        return (
            f"USER: <image>\n{question}\n"
            "To verify: is this object actually visible in the image? Answer yes or no.\nASSISTANT:"
        )

    def get_yn_logits(self, image, question, prompt_style="standard"):
        if prompt_style == "careful":
            import torch
            prompt  = self._build_prompt_careful(question)
            inputs  = self._inputs(image, prompt)
            with torch.no_grad():
                out = self.model(**inputs)
            logits = out.logits[0, -1, :].float()
            l_yes = max(logits[i].item() for i in self.yes_ids)
            l_no  = max(logits[i].item() for i in self.no_ids)
            return l_yes, l_no

        elif prompt_style == "verification":
            import torch
            prompt  = self._build_prompt_verification(question)
            inputs  = self._inputs(image, prompt)
            with torch.no_grad():
                out = self.model(**inputs)
            logits = out.logits[0, -1, :].float()
            l_yes = max(logits[i].item() for i in self.yes_ids)
            l_no  = max(logits[i].item() for i in self.no_ids)
            return l_yes, l_no

        else:
            return super().get_yn_logits(image, question, prompt_style)


def cw_opera_logits(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
) -> tuple[float, float]:
    """
    Confidence-weighted 5-prompt OPERA logits.

        conf_k     = |ℓ^(k)_yes − ℓ^(k)_no|
        w_k        = conf_k / Σ_j conf_j
        ℓ^cwOPERA  = Σ_k w_k · ℓ^(k)

    Returns (l_yes_cwopera, l_no_cwopera).
    Falls back gracefully if model doesn't support all 5 prompt styles.
    """
    logits_per_prompt: list[tuple[float, float]] = []

    for style in _PROMPT_STYLES_5:
        try:
            l_yes, l_no = model.get_yn_logits(image, question, style)
            logits_per_prompt.append((l_yes, l_no))
        except (ValueError, AttributeError):
            # model doesn't support this prompt style → skip
            pass

    if not logits_per_prompt:
        raise RuntimeError("No prompt styles produced logits in cw_opera_logits")

    # Confidence = logit margin magnitude
    confs = [abs(l_y - l_n) for l_y, l_n in logits_per_prompt]
    total_conf = sum(confs) + 1e-9
    weights = [c / total_conf for c in confs]

    l_yes_cw = sum(w * l_y for w, (l_y, _) in zip(weights, logits_per_prompt))
    l_no_cw  = sum(w * l_n for w, (_, l_n) in zip(weights, logits_per_prompt))

    return l_yes_cw, l_no_cw


def cw_opera_predict(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
) -> str:
    l_yes, l_no = cw_opera_logits(model, image, question)
    return "yes" if l_yes >= l_no else "no"
