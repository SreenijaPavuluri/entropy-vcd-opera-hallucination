"""
Stage 2 – Entropy-Guided Adaptive Routing (paper method).

Implements Algorithm 1 from the paper exactly:
  - Joint binary entropy from VCD and OPERA logit distributions
  - Hard routing with three regimes: high / medium / low confidence
  - Agreement-based conflict resolution in medium regime

Achieves avg F1 0.8413 across all POPE splits (paper Table II).
"""

import numpy as np
from PIL import Image
from .model_utils import ModelWrapper
from .vcd_lite import vcd_logits
from .opera_lite import opera_logits


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _binary_entropy(p: float, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def compute_joint_entropy(
    l_yes_vcd: float,
    l_no_vcd: float,
    l_yes_opera: float,
    l_no_opera: float,
) -> tuple[float, float, float]:
    """
    Compute joint entropy H = (H_VCD + H_OPERA) / 2 per Eqs. (7)–(8).
    Also returns p_yes for VCD and OPERA for downstream use.
    """
    p_vcd   = _sigmoid(l_yes_vcd  - l_no_vcd)
    p_opera = _sigmoid(l_yes_opera - l_no_opera)

    h_vcd   = _binary_entropy(p_vcd)
    h_opera = _binary_entropy(p_opera)
    h_joint = (h_vcd + h_opera) / 2.0
    return h_joint, p_vcd, p_opera


def adaptive_predict(
    model: ModelWrapper,
    image: Image.Image,
    question: str,
    alpha: float = 0.3,
    tau_lo: float = 0.45,
    tau_hi: float = 0.80,
) -> tuple[str, str]:
    """
    Entropy-guided hard routing (Algorithm 1).

    Returns (prediction, routing_regime) where routing_regime is one of:
      "vcd_strong"  – high confidence, routed to VCD
      "agreement"   – medium confidence, both methods agree
      "vcd_wins"    – medium confidence conflict, VCD margin stronger
      "opera_wins"  – medium confidence conflict, OPERA margin stronger
      "opera_low"   – low confidence, routed to OPERA
    """
    l_yes_vcd,   l_no_vcd   = vcd_logits(model, image, question, alpha)
    l_yes_opera, l_no_opera = opera_logits(model, image, question)

    h, p_vcd, p_opera = compute_joint_entropy(
        l_yes_vcd, l_no_vcd, l_yes_opera, l_no_opera
    )

    y_vcd   = "yes" if l_yes_vcd   >= l_no_vcd   else "no"
    y_opera = "yes" if l_yes_opera >= l_no_opera  else "no"

    if h < tau_lo:
        return y_vcd, "vcd_strong"

    elif h < tau_hi:
        if y_vcd == y_opera:
            return y_vcd, "agreement"
        else:
            vcd_margin   = abs(l_yes_vcd   - l_no_vcd)
            opera_margin = abs(l_yes_opera - l_no_opera)
            if vcd_margin >= opera_margin:
                return y_vcd, "vcd_wins"
            else:
                return y_opera, "opera_wins"

    else:
        return y_opera, "opera_low"
