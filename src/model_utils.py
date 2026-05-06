"""
LLaVA-1.5-7B loading and inference utilities.
Provides a single ModelWrapper used by all methods.
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration


LLAVA_MODEL_ID = "llava-hf/llava-1.5-7b-hf"

YES_TOKENS = ["yes", "Yes", "YES"]
NO_TOKENS  = ["no",  "No",  "NO"]


class ModelWrapper:
    """
    Thin wrapper around LLaVA-1.5-7B.

    Exposes:
        predict(image, question)            -> "yes" | "no"
        get_yn_logits(image, question)      -> (logit_yes, logit_no)  float32
    """

    def __init__(self, model_id: str = LLAVA_MODEL_ID, device: str = "cuda"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

        tok = self.processor.tokenizer
        self.yes_ids = [tok.encode(t, add_special_tokens=False)[0] for t in YES_TOKENS]
        self.no_ids  = [tok.encode(t, add_special_tokens=False)[0] for t in NO_TOKENS]

    # ------------------------------------------------------------------
    def _build_prompt(self, question: str) -> str:
        return f"USER: <image>\n{question}\nAnswer yes or no.\nASSISTANT:"

    def _build_prompt_strict(self, question: str) -> str:
        return (
            f"USER: <image>\n{question}\n"
            "Respond with only 'yes' or 'no'. No other words.\nASSISTANT:"
        )

    def _build_prompt_truth(self, question: str) -> str:
        return (
            f"USER: <image>\n{question}\n"
            "Answer truthfully with yes or no based strictly on what is visible.\nASSISTANT:"
        )

    def _inputs(self, image: Image.Image, prompt: str):
        return self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device, torch.float16)

    # ------------------------------------------------------------------
    def get_yn_logits(
        self,
        image: Image.Image,
        question: str,
        prompt_style: str = "standard",
    ) -> tuple[float, float]:
        """
        Returns (logit_yes, logit_no) from the last token position.
        Uses the maximum logit across all yes/no surface forms.
        """
        if prompt_style == "standard":
            prompt = self._build_prompt(question)
        elif prompt_style == "strict":
            prompt = self._build_prompt_strict(question)
        elif prompt_style == "truth":
            prompt = self._build_prompt_truth(question)
        else:
            raise ValueError(f"Unknown prompt_style: {prompt_style}")

        inputs = self._inputs(image, prompt)
        with torch.no_grad():
            out = self.model(**inputs)

        logits = out.logits[0, -1, :].float()
        l_yes = max(logits[i].item() for i in self.yes_ids)
        l_no  = max(logits[i].item() for i in self.no_ids)
        return l_yes, l_no

    def predict(
        self,
        image: Image.Image,
        question: str,
        prompt_style: str = "standard",
    ) -> str:
        l_yes, l_no = self.get_yn_logits(image, question, prompt_style)
        return "yes" if l_yes >= l_no else "no"

    # ------------------------------------------------------------------
    @staticmethod
    def blur_image(image: Image.Image, radius: int = 5) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
