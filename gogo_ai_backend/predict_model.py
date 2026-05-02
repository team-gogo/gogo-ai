import asyncio
import logging
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class ModelService:
    HF_MODEL = "kdyeon0309/gogo_forpanity_filter"
    TOKENIZER = "beomi/KcELECTRA-base"

    _model: Optional[AutoModelForSequenceClassification] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _device: Optional[torch.device] = None
    _load_lock = asyncio.Lock()

    @classmethod
    async def load(cls) -> None:
        async with cls._load_lock:
            if cls._model is not None:
                return

            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Loading profanity model '{cls.HF_MODEL}' on {cls._device}")

            def _load_blocking():
                model = AutoModelForSequenceClassification.from_pretrained(
                    cls.HF_MODEL, trust_remote_code=True, use_auth_token=False
                )
                tokenizer = AutoTokenizer.from_pretrained(cls.TOKENIZER)
                model.to(cls._device)
                model.eval()
                return model, tokenizer

            cls._model, cls._tokenizer = await asyncio.to_thread(_load_blocking)
            logging.info("Profanity model loaded")

    @classmethod
    def _predict_blocking(cls, sentence: str) -> int:
        tokenized_input = cls._tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128,
        ).to(cls._device)

        with torch.no_grad():
            outputs = cls._model(
                input_ids=tokenized_input["input_ids"],
                attention_mask=tokenized_input["attention_mask"],
                token_type_ids=tokenized_input.get("token_type_ids"),
            )

        return outputs.logits.detach().cpu().argmax(-1).item()

    @classmethod
    async def predict(cls, sentence: str) -> int:
        if cls._model is None:
            await cls.load()

        prediction = await asyncio.to_thread(cls._predict_blocking, sentence)
        if prediction == 2:
            prediction = 1
        return prediction


async def predictor(comment: str) -> int:
    return await ModelService.predict(comment)
