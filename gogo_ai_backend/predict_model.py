import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_MODELS_YAML = Path(__file__).parent / "models.yaml"


def _load_config() -> dict:
    with _MODELS_YAML.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["profanity_filter"]


class ModelService:
    _config = _load_config()
    HF_MODEL: str = _config["hf_repo"]
    HF_REVISION: str = _config["revision"]
    TOKENIZER: str = _config["tokenizer"]["hf_repo"]
    TOKENIZER_REVISION: str = _config["tokenizer"]["revision"]

    _model: Optional[AutoModelForSequenceClassification] = None
    _tokenizer: Optional[AutoTokenizer] = None
    _device: Optional[torch.device] = None
    _loaded_at: Optional[str] = None
    _load_lock = asyncio.Lock()

    @classmethod
    async def load(cls) -> None:
        async with cls._load_lock:
            if cls._model is not None:
                return

            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(
                f"Loading profanity model '{cls.HF_MODEL}@{cls.HF_REVISION[:8]}' on {cls._device}"
            )

            def _load_blocking():
                model = AutoModelForSequenceClassification.from_pretrained(
                    cls.HF_MODEL,
                    revision=cls.HF_REVISION,
                    trust_remote_code=True,
                    use_auth_token=False,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    cls.TOKENIZER,
                    revision=cls.TOKENIZER_REVISION,
                )
                model.to(cls._device)
                model.eval()
                return model, tokenizer

            cls._model, cls._tokenizer = await asyncio.to_thread(_load_blocking)
            cls._loaded_at = datetime.now(timezone.utc).isoformat()
            logging.info("Profanity model loaded")

    @classmethod
    def info(cls) -> dict:
        return {
            "model": cls.HF_MODEL,
            "revision": cls.HF_REVISION,
            "tokenizer": cls.TOKENIZER,
            "tokenizer_revision": cls.TOKENIZER_REVISION,
            "loaded_at": cls._loaded_at,
        }

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
