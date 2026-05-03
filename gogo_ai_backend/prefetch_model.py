"""빌드 타임에 모델·토크나이저를 HuggingFace Hub에서 받아 이미지에 캐싱.

런타임 외부 의존을 제거하기 위한 스크립트. Dockerfile에서만 호출됨.
"""
from pathlib import Path

import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

with (Path(__file__).parent / "models.yaml").open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)["profanity_filter"]

print(f"Prefetch model {cfg['hf_repo']}@{cfg['revision'][:8]}")
AutoModelForSequenceClassification.from_pretrained(
    cfg["hf_repo"],
    revision=cfg["revision"],
    trust_remote_code=True,
)

print(f"Prefetch tokenizer {cfg['tokenizer']['hf_repo']}@{cfg['tokenizer']['revision'][:8]}")
AutoTokenizer.from_pretrained(
    cfg["tokenizer"]["hf_repo"],
    revision=cfg["tokenizer"]["revision"],
)

print("Prefetch complete")
