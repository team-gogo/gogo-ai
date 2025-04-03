from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd
import asyncio

async def initialize_model_and_tokenizer(hf_model, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model, local_files_only=True, trust_remote_code=True, use_auth_token=False
    )
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    model.to(device)
    model.eval()
    return model, tokenizer

async def predict_sentence(model, tokenizer, sentence, device):
    tokenized_input = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        add_special_tokens=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=tokenized_input["input_ids"],
            attention_mask=tokenized_input["attention_mask"],
            token_type_ids=tokenized_input.get("token_type_ids")
        )

    logits = outputs.logits.detach().cpu()
    prediction = logits.argmax(-1).item()
    return prediction

async def predictor(comment):
    comment = comment
    hf_model = "kdyeon0309/gogo_forpanity_filter"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = await initialize_model_and_tokenizer(hf_model, device)
    prediction= await predict_sentence(model, tokenizer, comment, device)
    if prediction == 2:
        prediction = 0
    return prediction
