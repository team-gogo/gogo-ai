from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

# 데이터 및 모델 초기화
def initialize_model_and_tokenizer(hf_model, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model, local_files_only=True, trust_remote_code=True, use_auth_token=False
    )
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    model.to(device)
    model.eval()
    return model, tokenizer

# 문장 예측 함수
def predict_sentence(model, tokenizer, sentence, device):
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

    labels = {0: " >> 정상", 1: " >> 비속어", 2: " >> 공격적임"}
    return labels.get(prediction, " >> 알 수 없음")

def main():
    df = pd.read_csv('../Profanity_Filter/data/datasets/test.csv')
    hf_model = "kdyeon0309/gogo_forpanity_filter"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = initialize_model_and_tokenizer(hf_model, device)

    for comment in df['comments']:
        print(predict_sentence(model, tokenizer, comment, device))

if __name__ == "__main__":
    main()