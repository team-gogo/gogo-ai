import torch
import onnx
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_path = "../Profanity_Filter/output/checkpoint-52350"
tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

text = "나쁜 말로 너를 제압하고 정복해. 구원의 시기는 시기 질투를 걸복해, 아픈 달로 초원의 비기를 참벌해"
inputs = tokenizer(text, return_tensors="pt")

onnx_path = "../Profanity_Filter/onnx_output/filter_onnx.onnx"
torch.onnx.export(
    model,  
    (inputs["input_ids"], inputs["attention_mask"]),  # 입력 텐서
    onnx_path,  
    input_names=["input_ids", "attention_mask"],  
    output_names=["logits"],  
    dynamic_axes={"input_ids": {0: "batch_size"}, "attention_mask": {0: "batch_size"}},
    opset_version=12  
)

print(f"ONNX 모델 저장 완료: {onnx_path}")
