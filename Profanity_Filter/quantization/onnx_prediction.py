import numpy as np
import onnxruntime as ort
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

text = input("예측할 문장을 입력하세요: ")

inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=30)

quantized_onnx_path = "../Profanity_Filter/onnx_output/filter_onnx.onnx"
session = ort.InferenceSession(quantized_onnx_path)

inputs_onnx = {
    "input_ids": inputs["input_ids"].detach().cpu().numpy(),
    "attention_mask": inputs["attention_mask"].detach().cpu().numpy(),
}

outputs = session.run(["logits"], inputs_onnx)
logits = outputs[0]

pred = np.argmax(logits, axis=-1)
print(f"예측 결과: {pred}")
