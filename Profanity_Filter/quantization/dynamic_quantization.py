import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

onnx_model_path = "../Profanity_Filter/onnx_output/filter_onnx.onnx"
model = onnx.load(onnx_model_path)

quantized_model_path = "../Profanity_Filter/onnx_output/dynamic_quantized.onnx"
quantized_model = quantize_dynamic(
    model_input=onnx_model_path,  
    model_output=quantized_model_path,  
    weight_type=QuantType.QInt8  
)

print(f"Dynamic quantized model saved to {quantized_model_path}")
