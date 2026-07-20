import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

FP32_MODEL = "resnet18.onnx"
INT8_MODEL = "resnet18_int8.onnx"

quantize_dynamic(model_input=FP32_MODEL, model_output=INT8_MODEL, weight_type=QuantType.QInt8)
print(f"Quantized model saved to {INT8_MODEL}")

fp32_session = ort.InferenceSession(FP32_MODEL, providers=["CPUExecutionProvider"])
int8_session = ort.InferenceSession(INT8_MODEL, providers=["CPUExecutionProvider"])

dummy_input = np.load("dummy_input.npy")

fp32_output = fp32_session.run(None, {"input": dummy_input})[0]
int8_output = int8_session.run(None, {"input": dummy_input})[0]

def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

max_abs_error = np.max(np.abs(fp32_output - int8_output))
cos_sim = cosine_similarity(fp32_output, int8_output)
top_class_fp32, top_class_int8 = np.argmax(fp32_output), np.argmax(int8_output)

print("\n=== ACCURACY: FP32 vs INT8 ===")
print(f"Max absolute error: {max_abs_error:.6f}")
print(f"Cosine similarity: {cos_sim:.6f}")
print(f"Top-class match: {top_class_fp32 == top_class_int8}")

def benchmark(session, input_data, n_runs=50):
    for _ in range(5):
        session.run(None, {"input": input_data})
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {"input": input_data})
    elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000, n_runs / elapsed

fp32_latency, fp32_throughput = benchmark(fp32_session, dummy_input)
int8_latency, int8_throughput = benchmark(int8_session, dummy_input)

print("\n=== PERFORMANCE: FP32 vs INT8 ===")
print(f"FP32 -> {fp32_latency:.3f} ms | {fp32_throughput:.1f} inf/s")
print(f"INT8 -> {int8_latency:.3f} ms | {int8_throughput:.1f} inf/s")
print(f"Speedup: {fp32_latency / int8_latency:.2f}x")

import os
fp32_size = os.path.getsize(FP32_MODEL) / (1024 * 1024)
int8_size = os.path.getsize(INT8_MODEL) / (1024 * 1024)
print(f"\nFP32: {fp32_size:.2f} MB | INT8: {int8_size:.2f} MB | Reduction: {(1 - int8_size/fp32_size)*100:.1f}%")