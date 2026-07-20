import numpy as np
import onnxruntime as ort

dummy_input = np.load("dummy_input.npy")
pytorch_output = np.load("pytorch_output.npy")

session = ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])
onnx_outputs = session.run(None, {"input": dummy_input})
onnx_output = onnx_outputs[0]

print("PyTorch output shape:", pytorch_output.shape)
print("ONNX Runtime output shape:", onnx_output.shape)

atol, rtol = 1e-3, 1e-3
is_close = np.allclose(pytorch_output, onnx_output, atol=atol, rtol=rtol)
max_abs_error = np.max(np.abs(pytorch_output - onnx_output))

def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

cos_sim = cosine_similarity(pytorch_output, onnx_output)

print(f"\nnp.allclose (atol={atol}, rtol={rtol}): {is_close}")
print(f"Max absolute error: {max_abs_error:.8f}")
print(f"Cosine similarity: {cos_sim:.8f}")

pytorch_top_class = np.argmax(pytorch_output)
onnx_top_class = np.argmax(onnx_output)
print(f"\nPyTorch predicted class: {pytorch_top_class}")
print(f"ONNX Runtime predicted class: {onnx_top_class}")
print(f"Top-class match: {pytorch_top_class == onnx_top_class}")