"""
pytest suite for numerical validation of the PyTorch -> ONNX export.

Run with: pytest test_validation.py -v
"""
import numpy as np
import onnxruntime as ort
import torch
import torchvision.models as models
import pytest


# ---- Fixtures: reusable setup, run once, shared across tests ----
# A fixture avoids re-loading the model/session in every single test function.

@pytest.fixture(scope="module")
def pytorch_model():
    torch.manual_seed(0)
    model = models.resnet18(weights=None)
    model.eval()
    return model


@pytest.fixture(scope="module")
def onnx_session():
    return ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])


def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---- Parameterization: run the SAME test logic across multiple inputs ----
# This is how real validation scales -- not one input, but a batch of
# representative test cases, each run through the identical check.

@pytest.mark.parametrize("seed", [42, 123, 999])
def test_pytorch_onnx_numerical_match(pytorch_model, onnx_session, seed):
    """PyTorch and ONNX Runtime outputs must match within tolerance,
    for multiple different random inputs (not just one)."""
    torch.manual_seed(seed)
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    onnx_output = onnx_session.run(None, {"input": dummy_input.numpy()})[0]

    # This IS the automated pass/fail gate -- no human reads numbers here.
    assert np.allclose(pytorch_output, onnx_output, atol=1e-3, rtol=1e-3), \
        f"Outputs diverged beyond tolerance for seed={seed}"


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_top_class_matches(pytorch_model, onnx_session, seed):
    """The final predicted class must match -- the real-world outcome
    that actually matters, on top of raw numerical closeness."""
    torch.manual_seed(seed)
    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    onnx_output = onnx_session.run(None, {"input": dummy_input.numpy()})[0]

    assert np.argmax(pytorch_output) == np.argmax(onnx_output)


def test_cosine_similarity_above_threshold(pytorch_model, onnx_session):
    """Sanity check on output direction, using our fixed known-good input."""
    dummy_input = np.load("dummy_input.npy")
    pytorch_output = np.load("pytorch_output.npy")

    onnx_output = onnx_session.run(None, {"input": dummy_input})[0]
    sim = cosine_similarity(pytorch_output, onnx_output)

    assert sim > 0.999, f"Cosine similarity too low: {sim}"


def test_onnx_model_structurally_valid():
    """The ONNX checker must pass -- catches export corruption early,
    before we even get to numerical comparison."""
    import onnx
    onnx_model = onnx.load("resnet18.onnx")
    onnx.checker.check_model(onnx_model)  # raises if invalid


def test_output_shape_matches_expected():
    """Basic sanity: did we get 1000 ImageNet-class logits, not garbage?"""
    dummy_input = np.load("dummy_input.npy")
    session = ort.InferenceSession("resnet18.onnx", providers=["CPUExecutionProvider"])
    output = session.run(None, {"input": dummy_input})[0]
    assert output.shape == (1, 1000)