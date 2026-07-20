# AI Model Validation Report
**Project:** ResNet18 — PyTorch → ONNX Export, Numerical Validation, Quantization
**Author:** Kishor Saravanan
**Date:** 20/07/2001

---

## 1. Objective

Validate that converting a PyTorch model to ONNX, and subsequently quantizing it
to INT8, preserves numerical correctness within defined tolerance, and to
measure the resulting performance and size impact — before this conversion
would be considered "release qualified."

---

## 2. Pipeline Under Test

```
PyTorch Model (ResNet18, FP32)
        │
        ▼  torch.onnx.export() — opset 18, legacy exporter
ONNX Model (FP32)
        │
        ▼  ONNX Runtime (CPUExecutionProvider)
Inference Output
        │
        ▼  Post-Training Dynamic Quantization
ONNX Model (INT8)
```

---

## 3. Acceptance Criteria (defined before testing)

A conversion step is considered **PASS** if, against the same input:

| Metric | Threshold | Rationale |
|---|---|---|
| `np.allclose` (atol=1e-3, rtol=1e-3) | Must be `True` | Element-wise agreement within floating-point noise |
| Max absolute error | < 0.01 (FP32↔FP32 stage) | Bounds worst-case single-value drift |
| Cosine similarity | > 0.999 | Output direction/decision preserved |
| Top-1 predicted class | Must match | Real-world decision must be unaffected |

For the quantization stage specifically, a **higher** max-error tolerance
(< 0.2) is accepted, since INT8 has inherently lower numerical precision than
FP32 — this is expected accuracy drift, not a defect, provided top-1 class
still matches.

---

## 4. Results

### 4.1 PyTorch → ONNX (FP32) — Numerical Validation

| Metric | Result | Status |
|---|---|---|
| np.allclose | True | ✅ PASS |
| Max absolute error | 0.0000029 | ✅ PASS |
| Cosine similarity | 1.000000 | ✅ PASS |
| Top-1 class match | 238 == 238 | ✅ PASS |

Validated automatically across 3 independent random seeds via pytest
(9/9 tests passed — correctness × 3 seeds, top-class × 3 seeds, cosine
similarity, structural check, output-shape check).

### 4.2 FP32 → INT8 — Quantization (Post-Training, Dynamic)

| Metric | FP32 | INT8 | Status |
|---|---|---|---|
| Cosine similarity | — | 0.999812 | ✅ PASS |
| Max absolute error | — | 0.150652 | ✅ PASS (within relaxed quantization threshold) |
| Top-1 class match | 238 | 238 | ✅ PASS |
| Model size | 44.57 MB | 11.19 MB | 74.9% reduction |
| Avg latency (CPU) | 9.24 ms | 136.57 ms | ⚠️ 14.7x SLOWER |
| Throughput | 108.2 inf/s | 7.3 inf/s | ⚠️ Regression |

**Finding:** Quantization met all accuracy criteria and delivered a large size
reduction, but produced a significant *latency regression* on this CPU.
Dynamic INT8 quantization incurs per-inference quantize/dequantize overhead;
without hardware offering native low-precision execution paths, that overhead
is not offset by faster integer math. **Recommendation:** INT8 export should
only be qualified for release on hardware confirmed to have dedicated INT8
execution support (e.g. a target AI accelerator) — accuracy is release-ready,
performance is not, on this reference platform.

---

## 5. Automation & CI

- Full suite (export → validation → quantization checks) implemented as
  parameterized `pytest` tests with fixtures for model/session reuse.
- GitHub Actions CI runs the full pipeline from a clean environment on every
  push: dependency install → baseline generation → ONNX export → test suite.
- Current CI status: **green** (all runs passing on `main`).

---

## 6. Overall Release Verdict

| Stage | Verdict |
|---|---|
| ONNX Export (FP32) | ✅ Release qualified |
| INT8 Quantization — Accuracy | ✅ Release qualified |
| INT8 Quantization — Performance (CPU) | ❌ Not qualified — regression pending hardware-specific validation |
