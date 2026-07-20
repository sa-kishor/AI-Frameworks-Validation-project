import torch
import torchvision.models as models
import onnx

torch.manual_seed(0)
model = models.resnet18(weights=None)
model.eval()

torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "resnet18.onnx",
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False
)

print("Exported to resnet18.onnx")

onnx_model = onnx.load("resnet18.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model structurally valid (passed checker)")
print(f"Graph has {len(onnx_model.graph.node)} nodes (operators)")
print("First 5 op types:", [n.op_type for n in onnx_model.graph.node[:5]])