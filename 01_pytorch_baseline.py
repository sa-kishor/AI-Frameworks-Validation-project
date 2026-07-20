import torch
import torchvision.models as models
import numpy as np

torch.manual_seed(0)
model = models.resnet18(weights=None)   # random weights, no internet needed
model.eval()

torch.manual_seed(42)
dummy_input = torch.randn(1, 3, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)
print("First 5 logits:", output[0][:5])

np.save("dummy_input.npy", dummy_input.numpy())
np.save("pytorch_output.npy", output.numpy())
print("Saved dummy_input.npy and pytorch_output.npy")