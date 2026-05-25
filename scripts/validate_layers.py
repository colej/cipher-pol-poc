import torch
from scOT.model import ScOT

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T").to(device)
model.eval()

for name, module in model.named_modules():
    print(name, type(module).__name__)