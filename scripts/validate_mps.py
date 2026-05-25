import torch
from scOT.model import ScOT

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T").to(device)
model.eval()

# dummy input: (batch, channels, H, W) — Poseidon-T expects 128x128
# NS-Sines has 4 channels: u, v, pressure, tracer
x = torch.randn(1, 4, 128, 128).to(device)
t = torch.tensor([0.5]).to(device)  # lead time

with torch.no_grad():
    out = model(pixel_values=x, time=t)

print(out.output.shape)  # should be (1, 4, 128, 128)