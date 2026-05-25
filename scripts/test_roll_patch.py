"""
Verify that _cyclic_shift is bit-exact with torch.roll on CPU, then run a
full model forward pass to confirm the patch doesn't break anything.
"""
import torch
import numpy as np
from scOT.model import _cyclic_shift

torch.manual_seed(42)
np.random.seed(42)

# --- unit tests: _cyclic_shift vs torch.roll ---

def reference_roll(x, shift_h, shift_w):
    return torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))

cases = [
    # (B, H, W, C, shift_h, shift_w)
    (2, 8, 8, 16, -2, -2),   # typical Swin forward shift
    (2, 8, 8, 16, 2, 2),     # typical Swin reverse shift
    (1, 16, 16, 32, -4, -4),
    (1, 16, 16, 32, 4, 4),
    (1, 10, 12, 8, -3, -3),  # non-square
    (1, 10, 12, 8, 3, 3),
    (2, 8, 8, 16, 0, 0),     # no shift (branch should be skipped)
    (2, 8, 8, 16, -2, 0),    # only height shift
    (2, 8, 8, 16, 0, -2),    # only width shift
]

print("Unit tests: _cyclic_shift vs torch.roll")
for B, H, W, C, sh, sw in cases:
    x = torch.randn(B, H, W, C)
    got = _cyclic_shift(x, sh, sw)
    want = reference_roll(x, sh, sw)
    assert torch.equal(got, want), (
        f"FAIL shift=({sh},{sw}) shape=({B},{H},{W},{C}): "
        f"max diff = {(got - want).abs().max().item()}"
    )
    print(f"  PASS  shape=({B:1d},{H:2d},{W:2d},{C:2d})  shift=({sh:3d},{sw:3d})")

# --- forward pass smoke test ---

print("\nForward pass smoke test (CPU, no weights download needed if cached)")
from scOT.model import ScOT

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")
model.eval()

x = torch.randn(1, 4, 128, 128)
t = torch.tensor([0.5])

with torch.no_grad():
    out = model(pixel_values=x, time=t)

assert out.output.shape == (1, 4, 128, 128), f"Unexpected output shape: {out.output.shape}"
print(f"  PASS  output shape: {out.output.shape}")

# --- MPS smoke test (if available) ---

if torch.backends.mps.is_available():
    print("\nMPS smoke test")
    model_mps = model.to("mps")
    x_mps = x.to("mps")
    t_mps = t.to("mps")
    with torch.no_grad():
        out_mps = model_mps(pixel_values=x_mps, time=t_mps)
    assert out_mps.output.shape == (1, 4, 128, 128)
    print(f"  PASS  MPS output shape: {out_mps.output.shape}")
else:
    print("\nMPS not available — skipping MPS smoke test")

print("\nAll tests passed.")
