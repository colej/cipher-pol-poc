from scOT.model import ScOT

model = ScOT.from_pretrained("camlab-ethz/Poseidon-T")
model.eval()
print(model)  # inspect layer names here