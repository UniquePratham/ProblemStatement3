import torch, torch.nn as nn, base64, io, json
from PIL import Image

class Generator(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, 784), nn.Tanh(),
        )
    def forward(self, noise, labels):
        x = torch.cat((self.label_emb(labels), noise), -1)
        return self.model(x).view(-1, 1, 28, 28)

g = Generator()
g.load_state_dict(torch.load("trained_generator.pth", map_location="cpu", weights_only=True))
g.eval()

torch.manual_seed(42)
out = {}
with torch.no_grad():
    for digit in range(10):
        labels = torch.full((8,), digit, dtype=torch.long)
        noise = torch.randn(8, 64)
        imgs = g(noise, labels)
        imgs = ((imgs + 1) / 2 * 255).clamp(0, 255).byte()
        b64s = []
        for i in range(8):
            arr = imgs[i, 0].numpy()
            img = Image.fromarray(arr, mode="L").resize((112, 112), Image.NEAREST)
            buf = io.BytesIO(); img.save(buf, format="PNG")
            b64s.append(base64.b64encode(buf.getvalue()).decode())
        out[digit] = b64s

json.dump(out, open("samples.json", "w"), indent=0)
print("digits:", len(out), "images per digit:", len(out[0]))
print("total bytes:", sum(len(v) for vs in out.values() for v in vs))
