import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import os

# Hyperparameters
latent_dim = 64
num_classes = 10
batch_size = 256
epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        input = torch.cat((noise, self.label_emb(labels)), -1)
        out = self.model(input)
        return out.view(out.size(0), 1, 28, 28)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(784 + num_classes, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, imgs, labels):
        imgs_flat = imgs.view(imgs.size(0), -1)
        input = torch.cat((imgs_flat, self.label_emb(labels)), -1)
        validity = self.model(input)
        return validity

# Prepare Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
dataset = MNIST(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)
loss_fn = nn.BCELoss()
opt_G = torch.optim.Adam(generator.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

# Training loop
for epoch in range(epochs):
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Train Discriminator
        real = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        noise = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(noise, labels)

        d_real = discriminator(imgs, labels)
        d_fake = discriminator(gen_imgs.detach(), labels)

        d_loss = loss_fn(d_real, real) + loss_fn(d_fake, fake)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        g_loss = loss_fn(discriminator(gen_imgs, labels), real)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch+1}/{epochs} - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Save Generator
os.makedirs("models", exist_ok=True)
torch.save(generator.state_dict(), "trained_generator.pth")
print("✅ Generator saved as trained_generator.pth")
