import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Simple Generator model (Assume trained separately)
class Generator(nn.Module):
    def __init__(self, latent_dim=64, num_classes=10):
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
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Load trained model (You must train and save this externally)
generator = Generator()
generator.load_state_dict(torch.load("trained_generator.pth", map_location=torch.device("cpu")))
generator.eval()

st.title("Handwritten Digit Generator")
digit = st.selectbox("Select a digit (0-9):", list(range(10)))

if st.button("Generate"):
    noise = torch.randn(5, 64)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        images = generator(noise, labels).detach().cpu().numpy()

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(images[i][0], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
