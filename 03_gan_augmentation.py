# ============================================================
# 03. GAN Augmentation: cGAN-generated images + real MNIST -> ResNet-18
# GAN augmentation: O | FGSM attack: X
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# -----------------------------
# Hyperparameters
# -----------------------------
batch_size = 128
z_dim = 100
lr = 0.0002
epochs_gan = 30
epochs_cls = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Dataset (MNIST)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader_gan = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# -----------------------------
# Conditional Generator
# -----------------------------
class ConditionalGenerator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, 256, 4, 1, 0),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z, labels):
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        return self.net(torch.cat([z, label_embed], dim=1))

# -----------------------------
# Conditional Discriminator
# -----------------------------
class ConditionalDiscriminator(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Conv2d(1 + num_classes, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0),
            nn.Sigmoid()
        )
    def forward(self, img, labels):
        label_embed = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_embed = label_embed.expand(-1, -1, img.size(2), img.size(3))
        return self.net(torch.cat([img, label_embed], dim=1)).view(-1)

# -----------------------------
# Init GAN
# -----------------------------
G = ConditionalGenerator(z_dim=z_dim).to(device)
D = ConditionalDiscriminator().to(device)
criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# -----------------------------
# Train GAN
# -----------------------------
for epoch in range(epochs_gan):
    G.train(); D.train()
    total_d_loss = total_g_loss = 0
    for imgs, labels in train_loader_gan:
        bs = imgs.size(0)
        imgs, labels = imgs.to(device), labels.to(device)
        real = torch.ones(bs, device=device)
        fake = torch.zeros(bs, device=device)

        # Discriminator
        optimizer_D.zero_grad()
        loss_real = criterion(D(imgs, labels), real)
        z = torch.randn(bs, z_dim, 1, 1, device=device)
        fake_lab = torch.randint(0, 10, (bs,), device=device)
        fake_imgs = G(z, fake_lab)
        loss_fake = criterion(D(fake_imgs.detach(), fake_lab), fake)
        d_loss = loss_real + loss_fake
        d_loss.backward(); optimizer_D.step()

        # Generator
        optimizer_G.zero_grad()
        g_loss = criterion(D(fake_imgs, fake_lab), real)
        g_loss.backward(); optimizer_G.step()

        total_d_loss += d_loss.item(); total_g_loss += g_loss.item()
    print(f"[GAN {epoch+1}/{epochs_gan}] D_loss: {total_d_loss/len(train_loader_gan):.4f} "
          f"G_loss: {total_g_loss/len(train_loader_gan):.4f}")

# -----------------------------
# Generate fake images
# -----------------------------
def generate_fake_images(generator, num_images, z_dim):
    generator.eval()
    fake_imgs, fake_labels = [], []
    with torch.no_grad():
        while sum(f.size(0) for f in fake_imgs) < num_images:
            z = torch.randn(batch_size, z_dim, 1, 1).to(device)
            labels = torch.randint(0, 10, (batch_size,), device=device)
            fake_imgs.append(generator(z, labels).cpu())
            fake_labels.append(labels.cpu())
    return torch.cat(fake_imgs)[:num_images], torch.cat(fake_labels)[:num_images]

# -----------------------------
# Mixed dataset (real + fake, 50:50)
# -----------------------------
class MixedMNISTDataset(Dataset):
    def __init__(self, real_dataset, fake_images, fake_labels):
        self.real_dataset = real_dataset
        self.fake_images = fake_images
        self.fake_labels = fake_labels
    def __len__(self):
        return len(self.real_dataset) + len(self.fake_images)
    def __getitem__(self, idx):
        if idx < len(self.real_dataset):
            img, label = self.real_dataset[idx]
            label = torch.tensor(label, dtype=torch.long)
        else:
            j = idx - len(self.real_dataset)
            img = self.fake_images[j]
            label = torch.tensor(self.fake_labels[j], dtype=torch.long)
        return img, label

real_size = len(mnist)
fake_images, fake_labels = generate_fake_images(G, real_size, z_dim)
real_subset = Subset(mnist, torch.randperm(real_size)[:real_size])
mixed_dataset = MixedMNISTDataset(real_subset, fake_images, fake_labels)

train_size = int(0.8 * len(mixed_dataset))
val_size = len(mixed_dataset) - train_size
train_set, val_set = random_split(mixed_dataset, [train_size, val_size])
train_loader_cls = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader_cls = DataLoader(val_set, batch_size=batch_size)

# -----------------------------
# Classifier (ResNet-18)
# -----------------------------
resnet = models.resnet18(weights=None)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
resnet = resnet.to(device)
criterion_cls = nn.CrossEntropyLoss()
optimizer_cls = optim.Adam(resnet.parameters(), lr=lr)

# -----------------------------
# Train classifier
# -----------------------------
train_acc_hist, val_acc_hist = [], []
for epoch in range(epochs_cls):
    resnet.train()
    correct = total = 0
    for imgs, labels in train_loader_cls:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer_cls.zero_grad()
        outputs = resnet(imgs)
        loss = criterion_cls(outputs, labels)
        loss.backward(); optimizer_cls.step()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    train_acc_hist.append(100 * correct / total)

    resnet.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader_cls:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (resnet(imgs).argmax(1) == labels).sum().item()
            total += labels.size(0)
    val_acc_hist.append(100 * correct / total)
    print(f"[CLS {epoch+1}/{epochs_cls}] Train Acc: {train_acc_hist[-1]:.2f}% "
          f"Val Acc: {val_acc_hist[-1]:.2f}%")

# -----------------------------
# Plot accuracy
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_acc_hist, label='Train Accuracy')
plt.plot(val_acc_hist, label='Validation Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
plt.title('GAN-Augmented Training: Train vs Validation Accuracy')
plt.legend(); plt.grid(True); plt.show()
