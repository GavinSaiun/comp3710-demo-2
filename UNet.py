import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")

# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.middle = self.conv_block(512, 1024)
        
        self.upconv4 = self.upconv_block(1024, 512)
        self.upconv3 = self.upconv_block(512, 256)
        self.upconv2 = self.upconv_block(256, 128)
        self.upconv1 = self.upconv_block(128, 64)
        
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))
        
        # Middle
        m = self.middle(F.max_pool2d(e4, 2))
        
        # Decoder
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.conv_block(d4.size(1), 512)(d4)  # Adjust channels
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.conv_block(d3.size(1), 256)(d3)  # Adjust channels
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.conv_block(d2.size(1), 128)(d2)  # Adjust channels
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.conv_block(d1.size(1), 64)(d1)  # Adjust channels
        
        out = self.final_conv(d1)
        return out

# Define a dataset class for the brain segmentation task
class BrainSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])
        
        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        label = Image.open(label_path).convert('L')  # Convert label to grayscale
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        
        return image, label

# Define transforms for the dataset
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust normalization as needed
])

# Create datasets and dataloaders
train_dataset = BrainSegmentationDataset(
    image_dir='/home/groups/comp3710/OASIS/keras_png_slices_train',
    label_dir='/home/groups/comp3710/OASIS/keras_png_slices_seg_train',
    transform=transform
)

val_dataset = BrainSegmentationDataset(
    image_dir='/home/groups/comp3710/OASIS/keras_png_slices_validate',
    label_dir='/home/groups/comp3710/OASIS/keras_png_slices_seg_validate',
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)

# Initialize the model, criterion, and optimizer
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()  # Use appropriate loss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.5f}")
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss:.5f}")

# Validation loop
model.eval()
with torch.no_grad():
    running_loss = 0.0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
    
    val_loss = running_loss / len(val_loader.dataset)
    print(f'Validation Loss: {val_loss:.5f}')
