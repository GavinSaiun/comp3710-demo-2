import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torch.nn.functional as F

"""
Image segmentation is a computer vision technique that partitions a digital image
into discrete groups of pixels—image segments—to inform object detection and related tasks.
By parsing an image’s complex visual data into specifically shaped segments,
image segmentation enables faster, more advanced image processing.

https://www.ibm.com/topics/image-segmentation#:~:text=Image%20segmentation%20is%20a%20computer,faster%2C%20more%20advanced%20image%20processing.
"""
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU.")

# Define the UNet architecture
class UNet(nn.Module):
    """
    Basic structure of the UNet structure for brain mri segmentation

    https://www.youtube.com/watch?v=NhdzGfB1q74
    """
    def __init__(self):
        """
        the structure of the UNet, this is the structure used in the
        brain segmentation mri. Init, initializes encoding blocks for
        feature extraction, bottleneck, decoding blocks for upsampling
        and feature map concatenation (skip connections) and final layer
        to produce output segmentation map
        """
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

        self.conv_block_512 = self.conv_block(1024, 512)
        self.conv_block_256 = self.conv_block(512, 256)
        self.conv_block_128 = self.conv_block(256, 128)
        self.conv_block_64 = self.conv_block(128, 64)

    def conv_block(self, in_channels, out_channels):
        """
        The encoder is the first half of the UNet architecture.
        Each block has repeated 3x3 convolutional networks, followed by ReLu
        After, 2x2 max pooling layers are used to down sample/
        Each downsampling then doubles the amount of channels 

        Max Pooling is done in forward()
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        """
        The decoder is the second half of UNet.
        it consists of repeated 3x3 convolutional layers with ReLu
        each decoding block is upsampled, followed by a 2x2 conv layer
        channels are halved after each upsample
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Defines the forward pass through the UNet network.
        first step is encoding, followed by middle processing and decoding
        """
        # Encoder

        """
        Pass input image into the encoder, applies max pooling of a 2x2 then
        passes it through the next encoder block. Resulted image has
        reduced spatial dimensions with feature extracted.
        """
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # BottleNeck
        """
        1. DownSample wiht 2x2 max pooling
        BottleNeck layer processes features at its lowest spatial dimension
        """
        m = self.middle(F.max_pool2d(e4, 2))

        # Decoder
        """
        Decoding path. Each output is inputted into a decoding block, where
        it is upsampled using transposed conv. Skip connections are then used,
        concatenating the corresponding decoding feature map with its encoding
        feature map. Each concatenated feature map is then passed through the
        next blick.

        Decoder extracts semantic information (e.g. what hte item is). The 
        encoder contains the spatial information (e.g. this is where). 
        Skip connections allow the image to have both what the thing is and where

        """
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.conv_block_512(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.conv_block_256(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.conv_block_128(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.conv_block_64(d1)

        """
        Final Segmentation map, the number of channels is reduced to 1 for binary
        segmentation
        """
        out = self.final_conv(d1)
        return out

class BrainSegmentationDataset(Dataset):
    """
    Handles the initialization and prepping of the dataset
    """
    def __init__(self, image_dir, label_dir, transform=None):
        """
        initializes file path and transformatons
        Sorts file names and images into their respective directories
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.label_filenames = sorted([f for f in os.listdir(label_dir) if f.endswith('.png')])

    def __len__(self):
        """
        return the number of samples
        """
        return len(self.image_filenames)

    def __getitem__(self, idx):
        """
        Construct paths for the image and label using the provided index idx.
        Open and convert images and labels to grayscale.
        """
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        label_path = os.path.join(self.label_dir, self.label_filenames[idx])

        image = Image.open(image_path).convert('L')  # Convert image to grayscale
        label = Image.open(label_path).convert('L')  # Convert label to grayscale

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Convert label to one-hot encoding
        """
        Convert label to one-hot encoding
        """
        # labeli = torch.tensor(np.array(label)).long()
        # label_one_hoti = F.one_hot(label, num_classes=2).float()

        return image, label

class DiceLoss(nn.Module):
    """
    A Coefficient between 0 and 1 that determiens the similarity between 2 samples
    For brain MRI, it is used to evaluate the ground truth to the prediction made 
    by the UNet

    Formula = 2 * number of pixels in common between the 2/ total number of pixels

    https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient#:~:text=The%20Dice%2DS%C3%B8rensen%20coefficient%20(see,in%201945%20and%201948%20respectively.
    """
    def __init__(self, smooth=1e-6):
        """
        Initialize the Dice loss with a smoothing factor to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        """
        Compute the Dice loss between the outputs and targets.
        """
        # Apply sigmoid to the outputs to get probabilities
        outputs = torch.sigmoid(outputs)

        # Flatten the tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Ensure targets are binary
        targets = (targets > 0.5).float()

        # Calculate intersection and Dice coefficient
        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)

        return 1 - dice

    def dice_coefficient(self, outputs, targets):
        """
        Compute the Dice loss between the outputs and targets.
        """
        # Sigmoid activation is applied to the outputs to convert logits to probabilities
        outputs = torch.sigmoid(outputs)

        # Flatten the outputs and targets to calculate Dice coefficient
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # Ensure targets are binary
        targets = (targets > 0.5).float()

        intersection = (outputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (outputs.sum() + targets.sum() + self.smooth)

        return dice


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

# Creates Dataloaders, with 8 samples per batch, shuffle for randomness and 1 subprocess
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)


# Initialize the model, criterion, and optimizer
model = UNet().to(device)
criterion = DiceLoss()  # Use DiceLoss for binary segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(model)


# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_dice = 0.0  # Track total Dice coefficient for the epoch
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
        dice = criterion.dice_coefficient(outputs, labels).item()  # Calculate Dice coefficient for this batch
        running_dice += dice * images.size(0)

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.5f}, Dice: {dice:.5f}")

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_dice = running_dice / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {epoch_loss:.5f}, Average Dice: {epoch_dice:.5f}")

import matplotlib.pyplot as plt
import numpy as np

def visualize_batch(images, labels, outputs, idx=0):
    """
    Gives some examples of predicted brain segmentation and ground truth
    """
    image = images[idx].cpu().numpy().transpose(1, 2, 0)
    label = labels[idx].cpu().numpy()
    output = outputs[idx].cpu().numpy()

    # Denormalize image (if normalization was applied)
    image = (image * 0.5) + 0.5

    # Convert outputs and labels to binary
    output = np.squeeze(output) > 0.5
    label = np.squeeze(label) > 0.5

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[1].imshow(label, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(output, cmap='gray')
    axs[2].set_title('Prediction')

    for ax in axs:
        ax.axis('off')

    plt.show()


def save_visualization(images, labels, outputs, idx=0, file_name_prefix='visualization'):
    """
    Saves examples of predicted brain segmentation and ground truth as image files.
    """
    image = images[idx].cpu().numpy().transpose(1, 2, 0)
    label = labels[idx].cpu().numpy()
    output = outputs[idx].cpu().numpy()

    # Denormalize image (if normalization was applied)
    image = (image * 0.5) + 0.5

    # Convert outputs and labels to binary
    output = np.squeeze(output) > 0.5
    label = np.squeeze(label) > 0.5

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[1].imshow(label, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[2].imshow(output, cmap='gray')
    axs[2].set_title('Prediction')

    for ax in axs:
        ax.axis('off')

    # Save the figure
    plt.savefig(f'{file_name_prefix}_{idx}.png')
    plt.close(fig)


# Validation loop
model.eval()
with torch.no_grad():
    running_loss = 0.0
    running_dice = 0.0  # Track total Dice coefficient for the validation set
    total_samples = 0  # To track the total number of samples processed

    # Keep track of the number of saved visualizations
    saved_visualizations = 0
    max_visualizations = 5  # Number of visualizations to save

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        # Ensure outputs and labels are properly processed
        outputs = torch.sigmoid(outputs)  # Convert logits to probabilities
        dice = criterion.dice_coefficient(outputs, labels).item()  # Calculate Dice coefficient for this batch
        running_dice += dice * images.size(0)

        total_samples += images.size(0)

        # Save a few samples
        if saved_visualizations < max_visualizations:
            for i in range(min(images.size(0), max_visualizations - saved_visualizations)):
                save_visualization(images, labels, outputs, idx=i, file_name_prefix=f'visualization_{saved_visualizations + i}')
            saved_visualizations += min(images.size(0), max_visualizations - saved_visualizations)

        # Stop if we've saved enough visualizations
        if saved_visualizations >= max_visualizations:
            break

    # Compute average metrics
    val_loss = running_loss / total_samples
    val_dice = running_dice / total_samples
    print(f'Validation Loss: {val_loss:.5f}, Validation Dice: {val_dice:.5f}')

"""
Improvements:
- overfitting (regularization, early stopping)
- have to check validation set 
- Improper learning rate and optimizer
"""

