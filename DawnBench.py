import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

"""
Reference: Shakes Lecture 
"""
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

"""
Load CIFAR10 dataset and apply the defined transformations
"""
transform_train = transforms.Compose([
    # Convert image into tensor
    transforms.ToTensor(),
    # Normalize with mean and sd values
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    # Apply random horizontal flip and cropping for data augmentation
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect')
])

transform_test = transforms.Compose([
    # Convert image into tensor
    transforms.ToTensor(),
    # Normalizes tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

"""
Load CIFAR10 dataset and apply the defined transformations
"""
trainset = torchvision.datasets.CIFAR10(
    root='cifar10', train=True, download=True, transform=transform_train)
# Handles batching, shuffling and loading of data
train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False)

"""
Define the BasicBlock class for the ResNet architecture
"""
class BasicBlock(nn.Module):
    """
    Represents a block in the ResNet architecture.
    Contains two convolutional layers with batch normalization and ReLU activation.
    Includes a shortcut connection to add input to the block's output.
    """
    # no expansion of hte number of channels in the residual block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # takes in_planes input channels and outpuls planes channels
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # Normalizes the output of the first layer
        self.bn1 = nn.BatchNorm2d(planes)
        # number of input and output channels is equal to planes
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # Add the input to the output of the block
        self.shortcut = nn.Sequential()

        # Match Dimensions for residual conenctions
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """
        Forward pass of the BasicBlock. Applies convolutions, batch normalization, ReLU activation,
        and adds the shortcut connection.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add the shortcut connection to the second convolution
        out += self.shortcut(x)
        out = F.relu(out)
        return out

"""
Define the ResNet class which consists of multiple BasicBlocks stacked together
"""
class ResNet(nn.Module):
    """
    ResNet is formed by stacking multiple residual blocks (BasicBlocks) together.
    This architecture allows for deeper networks without encountering the vanishing gradient problem.

    3x3 convolution -> batch norm -> relu -> 3x3 convolution -> batch norm -> input connection -> relu

    stride of 2 halves the image (down samples), filters are doubled for double channels
    """

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        # Number of channels entering each block
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding = 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Final layer taht outputs predictiosn
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Creates a sequential layer of residual blocks with the specified parameters.
        Adjusts the number of channels and strides for each block.
        """
        # Strides refer to the stepsize at which the filter moves across the iamge
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            # Add Block in Layer
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 

    def forward(self, x):
        """
        Forward pass through the entire ResNet model.
        Applies initial convolution, passes through all layers, and applies average pooling and final linear layer.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # reduce spatial dimensions to 1x1
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    

"""
Define the ResNet18 model with 2 blocks in each layer
"""
def ResNet18():
    # 2 Blocks in each layer
    return ResNet(BasicBlock, [2, 2, 2, 2])


model = ResNet18()
model = model.to(device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
print("Model No. Of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

criterion = nn.CrossEntropyLoss()
# Stochastic Gradient Descent
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

total_step = len(train_loader)

"""
Define learning rate schedulers for dynamic learning rate adjustment
"""
# Cycles lr between base and max, with increase of 15in a triangle pattern (lr decrases faster than increases)
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.005, max_lr=0.1, step_size_up=15, step_size_down=15, mode="triangular2")
# Adjutss lr linearly
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.005/0.1, end_factor=0.005/5, verbose=False)

# Scheduler determines which lr scheduler to use, every 30 epoch, sche_lnear1 swaps to 3
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])

num_epochs = 35


"""
Training the model
"""
print("> Training")
model.train()


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move image and label to GPU
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss: {:.5f}".format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    scheduler.step()
    
"""
Training the model
"""
print("> Testing")
model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f} %')

