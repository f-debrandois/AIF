import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*4*4, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))       # First convolution followed by
        x = self.pool(x)                # a relu activation and a max pooling#
        x = F.relu(self.conv2(x))       # Second convolution followed by
        x = self.pool(x)                # a relu activation and a max pooling
        x = torch.flatten(x, 1)         # Flatten the tensor
        x = F.relu(self.fc1(x))         # First fully connected layer
        x = F.relu(self.fc2(x))         # Second fully connected layer
        x = self.fc3(x)                 # Third fully connected layer
        return x

    def get_features(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        return x
