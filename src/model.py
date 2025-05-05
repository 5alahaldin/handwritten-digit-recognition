import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitCNN(nn.Module):
  def __init__(self):
    super(DigitCNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(32)

    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)

    self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
    self.bn3 = nn.BatchNorm2d(128)

    self.dropout1 = nn.Dropout(0.25)
    self.dropout2 = nn.Dropout(0.5)

    self.fc1 = nn.Linear(128 * 7 * 7, 256)
    self.fc2 = nn.Linear(256, 10)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.max_pool2d(x, 2)

    x = F.relu(self.bn2(self.conv2(x)))
    x = F.max_pool2d(x, 2)

    x = F.relu(self.bn3(self.conv3(x)))
    x = self.dropout1(x)

    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.dropout2(x)
    x = self.fc2(x)
    return x
