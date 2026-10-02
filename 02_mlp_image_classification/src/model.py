import torch.nn as nn
import torch.nn.functional as F

class FASHIONMNISTMLP(nn.Module):

    def __init__(self, num_classes = 10):
        super().__init__()

        self.flatten = nn.Flatten()

        self.fc0 = nn.Linear(28 * 28, 512)
        self.bn0 = nn.BatchNorm1d(512)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)

        self.fc4 = nn.Linear(64, num_classes)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        x = self.flatten(x)

        x = F.relu(self.bn0(self.fc0(x)))
        x = self.dropout(x)

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn3(self.fc3(x)))

        x = F.log_softmax(self.fc4(x), dim=1)

        return x