import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, conv1_ch=32, conv2_ch=64, fc1_units=128, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(conv1_ch, conv2_ch, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(dropout)
        self.dropout2 = nn.Dropout(dropout * 2)
        self.fc1 = nn.Linear(conv2_ch * 7 * 7, fc1_units)
        self.fc2 = nn.Linear(fc1_units, 10)
        # For easily loading params later
        self.arch_params = {
            "conv1_ch": conv1_ch, "conv2_ch": conv2_ch,
            "fc1_units": fc1_units, "dropout": dropout,
        }

    def forward(self, x):
        x = x.float()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(self.pool(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout2(F.relu(self.fc1(x)))
        return F.log_softmax(self.fc2(x), dim=1)