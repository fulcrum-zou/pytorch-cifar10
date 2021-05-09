import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_class):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.S2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.C3 = nn.Conv2d(8, 32, 5)
        self.relu2 = nn.ReLU()
        self.S4 = nn.MaxPool2d(2, stride=2)
        self.C5 = nn.Linear(800, 120)
        self.relu3 = nn.ReLU()
        self.F6 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(84, num_class)
    
    def forward(self, data):
        output = self.relu1(self.C1(data))
        output = self.S2(output)
        output = self.relu2(self.C3(output))
        output = self.S4(output)
        output = output.view(output.shape[0], -1)
        output = self.relu3(self.C5(output))
        output = self.relu4(self.F6(output))
        output = self.out(output)
        return output