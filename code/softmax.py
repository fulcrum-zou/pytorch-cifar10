import torch.nn as nn
import torch.nn.functional as F

class Softmax(nn.Module):
    def __init__(self, input_dim, num_class):
        super(Softmax, self).__init__()
        self.linear = nn.Linear(input_dim, num_class)

    def forward(self, data):
        output = self.linear(data)
        return output