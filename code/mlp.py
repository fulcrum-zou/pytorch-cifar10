import torch
import torch.nn as nn
import torch.nn.functional as F

'''
batch_size = 20
num_class = 10
learning_rate = 1e-5
weight_decay = 1e-4
num_epoch = 100
hidden_dim = 100
'''

class MLP(nn.Module):
    def __init__(self, input_dim, num_class, hidden_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.tanh1 = nn.Tanh()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh2 = nn.Tanh()
        self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        self.tanh3 = nn.Tanh()
        self.linear4 = nn.Linear(hidden_dim, num_class)
        self.init_linears(method='normal')
    
    def init_linears(self, method):
        if(method == 'xavier'):
            nn.init.xavier_uniform_(self.linear1.weight)
            nn.init.zeros_(self.linear1.bias)
            nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.zeros_(self.linear2.bias)
            nn.init.xavier_uniform_(self.linear3.weight)
            nn.init.zeros_(self.linear3.bias)
            nn.init.xavier_uniform_(self.linear4.weight)
            nn.init.zeros_(self.linear4.bias)
        elif(method == 'normal'):
            nn.init.normal_(self.linear1.weight)
            nn.init.normal_(self.linear1.bias)
            nn.init.normal_(self.linear2.weight)
            nn.init.normal_(self.linear2.bias)
            nn.init.normal_(self.linear3.weight)
            nn.init.normal_(self.linear3.bias)
            nn.init.normal_(self.linear4.weight)
            nn.init.normal_(self.linear4.bias)

    def forward(self, data):
        output = self.linear1(data)
        output = self.tanh1(output)
        output = self.linear2(output)
        output = self.tanh2(output)
        output = self.linear3(output)
        output = self.tanh3(output)
        output = self.linear4(output)
        return output