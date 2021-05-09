import torch

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
file_path = '../src/'
batch_size = 32
input_dim = 32 * 32 * 3
num_class = 10
num_epoch = 100
hidden_dim = 50
learning_rate = 1e-4
weight_decay = 1e-4
momentum = 0.09

model_name = 'linear' # linear / mlp / cnn
optimizer_name = 'sgd' # sgd / sgdm / adam