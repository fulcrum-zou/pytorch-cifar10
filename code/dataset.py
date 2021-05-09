import torch
from config import *
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset

def load_data(train_size = 50000, test_size = 10000, filepath = "../src/"):
    train_images = np.load(filepath+"cifar10_train_images.npy") #(50000, 32, 32, 3)
    train_labels = np.load(filepath+"cifar10_train_labels.npy") #(50000, )
    test_images = np.load(filepath+"cifar10_test_images.npy")   #(10000, 32, 32, 3)
    test_labels = np.load(filepath+"cifar10_test_labels.npy")   #(10000, )

    return train_images[:train_size], train_labels[:train_size], test_images[:test_size], test_labels[:test_size]

def myDataLoader(data, label, linear):
    if(linear):
        data = data.reshape(len(data), -1)
    else:
        data = np.transpose(data, [0, 3, 1, 2])
    dataset = TensorDataset(torch.tensor(data), torch.tensor(label))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return dataloader

class CIFAR10(Dataset):
    def __init__(self, data, label, linear = True):
        if(linear):
            self.data = data.reshape(len(data), -1)
        else:
            self.data = data.reshape(len(data), 3, 32, 32)
        self.label = label
        self.len = len(self.data)
        self.label = self.label.reshape(self.len, 1)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx]), torch.LongTensor(self.label[idx])