import tqdm
from dataset import *
from utils import *
from softmax import *
from mlp import *
from cnn import *

def train():
    train_loss = 0
    train_acc = 0
    batch_num = len(train_loader)
    
    for i, item in tqdm.tqdm(enumerate(train_loader), desc='train', total=len(train_loader)):
        data, label = item[0].float().to(device), item[1].reshape(-1).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += (torch.argmax(output, 1) == label).sum().item()
    
    return train_loss / (batch_num * batch_size), train_acc / (batch_num * batch_size)

def test():
    test_loss = 0
    test_acc = 0
    batch_num = len(test_loader)

    for i, item in tqdm.tqdm(enumerate(test_loader), desc='test', total=len(test_loader)):
        data, label = item[0].float().to(device), item[1].reshape(-1).to(device)
        with torch.no_grad():
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            test_acc += (torch.argmax(output, 1) == label).sum().item()
    return test_loss / (batch_num * batch_size), test_acc / (batch_num * batch_size)

if __name__ == '__main__':
    train_data, train_label, test_data, test_label = load_data(filepath=file_path)

    isLinear = True
    if(model_name == 'cnn'):
        isLinear = False
    train_dataset = CIFAR10(train_data, train_label, linear=isLinear)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_dataset = CIFAR10(test_data, test_label, linear=isLinear)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()

    if(model_name == 'linear'):
        model = Softmax(input_dim, num_class).to(device)
    elif(model_name == 'mlp'):
        model = MLP(input_dim, num_class, hidden_dim).to(device)
    elif(model_name == 'cnn'):
        model = LeNet5(num_class).to(device)
    
    if(optimizer_name == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif(optimizer_name == 'sgdm'):
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif(optimizer_name == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)        

    train_result = []
    test_result = []
    train_max_acc, test_max_acc = 0, 0

    for i in range(num_epoch):
        train_result.append(train())
        test_result.append(test())
        train_max_acc = max(train_max_acc, train_result[-1][1])
        test_max_acc = max(test_max_acc, test_result[-1][1])
        print('epochs: ', i)
        print('train: -loss: %.4f  -acc: %.4f' %train_result[-1])
        print('test:  -loss: %.4f  -acc: %.4f' %test_result[-1])
    print('train_max_acc: %.4f' %train_max_acc)
    print('test_max_acc:  %.4f' %test_max_acc)
    file_name = str(optimizer_name) + '_' + str(batch_size) + '_'
    plot_result(model_name, file_name, train_result, test_result)