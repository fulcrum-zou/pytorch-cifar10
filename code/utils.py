import matplotlib.pyplot as plt
import numpy as np
import re

def plot_result(model_name, file_name, train_result, test_result):
    file_path = '../result/' + model_name + '/' + file_name

    plt.plot([i[1] for i in train_result], color='skyblue', label='Train', linewidth=1)
    plt.plot([i[1] for i in test_result], color='pink', label='Test', linewidth=1)
    plt.title('Accuracy')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(file_path + 'acc.png')
    plt.show()

    plt.clf()
    plt.plot([i[0] for i in train_result], color='skyblue', label='Train', linewidth=1)
    plt.plot([i[0] for i in test_result], color='pink', label='Test', linewidth=1)
    plt.title('Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(file_path + 'loss.png')
    plt.show()

def write_file(file_name, train_result, test_result):
    f = open(file_name, 'w')
    for i in range(len(train_result)):
        f.write('%.4f ' %train_result[i][0])
        f.write('%.4f\n' %train_result[i][1])
        f.write('%.4f ' %test_result[i][0])
        f.write('%.4f\n' %test_result[i][1])
    f.close()

def read_file(file_name):
    train_result, test_result = [], []
    with open(file_name, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            temp = re.findall(r'-?\d+\.?\d*e?-?\d*?', line)
            if i % 2 == 0:
                train_result.append([float(temp[0]), float(temp[1])])
            else:
                test_result.append([float(temp[0]), float(temp[1])])
                
    f.close()
    return train_result, test_result
