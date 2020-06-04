import os
import sys
import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net_cnn import classifier
from parameter_server import PS

from itertools import chain

import matplotlib.pyplot as plt
from numpy import math
# %matplotlib inline

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush() #每次写入后刷新到文件中，防止程序意外结束
    def flush(self):
        self.log.flush()
 
 
sys.stdout = Logger("res.txt")

train_set = CIFAR10('./cifardata', train=True, download=True)
test_set = CIFAR10('./cifardata', train=False, download=True)




transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 

def run_epoch(net, epoch, id):
        losses = []
        acces = []
        eval_losses = []
        eval_acces = []
        train_loss = 0
        train_acc = 0
        net.train()
        #torch.save(net.state_dict(), os.getcwd() + "/cifar1.pth")
        for im, label in net.train_data:
            im = Variable(im)
            label = Variable(label)
            # 前向传播
            out = net(im)
            loss = net.criterion(out, label)
            # 反向传播
            net.optimizer.zero_grad()
            loss.backward()
            net.optimizer.step()
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            train_acc += acc
            
        losses.append(train_loss / len(net.train_data))
        acces.append(train_acc / len(net.train_data))
        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        net.eval() # 将模型改为预测模式
        # torch.save(net.state_dict(), os.getcwd() + "/cifar1.pth")
        for im, label in net.test_data:
            im = Variable(im)
            label = Variable(label)
            out = net(im)
            loss = net.criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            eval_acc += acc
            
        eval_losses.append(eval_loss / len(net.test_data))
        eval_acces.append(eval_acc / len(net.test_data))
        print('epoch: {}, id:{},Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
            .format(epoch, id, train_loss / len(net.train_data), train_acc / len(net.train_data), 
                        eval_loss / len(net.test_data), eval_acc / len(net.test_data)))

train_set = CIFAR10('./cifardata', train=True, transform=transform, download=True) # 重新载入数据集，申明定义的数据变换
test_set = CIFAR10('./cifardata', train=False, transform=transform, download=True)

# split dataset to 10 shares
train_list = torch.utils.data.random_split(train_set, [5000, 5000,5000,5000,5000,5000,5000,5000,5000,5000])
print('train_0:', len(train_list[0]), 'train_1:', len(train_list[1]))

val_list= torch.utils.data.random_split(test_set, [1000, 1000,1000,1000,1000,1000,1000,1000,1000,1000])
print('val_0:', len(val_list[0]), 'val_1:', len(val_list[1]))




# global simple PS
ps_dict = torch.load(os.getcwd()+"/cifar1.pth")

# mynet list
mynet = []
# netwrok definition
for i in range(10):
    train_data = DataLoader(train_list[i], batch_size=64, shuffle=True)
    test_data = DataLoader(val_list[0], batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    new_net = classifier(train_data, test_data)
    #load model from base pth
    new_net.load_state_dict(torch.load(os.getcwd()+"/cifar1.pth"))
    new_net.set_criterion(criterion)
    optimizer = torch.optim.SGD(new_net.parameters(), 1e-2) # 使用随机梯度下降，学习率 0.1
    new_net.set_optimizer(optimizer)
    mynet.append(new_net)


def partial_max_changed_cifar(old, new, partial):
    first = True
    for k, v in old.items():
        if first == True:
            no = np.array(old[k]).flatten()
            nn = np.array(new[k]).flatten()
            first = False
        else:
            no = np.concatenate((no,np.array(old[k]).flatten()))
            nn = np.concatenate((nn,np.array(new[k]).flatten()))
    diff = abs(nn - no)
    max_index = np.argsort(-diff)
    needed = len(diff) * partial
    for i in range(len(diff)):
        if i < needed:
            if max_index[i] <= 196607:
                row = math.floor(max_index[i] / 3072)
                column = max_index[i] % 3072
                old['fc.0.weight'][row][column] = new['fc.0.weight'][row][column]
            elif max_index[i] >= 196608 and max_index[i] <= 196671 :
                pos = max_index[i] - 196608
                old['fc.0.bias'][pos] = new['fc.0.bias'][pos]
            elif max_index[i] >= 196672 and max_index[i] <= 199231:
                tmp = max_index[i] - 196672
                row = math.floor(tmp / 64)
                column = tmp % 64
                old['fc.2.weight'][row][column] = new['fc.2.weight'][row][column]
            elif max_index[i] >= 199232 and max_index[i] <= 199271:
                pos = max_index[i] - 199232
                old['fc.2.bias'][pos] = new['fc.2.bias'][pos]
            elif max_index[i] >=199272 and max_index[i] <= 199671:
                tmp = max_index[i] - 199272
                row = math.floor(tmp / 40)
                column = tmp % 40
                old['fc.4.weight'][row][column] = new['fc.4.weight'][row][column]
            elif max_index[i] >= 199672 and max_index[i] <= 199681:
                pos = max_index[i] - 199672
                old['fc.4.bias'][pos] = new['fc.4.bias'][pos]
        else:
            break
    return old

def partial_max_changed_cifar_cnn(old, new, partial):
    first = True
    for k, v in old.items():
        if first == True:
            no = np.array(old[k]).flatten()
            nn = np.array(new[k]).flatten()
            first = False
        else:
            no = np.concatenate((no,np.array(old[k]).flatten()))
            nn = np.concatenate((nn,np.array(new[k]).flatten()))
    diff = abs(nn - no)
    max_index = np.argsort(-diff)
    needed = len(diff) * partial
    for i in range(len(diff)):
        if i < needed:
            if max_index[i] <= 449:
                d1 = math.floor(max_index[i] / 75)
                d1r = max_index[i] % 75
                d2 = math.floor(d1r / 25)
                d2r = d1r % 25
                d3 = math.floor(d2r / 5)
                d4 = d2r % 5
                old['conv1.weight'][d1][d2][d3][d4] = new['conv1.weight'][d1][d2][d3][d4]
            elif max_index[i] >= 450 and max_index[i] <= 455 :
                pos = max_index[i] - 450
                old['conv1.bias'][pos] = new['conv1.bias'][pos]
            elif max_index[i] >= 456 and max_index[i] <= 2855:
                tmp = max_index[i] - 456
                d1 = math.floor(tmp / 150)
                d1r = tmp % 150
                d2 = math.floor(d1r / 25)
                d2r = d1r % 25
                d3 = math.floor(d2r / 5)
                d4 = d2r % 5
                old['conv2.weight'][d1][d2][d3][d4] = new['conv2.weight'][d1][d2][d3][d4]
            elif max_index[i] >= 2856 and max_index[i] <= 2871:
                pos = max_index[i] - 2856
                old['conv2.bias'][pos] = new['conv2.bias'][pos]
            elif max_index[i] >= 2872 and max_index[i] <= 50871:
                tmp = max_index[i] - 2872
                row = math.floor(tmp / 400)
                column = tmp % 400
                old['fc1.weight'][row][column] = new['fc1.weight'][row][column]
            elif max_index[i] >= 50872 and max_index[i] <= 50991:
                pos = max_index[i] - 50872
                old['fc1.bias'][pos] = new['fc1.bias'][pos]
            elif max_index[i] >= 50992 and max_index[i] <= 61071:
                tmp = max_index[i] - 50992
                row = math.floor(tmp / 120)
                column = tmp % 120
                old['fc2.weight'][row][column] = new['fc2.weight'][row][column]
            elif max_index[i] >= 61072 and max_index[i] <= 61155:
                pos = max_index[i] - 61072
                old['fc2.bias'][pos] = new['fc2.bias'][pos]
            elif max_index[i] >= 61156 and max_index[i] <= 61995:
                tmp = max_index[i] - 61156
                row = math.floor(tmp / 84)
                column = tmp % 84
                old['fc3.weight'][row][column] = new['fc3.weight'][row][column]
            elif max_index[i] >= 61996 and max_index[i] <= 62005:
                pos = max_index[i] - 61996
                old['fc3.bias'][pos] = new['fc3.bias'][pos]
        else:
            break
    return old




for i in range(120):
    for j in range(10):
        # download
        tmp_ps_dict = ps_dict
        model_dict = mynet[j].state_dict()
        model_dict.update(tmp_ps_dict)
        mynet[j].load_state_dict(model_dict)
        run_epoch(mynet[j], i, j)
        # upload 
        # find 10% max changed
        current_dict = mynet[j].state_dict()
        ps_dict = partial_max_changed_cifar_cnn(tmp_ps_dict,current_dict, 0.5)



# upload 1, download 1
# upload 1, download 0.1
# upload 0.1, download 0.1
print("Finish Train, Save Model!")
#torch.save(net.state_dict(), os.getcwd() + "/mnist-sub-save/mnist-train-2.pth")