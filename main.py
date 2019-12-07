import os
import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据

from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import classifier
from parameter_server import PS

from itertools import chain

import matplotlib.pyplot as plt
from numpy import math
# %matplotlib inline



train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)



def data_tf(x):
    x = np.array(x, dtype='float32') / 255
    x = (x - 0.5) / 0.5 # 标准化，这个技巧之后会讲到
    x = x.reshape((-1,)) # 拉平
    x = torch.from_numpy(x)
    return x

def run_epoch(net, epoch, id):
        losses = []
        acces = []
        eval_losses = []
        eval_acces = []
        train_loss = 0
        train_acc = 0
        net.train()
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

train_set = mnist.MNIST('./data', train=True, transform=data_tf, download=True) # 重新载入数据集，申明定义的数据变换
test_set = mnist.MNIST('./data', train=False, transform=data_tf, download=True)

# split dataset to 10 shares
train_list = torch.utils.data.random_split(train_set, [6000, 6000,6000,6000,6000,6000,6000,6000,6000,6000])
print('train_0:', len(train_list[0]), 'train_1:', len(train_list[1]))

val_list= torch.utils.data.random_split(test_set, [1000, 1000,1000,1000,1000,1000,1000,1000,1000,1000])
print('val_0:', len(val_list[0]), 'val_1:', len(val_list[1]))




# global simple PS
ps_dict = torch.load(os.getcwd()+"/mnist-6000-1000-base.pth")

# mynet list
mynet = []
# netwrok definition
for i in range(10):
    train_data = DataLoader(train_list[i], batch_size=64, shuffle=True)
    test_data = DataLoader(val_list[0], batch_size=128, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    new_net = classifier(train_data, test_data)
    #load model from base pth
    new_net.load_state_dict(torch.load(os.getcwd()+"/mnist-6000-1000-base.pth"))
    new_net.set_criterion(criterion)
    optimizer = torch.optim.SGD(new_net.parameters(), 1e-1) # 使用随机梯度下降，学习率 0.1
    new_net.set_optimizer(optimizer)
    mynet.append(new_net)

# add initial weight to ps
# ps_list = chain.from_iterable(mynet[0].state_dict()['0.weight']) + chain.from_iterable(mynet[0].state_dict()['0.bias']) + \
#     chain.from_iterable(mynet[0].state_dict()['2.weight']) + chain.from_iterable(mynet[0].state_dict()['2.bias']) + \
#     chain.from_iterable(mynet[0].state_dict()['4.weight']) + chain.from_iterable(mynet[0].state_dict()['4.bias']) + \
#     chain.from_iterable(mynet[0].state_dict()['6.weight']) + chain.from_iterable(mynet[0].state_dict()['6.bias'])

# def extract_ps_list(net):
#     Chain.from_iterable(net.state_dict()['0.weight']) + chain.from_iterable(net.state_dict()['0.bias']) + \
#     chain.from_iterable(net.state_dict()['2.weight']) + chain.from_iterable(net.state_dict()['2.bias']) + \
#     chain.from_iterable(net.state_dict()['4.weight']) + chain.from_iterable(net.state_dict()['4.bias']) + \
#     chain.from_iterable(net.state_dict()['6.weight']) + chain.from_iterable(net.state_dict()['6.bias'])
# myPS = PS(ps_list)
# def partial_max_changed(old, new, partial):
#     for k, v in old.items():
#         no = np.array(old[k])
#         nn = np.array(new[k])
#         diff = abs(nn - no)
#         if "bias" not in k:
#             min_index = np.argsort(diff, axis = 0)
#             max_index = min_index[:,::-1]
#             needed = len(old[k][0]) * partial
#             for i in range(len(old[k])):
#                 for j in range(len(old[k][0])):
#                     if j < needed:
#                         old[k][i][max_index[i][j]] = new[k][i][max_index[i][j]]
#                     else:
#                         break
#         else:
#             max_index = np.argsort(-diff)
#             needed = len(diff) * partial
#             for i in range(len(old[k])):
#                 if i < needed:
#                     old[k][max_index[i]] = new[k][max_index[i]]
#                 else:
#                     break
#     return old

def partial_max_changed(old, new, partial):
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
            if max_index[i] <= 313599:
                row = math.floor(max_index[i] / 784)
                column = max_index[i] % 784
                old['fc.0.weight'][row][column] = new['fc.0.weight'][row][column]
            elif max_index[i] >= 313600 and max_index[i] <= 313999 :
                pos = max_index[i] - 313600
                old['fc.0.bias'][pos] = new['fc.0.bias'][pos]
            elif max_index[i] >= 314000 and max_index[i] <= 393999:
                tmp = max_index[i] - 314000
                row = math.floor(tmp / 400)
                column = tmp % 400
                old['fc.2.weight'][row][column] = new['fc.2.weight'][row][column]
            elif max_index[i] >= 394000 and max_index[i] <= 394199:
                pos = max_index[i] - 394000
                old['fc.2.bias'][pos] = new['fc.2.bias'][pos]
            elif max_index[i] >=394200 and max_index[i] <= 414199:
                tmp = max_index[i] - 394200
                row = math.floor(tmp / 200)
                column = tmp % 200
                old['fc.4.weight'][row][column] = new['fc.4.weight'][row][column]
            elif max_index[i] >= 414200 and max_index[i] <= 414299:
                pos = max_index[i] - 414200
                old['fc.4.bias'][pos] = new['fc.4.bias'][pos]
            elif max_index[i] >= 414300 and max_index[i] <= 415299:
                tmp = max_index[i] - 414300
                row = math.floor(tmp / 100)
                column = tmp % 100
                old['fc.6.weight'][row][column] = new['fc.6.weight'][row][column]
            elif max_index[i] >= 415300 and max_index[i] <= 415309:
                pos = max_index[i] - 415300
                old['fc.6.bias'][pos] = new['fc.6.bias'][pos]
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
        ps_dict = partial_max_changed(tmp_ps_dict,current_dict, 0.1)



# upload 1, download 1
# upload 1, download 0.1
# upload 0.1, download 0.1
print("Finish Train, Save Model!")
#torch.save(net.state_dict(), os.getcwd() + "/mnist-sub-save/mnist-train-2.pth")