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
    
for i in range(120):
    for j in range(10):
        # download
        tmp_ps_dict = ps_dict
        model_dict = mynet[j].state_dict()
        model_dict.update(tmp_ps_dict)
        mynet[j].load_state_dict(model_dict)
        run_epoch(mynet[j], i, j)
        # upload 
        ps_dict = mynet[j].state_dict()



# upload 1, download 1
# upload 1, download 0.1
# upload 0.1, download 0.1
print("Finish Train, Save Model!")
#torch.save(net.state_dict(), os.getcwd() + "/mnist-sub-save/mnist-train-2.pth")