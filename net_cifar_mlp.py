import torch
from torchvision import models
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


class classifier(nn.Module):
    def __init__(self, train_data, test_data):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(3072, 64),
            nn.ReLU(),
            nn.Linear(64, 40),
            nn.ReLU(),
            nn.Linear(40, 10)
        )
        self.train_data = train_data
        self.test_data = test_data

    def forward(self, x):
        x = self.fc(x)
        return x
    
    def set_criterion(self, cr):
        self.criterion = cr

    def set_optimizer(self, op):
        self.optimizer = op
