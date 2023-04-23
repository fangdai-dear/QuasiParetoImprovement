import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# Fully connected neural network
class LabelPredictor (nn.Module):
    def __init__(self, num_classes):
        super(LabelPredictor , self).__init__()
        self.fc1 = nn.Linear(100, 70) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(70, 50) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 20) 
        self.relu3 = nn.ReLU()
        self.fc4 =  nn.Sequential(nn.Linear(20, num_classes), nn.Sigmoid())
    
    def forward(self, x):
        x1 = self.fc1(x)
        out1 = self.relu1(x1)
        x2 = self.fc2(out1)
        out2 = self.relu2(x2)
        x3 = self.fc3(out2)
        out3 = self.relu3(x3)
        out4 = self.fc4(out3)
        return out4
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data,gain=tanh_gain)


class DomainClassifier (nn.Module):
    def __init__(self, num_classes):
        super(DomainClassifier , self).__init__()
        self.fc1 = nn.Linear(100, 70) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(70, 50) 
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 20) 
        self.relu3 = nn.ReLU()
        self.fc4 =  nn.Sequential(nn.Linear(20, 2), nn.Sigmoid())
    
    def forward(self, x):
        x1 = self.fc1(x)
        out1 = self.relu1(x1)
        x2 = self.fc2(out1)
        out2 = self.relu2(x2)
        x3 = self.fc3(out2)
        out3 = self.relu3(x3)
        out4 = self.fc4(out3)
        return out4
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                tanh_gain = nn.init.calculate_gain('tanh')
                nn.init.xavier_uniform_(m.weight.data,gain=tanh_gain)
