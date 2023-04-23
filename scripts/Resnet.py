from cmath import phase
from doctest import FAIL_FAST
import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from model import train_model
import cxr_dataset as CXP


def begin(modelname,Name,PATH_TO_IMAGES,learning_rate, batch_size,num_epochs,gpu=0):
    if gpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    net = models.resnet101(pretrained=True)  # 导入预训练的模型

    if os.path.exists('./modelsaved/%s' % modelname) == False:  #
        os.makedirs('./modelsaved/%s' % modelname)

    features = net.fc.in_features
    net.fc = nn.Sequential(
                nn.Linear(features, 14),nn.Sigmoid())
    data_transforms = {
        'valid': transforms.Compose([
             transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
        'train': transforms.Compose([
            transforms.Resize(300),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=50),  # 旋转10度  顺，逆 -15，15
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    }

    print("%s Initializing Datasets and Dataloaders..." % modelname)

    transformed_datasets = {}
    transformed_datasets['train'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        Name = Name,
        transform=data_transforms['train'])
    transformed_datasets['valid'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='valid',
        Name = Name,
        transform=data_transforms['valid'])
    transformed_datasets['Female'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='Female',
        Name = Name,
        transform=data_transforms['valid'])
    transformed_datasets['Male'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='Male',
        Name = Name,
        transform=data_transforms['valid'])


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        transformed_datasets['valid'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['Female'] = torch.utils.data.DataLoader(
        transformed_datasets['Female'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['Male'] = torch.utils.data.DataLoader(
        transformed_datasets['Male'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train', 'valid','Female', 'Male']}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    params_to_update = net.parameters()
    optimizer = optim.Adam(params_to_update, lr=learning_rate, betas=(0.9, 0.99))
    criterion = nn.BCELoss()

    model_ft, train_acc, val_acc, testA_acc, testB_acc, \
    train_auc, val_auc, testA_auc, testB_auc, \
    train_loss, valid_loss, testA_loss, testB_loss = train_model(net, dataloaders,dataset_sizes, criterion, optimizer, num_epochs, gpu, modelname)



if __name__ == '__main__':
    
    begin(modelname='D1_0Model',Name = 'CheXpert-v1.0',PATH_TO_IMAGES = '/export/home/daifang/CXP',learning_rate=0.0005,batch_size=128, num_epochs=200)
















