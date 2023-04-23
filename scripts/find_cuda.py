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
# from model import train_model
# import cxr_dataset as CXP
from PIL import Image

def begin(modelname,Name,PATH_TO_IMAGES,learning_rate, batch_size,num_epochs,gpu=0):
    if gpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    net = models.resnet101(pretrained=True)  # 导入预训练的模型
    net.load_state_dict(torch.load("/export/home/daifang/CXP/modelsaved/MFLabel/epoch9_MFLabel_V0.761F0.759M0.746.pth",map_location=lambda storage, loc: storage),strict=False)
    

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


    """
    0-20 valid1
    20-40 valid2
    40-60 valid3
    60-80 valid4
    """

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
    transformed_datasets['valid1'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='valid1',
        Name = Name,
        transform=data_transforms['valid'])
    transformed_datasets['valid2'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='valid2',
        Name = Name,
        transform=data_transforms['valid'])
    transformed_datasets['valid3'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='valid3',
        Name = Name,
        transform=data_transforms['valid'])
    transformed_datasets['valid4'] = CXP.CXRDataset(
        path_to_images=PATH_TO_IMAGES,
        fold='valid4',
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
    dataloaders['valid1'] = torch.utils.data.DataLoader(
        transformed_datasets['valid1'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid2'] = torch.utils.data.DataLoader(
        transformed_datasets['valid2'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid3'] = torch.utils.data.DataLoader(
        transformed_datasets['valid3'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid4'] = torch.utils.data.DataLoader(
        transformed_datasets['valid4'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=24)

    dataset_sizes = {x: len(transformed_datasets[x]) for x in ['train','valid','valid1', 'valid2','valid3', 'valid4']}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    params_to_update = net.parameters()
    optimizer = optim.Adam(params_to_update, lr=learning_rate, betas=(0.9, 0.99))
    criterion = nn.BCELoss()

    model_ft, train_acc, val_acc, testA_acc, testB_acc, \
    train_auc, val_auc, testA_auc, testB_auc, \
    train_loss, valid_loss, testA_loss, testB_loss = train_model(net, dataloaders,dataset_sizes, criterion, optimizer, num_epochs, gpu, modelname)



if __name__ == '__main__':
    
    # begin(modelname='D',Name = 'CheXpert-v1.0-small',PATH_TO_IMAGES = '/export/home/daifang/CXP/',learning_rate=0.0005,batch_size=128, num_epochs=70)
    fold = ['train']
    PRED_LABEL = [
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']
    # df = pd.read_csv("/export/home/daifang/CXP/CheXpert-v1.0-small/csv/train_1_5.csv")
    # df_f = pd.read_csv("/export/home/daifang/CXP/CheXpert-v1.0-small/train.csv")
    # 删除符合条件的指定行，并替换原始df

    # df1 = df[df['Sex'].isin(['Female'])]

    # df1 = df[(df['Age']>=80)&(df['Age']<90)]
    # print(df1.shape[0])
    #     print(df1[PRED_LABEL[k]].value_counts())
    #     print('*' * 19)



    # df1 = df
    # list1 = []
    # for k in range(len(PRED_LABEL)):
    #     list1=list1+(df1[df1[PRED_LABEL[k]]==-1].index.values.tolist())
    # list2 = list(set(list1))
    # df2 = df1.drop(index = list2, inplace=False)
    # df3 = df2[df2['Sex'].isin(['Male'])]
    # print(len(df3))
    # df_m = df3[0:80000]
    # print(len(df_f))
    # print(len(df_m))
    # DF = pd.concat([df_f,df_m],axis=0)

    # outputpath='/export/home/daifang/CXP/CheXpert-v1.0-small/train1_8W.csv'
    # DF.to_csv(outputpath,sep=',',index=False,header=False)
    l = [[0.2],[0.2],[0.3],[0.5]]
    f=open("k.txt","w")
    for i in [0,1,2,3]:
        f.write('epoch %s ' % i + str(l[i]) + '\n')
    f.close()
