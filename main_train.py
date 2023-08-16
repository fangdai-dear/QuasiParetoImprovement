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
from scripts.model import train_model
import scripts.dataset as DATA
import scripts.config as config
import argparse
from efficientnet_pytorch import EfficientNet
from scripts.fc import LabelPredictor, DomainClassifier

def main(args):
    modelname = args.modelname
    image_path = args.image_path
    if modelname =='Thyroid_PF':
        label_num,subgroup_num = config.THYROID_PF()
        Datasets = DATA.Thyroid_PF_Datasets

    if modelname =='Thyroid_PM':
        label_num,subgroup_num = config.THYROID_PM()
        Datasets = DATA.Thyroid_PM_Datasets

    if modelname =='THYROID_TC':
        label_num,subgroup_num= config.THYROID_TC()
        Datasets = DATA.Thyroid_TC_Datasets

    if modelname =='CXP_Age':
        label_num, subgroup_num = config.CXP_Age()
        Datasets = DATA.CXP_Age_Datasets

    if modelname =='CXP_Race':
        label_num, subgroup_num = config.CXP_Race()   
        Datasets = DATA.CXP_Race_Datasets
    
    if modelname =='ISIC2019_Sex':
        label_num, subgroup_num = config.ISIC2019_Sex()
        Datasets = DATA.ISIC_Sex_Datasets
    
    if modelname =='ISIC2019_Age':
        label_num, subgroup_num = config.ISIC2019_Age()
        Datasets = DATA.ISIC_Age_Datasets


    if args.architecture =='resnet':
        net = models.resnet18(pretrained=True)  
        features = net.fc.in_features
        net.fc = nn.Sequential(
            nn.Linear(features, 100))
        
    if args.architecture =='densnet':
        net = models.densenet121(pretrained=True)  
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 100))

    if args.architecture == 'efficientnet':
        net = EfficientNet.from_name('efficientnet-b0')
        feature = net._fc.in_features
        net._fc = nn.Sequential(nn.Linear(in_features=feature, out_features=100, bias=False))
      
    if os.path.exists('./modelsaved/%s' % modelname) == False:  
        os.makedirs('./modelsaved/%s' % modelname)
    if os.path.exists('./result/%s' % modelname) == False:  
        os.makedirs('./result/%s' % modelname)


    data_transforms = config.Transforms(modelname)

    print("%s Initializing Datasets and Dataloaders..." % modelname)
    
    transformed_datasets = {}
    transformed_datasets['train'] = Datasets(
        path_to_images=image_path,
        fold=args.train_data,
        PRED_LABEL=label_num,
        transform=data_transforms['train'])
    transformed_datasets['valid'] = Datasets(
        path_to_images=image_path,
        fold=args.valid_data,
        PRED_LABEL=label_num,
        transform=data_transforms['valid'])
    transformed_datasets['test'] = Datasets(
        path_to_images=image_path,
        fold=args.test_data,
        PRED_LABEL=label_num,
        transform=data_transforms['valid'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        transformed_datasets['valid'],
        batch_size=64,
        shuffle=False,
        num_workers=24)
    dataloaders['test'] = torch.utils.data.DataLoader(
        transformed_datasets['test'],
        batch_size=64,
        shuffle=False,
        num_workers=24)


    if args.modelload_path:
        net.load_state_dict(torch.load('%s' % args.modelload_path , map_location=lambda storage, loc: storage),strict=False)
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device2 = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    fc_label= LabelPredictor(len(label_num)).to(device)
    fc_domain= DomainClassifier(len(subgroup_num)).to(device)
    fc_label.initialize()
    fc_domain.initialize()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)
    optimizer_label = optim.Adam(filter(lambda p: p.requires_grad, fc_label.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)
    optimizer_domain = optim.Adam(filter(lambda p: p.requires_grad, fc_domain.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)


    if len(label_num)>2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    gam_d = args.gamma_D
    gam_mmd = args.gamma_MMD
    train_model(net, fc_label, fc_domain, 
                dataloaders, label_num, subgroup_num, criterion, 
                optimizer, optimizer_label, optimizer_domain,
                gam_d, gam_mmd, 
                args.num_epochs, modelname, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, choices=["Thyroid_PF","Thyroid_PM","THYROID_TC","CXP_Age","CXP_Race","ISIC2019_Sex","ISIC2019_Age"], default="Thyroid_PF")
    parser.add_argument("--architecture", type=str, choices= ["resnet","densnet","efficientnet"], default="resnet")
    parser.add_argument("--modelload_path", type=str,  default= None)
    parser.add_argument("--image_path", type=str,  default="./dataset/")
    parser.add_argument("--train_data", type=str, default='thyroid_train')
    parser.add_argument("--valid_data", type=str, default='thyroid_valid')
    parser.add_argument("--test_data", type=str, default='thyroid_test')
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gamma_D", type=float, default=0.2)
    parser.add_argument("--gamma_MMD", type=float, default=0.8)
    args = parser.parse_args()
    main(args)

   
