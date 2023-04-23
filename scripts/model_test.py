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
from torch.autograd import Variable
from scripts.multiAUC import Metric, bootstrap_auc
import numpy
from random import sample


def train_model(model,  dataloaders,dataset_sizes, criterion, optimizer, num_epochs, gpu, modelname):
    global B_auc, A_auc,Val_auc
    since = time.time()
    train_loss_history, valid_loss_history, testA_loss_history, testB_loss_history = [], [], [], []
    train_acc_history, val_acc_history, testA_acc_history, testB_acc_history = [], [], [], []
    train_auc_history, val_auc_history, testA_auc_history, testB_auc_history = [], [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    bestA_auc = 0.0
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V_AUC, F_AUC, M_AUC = [],[],[]
    for epoch in range(num_epochs):
        start = time.time()
        print('{}  Epoch {}/{}  {}'.format('-' * 10, epoch, num_epochs - 1, '-' * 10))
        pred_df = pd.DataFrame(columns=["Image Index"])
        true_df = pd.DataFrame(columns=["Image Index"])

        with open("/export/home/daifang/CXP/result/%s/%s.txt" % (modelname,modelname), "a") as filewrite:   #”a"代表着每次运行都追加txt的内容
            for phase in ['train', 'valid', 'Female', 'Male']:
                running_loss = 0.0
                running_corrects = []
                prob_all, label_all = [], []
                B, train_acc_sum,n = 0, 0, 0
                for data in dataloaders[phase]:
                    inputs, labels= data
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda()).float()
                    batch_size = inputs.shape[0]
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)
                        running_loss += loss.data * batch_size
                        output = outputs.cpu().detach().numpy()
                        label = labels.cpu().detach().numpy()
                        if B == 0 :
                            Output = output
                            Label = label
                        Output = np.concatenate((Output,output),axis=0)
                        Label = np.concatenate((Label,label),axis=0)
                        B+=1
                print(Output.shape)
                print(Label.shape)
                myMetic = Metric(Output,Label)
                data_auc,auc = myMetic.auROC() 
                epoch_acc =  myMetic.accuracy()
                epoch_loss = running_loss / len(Output)
                statistics = bootstrap_auc(Label, Output, [0,1,2,3,4,5])
                mean_auc = np.mean(statistics, axis=1).max()
                max_auc = np.max(statistics, axis=1).max()
                min_auc = np.min(statistics, axis=1).max()

                print('{} Num: {} Loss: {:.4f} ACC: {:.4f} AUROC: {:.4f} ({:.4f} ~ {:.4f})'.format(
                        phase,
                        len(Output),
                        epoch_loss,
                        epoch_acc,
                        data_auc,
                        min_auc,
                        max_auc))

                PRED_LABEL = ['No Finding',
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
                
                if phase == 'train':
                    train_loss_history.append(epoch_loss)
                    train_acc_history.append(epoch_acc)
                    train_auc_history.append(data_auc)
                    for i in range(14):
                        print("%s  : %f"%(PRED_LABEL[i],auc[i]))
                print()
                if phase == 'valid':
                    valid_loss_history.append(epoch_loss)
                    val_acc_history.append(epoch_acc)
                    val_auc_history.append(data_auc)
                    Val_auc = format(data_auc, '.4f')
                    for i in range(14):
                        print("%s  : %f"%(PRED_LABEL[i],auc[i]))
                print()
                if phase == 'Female':
                    testA_loss_history.append(epoch_loss)
                    testA_acc_history.append(epoch_acc)
                    testA_auc_history.append(data_auc)
                    A_auc = format(data_auc, '.4f')
                    for i in range(14):
                        print("%s  : %f"%(PRED_LABEL[i],auc[i]))
                print()
                if phase == 'Male':
                    testB_loss_history.append(epoch_loss)
                    testB_acc_history.append(epoch_acc)
                    testB_auc_history.append(data_auc)
                    B_auc = format(data_auc, '.4f')
                    for i in range(14):
                        print("%s  : %f"%(PRED_LABEL[i],auc[i]))
                my_auc = ','.join('%s' % a for a in auc)
                filewrite.write("\n%s" % my_auc)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)
