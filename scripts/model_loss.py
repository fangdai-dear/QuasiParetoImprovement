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
from scripts.multiAUC import Metric
import numpy
from random import sample
from scripts.maximum_mean_discrepancies_loss import mmd_loss
from scripts.plot import bootstrap_auc, result_csv, plotimage

    

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
        for phase in ['train', 'valid', 'Female', 'Male']:
            if phase == 'train' and epoch != 0:
                model.train(True)
            else:
                model.train(False)
            running_loss = 0.0
            running_corrects = []
            prob_all, label_all = [], []
            B, train_acc_sum,n = 0, 0, 0
            for data in dataloaders[phase]:
                inputs, labels= data
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                optimizer.zero_grad(set_to_none=True) 
                batch_size = inputs.shape[0]
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    ### Enter 4loss progress
                    mmd = mmd_loss(outputs,labels)
                    print(mmd)






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
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            myMetic = Metric(Output,Label)
            data_auc,auc = myMetic.auROC() 
            epoch_acc =  myMetic.accuracy()
            epoch_loss = running_loss / len(Output)
            if phase in ['valid','Female', 'Male']:
                statistics = bootstrap_auc(Label, Output, [0,1,2,3,4,5])
                mean_auc = np.mean(statistics, axis=1).max()
                max_auc = np.max(statistics, axis=1).max()
                min_auc = np.min(statistics, axis=1).max()

            else: 
                mean_auc,max_auc,min_auc = 0.0,0.0,0.0

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

            if phase == 'valid':
                valid_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                val_auc_history.append(data_auc)
                Val_auc = format(data_auc, '.4f')
                V_AUC.append(auc)
                for i in range(14):
                    print("%s  : %f"%(PRED_LABEL[i],auc[i]))
                
            if phase == 'Female':
                testA_loss_history.append(epoch_loss)
                testA_acc_history.append(epoch_acc)
                testA_auc_history.append(data_auc)
                A_auc = format(data_auc, '.4f')
                F_AUC.append(auc)
                for i in range(14):
                    print("%s  : %f"%(PRED_LABEL[i],auc[i]))

            if phase == 'Male':
                testB_loss_history.append(epoch_loss)
                testB_acc_history.append(epoch_acc)
                testB_auc_history.append(data_auc)
                B_auc = format(data_auc, '.4f')
                M_AUC.append(auc)
                for i in range(14):
                    print("%s  : %f"%(PRED_LABEL[i],auc[i]))

            # if phase == 'valid'and float(Val_auc) > 0.75:
            #     print("In epoch%d,good valid = %.2f" % (epoch, float(Val_auc)))
            #     torch.save(model.state_dict(), './modelsaved/%s/epoch%d_%s_V%.3fF%.3fM%.3f.pth' % (modelname,epoch, modelname,val_auc_history[-1],testA_auc_history[-1],testB_auc_history[-1]))
                
        print("learning rate = %.6f" % optimizer.param_groups[-1]['lr'])
        if epoch != 0:
            scheduler.step()
        print("time: %.1f sec" % (time.time() - start))
        print()
        # fv=open("./result/%s_valid_auc.txt" % modelname,"w")
        # ff=open("./result/%s_female_auc.txt"% modelname,"w")
        # fm=open("./result/%s_male_auc.txt"% modelname,"w")

        # for i in range(len((V_AUC))):
        #     fv.write('epoch %s ' % i + str(V_AUC[i]) + '\n')
        #     ff.write('epoch %s ' % i + str(F_AUC[i]) + '\n')
        #     fm.write('epoch %s ' % i + str(M_AUC[i]) + '\n')
        # fv.close()
        # ff.close()
        # fm.close()
        # plotimage(train_auc_history, val_auc_history, testA_auc_history, testB_auc_history, "AUC", modelname)
        # plotimage(train_loss_history, valid_loss_history, testA_loss_history, testB_loss_history, "Loss", modelname)
        # result_csv(train_acc_history, val_acc_history, testA_acc_history, testB_acc_history, train_auc_history,
        #            val_auc_history, testA_auc_history, testB_auc_history, modelname)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)

   