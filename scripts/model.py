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
from tqdm import tqdm
from random import sample
from scripts.plot import bootstrap_auc,result_csv,plotimage
from scripts.QuasiPareto_loss import MMDLoss, BatchSpectralShrinkage, Divide
import pynvml
pynvml.nvmlInit()
from prettytable import PrettyTable


def train_model(model, LabelPredictor, DomainClassifier, 
                dataloaders, label_num, subgroup_num, criterion, 
                optimizer, optimizer_label, optimizer_domain,
                num_epochs, modelname, device):
    global VAL_auc,TEST_auc
    since = time.time()
    train_loss_history, valid_loss_history, test_loss_history= [], [], []
    train_maj_history, train_min_history, test_maj_history, test_min_history = [], [], [], []
    train_auc_history, val_auc_history, test_auc_history = [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    V_AUC, T_AUC = [],[]
    G =[]
    MMD = MMDLoss()
    BSS = BatchSpectralShrinkage()


    for epoch in range(num_epochs):
        start = time.time()

        print('{}  Epoch {}/{}  {}'.format('-' * 30, epoch, num_epochs - 1, '-' * 30))
        for phase in ['train','valid', 'test']:
            running_loss, Batch, gam_d, gam_mmd, gam_bss, gamma = 0.0, 0, 0.1, 0.8 ,0.9, 0.5
            with tqdm(range(len(dataloaders[phase])),desc='%s' % phase, ncols=100) as t:
                if Batch == 0 :
                    t.set_postfix(L = 0.000, Maj = 0.000, Min = 0.000, G = 0.5, n = 0)
                for data in dataloaders[phase]:
                    inputs, labels, subg = data
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device)).float()
                    subg = Variable(subg.cpu()).float()
                    optimizer.zero_grad(set_to_none=True) 
                    batch_size = inputs.shape[0]
                    with torch.set_grad_enabled(phase == 'train'):
                        # CNN feature extractor
                        outputs_512 = model(inputs)
                        torch.cuda.empty_cache()
                        # FCN label predictors
                        outputs_label = LabelPredictor(outputs_512.cpu())
                        outputs_maj_img, outputs_maj_lab, outputs_min_img, outputs_min_lab = Divide(outputs_label.cpu(),labels.cpu(),subg.cpu())
                        torch.cuda.empty_cache()
                        # FCN domain predictors
                        outputs_domain = DomainClassifier(outputs_512.cpu())
                        torch.cuda.empty_cache()
                         # Maj and Mix AUC
                        myMetic_maj = Metric(outputs_maj_img.cpu().detach().numpy(),outputs_maj_lab.cpu().detach().numpy())
                        data_auc_maj,auc = myMetic_maj.auROC()
                        myMetic_min = Metric(outputs_min_img.cpu().detach().numpy(),outputs_min_lab.cpu().detach().numpy())
                        data_auc_min,auc = myMetic_min.auROC()
                        torch.cuda.empty_cache()
                        
                        if phase =='train':
                            # Quasi-Pareto loss
                            
                            loss_bss = BSS(outputs_512.cpu().detach())
                            loss_mmd = MMD(subggroup = subg.cpu(), outputs = outputs_512.cpu())
                            loss_d = criterion(outputs_domain.cpu(), subg.cpu())
                            # Y_loss
                            if len(outputs_min_lab) > 5:
                                loss_y_maj = criterion(outputs_maj_img.cpu(), outputs_maj_lab.cpu())
                                loss_y_min = criterion(outputs_min_img.cpu(), outputs_min_lab.cpu())
                                loss_y = (gamma*loss_y_maj + (1-gamma)*loss_y_min).requires_grad_(True)
                            else:
                                loss_y = criterion(outputs_label.cpu(), labels.cpu())
                            
                            Loss = loss_y - gam_d*loss_d + gam_mmd*loss_mmd + gam_bss*loss_bss

                            fv = open("./result/%s/%s_Quasi-Pareto_loss.txt" % (modelname,modelname),"w")
                            fv.write('epoch%s' % epoch +'\t'+ str(loss_y.data.item()) +'\t'+str(loss_d.data.item())+'\t'+str(loss_mmd.data.item())+'\t'+str(loss_bss.data.item()) + '\n')
                            fv.close()

                            optimizer.zero_grad(), optimizer_label.zero_grad(), optimizer_domain.zero_grad()
                            Loss.backward(retain_graph=True), loss_d.backward(retain_graph=True), loss_y.backward(retain_graph=True)
                            optimizer.step(), optimizer_label.step(), optimizer_domain.step()

                            # get gamma
                            if epoch ==0 :
                                aucg = 0
                            else:
                                aucg = train_auc_history[-1]
                            x = (float(data_auc_maj) - float(aucg)) / 0.05
                            gamma = 1. / (1 + np.exp(x))
                            G.append(gamma)
                        else:
                            with torch.no_grad():
                                Loss = criterion(outputs_label.cpu(), labels.cpu())
                    # 
                    running_loss += Loss.cpu().detach()
                    if Batch == 0 :
                        Output = outputs_label.cpu().detach().numpy()
                        Label = labels.cpu().detach().numpy()
                    else:
                        Output = np.concatenate((Output, outputs_label.cpu().detach().numpy()),axis=0)
                        Label = np.concatenate((Label, labels.cpu().detach().numpy()),axis=0)
                    """
                    B:batch 
                    L:Loss
                    maj: Maj group AUC
                    min: Min group AUC
                    n: NVIDIA Memory used
                    """   
                    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(0)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
                    usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
                    n = usedMemory/meminfo              
                    t.set_postfix(L = Loss.data.item(), Maj = data_auc_maj, Min = data_auc_min, G = gamma, n = n)  # 
                    t.update()
                    Batch += 1

            myMetic = Metric(Output,Label)
            data_auc,auc = myMetic.auROC() 
            epoch_loss = running_loss / Batch
            statistics = bootstrap_auc(Label, Output, [0,1,2,3,4])
            max_auc = np.max(statistics, axis=1).max()
            min_auc = np.min(statistics, axis=1).max()

            print('{} --> Num: {} Loss: {:.4f}  AUROC: {:.4f} ({:.2f} ~ {:.2f}) (Maj {:.4f}, Min {:.4f})'.format(
                    phase, len(outputs_label), epoch_loss, data_auc, min_auc, max_auc, data_auc_maj,data_auc_min))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_auc_history.append(data_auc)

            if phase == 'valid':
                valid_loss_history.append(epoch_loss)
                val_auc_history.append(data_auc)

            if phase == 'test':
                test_loss_history.append(epoch_loss)
                test_auc_history.append(data_auc)
                test_maj_history.append(data_auc_maj)
                test_min_history.append(data_auc_min)
                table = PrettyTable()
                table.add_column('Label', label_num)
                table.add_column('AUC', auc)
                print(table)

            if phase == 'valid' and val_auc_history[-1] > 0.75:
                print("In epoch %d, good valid auc(%.3f) and save model. " % (epoch, float(VAL_auc)))
                PATH = './modelsaved/%s/e%d_%s_V%.3fT%.3f.pth' % (modelname,epoch,modelname,val_auc_history[-1],test_auc_history[-1])
                torch.save({
                            'Feature_state_dict': netA.state_dict(),
                            'Label_state_dict': netB.state_dict(),
                            'Domain_state_dict': netB.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'optimizer_label_state_dict': optimizer_label.state_dict(),
                            'optimizerr_domain_state_dict': optimizer_domain.state_dict(),
                            }, PATH)
            torch.cuda.empty_cache()     
            print()    
        print("learning rate = %.6f     time: %.1f sec" % (optimizer.param_groups[-1]['lr'], time.time() - start))
        if epoch != 0:
            scheduler.step()
        print()

        plotimage(train_auc_history, val_auc_history, test_maj_history, test_min_history, "AUC", modelname)        
        plotimage(train_loss_history, valid_loss_history, test_loss_history,test_loss_history, "Loss", modelname)
        result_csv( train_auc_history, val_auc_history, test_maj_history, test_min_history, modelname)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)

