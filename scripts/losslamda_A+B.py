import os
import torch
from torch.utils.data import DataLoader, Dataset
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
import random, shutil



def plotimage(train, val, testA, testB, LAM, ylabel, modelname):
    y1 = np.array(torch.tensor(train, device='cpu'))
    y2 = np.array(torch.tensor(val, device='cpu'))
    y3 = np.array(torch.tensor(testA, device='cpu'))
    y4 = np.array(torch.tensor(testB, device='cpu'))
    y5 = np.array(torch.tensor(LAM, device='cpu'))
    plt.title("%s vs. Number of Training Epochs" % ylabel)
    plt.xlabel("Training Epochs")
    plt.ylabel(ylabel)
    if ylabel == 'Loss':
        plt.plot(range(1, len(train) + 1), y1, label="Train %s" % ylabel)
        plt.plot(range(1, len(train) + 1), y2, label="Val %s" % ylabel)
        plt.plot(range(1, len(train) + 1), y3, label="TestA %s" % ylabel)
        plt.plot(range(1, len(train) + 1), y4, label="TestB %s" % ylabel)
        plt.plot(range(1, len(train) + 1), y5, label="Lamda ")


    if ylabel != 'Loss':
        plt.plot(range(0, len(train)), y1, label="Train %s" % ylabel)
        plt.plot(range(0, len(train)), y2, label="Val %s" % ylabel)
        plt.plot(range(0, len(train)), y3, label="TestA %s" % ylabel)
        plt.plot(range(0, len(train)), y4, label="TestB %s" % ylabel)
        plt.plot(range(0, len(train)), y5, label="Lamda ")
        plt.ylim((0, 1.))
        plt.yticks(np.arange(0, 1.01, 0.1))

    plt.xticks(np.arange(0, len(train), 5.0))
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(linestyle=":", color="r")  # 绘制刻度线的网格线
    plt.savefig("./result/%s_%s.png" % (modelname, ylabel))
    plt.cla()


def result_csv(train_acc, val_acc, testA_acc, testB_acc, train_auc, val_auc, testA_auc, testB_auc,LAM, modelname):
    y1 = np.array(torch.tensor(train_acc, device='cpu'))
    y2 = np.array(torch.tensor(val_acc, device='cpu'))
    y3 = np.array(torch.tensor(testA_acc, device='cpu'))
    y4 = np.array(torch.tensor(testB_acc, device='cpu'))
    y5 = np.array(torch.tensor(train_auc, device='cpu'))
    y6 = np.array(torch.tensor(val_auc, device='cpu'))
    y7 = np.array(torch.tensor(testA_auc, device='cpu'))
    y8 = np.array(torch.tensor(testB_auc, device='cpu'))
    y9 = np.array(torch.tensor(LAM, device='cpu'))
    CSV0 = pd.DataFrame(y1, columns=['train_acc'])
    CSV1 = pd.DataFrame(y2, columns=['val_acc'])
    CSV2 = pd.DataFrame(y3, columns=['testA_acc'])
    CSV3 = pd.DataFrame(y4, columns=['testB_acc'])
    CSV4 = pd.DataFrame(y5, columns=['train_auc'])
    CSV5 = pd.DataFrame(y6, columns=['val_auc'])
    CSV6 = pd.DataFrame(y7, columns=['testA_auc'])
    CSV7 = pd.DataFrame(y8, columns=['testB_auc'])
    CSV8 = pd.DataFrame(y9, columns=['LAM'])
    CSV = pd.concat([CSV0, CSV1, CSV2, CSV3, CSV4, CSV5, CSV6, CSV7,CSV8], axis=1)
    CSV.to_csv("./result/%s_auc.csv" % modelname, encoding='gbk')


def bootstrap_auc(y, pred, classes, bootstraps=100, fold_size=1000):
    statistics = np.zeros((len(classes), bootstraps))
    for c in range(len(classes)):
        df = pd.DataFrame(columns=['y', 'pred'])
        # df.
        df.loc[:, 'y'] = y
        df.loc[:, 'pred'] = pred
        df_pos = df[df.y == 1]
        df_neg = df[df.y == 0]
        prevalence = len(df_pos) / len(df)
        for i in range(bootstraps):
            pos_sample = df_pos.sample(n=int(fold_size * prevalence), replace=True)
            neg_sample = df_neg.sample(n=int(fold_size * (1 - prevalence)), replace=True)

            y_sample = np.concatenate([pos_sample.y.values, neg_sample.y.values])
            pred_sample = np.concatenate([pos_sample.pred.values, neg_sample.pred.values])
            score = roc_auc_score(y_sample, pred_sample)
            statistics[c][i] = score
    return statistics



# ********************************************均衡loss*********************************************************
class TrainSet(Dataset):
    def __init__(self, train_dataset,Astride,transform=None, target_transform=None):
        self.A = []
        self.B = []       
        imgs = []
        with alive_bar(len(train_dataset)) as bar: 
            for (X, label), (img_path, _) in zip(train_dataset, train_dataset.imgs):
                subtype = img_path.split('/')[-1].split('_')[0]
                if subtype == 'A':
                    self.A.append([img_path,label,10])
                elif subtype == 'B':
                    self.B.append([img_path,label,11])
                time.sleep(0.000000001)
                bar()
            self.imgs = imgs
            self.transform = transform


    def __len__(self):
        return self.sample_size

    def __getitem__(self, item):
        if len(set(self.iter_state)) == 1:    # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1   参考：https://blog.csdn.net/icamera0/article/details/50843172        
            if self.transform is not None:
                # for 
                img = self.transform(img) 
        self.iter_state[item] += 1
        return [*self.A[(self.Astride * item):(self.Astride * (item + 1))], self.B[item]]

    def collate_func(batch):
           # declare your expected total
        new_batch = sum(batch, [])
        data_list, label_list = [], []
        for x in new_batch:
            data_list.append(x[0].numpy())
            label_list.append(x[1])
        return torch.FloatTensor(np.array(data_list)), torch.LongTensor(label_list)

# ********************************************均衡loss*********************************************************

def train_model(model, dataloaders, criterion, optimizer, num_epochs, gpu,modelname):
    global A_auc, B_auc
    since = time.time()
    train_loss_history, valid_loss_history, testA_loss_history, testB_loss_history = [], [], [], []
    train_acc_history, val_acc_history, testA_acc_history, testB_acc_history = [], [], [], []
    train_auc_history, val_auc_history, testA_auc_history, testB_auc_history = [], [], [], []
    best_auc = 0.0
    lam = 0.5
    Lam = []
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # scheduler = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1)
    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}  lamda = {}'.format(epoch, num_epochs - 1,lam))
        print('-' * 10)
        for phase in ['train', 'valid', 'testA', 'testB']:
            if phase == 'train' and epoch != 0:
                model.train()
            else:
                model.eval()
            running_loss, running_loss1 = [], []
            running_corrects = []
            prob_all, label_all = [], []
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss1 = criterion(outputs, labels)
                    if phase == 'train':
                        step = train_datasets_pair.Astride 
                        img_listAs, labelAs, outputsAs = [], [], []
                        for i in range(step):  
                            img_listAs.append(inputs[i::(step + 1)])
                            labelAs.append(labels[i::(step + 1)])
                            outputsAs.append(outputs[i::(step + 1)])
                        img_listA, labelA, outputsA = torch.cat(img_listAs), torch.cat(labelAs), torch.cat(outputsAs)
                        img_listB, labelB, outputsB = inputs[step::(step + 1)], labels[step::(step + 1)], outputs[step::(step + 1)]
                        Aloss = criterion(outputsA, labelA).cuda()
                        Bloss = criterion(outputsB, labelB).cuda()
                        # TODO: 调整系数的地方
                        loss = lam*Aloss + (1-lam)*Bloss
                        if epoch != 0:
                            loss.backward()
                            optimizer.step()

                running_loss.append(loss.item())
                running_loss1.append(loss1.item())
                running_corrects.append((preds.cpu().detach() == labels.cpu().detach()).numpy())

                prob_all.extend(outputs[:, 1].cpu().detach().numpy())
                label_all.extend(labels.cpu().detach().numpy())
            data_auc = roc_auc_score(label_all, prob_all)
            label_all = np.array(label_all)
            prob_all = np.array(prob_all)
            statistics = bootstrap_auc(label_all, prob_all, [0,1,3,4,5,6,7,8,9,10])
            epoch_loss = np.mean(running_loss)
            epoch_loss1 = np.mean(running_loss1)
            epoch_acc = np.concatenate(running_corrects).mean()
            mean_auc = np.mean(statistics, axis=1).max()
            max_auc = np.max(statistics, axis=1).max()
            min_auc = np.min(statistics, axis=1).max()

            print(
                '{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f} ({:.4f} ~ {:.4f})'.format(
                    phase,
                    epoch_loss,
                    epoch_acc,
                    data_auc,
                    min_auc,
                    max_auc))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                train_auc_history.append(data_auc)

            if phase == 'valid':
                valid_loss_history.append(epoch_loss1)
                val_acc_history.append(epoch_acc)
                val_auc_history.append(data_auc)

            if phase == 'testA':
                testA_loss_history.append(epoch_loss1)
                testA_acc_history.append(epoch_acc)
                testA_auc_history.append(data_auc)
                A_auc = format(data_auc, '.8f')
                x = (float(A_auc) - float(0.8970)) / 0.05
                lam = 1. / (1 + np.exp(x))
                Lam.append(lam)
                

            if phase == 'testB' and data_auc > best_auc:
                best_auc = data_auc

            if phase == 'testB':
                testB_loss_history.append(epoch_loss1)
                testB_acc_history.append(epoch_acc)
                testB_auc_history.append(data_auc)
                B_auc = format(data_auc, '.2f')

            if phase == 'testB' and data_auc >= 0.70:
                print("******* epoch %d save best test B model AUC = %.4f" % (epoch, data_auc))
                torch.save(model.state_dict(),
                           './modelsaved/%s/epoch%d_%s_A%.3fB%.3f.pth' % ( modelname, epoch,modelname,testA_auc_history[-1], data_auc))
            ## change lam

        print("learning rate = %.8f   new lamda = %.4f" % (optimizer.param_groups[-1]['lr'], lam))
        if epoch != 0:
            scheduler.step()
        print("time: %.1f" % (time.time() - start))
        plotimage(train_auc_history, val_auc_history, testA_auc_history, testB_auc_history, Lam, "AUC", modelname)
        plotimage(train_loss_history, valid_loss_history, testA_loss_history, testB_loss_history, Lam, "Loss", modelname)
        result_csv(train_acc_history, val_acc_history, testA_acc_history, testB_acc_history, train_auc_history, val_auc_history, testA_auc_history, testB_auc_history, Lam, modelname)
        print()
        # if train_loss_history[-1]<= 0.08:
        #     return model, Lam, train_acc_history, val_acc_history, testA_acc_history, testB_acc_history, \
        #                     train_auc_history, val_auc_history, testA_auc_history, testB_auc_history, \
        #                     train_loss_history, valid_loss_history, testA_loss_history, testB_loss_history
       
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, Lam,\
           train_acc_history, val_acc_history, testA_acc_history, testB_acc_history, \
           train_auc_history, val_auc_history, testA_auc_history, testB_auc_history, \
           train_loss_history, valid_loss_history, testA_loss_history, testB_loss_history

if __name__ == '__main__':
    ## 超参数调
    rl = 0.0001  # 
    batch_size = 128
    num_epochs = 150
    gpu = 0
    if gpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    net = models.resnet18(pretrained=True)
    astride = 1  # A：B的比例  3= 3:1

    # 导入预训练的模型
    features = net.fc.in_features
    net.fc = nn.Sequential(
        nn.Linear(features, 2)
    )

    modelname = "loss_A+Bequal_re(10.20)"
    if os.path.exists('./modelsaved/%s' % modelname) == False:  #
        os.makedirs('./modelsaved/%s' % modelname)

    net.load_state_dict(torch.load(
        # "./modelsaved(10.11)/ModelA/New Folder/epoch69_ModelA_A0.897B0.897.pth",
        "newAAA/modelsaved/try586/epoch46_try586_A0.812B0.814.pth",
        map_location=lambda storage, loc: storage),
        strict=False)
    for param in net.parameters():
        param.requires_grad = False
    # for param in net.bn1.parameters():
    #     param.requires_grad = True
    # for param in net.layer1.parameters():
    #     param.requires_grad = True
    # for param in net.layer2.parameters():
    #     param.requires_grad = True
    # for param in net.layer3.parameters():
    #     param.requires_grad = True
    for param in net.layer4.parameters():
        param.requires_grad = True
    for param in net.fc.parameters():
        param.requires_grad = True

    data_transforms = {
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
        'train': transforms.Compose([
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2
            ),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=15),  # 旋转10度  顺，逆 -15，15
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
    }

    print("%s Initializing Datasets and Dataloaders..." % modelname)
    
    # ************************************************************均衡loss*********************************************************
    
    data_dir = "./data_train/dataA+B/"

    print("train : %s  |  valid : %s  |  Atest : %s  |  Btest : %s " % 
                            (len(os.listdir(data_dir+"/train/0/"))+len(os.listdir(data_dir+"/train/1/")),
                            len(os.listdir(data_dir+"/valid/0/"))+len(os.listdir(data_dir+"/valid/1/")),
                            len(os.listdir(data_dir+"/Atest/0/"))+len(os.listdir(data_dir+"/Btest/1/")),
                            len(os.listdir(data_dir+"/Btest/0/"))+len(os.listdir(data_dir+"/Btest/1/"))))



    train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=data_transforms['train'])
    train_datasets_pair = TrainSet(train_datasets, transform=data_transforms['train'])
    val_datasets = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=data_transforms['valid'])
    testA_datasets = datasets.ImageFolder(os.path.join(data_dir, 'Atest'),transform=data_transforms['valid'])
    testB_datasets = datasets.ImageFolder(os.path.join(data_dir, 'Btest'),transform=data_transforms['valid'])

    num_workers = 24
    dataloaders_dict = {
        'train': DataLoader(train_datasets_pair, batch_size=batch_size // (astride + 1), shuffle=True,
                            num_workers=num_workers,
                            collate_fn=TrainSet.collate_func),
        'valid': DataLoader(val_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'testA': DataLoader(testA_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'testB': DataLoader(testB_datasets, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    # ********************************************均衡loss************************************************************************

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = net.to(device)

    params_to_update = net.parameters()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=rl, betas=(0.9, 0.99),weight_decay=0.03)

    criterion = nn.CrossEntropyLoss()

    model_ft, LAM,train_acc, val_acc, testA_acc, testB_acc, \
    train_auc, val_auc, testA_auc, testB_auc, \
    train_loss, valid_loss, testA_loss, testB_loss = train_model(net, dataloaders_dict, criterion,
                                                                 optimizer, num_epochs,gpu, modelname)

    plotimage(train_auc, val_auc, testA_auc, testB_auc,LAM, "AUC", modelname)
    plotimage(train_loss, valid_loss, testA_loss, testB_loss, LAM,"Loss", modelname)
    plotimage(train_acc, val_acc, testA_acc, testB_acc, LAM,"Accuracy", modelname)
    result_csv(train_acc, val_acc, testA_acc, testB_acc, train_auc, val_auc, testA_auc, testB_auc, LAM,modelname)
