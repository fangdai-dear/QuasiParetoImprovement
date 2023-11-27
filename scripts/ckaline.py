
from email.mime import image
from operator import le
import torch
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import os
from torchvision import datasets, transforms
import random
import sys
from torch_cka import CKA
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image 
import pandas as pd

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def Combine(path1,path2,path):
    im_list = [Image.open(path1),Image.open(path2)]
    width0, height0 = im_list[0].size
    width1, height1 = im_list[1].size
    im_list[1] = im_list[1].resize((width0, round(height0/3)), Image.BILINEAR)
    width, height = im_list[0].size
    result = Image.new(im_list[0].mode, (width0, height +round(height0/3) ))
    for i, im in enumerate(im_list):
        result.paste(im, box=(0, i * height))
    result.save(path)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)


model1 = resnet18(pretrained=False)
model2 = resnet18(pretrained=False)

in_features1 = model1.fc.in_features
model1.fc = nn.Sequential(nn.Linear(in_features1, 2),)
in_features2 = model2.fc.in_features
model2.fc = nn.Sequential(nn.Linear(in_features2, 2),)

path = "model path"
for i in [0,1,3]:        
    if i ==0: # 'ImageNet --- Minority'
        filename= ["ImageNet Model.pth","Minority Model.pth"]
        file_nameY ="ImageNet Model" 
        file_nameX ="Minority Model"  
        
    if i ==1: # 'Majority --- Minority(mix training)'
        filename= ["Majority Model.pth","Mix training.pth"]
        file_nameY ="Majority Model"  
        file_nameX ="Mix training"
    if i ==3: # 'Majority --- Minority(QuasiPareto)'
        filename= ["Majority Model.pth","QuasiPareto.pth"]
        file_nameY ="Majority Model"  
        file_nameX ="QuasiPareto"
    model1.load_state_dict(torch.load(path+filename[0],map_location=lambda storage, loc: storage),strict=False)
    model2.load_state_dict(torch.load(path+filename[1],map_location=lambda storage, loc: storage),strict=False)  


    transform = transforms.Compose(
        [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]
    )


    dataset_B = datasets.ImageFolder(os.path.join(" ", 'minority'),
                                transform=transform)

    dataloader_B = DataLoader(dataset_B,
                            batch_size=80,
                            shuffle=False,
                            worker_init_fn=seed_worker,
                            generator=g,)

    layer1_list = ['conv1',
                'layer1.0.conv1','layer1.0.conv2',
                'layer1.1.conv1','layer1.1.conv2', 
                'layer2.0.conv1','layer2.0.conv2', 
                'layer2.1.conv1','layer2.1.conv2', 
                'layer3.0.conv1','layer3.0.conv2', 
                'layer3.1.conv1','layer3.1.conv2', 
                'layer4.0.conv1','layer4.0.conv2',
                'layer4.1.conv1','layer4.1.conv2','fc'] 


    cka = CKA(model1, model2,  
            model1_name="%s" % file_nameY,
            model2_name="%s" % file_nameX, 
            model1_layers = layer1_list,
            model2_layers = layer1_list,
            device='cuda')
    
    if i in [0,1,3]:
        print()
        print("Data:minority draw image %s  VS  %s" % (file_nameY,file_nameX))
        cka.compare(dataloader_A)
        cka.plot_results(save_path="./CKAimage/Baseline_compare.png")
        cka.plot_line(save_path="./CKAimage/Baseline_line_compare.png")
        cka.plot_data_line(save_path="./CKAimage/%s_VS_%s.png" %(file_nameY,file_nameX))
        Combine("./CKAimage/Baseline_compare.png","./CKAimage/Baseline_line_compare.png","./CKAimage/combine_%s_VS_%s.png" %(file_nameY,file_nameX))  # Y baseline X finetune


def plot_line(t1, t2, t3, k1, k2):
    plt.figure(dpi=600, figsize=(8, 3))
    plt.grid(linestyle=":", color="r")  # 绘制刻度线的网格线
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    plt.plot(x, np.array(torch.tensor(t3, device='cpu')), linestyle='-', color='deepskyblue', marker='.', mec='b', mfc='b',
             linewidth=2.5, alpha=0.8, label='△CKA   ')
    plt.plot(x, np.array(torch.tensor(k2, device='cpu')), linestyle='-', color='red', marker='.', mec='darkred', mfc='darkred',
             linewidth=2.5, alpha=0.8, label='△CKA   ')

    plt.axvspan(0.0, 1.0, facecolor='pink', alpha=0.1, **dict())  # 垂直y轴区域
    plt.axvspan(1.0, 4.0, facecolor='yellow', alpha=0.1, **dict())  # 垂直y轴区域
    plt.axvspan(5.0, 8.0, facecolor='blue', alpha=0.1, **dict())  # 垂直y轴区域
    plt.axvspan(9.0, 12.0, facecolor='red', alpha=0.1, **dict())  # 垂直y轴区域
    plt.axvspan(13.0, 16.0, facecolor='green', alpha=0.1, **dict())  # 垂直y轴区域
    plt.axvspan(16.0, 17.0, facecolor='pink', alpha=0.1, **dict())  # 垂直y轴区域
    plt.axhline(y=0, ls="-", c="black", linewidth=0.8)
    plt.xticks(np.arange(0, 18, 1.0), fontsize=8)
    # plt.yticks(np.arange(-0.4, 1.15, 0.2), fontsize=8)
    plt.yticks(np.arange(-0.4, 0.61, 0.2), fontsize=8)
    plt.legend(fontsize=12, loc=2, bbox_to_anchor=(0.05, 0.80), borderaxespad=0.)
    # plt.show()
    plt.savefig('/Figure5.pdf', format="pdf")
    plt.cla()

df1 = pd.read_excel('./CKAexample.xlsx')
t1 = df1['ImageNet VS Minority'].tolist()

t2 = df1['Majority VS Mix training'].tolist()
t3 = []
for i in range(len(t1)):
    m = t2[i] - t1[i]
    t3.append(m)

k1 = df1['Majority VS QuasiPareto'].tolist()
k2 = []
for i in range(len(t1)):
    m = k1[i] - t1[i]
    k2.append(m)

plot_line(t1, t2, t3, k1, k2)