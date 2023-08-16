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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scripts.plot import create_multi_bars

label = ['No Finding',
            'Enlarged Card..',
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

df1 = pd.read_csv('./result/Male_age.txt', sep=',', names=label)
data1 = []
# for indexs in df1.index:
#         rowData = df1.loc[indexs].values[0:14]
#         rowData = rowData.tolist()
#         data1.append(rowData)
df2 = pd.read_csv('./result/Female_age.txt', sep=',', names=label)
for indexs in df1.index:
        rowData = df2.loc[indexs].values[0:14]
        rowData = rowData.tolist()
        data1.append(rowData)
AUC1 = []
auc1 = pd.read_csv('./result/Male_Model_80-all/Male_Model_80-all_auc.txt', sep=',', names=label)
auc2 = pd.read_csv(./result/Female_Model_80-all/Female_Model_80-all_auc.txt', sep=',', names=label)
# for indexs in auc1.index:
#         rowData = auc1.loc[indexs].values[0:14]
#         rowData = rowData.tolist()
#         AUC1 .append(rowData)
for indexs in auc2.index:
        rowData = auc2.loc[indexs].values[0:14]
        rowData = rowData.tolist()
        AUC1 .append(rowData)

# color1 = ['#FFD9Da','#FFC0BE','#FF82A9','#7F95D1']
# color1 = ['#CDB4DB','#FFC8DD','#FFAFFC','#BDE0FE']
color1 = ['#CDB4DB','#FFC0BE','#FF82A9','#A2D2FF']
text_label1 = [ 'Female:18-40 (9997)','Female:40-60 (17364)','Female:60-80 (21908)','Female:>80 (7840)']
base_auc1 = []
title1 = 'CXP datasets subgroup: Female 18-40, 40-60, 60-80, >80)'
save1 = "./figure/test_age(female).png"
create_multi_bars(label, data1, base_auc1, AUC1, color1, text_label1, title1, save1, bar_gap=0.03)
