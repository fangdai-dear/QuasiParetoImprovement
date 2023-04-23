import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage import io, transform
from torchvision import models, datasets, transforms
from alive_progress import alive_bar
# from tqdm import tqdm
# import time
class Thyroid_PF_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['Path'])
        self.df = self.df.set_index("Path")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = self.df.index[idx]
            if str(X) is not None:
                image = Image.open(os.path.join(self.path_to_images,str(X)))
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=int)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                if(self.df["Sex".strip()].iloc[idx] == 'Female'):
                    if(self.df["Age".strip()].iloc[idx].astype('int') > 80 and self.df["Sex".strip()].iloc[idx] == 'Female') or \
                        (self.df["Age".strip()].iloc[idx].astype('int') < 40 and self.df["Sex".strip()].iloc[idx] == 'Female'):
                        subg[0] = 1
                        subg[1] = 0
                    else:
                        subg[0] = 0
                        subg[1] = 1
                else:
                        subg[0] = 0
                        subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg)


class Thyroid_PM_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['Path'])
        self.df = self.df.set_index("Path")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = self.df.index[idx]
            if str(X) is not None:
                image = Image.open(os.path.join(self.path_to_images,str(X)))
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=int)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                if(self.df["Sex".strip()].iloc[idx] == 'Female'):
                    if(self.df["Age".strip()].iloc[idx].astype('int') > 80 and self.df["Sex".strip()].iloc[idx] == 'Female') or \
                        (self.df["Age".strip()].iloc[idx].astype('int') < 40 and self.df["Sex".strip()].iloc[idx] == 'Female'):
                        subg[0] = 1
                        subg[1] = 0
                    else:
                        subg[0] = 0
                        subg[1] = 1
                else:
                        subg[0] = 0
                        subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg)


class Thyroid_TC_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['Path'])
        self.df = self.df.set_index("Path")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = self.df.index[idx]
            if str(X) is not None:
                image = Image.open(os.path.join(self.path_to_images,str(X)))
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=int)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                if(self.df["Sex".strip()].iloc[idx] == 'Female'):
                    if(self.df["Age".strip()].iloc[idx].astype('int') > 80 and self.df["Sex".strip()].iloc[idx] == 'Female') or \
                        (self.df["Age".strip()].iloc[idx].astype('int') < 40 and self.df["Sex".strip()].iloc[idx] == 'Female'):
                        subg[0] = 1
                        subg[1] = 0
                    else:
                        subg[0] = 0
                        subg[1] = 1
                else:
                        subg[0] = 0
                        subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg)


class CXP_Age_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['Path'])
        self.df = self.df.set_index("Path")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = self.df.index[idx]
            if str(X) is not None:
                image = Image.open(os.path.join(self.path_to_images,str(X)))
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=int)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                if(self.df["Sex".strip()].iloc[idx] == 'Female'):
                    if(self.df["Age".strip()].iloc[idx].astype('int') > 80 and self.df["Sex".strip()].iloc[idx] == 'Female') or \
                        (self.df["Age".strip()].iloc[idx].astype('int') < 40 and self.df["Sex".strip()].iloc[idx] == 'Female'):
                        subg[0] = 1
                        subg[1] = 0
                    else:
                        subg[0] = 0
                        subg[1] = 1
                else:
                        subg[0] = 0
                        subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg)


class CXP_Race_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['Path'])
        self.df = self.df.set_index("Path")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = self.df.index[idx]
            if str(X) is not None:
                image = Image.open(os.path.join(self.path_to_images,str(X)))
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=int)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                if(self.df["Sex".strip()].iloc[idx] == 'Female'):
                    if(self.df["PRIMARY_RACE".strip()].iloc[idx] in ['Asian', 'Asian, Hispanic', 'Asian, non-Hispanic','Black or African American','Black, Hispanic', 'Black, non-Hispanic'] and self.df["Sex".strip()].iloc[idx] == 'Female'):
                        subg[0] = 1
                        subg[1] = 0
                    else:
                        subg[0] = 0
                        subg[1] = 1
                else:
                        subg[0] = 0
                        subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg)


class ISIC_Sex_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['image'])
        self.df = self.df.set_index("image")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = str(self.df.index[idx]) + '.jpg'
            if str(X) is not None:
                image = Image.open('/export/home/daifang/fairness/ISIC_2019/image/ISIC_2019_Training_Input/%s' % X)
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=float)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                if(self.df["sex".strip()].iloc[idx] == 'female'):
                    subg[0] = 1
                    subg[1] = 0
                else:
                    subg[0] = 0
                    subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg)


class ISIC_Age_Datasets(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            PRED_LABEL,
            transform=None,
            sample=0,
            finding="any"):
        self.transform = transform
        self.path_to_images = path_to_images
        self.PRED_LABEL = PRED_LABEL
        self.df = pd.read_csv("%s/CSV/%s.csv" % (path_to_images,fold))
        print("%s/CSV/%s.csv " % (path_to_images,fold), "num of images:  %s" % len(self.df))
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
            self.df = self.df.dropna(subset = ['image'])
        self.df = self.df.set_index("image")
    def __len__(self):
        return len(self.df)   
    def __getitem__(self, idx):
            X = self.df.index[idx] + '.jpg'
            if str(X) is not None:
                image = Image.open('/export/home/daifang/fairness/ISIC_2019/image/ISIC_2019_Training_Input/%s' % X)
                image = image.convert('RGB')
                label = np.zeros(len(self.PRED_LABEL), dtype=int)
                for i in range(0, len(self.PRED_LABEL)):
                    if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') == 1):
                        label[i] = 1
                    else:
                        label[i] = 0
                subg = np.zeros(2, dtype=int)
                
                if(self.df["age_approx".strip()].iloc[idx].astype('int') > 59 or self.df["age_approx".strip()].iloc[idx].astype('int') < 40):
                    subg[0] = 1
                    subg[1] = 0
                else:
                    subg[0] = 0
                    subg[1] = 1
                if self.transform:
                    image = self.transform(image)
                return (image, label, subg)