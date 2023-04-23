from torchvision import transforms

THYROID_LABEL = ['Benign', 
                'Malignant']

CHEXPERT_LABEL = ['No Finding',
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

ISIC2019_LABEL = ['MEL',
                'NV',
                'BCC',
                'AK',
                'BKL',
                'DF',
                'VASC',
                'SCC',
                'UNK']

THYROID_SUBGROUP = [['Papillary','Follicular'],
                    ['Papillary','Medullary'],
                    ['Tertiary','Community']]

CHEXPERT_SUBGROUP = [['Female18-40','Female40-60','Female60-80','Female80'],
                    ['Female-Asian','Female-Blaok','Female-White','Female-Other']]

ISIC2019_SUBGROUP = [['Male','Female'], 
                     ['0~59', '60~85']]



def THYROID_PF():
    return (THYROID_LABEL, THYROID_SUBGROUP[0])
def THYROID_PM():
    return (THYROID_LABEL, THYROID_SUBGROUP[1])
def THYROID_TC():
    return (THYROID_TC, THYROID_SUBGROUP[2])

def CXP_Age():
    return (CHEXPERT_LABEL, CHEXPERT_SUBGROUP[0])
def CXP_Race():
    return (CHEXPERT_LABEL, CHEXPERT_SUBGROUP[1])

def ISIC2019_Sex():
    return (ISIC2019_LABEL, ISIC2019_SUBGROUP[0])
def ISIC2019_Age():
    return (ISIC2019_LABEL, ISIC2019_SUBGROUP[1])


def Transforms(name):
    if name in ["Thyroid_PF","Thyroid_PM","THYROID_TC"]:
        data_transforms_ONE = {
            'valid': transforms.Compose([
                    transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])
            ]),
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.RandomRotation(degrees=90),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])
            ])
        }
        return data_transforms_ONE
    
    if name in ["CXP_Age","CXP_Race"]:
        data_transforms_TWO = {
        'valid': transforms.Compose([
                transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ]),
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomRotation(degrees=90),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])
        }
        return data_transforms_TWO
    
    if name in ["ISIC2019_Sex","ISIC2019_Age"]:
        data_transforms_THREE = {
            'valid': transforms.Compose([
                transforms.CenterCrop(768),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])
            ]),
            'train': transforms.Compose([
                transforms.CenterCrop(768),
                transforms.Resize(224),
                transforms.RandomRotation(degrees=90), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5])
            ])
        }
        return data_transforms_THREE
        