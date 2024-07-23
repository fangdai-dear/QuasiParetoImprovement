# Quasi-Pareto Improvement
[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

## Enhancing the Generalizability and Fairness of Ultrasonographical AI Model among Heterogeneous Thyroid Nodule Population by a Novel Quasi-Pareto Improvement

Siqiong Yao, Fang Dai, Peng Sun, Weituo Zhang, Biyun Qian, Hui Lu
### Abstract

Artificial Intelligence (AI) models for medical diagnosis often face challenges of generalizability and fairness. We highlighted the algorithmic unfairness in a large thyroid ultrasound dataset with significant diagnostic performance disparities across subgroups linked causally to sample size imbalances. To address this, we introduced the Quasi-Pareto Improvement (QPI) approach and a deep learning implementation (QP-Net) combining multi-task learning and domain adaptation to improve model performance among disadvantaged subgroups without compromising overall population performance. On the thyroid ultrasound dataset, our method significantly mitigated the area under curve (AUC) disparity for three less-prevalent subgroups by 0.213, 0.112, and 0.173 while maintaining the AUC for dominant subgroups; we also further confirmed the generalizability of our approach on two public datasets: the ISIC2019 skin disease dataset and the CheXpert chest radiograph dataset. Here we show the QPI approach to be widely applicable in promoting AI for equitable healthcare outcomes.


For details, see[[Nature Communications Paper](https://www.nature.com/articles/s41467-024-44906-y#citeas)].


![https://github.com/fangdai-dear/QuasiParetoImprovement/scripts/Figure/figure.png](https://github.com/fangdai-dear/QuasiParetoImprovement/blob/8ca0df7baecf650baec5305c55bd0758843bf94c/scripts/Figure/figure.png)

This repository contains:

1.   This is a code for the work being submitted, we provide only a brief description
2.   This includes model structure, training code

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Enhancements](#enhancements)
- [Reference](#reference)
- [License](#license)

## Model architecture

![figure2](https://github.com/fangdai-dear/QuasiParetoImprovement/blob/master/scripts/Figure/figure2.png)


## Install

This project uses requirements.txt.

```sh
$ pip install -r requirements.txt
```

## Datasets
1. We have shared part of the thyroid ultrasound dataset for verification. Please refer to this article for other studies using this dataset.
```sh
├─Thyroid
    └─PTC
    └─FTC
    └─MTC
```
1. MICCAI 2020 TN-SCUI ultrasound image dataset (This study took into account the clinical significance of the contest and segmented according to the data segmentation style of the contest)
```sh
├─Thyroid
    └─TNS
        ├─test
        │  ├─0
        │  └─1
        ├─train
        │  ├─0
        │  └─1
```
2. Chexpert chest radiograph multi-classification dataset
```sh
├─CheXpert-v1.0
│  ├─train
│  │  └─patient00001
│  │      └─study1
│  │              view1_frontal.jpg
│  │              
│  └─valid
│      └─patient64541
│          └─study1
│                  view1_frontal.jpg
```
3. ISIC2019 skin disease multi-classification dataset
```     sh              
├─ISIC
│  ├─ISIC_2018
│  │      ISIC_0024306.jpg
│  │      
│  └─ISIC_2019
│          ISIC_0000000.jpg
│          
```
Partial thyroid ultrasonography data used in this study are subject to privacy restrictions, but may be anonymized and made available upon reasonable request to the corresponding author.

## Usage

This  

```sh
$ sh ./main.sh
```
```sh
├─CSV
│      CXP_female_age.csv
│      CXP_female_race.csv
│      CXP_male_age.csv
│      CXP_male_race.csv
│      CXP_test_age.csv
│      CXP_train_age.csv
│      CXP_train_race.csv
│      CXP_valid_race.csv
│      ISIC_2019_Test.csv
│      ISIC_2019_Training_age.csv
│      ISIC_2019_Training_sex.csv
│      ISIC_2019_valid.csv
```
## Citing
If you use our code and any information in your research, please consider citing with the following BibTex.
```text
@article{yao2024enhancing,
  title={Enhancing the fairness of AI prediction models by Quasi-Pareto improvement among heterogeneous thyroid nodule population},
  author={Yao, Siqiong and Dai, Fang and Sun, Peng and Zhang, Weituo and Qian, Biyun and Lu, Hui},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={1--13},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## Reference
All references are listed in the article

## Licence
[Licence](https://github.com/fangdai-dear/QuasiParetoImprovement/blob/master/LICENSE)

