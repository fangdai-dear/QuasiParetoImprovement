# Quasi-Pareto Improvement

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Imbalanced subgroups are widely prevalent in medical applications, caused by insufficient model generalization and unfair predictions that limit the application of medical AI. Current research pays less attention to the prediction differences among subgroups and lacks guidelines on addressing subgroup fairness. In this study, we trained a thyroid cancer diagnostic model on 377, 917 ultrasound images of thyroid nodules from 132, 001 patients and found that the predictive ineffectiveness of rare subtypes was masked by the overall effectiveness in the entire population. We propose the Quasi-Pareto Improvement method to enhance the prediction accuracy of imbalanced subgroups without compromising the predictive performance of the overall population. Innovatively, we introduce a domain-adaptive component and a variable-weight loss function within the Quasi-Pareto Improvement framework, which strengthens the model's ability to represent the features of imbalanced subgroups while maintaining overall predictive performance stability. Empirical results show that our method improved the predictive performance for two rare subtypes of thyroid cancer by 21.2% and 10.2%, respectively, and reduced the AUC prediction difference to 0.107 and 0.069 compared to the prediction model trained on the overall population. Additionally, we also evaluated our approach on two public datasets (the ISIC2019 skin disease multi-classification dataset and the Chexpert chest radiograph multi-classification dataset).

The Quasi-Pareto Improvement method can be broadly applied to imbalanced subgroup prediction problems in biomedical imaging, providing new insights for enhancing model generalization and addressing unfairness.

![https://github.com/fangdai-dear/QuasiParetoImprovement/scripts/Figure/figure2.png](https://github.com/fangdai-dear/QuasiParetoImprovement/blob/8ca0df7baecf650baec5305c55bd0758843bf94c/scripts/Figure/figure.png)

This repository contains:

1.   This is a code for the work being submitted, we provide only a brief description
2.   This includes model structure, training code, and partial data

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [Related Efforts](#related-efforts)
- [Maintainers](#maintainers)
- [Contributing](#contributing)
- [License](#license)

## Background

![https://github.com/fangdai-dear/QuasiParetoImprovement/scripts/Figure/figure2.png](https://github.com/fangdai-dear/QuasiParetoImprovement/blob/8ca0df7baecf650baec5305c55bd0758843bf94c/scripts/Figure/figure2.png)


## Install

This project uses requirements.txt.

```sh
$ pip install -r requirements.txt
```

## Datasets
1. MICCAI 2020 TN-SCUI ultrasound image dataset
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
### Enhancements

- [ ] The model schema diagram is updated later
- [ ] Experimental results will be updated later

## Reference
All references are listed in the article

## License
No License and copyrighted until work publication
