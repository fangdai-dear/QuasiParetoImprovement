# Quasi-Pareto Improvement

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/RichardLitt/standard-readme)

Imbalanced subgroups are widely prevalent in medical applications, resulting in insufficient model generalization and unfair predictions that limit the application of medical AI. Current research pays less attention to the prediction differences among subgroups and lacks guidelines on addressing subgroup fairness. In this study, we trained a thyroid cancer diagnostic model on 377, 917 ultrasound images of thyroid nodules from 132, 001 patients and found that the predictive ineffectiveness of rare subtypes was masked by the overall effectiveness in the entire population. We propose the Quasi-Pareto Improvement method to enhance the prediction accuracy of imbalanced subgroups without compromising the predictive performance of the overall population. Innovatively, we introduce a domain-adaptive component and a variable-weight loss function within the Quasi-Pareto Improvement framework, which strengthens the model's ability to represent the features of imbalanced subgroups while maintaining overall predictive performance stability. Empirical results show that our method improved the predictive performance for two rare subtypes of thyroid cancer by 21.2% and 10.2%, respectively, and reduced the AUC prediction difference to 0.107 and 0.069 compared to the prediction model trained on the overall population. Additionally, we also evaluated our approach on two public datasets (the ISIC2019 skin disease multi-classification dataset and the Chexpert chest radiograph multi-classification dataset).

The Quasi-Pareto Improvement method can be broadly applied to imbalanced subgroup prediction problems in biomedical imaging, providing new insights for enhancing model generalization and addressing unfairness.

![https://github.com/fangdai-dear/QuasiParetoImprovement/scripts/Figure/figure2.png](https://github.com/fangdai-dear/QuasiParetoImprovement/blob/18fdc17629e8b471657dda916554765205995347/scripts/Figure/figure.png)

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

![https://github.com/fangdai-dear/QuasiParetoImprovement/scripts/Figure/figure2.png](https://github.com/fangdai-dear/QuasiParetoImprovement/blob/18fdc17629e8b471657dda916554765205995347/scripts/Figure/figure2.png)


## Install

This project uses [node]( ) and [npm]( ). Go check them out if you don't have them locally installed.

```sh
$ pip install -r requirements.txt
```

## Datasets
1. MICCAI 2020 TN-SCUI ultrasound image dataset

2. Chexpert chest radiograph multi-classification dataset
3.
4. ISIC2019 skin disease multi-classification dataset
## Usage

This  

```sh
$ sh ./main.sh
```


## Related Efforts

- [Art of Readme]( ) - 💌 Learn the art of writing quality READMEs.
- [open-source-template]( ) - A README template to encourage open-source contributions.

## Maintainers

[@RichardLitt](https://github.com/RichardLitt).

### Contributors

This project exists thanks to all the people who contribute. 
<a href="https://github.com/RichardLitt/standard-readme/graphs/contributors"><img src="https://opencollective.com/standard-readme/contributors.svg?width=890&button=false" /></a>


## License

[MIT](LICENSE) © Richard Littauer
