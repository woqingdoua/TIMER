# R2GenCMN

This is the implementation of [Token Imbalance Adaptation for Radiology Report Generation](https://arxiv.org/user/) at CHIL-2023.

## Citations

If you use or extend our work, please cite our paper at CHIL-2023.
```
Bibtex
```

## Requirements

- `torch==1.7.1`
- `torchvision==0.8.2`
- `opencv-python==4.4.0.42`

## Download TIMER
You can download the models we trained for each dataset from [here](https://github.com/zhjohnchan/R2GenCMN/blob/main/data/r2gencmn.md).

## Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://drive.google.com/file/d/1DS6NYirOXQf8qYieSVMvqNwuOlgAbM_E/view?usp=sharing) and then put the files in `data/mimic_cxr`.

NOTE: The `IU X-Ray` dataset is of small size, and thus the variance of the results is large.
There have been some works using `MIMIC-CXR` only and treating the whole `IU X-Ray` dataset as an extra test set.

## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

Run `bash train_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

Run `bash test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.

## Contacts
Because the experimental datasets are too large to share all of them. Please send any requests or questions to my email: [ywu10@memphis.edu](ywu10@memphis.edu).


