## This is a NAS-based search program for hyperspectral image classification.

## Installation

-   `Python 3.11`
-   `Pytorch 2.3.1`
-   `pip install -r requirements.txt`

## Datasets preparing

If you would like to generate them by yourself, run the samples_extraction.py script to assign the training, test and val samples.

-   `python samples_extraction.py --data_root data_dir --dist_dir output_dir --dataset dataset_name --train_num number_training_samples --val_num number_val_samples`

Then, you should set the path of sample assignment files e,g("HoustonU_dist_per_train-20.0_val-10.0.h5") in the config files.

## Architucture Searching

-   `python search.py --config-file './configs/Houston2018/search.yaml' --device '0'`
-   `python search.py --config-file './configs/MUUFL/search.yaml' --device '0'`
-   `python search.py --config-file './configs/XiongAn/search.yaml' --device '0'`

## Model Training and Inference

-   `python train.py --config-file './configs/Houston2018/train.yaml' --device '0'`
-   `python train.py --config-file './configs/MUUFL/train.yaml' --device '0'`
-   `python train.py --config-file './configs/XiongAn/train.yaml' --device '0'`
