# CryoRL
Reinforcement Learning Enables Efficient CryoEM Data Collection

## Pre-requisite

Tested in Linux.
Please find all package requirements in `requirements.txt` (tested).

## Installation

To create the usable conda environment:  
`conda create --name <env> --file requirements.txt`  
Typical installation should take only several minutes.


## Dataset  

To add new dataset, you should edit the file  `cryoRL/cryoEM_dataset.py`, and use the corresponding files in the folder `cryoRL/CryoEM_data` (and put new files into it if necessary).

This repo expect you already have the "hole target -- quality metric" pairs ready from the upstream prediction step. For the example format,
please see the files in the folder `CryoEM_data`.

Example hole images can be found from https://github.com/yilaili/cryoRL-pytorch-data. Train-test split is slightly different from [Optimized path planning surpasses human efficiency in cryo-EM imaging](https://doi.org/10.1101/2022.06.17.496614).


## Hole regressor  

The dataset folder needs to be placed paralell to this repository (CryoRL-pytorch). If you want to change that, you should also change the `config/regress_valY1` to make the path valid.

To train a regressor (sample code):  
``python train.py --backbone_net resnet50 --config configs/regress_valY1.yaml --lr 0.0005 --epoch 50 --logdir exp --loss_function l2 --batch-size 128``

To evaluate (sample code):  
``python train.py --backbone_net resnet50 --config configs/regress_valY1.yaml --lr 0.0005 --epoch 50 --logdir exp --loss_function l2 --batch-size 128 --evaluate --pretrained exp/Y1Data-resnet50-cosine-bs128-l2-e50-l0.0005/model_best.pth.tar > Y1_2_regress_8bit_res50_val_by_hl.txt``


## CryoRL (DQN)

To train a DQN (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --duration 120 --prediction-type regress --train-prediction --test-prediction``  

Note that training would usually take several hours on a typical computer with supported GPU acceleration.

To evaluate (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --train-prediction --test-prediction --eval --duration 480 --print-trajectory``

Note that evaluation would usually take less than 1 hour on a typical computer with supported GPU acceleration. This is because we were running many paralell runs on the same dataset with a random starting position.
