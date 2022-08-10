# CryoRL: Reinforcement Learning-powered cryoEM data collection

![IMG_0029](https://user-images.githubusercontent.com/109689432/183966746-42acd7a4-f482-4d35-b107-108bfc764c3d.jpg)

CryoRL utilizes a two-fold regressor and reinforcement learning network to determine an optimized cryoEM microscope trajectory from low-magnification images. CryoRL's image regressor predicts micrograph quality from cropped hole-level images. The resulting quality scores are used by cryoRL's DQN-based reinforcement learning policy to map an optimized trajectory of target holes. CryoRL is currently still in testing; more can be found in ["Outperforming cryoEM experts in data collection using artificial intelligence", Li Y, Fan Q, et al.](https://www.biorxiv.org/content/10.1101/2022.06.17.496614v1.full).

## Step 0: Installation

To start, create a new suitable conda environment :  
`conda create --name <env> --file requirements.txt`  
Typical installation should take only several minutes.

For the full list of dependencies as tested in Linux, see 'requirements.txt'.

## Step 1: CryoEM Grid Survey Data Preparation

<p align="center">
  <img src="https://user-images.githubusercontent.com/109689432/183967204-659c0aa2-34e4-471b-9b85-309b5d7869df.jpg" width="400" height="250">
</p>

CryoEM grids were surveyed at the patch level, and the resulting .mrc files converted to 8-bit .png format using e2proc2d.py from EMAN2. Individual hole images were identified with Leginon hole coordinates and cropped to boxes of 150x150 px.

Depending on the microscope setup used, these steps may differ; cryoRL will accept cropped hole-level images in .png format. Example hole images can be found from https://github.com/yilaili/cryoRL-pytorch-data.

## Step 2: Hole-Level Image Regression

## Step 3: 

# Retraining CryoRL Models

To add new dataset, you should edit the file  `cryoRL/cryoEM_dataset.py`, and use the corresponding files in the folder `cryoRL/CryoEM_data` (and put new files into it if necessary).

This repo expect you already have the "hole target -- quality metric" pairs ready from the upstream prediction step. For the example format,
please see the files in the folder `CryoEM_data`.

Example hole images can be found from https://github.com/yilaili/cryoRL-pytorch-data. Train-test split is slightly different from [Optimized path planning surpasses human efficiency in cryo-EM imaging](https://doi.org/10.1101/2022.06.17.496614).


## Hole regressor  

To train a regressor (sample code):  
``python train.py --backbone_net resnet50 --config configs/regress_valY1.yaml --lr 0.0005 --epoch 50 --logdir exp --loss_function l2 --batch-size 128``

To evaluate (sample code):  
``python train.py --backbone_net resnet50 --config configs/regress_valY1.yaml --lr 0.0005 --epoch 50 --logdir exp --loss_function l2 --batch-size 128 --evaluate --pretrained exp/YData-resnet50-cosine-bs128-l2-e50-l0.0005/model_best.pth.tar > Y1_2_regress_8bit_res50_val_by_hl.txt``


## CryoRL (DQN)

To train a DQN (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --duration 120 --prediction-type regress --train-prediction --test-prediction``  

Note that training would usually take several hours on a typical computer with supported GPU acceleration.

To evaluate (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --train-prediction --test-prediction --eval --duration 480 --print-trajectory``

Note that evaluation would usually take less than 1 hour on a typical computer with supported GPU acceleration. This is because we were running many parallel runs on the same dataset with a random starting position.
