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

To add new dataset, you should edit the file  `cryoEM_dataset.py`, and use the corresponding files in the folder `CryoEM_data` (and put new files
into it if necessary).

This repo expect you already have the "hole target -- quality metric" pairs ready from the upstream prediction step. For the example format,
please see the files in the folder `CryoEM_data`.


## Sample Code

### DQN

Train a DQN: ``python train.py --dataset CryoEM-8bit-resnet18 --lr 0.001 --training-num 10 --test-num 10 --step-per-epoch 500 --seed 2 --ctf-thresh 6 --train-prediction --duration 120 --epoch 10``  

Note that training would usually take several hours on a typical computer with supported GPU acceleration.

Evaluate: ``python train.py --dataset CryoEM-8bit-resnet18 --lr 0.001 --training-num 10 --test-num 10 --step-per-epoch 500 --seed 2 --ctf-thresh 6 --train-prediction --duration 120 --epoch 10 --eval``

Note that evaluation would usually take less than 1 hour on a typical computer with supported GPU acceleration. This is because we were running many paralell runs on the same dataset with a random starting position.
