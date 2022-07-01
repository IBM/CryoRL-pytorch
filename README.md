# CryoRL
Reinformcement Learning Enables Efficient CryoEM Data Collection

## Pre-requisite
pip install tianshou

## Sample Code

``python train.py --dataset CryoEM-8bit-resnet18 --lr 0.001 --training-num 10 --test-num 10 --step-per-epoch 500 --seed 2 --ctf-thresh 6 --train-prediction --duration 120``
