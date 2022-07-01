# CryoRL
Reinformcement Learning Enables Efficient CryoEM Data Collection

## Pre-requisite
pip install tianshou

## Sample Code

### DQN

Train a DQN: ``python train.py --dataset CryoEM-8bit-resnet18 --lr 0.001 --training-num 10 --test-num 10 --step-per-epoch 500 --seed 2 --ctf-thresh 6 --train-prediction --duration 120 --epoch 10``
Evaluate: ``python train.py --dataset CryoEM-8bit-resnet18 --lr 0.001 --training-num 10 --test-num 10 --step-per-epoch 500 --seed 2 --ctf-thresh 6 --train-prediction --duration 120 --epoch 10 --eval``
