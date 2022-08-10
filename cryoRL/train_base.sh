#!/bin/sh

python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-30t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 30.0 --duration 480 --prediction-type regress --test-prediction --dynamic-reward --gaussian-filter-size 0.5 --gpu 0 > ./train-result/Y1-Y1-Y1-30.txt 2>&1 &

