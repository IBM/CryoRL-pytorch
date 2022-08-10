#!/bin/sh

# python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-4t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 4.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 0 > ./test-result/Y1-Y1-Y1-4-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-5t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 5.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 1 > ./test-result/Y1-Y1-Y1-5-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 2 > ./test-result/Y1-Y1-Y1-6-traj.txt

# python train.py --dataset CryoEM-8bit-resnet50-Y3 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-4t/CryoEM-8bit-resnet50-Y3 --step-per-epoch 2500 --ctf-thresh 4.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 0 > ./test-result/Y3-Y3-Y3-4-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-Y3 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-5t/CryoEM-8bit-resnet50-Y3 --step-per-epoch 2500 --ctf-thresh 5.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 1 > ./test-result/Y3-Y3-Y3-5-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-Y3 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y3 --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 2 > ./test-result/Y3-Y3-Y3-6-traj.txt &

# python train.py --dataset CryoEM-8bit-resnet50-M-Y2-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-4t/CryoEM-8bit-resnet50-M-Y2-new --step-per-epoch 2500 --ctf-thresh 4.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 3 > ./test-result/Y2-M-Y2-4-traj.txt
# python train.py --dataset CryoEM-8bit-resnet50-M-Y2-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-5t/CryoEM-8bit-resnet50-M-Y2-new --step-per-epoch 2500 --ctf-thresh 5.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 0 > ./test-result/Y2-M-Y2-5-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-M-Y2-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-M-Y2-new --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 1 > ./test-result/Y2-M-Y2-6-traj.txt &

# python train.py --dataset CryoEM-8bit-resnet50-M-Y2-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-4t-trans/CryoEM-8bit-resnet50-M-Y2-new --step-per-epoch 2500 --ctf-thresh 4.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 2 > ./test-result/Y2-M-Y1-4-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-M-Y2-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-5t-trans/CryoEM-8bit-resnet50-M-Y2-new --step-per-epoch 2500 --ctf-thresh 5.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 3 > ./test-result/Y2-M-Y1-5-traj.txt
# python train.py --dataset CryoEM-8bit-resnet50-M-Y2-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t-trans/CryoEM-8bit-resnet50-M-Y2-new --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 0 > ./test-result/Y2-M-Y1-6-traj.txt &

# python train.py --dataset CryoEM-8bit-resnet50-M-Y3-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-4t-trans/CryoEM-8bit-resnet50-M-Y3-new --step-per-epoch 2500 --ctf-thresh 4.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 1 > ./test-result/Y3-M-Y1-4-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-M-Y3-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-5t-trans/CryoEM-8bit-resnet50-M-Y3-new --step-per-epoch 2500 --ctf-thresh 5.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 2 > ./test-result/Y3-M-Y1-5-traj.txt &
# python train.py --dataset CryoEM-8bit-resnet50-M-Y3-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t-trans/CryoEM-8bit-resnet50-M-Y3-new --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 3 > ./test-result/Y3-M-Y1-6-traj.txt

python train.py --dataset CryoEM-8bit-resnet50-M-Y3ft-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-4t-trans/CryoEM-8bit-resnet50-M-Y3ft-new --step-per-epoch 2500 --ctf-thresh 4.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 1 > ./test-result/Y3-Y3ft-Y1-4-traj.txt &
python train.py --dataset CryoEM-8bit-resnet50-M-Y3ft-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-5t-trans/CryoEM-8bit-resnet50-M-Y3ft-new --step-per-epoch 2500 --ctf-thresh 5.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 2 > ./test-result/Y3-Y3ft-Y1-5-traj.txt &
python train.py --dataset CryoEM-8bit-resnet50-M-Y3ft-new --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t-trans/CryoEM-8bit-resnet50-M-Y3ft-new --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --test-prediction --eval --duration 480 --dynamic-reward --gaussian-filter-size 0.5 --print-trajectory --gpu 3 > ./test-result/Y3-Y3ft-Y1-6-traj.txt
