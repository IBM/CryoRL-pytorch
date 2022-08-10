#!/bin/sh

program=$1
dataset=$2
output_dir=$3
test_num=10
ctf_thresh=$4
gpu=$5
others=${@:6}

#penalty=$5

use_penalty=''
hidden_sizes='128 256 128'
#hidden_sizes='256 512 256'

#for duration in 60 120 180 240 300 360; do
#for duration in 480; do
for duration in 120 240 360 480 600; do
    #result=`CUDA_VISIBLE_DEVICES=${gpu} python train.py --dataset ${dataset} --lr 0.01 --training-num 10 --test-num ${test_num} --logdir ${output_dir}/${dataset} --step-per-epoch 2500 --hidden-sizes ${hidden_sizes} ${use_penalty} --random-policy --eval --duration ${duration} --print-trajectory|grep 'End of Trajectory'|tail -1`
    #echo 'random-policy'${penalty}' '${duration}' '${result}

    #gt-gt-hard
#    result=`CUDA_VISIBLE_DEVICES=${gpu} python train.py --dataset ${dataset} --lr 0.01 --training-num 10 --test-num ${test_num} --logdir ${output_dir}/${dataset} --step-per-epoch 2500 --hidden-sizes ${hidden_sizes} ${use_penalty} --ctf-thresh ${ctf_thresh} --eval --duration ${duration} ${others} --print-trajectory|grep 'End of Trajectory'|tail -1`
#    echo 'gt-gt-hard'${penalty}' '${duration}' '${ctf_thresh}' '${result}

    #gt-pred-hard
    #result=`CUDA_VISIBLE_DEVICES=${gpu} python ${program} --dataset ${dataset} --lr 0.01 --training-num 10 --test-num ${test_num} --logdir ${output_dir}/${dataset} --step-per-epoch 2500 --hidden-sizes ${hidden_sizes} ${use_penalty} --test-prediction --ctf-thresh ${ctf_thresh} --use-one-hot --eval --duration ${duration} ${others} --print-trajectory|grep 'End of Trajectory' |tail -1`
    #echo 'gt-pred-hard'${penalty}' '${duration}' '${ctf_thresh}' '${result}

    #gt-pred-hard
    result=`CUDA_VISIBLE_DEVICES=${gpu} python ${program} --dataset ${dataset} --lr 0.01 --training-num 10 --test-num ${test_num} --logdir ${output_dir}/${dataset} --step-per-epoch 2500 --hidden-sizes ${hidden_sizes} ${use_penalty} --prediction-type regress --test-prediction --dynamic-reward --gaussian-filter-size 0.5 --ctf-thresh ${ctf_thresh} --eval --duration ${duration} ${others} --print-trajectory|grep 'End of Trajectory' |tail -1`
    echo 'gt-pred-hard'${penalty}' '${duration}' '${ctf_thresh}' '${result}

    #pred-pred-hard
    #result=`CUDA_VISIBLE_DEVICES=${gpu} python ${program} --dataset ${dataset} --lr 0.01 --training-num 10 --test-num ${test_num} --logdir ${output_dir}/${dataset} --step-per-epoch 2500 --hidden-sizes ${hidden_sizes} ${use_penalty} --train-prediction --test-prediction --use-one-hot --ctf-thresh ${ctf_thresh} --eval --duration ${duration} ${others} --print-trajectory |grep 'End of Trajectory'|tail -1`
   #echo 'pred-pred-hard'${penalty}' '${duration}' '${ctf_thresh}' '${result}

    #pred-pred-soft
    #result=`CUDA_VISIBLE_DEVICES=${gpu} python train.py --dataset ${dataset} --lr 0.01 --training-num 10 --test-num ${test_num} --logdir ${output_dir}/${dataset} --step-per-epoch 2500 --hidden-sizes ${hidden_sizes} ${use_penalty} --train-prediction --test-prediction --ctf-thresh ${ctf_thresh} --eval --duration ${duration} --print-trajectory|grep 'End of Trajectory'|tail -1`
    #echo 'pred-pred-soft'${penalty}' '${duration}' '${ctf_thresh}' '${result}

done
