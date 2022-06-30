#!/bin/sh

if [ "$#" -lt 5 ]; then
    echo "Illegal number of parameters"
    echo "run_greedy_policy.sh [dataset] [gt file] [classification file] [grid|patch] [ctf] [use_gt]"
    echo "run_greedy_policy.sh resnet18 xxx.csv yyy.txt grid 4.0 gt"
    exit
fi


dataset=$1
gt_file=$2
prediction_file=$3
sort=$4
ctf=$5
others=${@:6}

echo 'dataset:'$dataset
echo 'gt:'$gt_file
echo 'prediction:'$prediction_file
echo 'sort:'$sort
echo 'ctf:'$ctf

#
#if [ -z "$gt" ]; then
#    use_gt=''
#else
#    use_gt='--use-gt'
#    dataset='GT'
#fi

echo ${dataset}' sort='${sort}' ctf='${ctf} 

#for duration in 60 120 180 240 300 360 420 480 540 600; do
for duration in 120 240 360 480 600; do
#for duration in 120; do
#for duration in 420 480; do
     python cryoEM_greedy_policy.py --prediction ${prediction_file} --annotation ${gt_file} --duration ${duration}  --sort ${sort} --ctf ${ctf} ${others}

done
