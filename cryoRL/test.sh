#!/bin/sh

bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-Y1 test-4t 4 0 > ./test-result/Y1-Y1-Y1-4.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-Y1 test-5t 5 1 > ./test-result/Y1-Y1-Y1-5.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-Y1 test-6t 6 2 > ./test-result/Y1-Y1-Y1-6.txt 

bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-Y3 test-4t 4 0 > ./test-result/Y3-Y3-Y3-4.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-Y3 test-5t 5 1 > ./test-result/Y3-Y3-Y3-5.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-Y3 test-6t 6 2 > ./test-result/Y3-Y3-Y3-6.txt &

bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y2-new test-4t-long 4 3 > ./test-result/Y2-M-Y2-4.txt
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y2-new test-5t-long 5 0 > ./test-result/Y2-M-Y2-5.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y2-new test-6t-long 6 1 > ./test-result/Y2-M-Y2-6.txt &

bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y2-new test-4t-trans 4 2 > ./test-result/Y2-M-Y1-4.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y2-new test-5t-trans 5 3 > ./test-result/Y2-M-Y1-5.txt
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y2-new test-6t-trans 6 0 > ./test-result/Y2-M-Y1-6.txt &

bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y3-new test-4t-trans 4 1 > ./test-result/Y3-M-Y1-4.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y3-new test-5t-trans 5 2 > ./test-result/Y3-M-Y1-5.txt &
bash run_cryoEM.sh train.py CryoEM-8bit-resnet50-M-Y3-new test-6t-trans 6 3 > ./test-result/Y3-M-Y1-6.txt
