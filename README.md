# CryoRL: Reinforcement Learning-powered cryoEM data collection

![IMG_0029](https://user-images.githubusercontent.com/109689432/183966746-42acd7a4-f482-4d35-b107-108bfc764c3d.jpg)

CryoRL utilizes a two-fold regressor and reinforcement learning network to determine an optimized cryoEM microscope trajectory from low-magnification images. CryoRL's regressor predicts and assigns CTF-associated quality scores to hole-level images. The resulting scores are then used by CryoRL's DQN-based reinforcement learning policy to map a trajectory of target holes. CryoRL is currently still in testing; more can be found in ["Outperforming cryoEM experts in data collection using artificial intelligence", Li Y, Fan Q, et al.](https://www.biorxiv.org/content/10.1101/2022.06.17.496614v1.full).

## Step 0: Installation

To start, create a new suitable conda environment :  
`conda create --name <env> --file requirements.txt`  
Typical installation should take only several minutes.

For the full list of dependencies as tested in Linux, see 'requirements.txt'.

## Step 1: CryoEM Grid Survey Data Preparation


  <img src="https://user-images.githubusercontent.com/109689432/183967204-659c0aa2-34e4-471b-9b85-309b5d7869df.jpg" width="400" height="250">


CryoEM grids were surveyed at the patch level, and the resulting .mrc files converted to 8-bit .png format using [e2proc2d.py](https://blake.bcm.edu/emanwiki/EMAN2/Programs/e2proc2d) from EMAN2. Individual hole images were identified with [Leginon](https://emg.nysbc.org/redmine/projects/leginon/wiki/Leginon_Homepage) hole coordinates and cropped to boxes of 150x150 px.

Depending on the microscope setup used, these steps may differ; CryoRL will accept cropped hole-level images only in .png format. Example hole-level images can be found in https://github.com/yilaili/cryoRL-pytorch-data.

## Step 2: Hole-Level Image Regression

{run current regressor on .png files, no supervision}
{convert predicted good targets to quality scores}

## Step 3: DQN Policy Enforcement

{input two files: (.png name : quality score), (.png name : x-coord : y-coord : quality score)}
{output micrograph trajectory file}

# Re-training CryoRL Models

CryoRL's components can easily be re-trained and evaluated on a dataset of your own. With supported GPU acceleration, expect the image regressor to take well under an hour to train and evaluate. Since the DQN is more computationally intensive, expect training to take several hours and evaluation under an hour.

## Re-training Hole-Level Regressor

By retraining the image regressor on custom ground-truth labels : cropped hole-level .png files, CryoRL can be closer fit to your data.

### Step 0: Assembling Dataset

Firstly to have the .png files understood by our model, they must be organized into "training" and "validate" folders based on their respective label. See the file structure below for reference:

<img width="213" alt="Screen Shot 2022-08-10 at 3 21 30 PM" src="https://user-images.githubusercontent.com/109689432/184002307-ae7eb954-aeba-4f7f-b88a-a98433a6bc47.png">

### Step 1: Writing Configuration .yaml

Download {example config.yaml file} as a template, and reassign the `datadir:` field to point to your dataset's host folder. Your .yaml file should mirror the following:

<img width="201" alt="Screen Shot 2022-08-10 at 3 28 22 PM" src="https://user-images.githubusercontent.com/109689432/184003641-3c2fbaa8-48b8-4341-ba30-2c4aec35841f.png">

### Step 2: Training

Execute the command `python train.py --config /YOUR_CONFIG/.yaml --lr 0.001 --backbone_net resnet18 -b 32 --epochs 100` to train a new image regressor model. Details of the training process are saved into a folder named `cryoRL/log/YOUR_DATA/`. The model log directory is also returned in the command's output, as can be seen below:

<img width="809" alt="Screen Shot 2022-08-10 at 3 52 35 PM" src="https://user-images.githubusercontent.com/109689432/184008674-86a2c558-8af3-4ba8-8f2e-94b4401dec03.png">

### Step 3: Evaluating

Execute the command `python train.py --config /YOUR_CONFIG/.yaml --lr 0.001 --backbone_net resnet18 -b 32 --epochs 100 --pretrained /YOUR_MODEL_PATH/`, where `/YOUR_MODEL_PATH/` in the previous example is `/cryoRL/log/YOUR_DATA/SL_resnet18-cosine-bs64-e2/0/`. To quickly retrieve `sk.learn` model metrics, execute `python tools/get_clf_metrics.py --dir /YOUR_MODEL_PATH/` to print a summary similar to the following:

<img width="679" alt="Screen Shot 2022-08-10 at 9 48 29 PM" src="https://user-images.githubusercontent.com/109689432/184051481-063c64d8-1b78-4358-8aa5-20fb763615ba.png">

## Re-training DQN Policy

To train a DQN (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --duration 120 --prediction-type regress --train-prediction --test-prediction``  

To evaluate (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --train-prediction --test-prediction --eval --duration 480 --print-trajectory``
