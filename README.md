# CryoRL: Reinforcement Learning-powered cryo-EM data collection

![IMG_0029](https://user-images.githubusercontent.com/109689432/183966746-42acd7a4-f482-4d35-b107-108bfc764c3d.jpg)

CryoRL utilizes a two-fold regressor and reinforcement learning networks to determine an optimized cryo-EM trajectory from low-magnified images. CryoRL's regressor predicts CTF-associated quality scores to hole-level images. The resulting scores are then used by CryoRL's DQN-based reinforcement learning to map a trajectory of target holes. CryoRL is currently still in testing; more can be found in ["Outperforming cryoEM experts in data collection using artificial intelligence", Li Y, Fan Q, et al.](https://www.biorxiv.org/content/10.1101/2022.06.17.496614v1.full).

## Step 0: Installation

To start, create a new suitable conda environment :  
`conda create --name <env> --file requirements.txt`  
Typical installation should take only several minutes.

For the full list of dependencies as tested in Linux, see `requirements.txt`.

## Step 1: CryoEM Grid Survey Data Preparation


  <img src="https://user-images.githubusercontent.com/109689432/183967204-659c0aa2-34e4-471b-9b85-309b5d7869df.jpg" width="400" height="250">


Cryo-EM grids were surveyed at the patch level, and the resulting mrc files converted to 8-bit png format using [e2proc2d.py](https://blake.bcm.edu/emanwiki/EMAN2/Programs/e2proc2d) from EMAN2. Individual hole images were identified with [Leginon](https://emg.nysbc.org/redmine/projects/leginon/wiki/Leginon_Homepage) hole coordinates and cropped to boxes of 150x150 px.

Depending on the microscope setup used, these steps may differ; CryoRL will accept cropped hole-level images only in 8-bit png format.

## Step 2: Hole-Level Image Regression

### Assembling Dataset

Example hole-level images, along with the desired file organization can be found in https://github.com/yilaili/cryoRL-pytorch-data.

You can also download it from our shared Google drive: https://drive.google.com/drive/folders/1znPXk5fJ9aujWDfeaU3LJlLyVjnuod_Y?usp=sharing.

The dataset folder needs to be placed parallel to this repository (CryoRL-pytorch). If you want to change that, you should modify the `config/regress_valY1.yaml` to indicate a valid path.

### The configuration yaml file

The example config yaml file is usually as in `config/regress_valY1.yaml`:

```
dataset: 'Y1Data'
datadir: '../../cryoRL-pytorch-data/Aldolase'
num_classes: 1
train_folder: 'train'
val_folder: 'val'
```
**dataset**: when changing this, you should also modify the `get_dataset` function in `train.py`. For example, `Y1Data` is defined in `train.py` as:
```
if dataset == 'Y1Data':
    train_img_dir = os.path.join(datadir, 'train')
    val_img_dir = os.path.join(datadir, 'val')
    train_ctf_file = os.path.join(datadir, 'target_CTF_Y_train.csv')
    val_ctf_file = os.path.join(datadir, 'target_CTF_Y_val.csv')
```

**datadir**: needs to be a valid data directory relative to the yaml file.

**num_classes**: always 1 for regression.

**train_folder**: the name of the subfolder including the training data.

**val_folder**: the name of the subfolder including the validation data.


### Running the regressor

To train a regressor (sample code):  
``python train.py --backbone_net resnet50 --config configs/regress_valY1.yaml --lr 0.0005 --epoch 50 --logdir exp --loss_function l2 --batch-size 128``

Replace the `--config` to any valid yaml file you created as described in the previous section.

To evaluate the regressor given a trained model (sample code):  
``python train.py --backbone_net resnet50 --config configs/regress_valY1.yaml --lr 0.0005 --epoch 50 --logdir exp --loss_function l2 --batch-size 128 --evaluate --pretrained exp/Y1Data-resnet50-cosine-bs128-l2-e50-l0.0005/model_best.pth.tar > Y1_2_regress_8bit_res50_val_by_hl.txt``

Note that this will redirect the stdout and stderr to the `Y1_2_regress_8bit_res50_val_by_hl.txt` file.

You can replace the `--config` to the corresponding yaml file and `--pretrained` to any other trained model you want to evaluate.

With supported GPU acceleration, expect the image regressor to take well under an hour to evaluate. Training a new regressor from scratch may take several hours, depending on the epoch number and batch size being used.

## Step 3: Trajectory planning with reinforcement learning

### Preparing dataset
Multiple datasets were compiled in `cryoRL/cryoEM_dataset.py`. For example, `CryoEM-8bit-resnet50-Y1`:
```
'CryoEM-8bit-resnet50-Y1': {
    'regress_feature': {
        'train_timestamps': './CryoEM_data/target_CTF_Y1.csv',
        'val_timestamps': 'CryoEM_data/target_CTF_Y1.csv',
        'train_ctf_file': './CryoEM_data/target_CTF_Y1_train.csv',
        'val_ctf_file': './CryoEM_data/target_CTF_Y1_val.csv',
        'train_prediction_file': './CryoEM_data/Y1_2_regress_8bit_res50_train_by_hl.txt',
        'val_prediction_file': './CryoEM_data/Y1_2_regress_8bit_res50_val_by_hl.txt',
        'train_visual_file': None,
        'val_visual_file': None,
        'category_bins': [0, 4, 6, 8, 10, 999999],
        'feature_dim': 32
    }
},
```

To use a different image regressor, replace `train_timestamps`, `val_timestamps`, `train_ctf_file`, `val_ctf_file`, `train_prediction_file` and `val_prediction_file` with the ground truth and the evaluation results from your own model. `train_visual_file` is currently repressed.

### Input files

The files specified in `CryoEM-8bit-resnet50-Y1` were uploaded to the `CryoEM_data` folder in this repo. For csv files, column 1 is the target name, column 4 is the CTFMaxRes (i.e. quality score) for the target. Txt files include the target name and the predicted CTFMaxRes, which were generated from the regressor. Please keep the same format if you'd like to use your own prediction.

### Running the DQN

To train a DQN (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --duration 480 --prediction-type regress --train-prediction --test-prediction``  

Note that training would usually take several hours on a typical computer with supported GPU acceleration.

Replace the `--dataset` to any dataset you created `cryoRL/cryoEM_dataset.py`. You can also specify the good/bad micrograph threshold by changing `--ctf-thresh`.


To evaluate a trained DQN (sample code):  
``python train.py --dataset CryoEM-8bit-resnet50-Y1 --lr 0.01 --epoch 20 --training-num 10 --test-num 10 --logdir test-6t/CryoEM-8bit-resnet50-Y1 --step-per-epoch 2500 --ctf-thresh 6.0 --prediction-type regress --train-prediction --test-prediction --eval --duration 480 --print-trajectory``

The trained model to evaluate is specified by `--logdir`.

Note that evaluation would usually take less than 1 hour on a typical computer with supported GPU acceleration. This is because we were running many parallel runs on the same dataset with a random starting position.

The output of the evaluation would be a sequence of targets and the resulting statistics from 50 parallel runs each starting from a random position.
