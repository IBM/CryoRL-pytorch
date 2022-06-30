from cryoEM_data import CryoEMData
from cryoEM_config import CryoEMConfig
import pickle
import csv
import numpy as np

from tools.data_utils import get_img_names, split_by_name_ts, split_by_name_ctf

CryoEM_DATASETS = {
    'CryoEM-8bit-resnet18': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_cat_8bit_res18_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_cat_8bit_res18_val_by_hl.txt',
            'train_visual_file': '',
            'val_visual_file': '',
            'category_bins': [0, 6, 999999],
            'feature_dim': 17
        }
    },
    'CryoEM-8bit-resnet50': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_cat_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_cat_8bit_res50_val_by_hl.txt',
            'train_visual_file': '',
            'val_visual_file': '',
            'category_bins': [0, 6, 999999],
            'feature_dim': 17
        },
        'regress_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_regress_8bit_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
}

def get_cnn_features(filename, image_list):
    with open(filename , 'rb') as f:
        features = pickle.load(f)

    output = np.stack([features[item] for item in image_list])
    print ('----', output.shape)
    return output

def get_dataset(dataset, prediction_type, use_one_hot=False):
    dataset = CryoEM_DATASETS[dataset]
    feature_set = dataset['cat_feature'] if prediction_type == CryoEMConfig.CLASSIFICATION else dataset['regress_feature']
    feature_dim = feature_set['feature_dim']
    category_bins = feature_set['category_bins']

    # set a mode
    if dataset['simple_configure'] is not None:
        train_idx = get_img_names(feature_set['train_prediction_file'])
        val_idx = get_img_names(feature_set['val_prediction_file'])
        ctfs_train = split_by_name_ctf(feature_set['ctf_file'], train_idx)
        ctfs_val = split_by_name_ctf(feature_set['ctf_file'], val_idx)
        train_dataset = CryoEMData(feature_set['timestamps'],
                                ctf_file=ctfs_train,
                                prediction_file=feature_set['train_prediction_file'],
                                prediction_type=prediction_type,
                                category_bins=category_bins,
                                use_one_hot=use_one_hot)

        val_dataset = CryoEMData(feature_set['timestamps'],
                       ctf_file=ctfs_val,
                       prediction_file=feature_set['val_prediction_file'],
                       prediction_type=prediction_type,
                       category_bins=category_bins,
                       use_one_hot=use_one_hot) 
        
    else:
        train_dataset = CryoEMData(feature_set['train_timestamps'],
                                ctf_file=feature_set['train_ctf_file'],
                                prediction_file=feature_set['train_prediction_file'],
                                prediction_type=prediction_type,
                                category_bins=category_bins,
                                use_one_hot=use_one_hot)

        val_dataset = CryoEMData(feature_set['val_timestamps'],
                       ctf_file=feature_set['val_ctf_file'],
                       prediction_file=feature_set['val_prediction_file'],
                       prediction_type=prediction_type,
                       category_bins=category_bins,
                       use_one_hot=use_one_hot)

    print(train_dataset.num_holes())
    print(val_dataset.num_holes())
    return train_dataset, val_dataset, feature_dim, category_bins