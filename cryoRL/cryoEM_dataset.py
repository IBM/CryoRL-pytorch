from cryoEM_data import CryoEMData
from cryoEM_config import CryoEMConfig
import pickle

CryoEM_DATASETS = {
    'CryoEM-resnet18': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_categorization_res18_train_by_hl.txt',
           'val_prediction_file': './CryoEM_data/2_categorization_res18_val_by_hl.txt',
            'train_visual_file': './CryoEM_data/resnet18-train-cnn-feature.pickle',
           'val_visual_file': './CryoEM_data/resnet18-val-cnn-feature.pickle',
           'category_bins': [0, 6, 999999],
           'feature_dim': 17
        },
    },
    'CryoEM-resnet50': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
           'train_prediction_file': './CryoEM_data/2_categorization_res50_train_by_hl.txt',
           'val_prediction_file': './CryoEM_data/2_categorization_res50_val_by_hl.txt',
            'train_visual_file': './CryoEM_data/resnet50-train-cnn-feature.pickle',
           'val_visual_file': './CryoEM_data/resnet50-val-cnn-feature.pickle',
           'category_bins': [0, 6, 999999],
           'feature_dim': 17
        },
    },
    'CryoEM-weighted-resnet50': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_categorization_weighted_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_categorization_weighted_res50_val_by_hl.txt',
            'train_visual_file': './CryoEM_data/resnet50-train-cnn-feature.pickle',
            'val_visual_file': './CryoEM_data/resnet50-val-cnn-feature.pickle',
            'category_bins': [0, 6, 999999],
            'feature_dim': 17
        },
    },
    'CryoEM-8bit-resnet18': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_cat_8bit_res18_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_cat_8bit_res18_val_by_hl.txt',
            'train_visual_file': './CryoEM_data/resnet18-train-cnn-feature-8bit.pickle',
            'val_visual_file': './CryoEM_data/resnet18-val-cnn-feature-8bit.pickle',
            'category_bins': [0, 6, 999999],
            'feature_dim': 17
        }
    },
    'CryoEM-8bit-ctf4-resnet18': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_cat_8bit_ctf4_res18_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_cat_8bit_ctf4_res18_val_by_hl.txt',
            'train_visual_file': './CryoEM_data/resnet18-train-cnn-feature-8bit-ctf4.pickle',
            'val_visual_file': './CryoEM_data/resnet18-val-cnn-feature-8bit-ctf4.pickle',
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
            'train_visual_file': './CryoEM_data/resnet50-train-cnn-feature-8bit.pickle',
            'val_visual_file': './CryoEM_data/resnet50-val-cnn-feature-8bit.pickle',
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
    'CryoEM-8bit-ctf4-resnet50': {
        'cat_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_cat_8bit_ctf4_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_cat_8bit_ctf4_res50_val_by_hl.txt',
            'train_visual_file': './CryoEM_data/resnet50-train-cnn-feature-8bit-ctf4.pickle',
            'val_visual_file': './CryoEM_data/resnet50-val-cnn-feature-8bit-ctf4.pickle',
            'category_bins': [0, 6, 999999],
            'feature_dim': 17
        },
        'regress_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
            'train_ctf_file': './CryoEM_data/CTF_train_by_hl.csv',
            'val_ctf_file': './CryoEM_data/CTF_val_by_hl.csv',
            'train_prediction_file': './CryoEM_data/2_regress_8bit_ctf4_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/2_regress_8bit_ctf4_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
    'CryoEM-8bit-resnet50-Y1': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/timestamps.csv',
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
    'CryoEM-8bit-resnet50-Y2': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'val_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'train_ctf_file': './CryoEM_data/target_CTF_Y2_train.csv',
            'val_ctf_file': './CryoEM_data/target_CTF_Y2_val.csv',
            'train_prediction_file': './CryoEM_data/Y2Y2_2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/Y2Y2_2_regress_8bit_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
    'CryoEM-8bit-resnet50-Y3': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/target_CTF_Y3.csv',
            'val_timestamps': './CryoEM_data/target_CTF_Y3.csv',
            'train_ctf_file': './CryoEM_data/target_CTF_Y3_train.csv',
            'val_ctf_file': './CryoEM_data/target_CTF_Y3_val.csv',
            'train_prediction_file': './CryoEM_data/Y3Y3_2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/Y3Y3_2_regress_8bit_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
    'CryoEM-8bit-resnet50-M-Y2-new': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'val_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'train_ctf_file': './CryoEM_data/target_CTF_Y2_train.csv',
            'val_ctf_file': './CryoEM_data/target_CTF_Y2_val.csv',
            'train_prediction_file': './CryoEM_data/Y2_2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/Y2_2_regress_8bit_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
    'CryoEM-8bit-resnet50-MFT2-Y2-new': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/timestamps.csv',
            'val_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'train_ctf_file': './CryoEM_data/target_CTF_Y1_train.csv',
            'val_ctf_file': './CryoEM_data/target_CTF_Y2_val.csv',
            'train_prediction_file': './CryoEM_data/Y1_2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/Y2FT_2_regress_8bit_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
    'CryoEM-8bit-resnet50-M-Y3-new': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'val_timestamps': './CryoEM_data/target_CTF_Y3.csv',
            'train_ctf_file': './CryoEM_data/target_CTF_Y2_train.csv',
            'val_ctf_file': './CryoEM_data/target_CTF_Y3_val.csv',
            'train_prediction_file': './CryoEM_data/Y2_2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/Y3_2_regress_8bit_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
    'CryoEM-8bit-resnet50-M-Y3ft-new': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'val_timestamps': './CryoEM_data/target_CTF_Y3.csv',
            'train_ctf_file': './CryoEM_data/target_CTF_Y2_train.csv',
            'val_ctf_file': './CryoEM_data/target_CTF_Y3_val.csv',
            'train_prediction_file': './CryoEM_data/Y2_2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/Y3Y3ft_2_regress_8bit_res50_val_by_hl.txt',
            'train_visual_file': None,
            'val_visual_file': None,
            'category_bins': [0, 4, 6, 8, 10, 999999],
            'feature_dim': 32
        }
    },
    'CryoEM-8bit-resnet50-M-Y4-new': {
        'regress_feature': {
            'train_timestamps': './CryoEM_data/target_CTF_Y2.csv',
            'val_timestamps': './CryoEM_data/target_CTF_Y4.csv',
            'train_ctf_file': './CryoEM_data/target_CTF_Y4_train.csv',
            'val_ctf_file': './CryoEM_data/target_CTF_Y4_val.csv',
            'train_prediction_file': './CryoEM_data/Y2_2_regress_8bit_res50_train_by_hl.txt',
            'val_prediction_file': './CryoEM_data/Y4_2_regress_8bit_res50_val_by_hl.txt',
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

def get_dataset(dataset, prediction_type, use_one_hot=False, visual_similarity=False):
    dataset = CryoEM_DATASETS[dataset]
    feature_set = dataset['cat_feature'] if prediction_type == CryoEMConfig.CLASSIFICATION else dataset['regress_feature']
    feature_dim = feature_set['feature_dim']
    category_bins = feature_set['category_bins']

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

    train_visual_feature = None
    val_visual_feature = None
    if visual_similarity:
        train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
        train_visual_feature = get_cnn_features(feature_set['train_visual_file'], train_file_list)
        val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
        val_visual_feature = get_cnn_features(feature_set['val_visual_file'], val_file_list)

    return train_dataset, val_dataset, feature_dim, category_bins, train_visual_feature, val_visual_feature


'''
def get_dataset(dataset, prediction_type, use_one_hot=False, visual_similarity=False, gaussian_filter_size=0.0):
    train_visual_feature = None
    val_visual_feature = None
    category_bins = [0, 6, 999999]
    feature_dim = 17
    elif dataset == 'CryoEM-resnet18':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_res18_train_by_hl.txt',
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_res18_val_by_hl.txt',
                                 use_one_hot=use_one_hot)

        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet18-train-cnn-feature.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet18-val-cnn-feature.pickle', val_file_list)

    elif dataset == 'CryoEM-resnet50':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_res50_train_by_hl.txt',
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_res50_val_by_hl.txt',
                                 use_one_hot=use_one_hot)
        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet50-train-cnn-feature.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet50-val-cnn-feature.pickle', val_file_list)
    elif dataset == 'CryoEM-weighted-resnet50':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_categorization_weighted_res50_train_by_hl.txt',
                                    use_one_hot=use_one_hot)
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_categorization_weighted_res50_val_by_hl.txt',
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-8bit-resnet18':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_cat_8bit_res18_train_by_hl.txt',
                                    category_bins=category_bins,
                                    gaussian_filter_size = gaussian_filter_size,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_cat_8bit_res18_val_by_hl.txt',
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)

        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet18-train-cnn-feature-8bit.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet18-val-cnn-feature-8bit.pickle', val_file_list)

    elif dataset == 'CryoEM-8bit-resnet50':
        if prediction_type == CryoEMConfig.CLASSIFICATION:
            feature_dim = 17
            category_bins = [0, 6, 999999]
            train_file = './CryoEM_data/2_cat_8bit_res50_train_by_hl.txt'
            val_file =  './CryoEM_data/2_cat_8bit_res50_val_by_hl.txt'
        else:
            feature_dim = 32
            category_bins = [0, 4, 6, 8, 10, 999999]
            train_file = './CryoEM_data/2_regress_8bit_res50_train_by_hl.txt'
            val_file =  './CryoEM_data/2_regress_8bit_res50_val_by_hl.txt'

        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file=train_file,
                                    prediction_type=prediction_type,
                                    category_bins=category_bins,
                                    gaussian_filter_size = gaussian_filter_size,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file=val_file,
                                 prediction_type=prediction_type,
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)
        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet50-train-cnn-feature-8bit.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet50-val-cnn-feature-8bit.pickle', val_file_list)
    elif dataset == 'CryoEM-8bit-weighted-resnet18':
        feature_dim = 17
        category_bins = [0, 6, 999999]
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_cat_8bit_weighted_res18_train_by_hl.txt',
                                    category_bins=category_bins,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_cat_8bit_weighted_res18_val_by_hl.txt',
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-8bit-weighted-resnet50':
        feature_dim = 17
        category_bins = [0, 6, 999999]
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_cat_8bit_weighted_res50_train_by_hl.txt',
                                    category_bins=category_bins,
                                    gaussian_filter_size = gaussian_filter_size,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_cat_8bit_weighted_res50_val_by_hl.txt',
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-8bit-ctf4-resnet18':
        feature_dim = 17
        category_bins = [0, 4, 999999]
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_cat_8bit_ctf4_res18_train_by_hl.txt',
                                    category_bins=category_bins,
                                    gaussian_filter_size = gaussian_filter_size,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_cat_8bit_ctf4_res18_val_by_hl.txt',
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)

        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet18-train-cnn-feature-8bit-ctf4.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet18-val-cnn-feature-8bit-ctf4.pickle', val_file_list)

    elif dataset == 'CryoEM-8bit-ctf4-resnet50':
        feature_dim = 17
        category_bins = [0, 4, 999999]
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_cat_8bit_ctf4_res50_train_by_hl.txt',
                                     category_bins=category_bins,
                                    gaussian_filter_size = gaussian_filter_size,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_cat_8bit_ctf4_res50_val_by_hl.txt',
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)
        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet50-train-cnn-feature-8bit-ctf4.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet50-val-cnn-feature-8bit-ctf4.pickle', val_file_list)
    elif dataset == 'CryoEM-8bit-ctf4-weighted-resnet18':
        feature_dim = 17
        category_bins = [0, 4, 999999]
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_cat_8bit_ctf4_weighted_res18_train_by_hl.txt',
                                    category_bins=category_bins,
                                    gaussian_filter_size = gaussian_filter_size,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_cat_8bit_ctf4_weighted_res18_val_by_hl.txt',
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-8bit-ctf4-weighted-resnet50':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/2_cat_8bit_ctf4_weighted_res50_train_by_hl.txt',
                                    category_bins=category_bins,
                                    gaussian_filter_size = gaussian_filter_size,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/2_cat_8bit_ctf4_weighted_res50_val_by_hl.txt',
                                 category_bins=category_bins,
                                 gaussian_filter_size=gaussian_filter_size,
                                 use_one_hot=use_one_hot)
    elif dataset == 'CryoEM-8bit-ctf4-cat3-resnet18':
        feature_dim = 21
        category_bins = [0, 4, 8, 999999]
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/3_cat_8bit_ctf4_res18_train_by_hl.txt',
                                    category_bins=category_bins,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/3_cat_8bit_ctf4_res18_val_by_hl.txt',
                                 category_bins=category_bins,
                                 use_one_hot=use_one_hot)

        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet18-train-cnn-feature-8bit-ctf4-cat3.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet18-val-cnn-feature-8bit-ctf4-cat3.pickle', val_file_list)

    elif dataset == 'CryoEM-8bit-ctf4-cat3-resnet50':
        feature_dim = 21
        category_bins = [0, 4, 8, 999999]
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train_by_hl.csv',
                                    prediction_file='./CryoEM_data/3_cat_8bit_ctf4_res50_train_by_hl.txt',
                                     category_bins=category_bins,
                                    use_one_hot=use_one_hot)

        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val_by_hl.csv',
                                 prediction_file='./CryoEM_data/3_cat_8bit_ctf4_res50_val_by_hl.txt',
                                 category_bins=category_bins,
                                 use_one_hot=use_one_hot)
        if visual_similarity:
            train_file_list = [train_dataset.get_hole(k).name for k in range(train_dataset.num_holes())]
            train_visual_feature = get_cnn_features('./CryoEM_data/resnet50-train-cnn-feature-8bit-ctf4.pickle', train_file_list)
            val_file_list = [val_dataset.get_hole(k).name for k in range(val_dataset.num_holes())]
            val_visual_feature = get_cnn_features('./CryoEM_data/resnet50-val-cnn-feature-8bit-ctf4.pickle', val_file_list)
    elif dataset == 'CryoEM-7-3-Category':
        train_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                    ctf_file='./CryoEM_data/CTF_train.csv',
                                    prediction_file='./CryoEM_data/2_categorization_train.txt',
                                    use_one_hot=use_one_hot)
        val_dataset = CryoEMData('./CryoEM_data/timestamps.csv',
                                 ctf_file='./CryoEM_data/CTF_val.csv',
                                 prediction_file='./CryoEM_data/2_categorization_val.txt',
                                 use_one_hot=use_one_hot)
    else:
        raise ValueError('Dataset {} not available.'.format(dataset))

    return train_dataset, val_dataset, feature_dim, category_bins, train_visual_feature, val_visual_feature
'''
