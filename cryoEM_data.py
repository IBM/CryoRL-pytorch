from cryoEM_object import CTFValue, CTFCategory, CryoEMHole, CryoEMPatch, CryoEMSquare, CryoEMGrid
from cryoEM_config import CryoEMConfig

import csv
import numpy as np

import pdb

# CryoEM dataset (store all CryoEM data)
class CryoEMData:
#    def __init__(self, gridList=[], indexList=[]):
    def __init__(self, timestamp_file, ctf_file, prediction_file,
                 prediction_type=CryoEMConfig.CLASSIFICATION,
                 use_one_hot=False,
                 category_bins=[0, 6, CryoEMConfig.MAX_CTF_VALUE],
                 img_basedir=''):
        self._prediction_type = prediction_type
        self._use_one_hot = use_one_hot
        self._category_bins = category_bins
        self._img_basedir = img_basedir
        self._grid_list, self._index_list = self.load_cryoEM_data(timestamp_file, ctf_file, prediction_file, \
                                                                  prediction_type=prediction_type,
                                                                  use_one_hot=use_one_hot,
                                                                  category_bins=category_bins)
        print (category_bins)

    def num_holes(self):
        return len(self._index_list)

    @property
    def prediction_type(self):
        return self._prediction_type

    @property
    def use_one_hot(self):
        return self._use_one_hot

    @property
    def category_bins(self):
        return self._category_bins

    @property
    def grids(self):
        return self._grid_list

    @property
    def image_basedir(self):
        return self._img_basedir

    @property
    def index_list(self):
        return self._index_list

    def idx(self, k):
        return self._index_list[k]

    def get_grid(self, k):
        g_idx, _, _ , _ = self.idx(k)
        return self._grid_list[g_idx]

    def get_square(self, k):
        g_idx, s_idx, _, _ = self.idx(k)
        return self._grid_list[g_idx][s_idx]

    def get_patch(self, k):
        g_idx, s_idx, p_idx, _ = self.idx(k)
        return self._grid_list[g_idx][s_idx][p_idx]

    def get_hole(self, k):
        g_idx, s_idx, p_idx, h_idx = self.idx(k)
        return self._grid_list[g_idx][s_idx][p_idx][h_idx]

    def get_all_hole_values(self, is_prediction=True):
        if self.prediction_type == CryoEMConfig.CLASSIFICATION:
            results = [self.get_hole(k).gt_category.value[0] for k in range(self.num_holes())] if is_prediction \
                else [self.get_hole(k).category.value[0] for k in range(self.num_holes())]
        else:
            results = [self.get_hole(k).gt_ctf.value for k in range(self.num_holes())] if is_prediction \
                else [self.get_hole(k).ctf.value for k in range(self.num_holes())]
        return results

    def set_hole_status(self, k, status=False):
        g_idx, s_idx, p_idx, h_idx = self._index_list[k]
        self._grid_list[g_idx][s_idx][p_idx][h_idx].set_status(status)

    def init_status(self):
        for k in range(self.num_holes()):
            self.set_hole_status(k, False)

    def is_grid_same(self, k1, k2):
        return self.idx(k1)[0] == self.idx(k2)[0]

    def is_square_same(self, k1, k2):
        return self.idx(k1)[0] == self.idx(k2)[0] and \
               self.idx(k1)[1] == self.idx(k2)[1]
#        return self.idx(k1)[1] == self.idx(k2)[1]

    def is_patch_same(self, k1, k2):
        return self.idx(k1)[0] == self.idx(k2)[0] and \
               self.idx(k1)[1] == self.idx(k2)[1] and \
               self.idx(k1)[2] == self.idx(k2)[2]
 #       return self.idx(k1)[2] == self.idx(k2)[2]

    def load_cryoEM_data(self, ts_filename, ctf_filename, prediction_file,prediction_type, use_one_hot, category_bins):
        print ('--------------', prediction_file)
        use_one_hot = use_one_hot if prediction_type == CryoEMConfig.CLASSIFICATION else False
        predictions = self.prediction_loader(prediction_file, use_one_hot)
        names, CTFs = self.CTF_loader(ts_filename, ctf_filename)
        data, idx = self.cryoEM_loader(names, CTFs, predictions=predictions, \
                                       prediction_type=prediction_type, category_bins=category_bins)
        # pdb.set_trace()
        return data, idx

    def one_hot_vector(self, prediction):
        max_id = np.argmax(np.array(prediction))
        one_hot_prediction = np.zeros(len(prediction))
        one_hot_prediction[max_id] = 1.0
        return tuple(one_hot_prediction)

    def prediction_loader(self, prediction_file, prediction_type, use_one_hot=False):
        if type(prediction_file) == str:
            data = np.loadtxt(prediction_file, delimiter=' ', dtype=str)
        else:
            data = prediction_file

        output = {}
        for p in data:
            # in this new file I generated, the last two values are true and predicted labels
            prediction = tuple(float(item) for item in p[1:])
            if use_one_hot:
                if prediction_type == CryoEMConfig.CLASSIFICATION:
                    prediction = self.one_hot_vector(prediction)
                else:
                    raise ValueError('Prediction type must be classification')

            output[p[0]] = prediction

        return output

    '''
    load cryoEM data from a csv file
    TimeStamp_FILE = "cryo_em/timestamps.csv"
    '''
    def CTF_loader(self, TimeStamp_FILE, CTF_FILE):
        if type(TimeStamp_FILE) == str:
            with open(TimeStamp_FILE, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                timestamps = [(row[0],row[1]) for row in csvreader]
        else:
            timestamps = TimeStamp_FILE
        timeStamps = {}
        names = []
        for i in timestamps:
            timeStamps[i[0]] = i[1]
            names.append(i[0]) # name list

        # Read CTF values of env
        if type(CTF_FILE) == str:
            with open(CTF_FILE, 'r') as csvfile:
                csvreader = csv.reader(csvfile)
                ctfs = [(row[0],row[3], row[4]) for row in csvreader]
        else:
            ctfs = CTF_FILE

        CTFs = {}
        for i in ctfs:
            CTFs[i[0]] = (float(i[1]), float(i[2]))

        return names, CTFs

    # use absolute indexing for elements, and relative indexing for accessing elements
    def cryoEM_loader(self, names, CTFs, predictions, prediction_type,  category_bins):
        grids = set()
        squares = set()
        patches = set()
        for j in CTFs.keys():
            for i in names:
                if i.endswith('gr') and j.startswith(i):
                    grids.add(i)
                if i.endswith('sq') and j.startswith(i):
                    squares.add(i)
                if i.endswith('hl') and j.startswith(i):
                    patches.add(i)

        grids = list(grids)
        grids.sort()
        squares = list(squares)
        squares.sort()
        patches = list(patches)
        patches.sort()
        hl_en = {}
        for i in patches:
            hl_en[i] = {en: CTFs[en] for en in CTFs.keys() if en.startswith(i)}
        # sq,hl,en: ctf
        sq_hl_en = {}
        for i in squares:
            sq_hl_en[i] = {hl: hl_en[hl] for hl in patches if hl.startswith(i)}
        # gr, sq,hl,en: ctf
        gr_sq_hl_en={}
        for i in grids:
            gr_sq_hl_en[i] = {sq: sq_hl_en[sq] for sq in squares if sq.startswith(i)}


        cryoEM_data = list()  # List of CryoEMSquares
        index_list = list()
        hole_cnt = 0
        patch_cnt = 0
        square_cnt = 0
        grid_cnt = 0
        for g_th, (grid_name, grid) in enumerate(gr_sq_hl_en.items()):
            squareList = list()
            local_square_cnt = 0
            for i_th, (square_name, square) in enumerate(grid.items()):
                patchList = list()
                local_patch_cnt = 0
                for j_th, (patch_name, patch) in enumerate(square.items()):
                    holeList = list()
                    local_hole_cnt = 0
                    for k_th, (hole_name, hole_CTF) in enumerate(patch.items()):
                        ctf_value = min(hole_CTF[0], CryoEMConfig.MAX_CTF_VALUE)
                        ctf_conf = hole_CTF[1]
                        gt_ctf = CTFValue(ctf_value, ctf_conf)
                        if not hole_name in predictions.keys():
                            continue
                        if prediction_type == CryoEMConfig.CLASSIFICATION:
                            ctf_category = CTFCategory(predictions[hole_name], 1.0)
                            ctf = None
                        else:
                            ctf_category = None
                            ctf = CTFValue(predictions[hole_name][0], 1.0)
                        #print (gt_ctf.value, ctf.value, ctf_category)
                        hole_k = CryoEMHole(hole_name, hole_cnt, patch_cnt, gt_ctf=gt_ctf, ctf=ctf, category_bins=category_bins, ctf_category=ctf_category)

                        holeList.append(hole_k)
                        index_list.append((grid_cnt, local_square_cnt, local_patch_cnt, local_hole_cnt)) # relative indexing
                        hole_cnt += 1
                        local_hole_cnt += 1
                    if local_hole_cnt == 0:
                        continue
                    
                    patch_j = CryoEMPatch(patch_name, patch_cnt, square_cnt, holeList=holeList)
                    patchList.append(patch_j)
                    local_patch_cnt += 1
                    patch_cnt += 1
                if local_patch_cnt == 0:
                    continue
                square_i = CryoEMSquare(square_name, square_cnt, grid_cnt, patchList=patchList)
                squareList.append(square_i)
                square_cnt += 1
                local_square_cnt += 1
            if local_square_cnt == 0:
                continue
            grid_g = CryoEMGrid(grid_name, grid_cnt, squareList=squareList)
            cryoEM_data.append(grid_g)
            grid_cnt += 1

        return cryoEM_data, index_list