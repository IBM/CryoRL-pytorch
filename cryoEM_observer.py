import numpy as np

import cryoEM_config
from cryoEM_config import *
from cryoEM_feature import CTFValueFeature, CTFCategoryFeature

class CryoEMObserver:
    def __init__(self, cryoEM_data, use_prediction=False, duration=240, action_elimination=False, action_elimination_factor=1.0):
        self.cryoEM_data = cryoEM_data
        self._duration = duration
        self._use_prediction = use_prediction
        self._action_elimination = action_elimination
        self._action_elimination_factor = action_elimination_factor
        self._actions = self._get_action_space()
        self._num_holes = len(self._actions)

        if cryoEM_data.prediction_type == CryoEMConfig.CLASSIFICATION:
            self.cryoEM_feature = CTFCategoryFeature(cryoEM_data,
                                                     ctf_low_thresh=CryoEMConfig.LOW_CTF_THRESH,
                                                     hist_bins=CryoEMConfig.FEATURE_HISTOGRAM_BIN,
                                                     prediction=use_prediction,
                                                     hole_idx_list=self._actions)
        else:
            self.cryoEM_feature = CTFValueFeature(cryoEM_data,
                                                  hist_bins=CryoEMConfig.FEATURE_HISTOGRAM_BIN,
                                                  ctf_low_thresh=CryoEMConfig.LOW_CTF_THRESH,
                                                  prediction=use_prediction,
                                                  hole_idx_list=self._actions)

    @property
    def action_elimination(self):
        return self._action_elimination

    @property
    def num_holes(self):
        return self._num_holes

    @property
    def observation(self):
        return self._observation

    @property
    def actions(self):
        return self._actions

    def _max_hole_visits(self):
        t = 0
        max_visits = 0
        square_cnt = 0
        grid_cnt = 0
        while t < self._duration:
            for _ in range(int(CryoEMConfig.MAX_HOLE_CNT_PER_PATCH)):
                t += CryoEMConfig.SWITCH_HOLE_LOSS
                if t >= self._duration:
                    break
                max_visits += 1
                square_cnt += 1
                grid_cnt += 1

            if square_cnt >= CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE:
                t += CryoEMConfig.SWITCH_SQUARE_LOSS
                square_cnt = 0

            if grid_cnt >= CryoEMConfig.MAX_HOLE_CNT_PER_GRID:
                t += CryoEMConfig.SWITCH_SQUARE_LOSS  + CryoEMConfig.SWITCH_GRID_LOSS
                grid_cnt = 0

            t += CryoEMConfig.SWITCH_PATCH_LOSS

        return max_visits

    '''
    def _preselect_holes(self):
        max_visits = self._max_hole_visits()
        num_selected = min(self.cryoEM_data.num_holes(), int(max_visits * CryoEMConfig.PRESELECTION_FACTOR))
        # use all the holes
        if num_selected == self.cryoEM_data.num_holes():
            return [k for k in range(num_selected)]

        #sort all the holes by their estimations (either classification or regression)
        hole_idx_list = np.argsort(self.cryoEM_data.get_all_hole_values(is_prediction=self._use_prediction))
        hole_idx_list = hole_idx_list[:max_visits] if self.cryoEM_data.prediction_type == CryoEMConfig.REGRESSION else hole_idx_list[-max_visits:]

        # now collect all the holes in the selected patches
        patch_idx_list = [self.cryoEM_data.get_hole(hole_idx).parent_idx for hole_idx in hole_idx_list]
        patch_idx_list = set(patch_idx_list)

        print (hole_idx_list)
        print (patch_idx_list)

        selected = [ k for k in range(self.cryoEM_data.num_holes()) if self.cryoEM_data.get_hole(k).parent_idx in patch_idx_list]
        print (len(selected))
        return selected
    '''

    def _get_action_space(self):
        if not self._action_elimination:
            return [k for k in range(self.cryoEM_data.num_holes())]

        # action elimination
        max_visits = self._max_hole_visits()
        num_selected = min(self.cryoEM_data.num_holes(), int(max_visits * self._action_elimination_factor))
        # use all the holes
        if num_selected == self.cryoEM_data.num_holes():
            return [k for k in range(num_selected)]

        prediction_type = self.cryoEM_data.prediction_type
        all_values =  self.cryoEM_data.get_all_hole_values(is_prediction=self._use_prediction)
        all_values = [item >=0.5 for item in all_values] if prediction_type == CryoEMConfig.CLASSIFICATION else \
                      [item <= CryoEMConfig.LOW_CTF_THRESH for item in all_values]

        patches = {}
        # count the low-ctf holes in each patch
        for k, v in enumerate(all_values):
            parent_idx = self.cryoEM_data.get_hole(k).parent_idx
            if parent_idx not in patches:
                patches[parent_idx] = 0
            if v:
                patches[parent_idx] += 1

        sorted_patches = {k: v for k, v in sorted(patches.items(), key=lambda item: item[1], reverse=True)}
        tot_cnt = 0
        patch_idx_list = []
        for k, v in sorted_patches.items():
            tot_cnt += v
        #    print (k, v, tot_cnt, num_selected)
            patch_idx_list.append(k)
            if tot_cnt >= num_selected:
                break

       # print (patch_idx_list)
        selected = [ k for k in range(self.cryoEM_data.num_holes()) if self.cryoEM_data.get_hole(k).parent_idx in patch_idx_list]
        #print (selected)
        return selected

    def initialization(self):
        # initial random observation. shape: history_size * num_holes * feature_dim
        self._observation = np.zeros((CryoEMConfig.HISTORY_SIZE, self._num_holes, CryoEMConfig.FEATURE_DIM), dtype=np.float16)

    def update_observation(self, current_hole, is_shift=True):
        feature = self.cryoEM_feature.compute_CryoEMdata_features(current_hole)

        # left shift the history and append the new one
        if is_shift is not True:
            self.observation[:, ...] = feature
            return

        self._observation[:-1, ...] = self._observation[1:, ...]  # whether shift?
        self._observation[-1, ...] = feature

        #return self._observation
