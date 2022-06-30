import numpy as np
import random
from cryoEM_config import *

class CryoEMSampler:
    def __init__(self, cryoEM_data, num_holes, use_prediction=True, sample_method='sample_by_random'):
        self.cryoEM_data = cryoEM_data
        self._num_holes = num_holes
        self._use_prediction=use_prediction
        self._sample_method = sample_method
        self._good_samples = self._get_good_samples() if self._sample_method == 'sample_by_patch' else None

    @property
    def sample_method(self):
        return self._sample_method

    def _get_good_samples(self):
        patches = {}
        prediction_type = self.cryoEM_data.prediction_type
        all_values =  self.cryoEM_data.get_all_hole_values(is_prediction=self._use_prediction)
        all_values = [item >=0.5 for item in all_values] if prediction_type == CryoEMConfig.CLASSIFICATION else \
                      [item <= CryoEMConfig.LOW_CTF_THRESH for item in all_values]
        for k, value in enumerate(all_values):
            idx = self.cryoEM_data.idx(k)
            idx = idx[:-1] # keep the parent indice and drop the hole index
            if idx not in patches:
                patches[idx] = []
            if value:
                patches[idx].append(k)

        good_samples = [hole_list for _,hole_list in patches.items()]
        num_good_samples = [len(hole_list) for hole_list in good_samples]
        print (num_good_samples)
        #sample_idx = np.argsort(num_good_samples)[-5:]
        sample_idx = np.argsort(num_good_samples)[-1:]
        good_samples = [good_samples[idx] for idx in sample_idx]
        print (good_samples)
        return [item for sublist in good_samples for item in sublist]

    def sample(self):
        if self._sample_method == 'sample_by_random' or self._good_samples is None:
            return np.random.randint(0, self._num_holes - 1)

        return random.choice(self._good_samples)
