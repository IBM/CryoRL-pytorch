import numpy as np
from cryoEM_config import CryoEMConfig
from scipy.ndimage import gaussian_filter1d

class CryoEMFeature:
    def __init__(self, cryoEM_data, hist_bins, ctf_low_thresh, prediction, hole_idx_list=None):
        self._cryoEM_data = cryoEM_data
        self._hist_bins = hist_bins
        self._ctf_low_thresh = ctf_low_thresh
        self._prediction = prediction
        self._hole_idx_list = hole_idx_list if hole_idx_list is not None \
            else [k for k in range(self.cryoEM_data.num_holes())]

    @property
    def cryoEM_data(self):
        return self._cryoEM_data

    @property
    def hist_bins(self):
        return self._hist_bins

    @property
    def ctf_low_thresh(self):
        return self._ctf_low_thresh

    @property
    def prediction(self):
        return self._prediction

    @property
    def hole_idx_list(self):
        return self._hole_idx_list

    def hole_feature(self, hole):
        raise NotImplementedError()

    def patch_feature(self, patch):
        raise NotImplementedError()

    def square_feature(self, square):
        raise NotImplementedError()

    # currrent_state ----> next state
    def compute_CryoEMdata_single_feature(self, current_k, next_k, feature_storage=None):
        # hole features
        next_hole = self.cryoEM_data.get_hole(next_k)
        next_patch = self.cryoEM_data.get_patch(next_k)
        next_square = self.cryoEM_data.get_square(next_k)
        next_grid = self.cryoEM_data.get_grid(next_k)
        patch_idx = next_patch.idx
        square_idx = next_square.idx
        grid_idx = next_grid.idx
        assert patch_idx >= 0 and square_idx >= 0 and grid_idx >= 0

        hole_feature = self.hole_feature(next_hole)

        # patch features
        if feature_storage is not None and patch_idx in feature_storage['patch']:
            patch_feature = np.array(feature_storage['patch'][patch_idx])
        else:
            next_patch = self.cryoEM_data.get_patch(next_k)
            patch_feature = self.patch_feature(next_patch)
            if feature_storage is not None: # indexing features for efficiency
                feature_storage['patch'][patch_idx] = patch_feature

        # square_features
        if feature_storage is not None and square_idx in feature_storage['square']:
            square_feature = np.array(feature_storage['square'][square_idx])
        else:
            next_square = self.cryoEM_data.get_square(next_k)
            square_feature = self.square_feature(next_square)
            if feature_storage is not None: # indexing features for efficiency
                feature_storage['square'][square_idx] = square_feature

        # grid_features
        if feature_storage is not None and grid_idx in feature_storage['grid']:
            grid_feature = np.array(feature_storage['grid'][grid_idx])
        else:
            next_grid = self.cryoEM_data.get_grid(next_k)
            grid_feature = self.grid_feature(next_grid)
            if feature_storage is not None: # indexing features for efficiency
                feature_storage['grid'][grid_idx] = grid_feature

        status_changes = [ self.cryoEM_data.is_grid_same(current_k, next_k),
                           self.cryoEM_data.is_square_same(current_k, next_k),
                           self.cryoEM_data.is_patch_same(current_k, next_k)]

        feature = np.concatenate((hole_feature, patch_feature, square_feature, grid_feature, status_changes))

        return feature

    def compute_CryoEMdata_features(self, current_k):
        # keep features to avoid re-computing
        feature_storage = {'patch':{}, 'square':{}, 'grid':{}}
#        feat = [self.compute_CryoEMdata_single_feature(current_k, k, feature_storage) for k in range(self.cryoEM_data.num_holes())]
        feat = [self.compute_CryoEMdata_single_feature(current_k, k, feature_storage) for k in self.hole_idx_list]
        feat = np.array(feat)

        del feature_storage

        return feat

'''
Use CTF values (from regression or measurement) to compute features
'''
class CTFValueFeature(CryoEMFeature):
    def __init__(self, cryoEM_data, hist_bins=[0, 4, 6, 8, 10, 999999], ctf_low_thresh=CryoEMConfig.LOW_CTF_THRESH, prediction=True, hole_idx_list=None):
        super().__init__(cryoEM_data, hist_bins, ctf_low_thresh,  prediction, hole_idx_list)

    def hole_feature(self, hole):
        ctf = hole.gt_ctf if not self.prediction else hole.ctf
        return np.array([min(ctf.value, CryoEMConfig.MAX_CTF_VALUE)/CryoEMConfig.MAX_CTF_VALUE, float(hole.status)])

    def compute_patch_feature(self, patch, normalized=False):
        unvisited_ctfs = patch.get_ctfs(prediction=self.prediction, visited=False)
        # compute for unvisited holes only
        hist, _ = np.histogram([c.value for c in unvisited_ctfs], self.hist_bins)
        hist = hist.astype(float)
        #print ('---', hist)
        if CryoEMConfig.GAUSS_FILTER_SIZE > 0:
            hist = gaussian_filter1d(hist, CryoEMConfig.GAUSS_FILTER_SIZE)
            #print ('xxxx', hist)

        # unvisited hole counts
        unvisited_ctf_counts = len(unvisited_ctfs)
        # unvisited low ctf counts
        unvisited_lCTF_counts = len(patch.get_ctfs(prediction=self.prediction, threshold=self.ctf_low_thresh, visited=False))
        # visited hole counts
        visited_ctf_counts = len(patch) - unvisited_ctf_counts
        # visited low ctf counts
        visited_lCTF_counts = len(patch.get_ctfs(prediction=self.prediction, threshold=self.ctf_low_thresh, visited=True))

        if not normalized:
            return hist, np.array([unvisited_ctf_counts,
                                   unvisited_lCTF_counts,
                                   visited_ctf_counts,
                                   visited_lCTF_counts], dtype=np.float32)
        else:
            return hist/(np.sum(hist)+0.0001), \
                   np.array([min(unvisited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(visited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH],
                            dtype=np.float32)
                            #unvisited_lCTF_counts / (unvisited_ctf_counts + 0.0001), # ratio of unvisited low CTF
                            #       visited_lCTF_counts / (visited_ctf_counts + 0.0001)], dtype=np.float32) # ratio

    def patch_feature(self, patch):
        results = self.compute_patch_feature(patch, normalized=True)
        return np.concatenate(results)

    def compute_square_feature(self, square, normalized=False):
        #hist_list=[]
        #patch_stats_list = []

        #for item in square:
        #    hist, patch_stats = self.compute_patch_feature(item, normalized=False)
        #    hist_list.append(hist)
        #    patch_stats_list.append(patch_stats)
        #hist = sum(hist_list)
        #patch_stats = sum(patch_stats_list)

        patch_info = [self.compute_patch_feature(item, normalized=False) for item in square]
        square_hist = sum([item[0] for item in patch_info])
        square_stats = sum([item[1] for item in patch_info])

        if not normalized:
            return square_hist, square_stats
        else:
            square_hist = square_hist / (np.sum(square_hist) + 0.0001)
            square_stats = np.array([min(square_stats[0], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                      min(square_stats[1], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                      min(square_stats[2], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                      min(square_stats[3], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE],
                      dtype = np.float32)

            #                       patch_stats[1] / (patch_stats[0] + 0.0001),
#                       patch_stats[3] / (patch_stats[2] + 0.0001)], dtype=np.float32)
        return square_hist, square_stats

    def square_feature(self, square):
        results = self.compute_square_feature(square, normalized=True)
        return np.concatenate(results)

    def compute_grid_feature(self, grid):
        grid_info = [self.compute_square_feature(item, normalized=False) for item in grid]
        grid_hist = sum([item[0] for item in grid_info])
        grid_stats = sum([item[1] for item in grid_info])

        grid_hist = grid_hist / (np.sum(grid_hist) + 0.0001)
        grid_stats = np.array([min(grid_stats[0], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[1], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[2], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[3], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID],
                      dtype = np.float32)

        return np.concatenate((grid_hist, grid_stats))
    
    def grid_feature(self, grid):
        return self.compute_grid_feature(grid)

'''
Use CTF categorized results to compute features
'''
class CTFCategoryFeature(CryoEMFeature):
    def __init__(self, cryoEM_data, hist_bins=[0,6,999999], ctf_low_thresh=CryoEMConfig.LOW_CTF_THRESH, prediction=False, hole_idx_list=None):
        super().__init__(cryoEM_data, hist_bins, ctf_low_thresh, prediction, hole_idx_list)

    @property
    def cryoEM_data(self):
        return self._cryoEM_data

    def hole_feature(self, hole):
        # the sum of the scores of all categories is 1.0, so we drop off the score of the last category
        value = hole.gt_category.value[0:-1] if not self.prediction else \
                hole.category.value[0:-1]
        return np.array(value + (float(hole.status),), dtype=np.float32)

    def compute_patch_feature(self, patch, normalized=False):
        unvisited_categories = patch.get_categories(prediction=self.prediction, visited=False)

        # unvisited hole counts
        unvisited_ctf_counts = len(unvisited_categories)
        # [print(c.value) for c in unvisited_categories]
        # unvisited low ctf counts
        unvisited_lCTF_counts = sum([c.value[0] for c in unvisited_categories])

        # visited hole counts
        visited_ctf_counts = len(patch) - unvisited_ctf_counts
        # visited low ctf counts
        # for visisted holes, use measured CTFs
        visited_lCTF_counts = len(patch.get_ctfs(prediction=False, threshold=self.ctf_low_thresh, visited=True))

        #if normalized:
        #    print ('{:30s} {:.2f} {:.2f} {:.2f}'.format(patch.name, unvisited_ctf_counts, unvisited_lCTF_counts, unvisited_lCTF_counts/unvisited_ctf_counts))

#        print(unvisited_ctf_counts, unvisited_lCTF_counts, visited_ctf_counts, visited_lCTF_counts, unvisited_lCTF_counts / (unvisited_ctf_counts + 0.0001),visited_lCTF_counts / (visited_ctf_counts + 0.0001) )
        if not normalized:
            return np.array([unvisited_ctf_counts,
                                   unvisited_lCTF_counts,
                                   visited_ctf_counts,
                                   visited_lCTF_counts], dtype=np.float32)

        return np.array([min(unvisited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(visited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH],
                             dtype = np.float32)
            #                       unvisited_lCTF_counts / (unvisited_ctf_counts + 0.0001), # ratio of unvisited low CTF
#                                   visited_lCTF_counts / (visited_ctf_counts + 0.0001)], dtype=np.float32) # ratio

    def patch_feature(self, patch):
        return self.compute_patch_feature(patch, normalized=True)

    def compute_square_feature(self, square, normalized=False):
        # hist_list=[]
#        patch_stats_list = []
#        for item in square:
#            patch_stats = self.compute_patch_feature(item, normalized=False)
#            # hist_list.append(hist)
#            patch_stats_list.append(patch_stats)

        # hist = sum(hist_list)
#        patch_stats = sum(patch_stats_list)

        square_stats = sum([self.compute_patch_feature(item, normalized=False) for item in square])
        # hist = hist / (np.sum(hist) + 0.0001)

        if not normalized:
            return square_stats

        return np.array([min(square_stats[0], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                min(square_stats[1], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                min(square_stats[2], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                min(square_stats[3], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE],
                dtype=np.float32)

    def square_feature(self, square):
        return self.compute_square_feature(square, normalized=True)

    def grid_feature(self, grid):
        #hist_list=[]
        square_stats_list = []
#        for item in grid:
#            square_stats = self.compute_square_feature(item, normalized=False)
#            #hist_list.append(hist)
#            square_stats_list.append(square_stats)

        grid_stats = sum([self.compute_square_feature(item, normalized=False) for item in grid])
#        print ('{:30s} {:.2f} {:.2f} {:.2f}'.format(grid.name, grid_stats[0], grid_stats[1], float(grid_stats[1]/grid_stats[0])))

        #hist = hist / (np.sum(hist) + 0.0001)
        grid_stats = np.array([min(grid_stats[0], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[1], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[2], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[3], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID],
                      dtype = np.float32)

        return grid_stats

'''
Use CTF categorized results to compute features
'''
class CTFCategoryFeature_new(CryoEMFeature):
    def __init__(self, cryoEM_data, ctf_low_thresh=CryoEMConfig.LOW_CTF_THRESH, prediction=False):
        super().__init__(cryoEM_data, [0, 6, 99999], ctf_low_thresh, prediction)

    @property
    def cryoEM_data(self):
        return self._cryoEM_data

    def hole_feature(self, hole):
        # the sum of the scores of all categories is 1.0, so we drop off the score of the last category
        value = hole.gt_category.value[0:-1] if not self.prediction else \
                hole.category.value[0:-1]
        return np.array(value + (float(hole.status),), dtype=np.float32)

    def compute_patch_feature(self, patch, normalized=False):
        unvisited_categories = patch.get_categories(prediction=self.prediction, visited=False)

        # unvisited hole counts
        unvisited_ctf_counts = len(unvisited_categories)
        # unvisited low ctf counts
        unvisited_lCTF_counts = sum([c.value[0] for c in unvisited_categories])

        # visited hole counts
        visited_ctf_counts = len(patch) - unvisited_ctf_counts
        # visited low ctf counts
        # for visisted holes, use measured CTFs
        visited_lCTF_counts = len(patch.get_ctfs(prediction=False, threshold=self.ctf_low_thresh, visited=True))

        patch_stats = np.array([item.value for item in unvisited_categories])
        patch_stats = np.sum(patch_stats, axis=0) if len(patch_stats) > 0 else np.array([0.0,0.0,0.0])

        #if normalized:
        #    print ('{:30s} {:.2f} {:.2f} {:.2f}'.format(patch.name, unvisited_ctf_counts, unvisited_lCTF_counts, unvisited_lCTF_counts/unvisited_ctf_counts))

#        print(unvisited_ctf_counts, unvisited_lCTF_counts, visited_ctf_counts, visited_lCTF_counts, unvisited_lCTF_counts / (unvisited_ctf_counts + 0.0001),visited_lCTF_counts / (visited_ctf_counts + 0.0001) )
        if not normalized:
            return np.array([patch_stats[0], # low
                            patch_stats[1],  # mid
                            patch_stats[2],  # high
                            visited_ctf_counts,
                            visited_lCTF_counts], dtype=np.float32)

        return np.array([min(patch_stats[0], CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(patch_stats[1],  CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(patch_stats[2],  CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(visited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH,
                             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH],
                             dtype = np.float32)
            #                       unvisited_lCTF_counts / (unvisited_ctf_counts + 0.0001), # ratio of unvisited low CTF
#                                   visited_lCTF_counts / (visited_ctf_counts + 0.0001)], dtype=np.float32) # ratio

    def patch_feature(self, patch):
        return self.compute_patch_feature(patch, normalized=True)

    def compute_square_feature(self, square, normalized=False):
        # hist_list=[]
#        patch_stats_list = []
#        for item in square:
#            patch_stats = self.compute_patch_feature(item, normalized=False)
#            # hist_list.append(hist)
#            patch_stats_list.append(patch_stats)

        # hist = sum(hist_list)
#        patch_stats = sum(patch_stats_list)

        square_stats = sum([self.compute_patch_feature(item, normalized=False) for item in square])
        # hist = hist / (np.sum(hist) + 0.0001)

        if not normalized:
            return square_stats

        return np.array([min(square_stats[0], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                min(square_stats[1], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                min(square_stats[2], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                min(square_stats[3], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE,
                min(square_stats[4], CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE],
                dtype=np.float32)

    def square_feature(self, square):
        return self.compute_square_feature(square, normalized=True)

    def grid_feature(self, grid):
        #hist_list=[]
        square_stats_list = []
#        for item in grid:
#            square_stats = self.compute_square_feature(item, normalized=False)
#            #hist_list.append(hist)
#            square_stats_list.append(square_stats)

        grid_stats = sum([self.compute_square_feature(item, normalized=False) for item in grid])
#        print ('{:30s} {:.2f} {:.2f} {:.2f}'.format(grid.name, grid_stats[0], grid_stats[1], float(grid_stats[1]/grid_stats[0])))

        #hist = hist / (np.sum(hist) + 0.0001)
        grid_stats = np.array([min(grid_stats[0], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[1], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[2], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[3], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID,
                      min(grid_stats[4], CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID],
                      dtype = np.float32)

        return grid_stats

'''
Use CTF categorized results to compute features
'''
'''
class CTFCategoryFeature_new(CryoEMFeature):
    def __init__(self, cryoEM_data, hist_bins=[0,6,999999], ctf_low_thresh=CryoEMConfig.LOW_CTF_THRESH, prediction=False):
        super().__init__(cryoEM_data, hist_bins, ctf_low_thresh, prediction)

    @property
    def cryoEM_data(self):
        return self._cryoEM_data

    def hole_feature(self, hole):
        # the sum of the scores of all categories is 1.0, so we drop off the score of the last category
        value = hole.gt_category.value[0:-1] if not self.prediction else \
                hole.category.value[0:-1]

        return np.array(value + (float(hole.status),), dtype=np.float32)

    def compute_patch_feature(self, patch, normalized=False):
        unvisited_categories = patch.get_categories(prediction=self.prediction, visited=False)

        # unvisited hole counts
        unvisited_ctf_counts = len(unvisited_categories)
        # unvisited low ctf counts
        unvisited_lCTF_counts = sum([c.value[0] for c in unvisited_categories])

        # visited hole counts
        visited_ctf_counts = len(patch) - unvisited_ctf_counts
        # visited low ctf counts
        # for visisted holes, use measured CTFs
        visited_lCTF_counts = len(patch.get_ctfs(prediction=False, threshold=self.ctf_low_thresh, visited=True))

        #if normalized:
        #    print ('{:30s} {:.2f} {:.2f} {:.2f}'.format(patch.name, unvisited_ctf_counts, unvisited_lCTF_counts, unvisited_lCTF_counts/unvisited_ctf_counts))

#        print(unvisited_ctf_counts, unvisited_lCTF_counts, visited_ctf_counts, visited_lCTF_counts, unvisited_lCTF_counts / (unvisited_ctf_counts + 0.0001),visited_lCTF_counts / (visited_ctf_counts + 0.0001) )
        if not normalized:
            return np.array([unvisited_ctf_counts,
                                   unvisited_lCTF_counts,
                                   visited_ctf_counts,
                                   visited_lCTF_counts], dtype=np.float32)

        return np.array([min(unvisited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / (visited_ctf_counts + unvisited_ctf_counts), # ratio
                             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / (unvisited_ctf_counts + 1e-4),  # ratio
                             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH, # number
                             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / (visited_ctf_counts + 1e-4),
                             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_PATCH) / CryoEMConfig.MAX_HOLE_CNT_PER_PATCH],
                             dtype = np.float32)
            #                       unvisited_lCTF_counts / (unvisited_ctf_counts + 0.0001), # ratio of unvisited low CTF
#                                   visited_lCTF_counts / (visited_ctf_counts + 0.0001)], dtype=np.float32) # ratio

    def patch_feature(self, patch):
        return self.compute_patch_feature(patch, normalized=True)

    def compute_square_feature(self, square, normalized=False):
        # hist_list=[]
#        patch_stats_list = []
#        for item in square:
#            patch_stats = self.compute_patch_feature(item, normalized=False)
#            # hist_list.append(hist)
#            patch_stats_list.append(patch_stats)

        # hist = sum(hist_list)
#        patch_stats = sum(patch_stats_list)

        square_stats = sum([self.compute_patch_feature(item, normalized=False) for item in square])
        # hist = hist / (np.sum(hist) + 0.0001)

        if not normalized:
            return square_stats

        unvisited_ctf_counts = square_stats[0]
        unvisited_lCTF_counts = square_stats[1]
        visited_ctf_counts = square_stats[2]
        visited_lCTF_counts = square_stats[3]

        return np.array([min(unvisited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / (visited_ctf_counts + unvisited_ctf_counts), # ratio
             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / (unvisited_ctf_counts + 1e-4),  # ratio
             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE, # number
             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / (visited_ctf_counts + 1e-4),
             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE) / CryoEMConfig.MAX_HOLE_CNT_PER_SQUARE],
            dtype=np.float32)


    def square_feature(self, square):
        return self.compute_square_feature(square, normalized=True)

    def grid_feature(self, grid):
        #hist_list=[]
        square_stats_list = []
#        for item in grid:
#            square_stats = self.compute_square_feature(item, normalized=False)
#            #hist_list.append(hist)
#            square_stats_list.append(square_stats)

        grid_stats = sum([self.compute_square_feature(item, normalized=False) for item in grid])
#        print ('{:30s} {:.2f} {:.2f} {:.2f}'.format(grid.name, grid_stats[0], grid_stats[1], float(grid_stats[1]/grid_stats[0])))

        unvisited_ctf_counts = grid_stats[0]
        unvisited_lCTF_counts = grid_stats[1]
        visited_ctf_counts = grid_stats[2]
        visited_lCTF_counts = grid_stats[3]


        return np.array([min(unvisited_ctf_counts, CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / (visited_ctf_counts + unvisited_ctf_counts), # ratio
             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / (unvisited_ctf_counts + 1e-4),  # ratio
             min(unvisited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID, # number
             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / (visited_ctf_counts + 1e-4),
             min(visited_lCTF_counts, CryoEMConfig.MAX_HOLE_CNT_PER_GRID) / CryoEMConfig.MAX_HOLE_CNT_PER_GRID],
            dtype=np.float32)

        return grid_stats
'''

def test_feature():
    x = CryoEMData('CryoEM_data/timestamps.csv', ctf_file='CryoEM_data/ctf_train.csv',
                   prediction_file='CryoEM_data/2_categorization_train_new.txt')
    f = CTFCategoryFeature(x, [0, 6, 999999], ctf_low_thresh=6, prediction=True)
    patch = x.get_patch(0)
    print(patch)
    print (f.patch_feature(patch))

    square1 = x.get_square(200)
    print (square1)
    print(f.square_feature(square1))
    #patch1 = square1[0]


if __name__ == '__main__':
    from cryoEM_data import CryoEMData
    test_feature()
