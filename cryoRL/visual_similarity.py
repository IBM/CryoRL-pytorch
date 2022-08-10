import torch
import numpy as np
from collections import deque

class AverageVectorMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, size=5):
        self.size = size
        self.reset()

    def reset(self):
        self.val = np.zeros((self.size,), dtype = np.float32)
        self.avg = np.zeros((self.size,), dtype = np.float32)
        self.sum = np.zeros((self.size,), dtype = np.float32)
        self.count = 0
        self.local_history = deque([])
        self.local_avg = np.zeros((self.size,), dtype = np.float32)
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history)

    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def __len__(self):
        return self.count

class FastVisualSimilarity(object):
    def __init__(self, feature_dim, size=5, coeff=10.0):
        super(FastVisualSimilarity, self).__init__()
        # the last dim
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        self.positive_meter = AverageVectorMeter(size=feature_dim)
        self.negative_meter = AverageVectorMeter(size=feature_dim)
        self.coeff = coeff

    def reset(self):
        self.positive_meter.reset()
        self.negative_meter.reset()

    def compute_visual_similarity(self, features, positive_idx, negative_idx):
        num_features = features.shape[1]  # second dim

        #print (len(self.positive_meter), len(self.negative_meter)) 

        assert (len(positive_idx) <= 1 and len(negative_idx) <= 1)
        if len(positive_idx) > 0:
            positive_sim = self.cos(features[:, positive_idx], features).flatten()
            positive_sim += 1.0
            positive_sim /= 2.0
            self.positive_meter.update(val=positive_sim.numpy(), n=len(positive_idx))

        if len(negative_idx) > 0:
            negative_sim = self.cos(features[:, negative_idx], features).flatten()
            negative_sim += 1.0
            negative_sim /= 2.0
            self.negative_meter.update(val=negative_sim.numpy(), n=len(negative_idx))

        # print (positive_sim.shape, negative_sim.shape)
        contrastive_sim = self.positive_meter.avg - self.negative_meter.avg
        #contrastive_sim = np.clip(contrastive_sim, a_max=0.5, a_min=-0.5)
        return  contrastive_sim * self.coeff
        #return (self.positive_meter.avg / (self.positive_meter.avg + self.negative_meter.avg + 1e-5)) * self.coeff

class VisualSimilarity:
    def __init__(self, coeff=1.0):
        super(VisualSimilarity, self).__init__()
        # the last dim
        self.cos = torch.nn.CosineSimilarity(dim=2, eps=1e-6)
        self.coeff = coeff

    def compute_visual_similarity(self, features, positive_idx, negative_idx):

        num_features = features.shape[1]  # second dim

        if len(positive_idx) > 0:
            positive_sim = [torch.mean(self.cos(features[:, k:k + 1], features[:, positive_idx]), dim=1) for k in
                            range(num_features)]
            # print (positive_sim[0])
            positive_sim = torch.stack(positive_sim).transpose(0, 1)
            positive_sim += 1.0
            positive_sim /= 2.0
        else:
            positive_sim = torch.zeros(features.shape[:2])

        if len(negative_idx) > 0:
            negative_sim = [torch.mean(self.cos(features[:, k:k + 1], features[:, negative_idx]), dim=1) for k in
                            range(num_features)]
            negative_sim = torch.stack(negative_sim).transpose(0, 1)
            negative_sim += 1.0
            negative_sim /= 2.0
        else:
            negative_sim = torch.zeros(features.shape[:2])

        # print (positive_sim.shape, negative_sim.shape)
        return (positive_sim / (positive_sim + negative_sim + 1e-5)) * self.coeff
