import gym
from gym import spaces
from gym.utils import seeding
import random
import numpy as np
from cryoEM_feature import CTFValueFeature, CTFCategoryFeature, CTFCategoryFeature_new
from cryoEM_config import *
import math
import time
from visual_similarity import FastVisualSimilarity

class CryoEMEnv(gym.Env):
    # ?
    metadata = {'render.modes': ['human']}

    def __init__(self, cryoEM_data,
                 id=-1,
                 cryoEM_visual_feature=None,
                 visual_similarity_coeff=10.0,
                 history_size=4,
                 ctf_thresh=6.0,
                 hist_bins=[0, 6.0, 999999],
                 use_prediction=False,
                 use_penalty=False,
                 dynamic_reward=False,
                 planning=False,
                 evaluation=False,
                 print_trajectory=False):
        super(CryoEMEnv, self).__init__()
        self.cryoEM_data = cryoEM_data
        self.id = id
        self.use_prediction = use_prediction
        self.use_penalty = use_penalty
        self.dynamic_reward = dynamic_reward
        # planning mode use a dummy DQN that only outputs classification results
        self.planning = planning
        self.print_trajectory = print_trajectory
        self.prediction_type = cryoEM_data.prediction_type

        if cryoEM_data.prediction_type == CryoEMConfig.CLASSIFICATION:
            self.cryoEM_feature = CTFCategoryFeature(cryoEM_data,
                                                     ctf_low_thresh=ctf_thresh,
                                                     hist_bins=hist_bins,
                                                     prediction=use_prediction)
            #self.cryoEM_feature = CTFCategoryFeature_new(cryoEM_data,
            #                                         ctf_low_thresh=ctf_thresh,
            #                                         prediction=use_prediction)
        else:
            self.cryoEM_feature = CTFValueFeature(cryoEM_data,
                                                  hist_bins=hist_bins,
                                                  ctf_low_thresh=ctf_thresh,
                                                  prediction=use_prediction)

        self.cryoEM_visual_feature = cryoEM_visual_feature
        self.history_size = history_size
        self.base_feature_dim = CryoEMConfig.FEATURE_DIM
        self.feature_dim = self.base_feature_dim * self.history_size
        self.ctf_thresh = ctf_thresh
        self.visited_holes = list()  #a list of visited holes
        self.num_holes = self.cryoEM_data.num_holes()
        self.evaluation = evaluation
        self.observation = None # keep the history of observations
        self.action_space = spaces.Discrete(self.num_holes)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_holes, self.feature_dim), dtype=np.float16)

        self.visual_sim = None
        if self.cryoEM_visual_feature is not None:
            self.visual_sim = FastVisualSimilarity(self.num_holes, coeff=visual_similarity_coeff)

        self.total_rewards = 0
        self.total_lctf = 0
        self.total_visits = 0
        self.hole_reward_list =[]
        self.hole_lctf_list = []
        self.hole_visit_list = []

#        self.index_list = self.cryoEM_data.index_list

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def current_state(self):
        current_k = self.visited_holes[-1]
        return self.cryoEM_data.get_hole(current_k)

    def next_state(self, action):
        return self.cryoEM_data.get_hole(action)

    # SWITCH_HOLES: 1; SWITCH_PATCHES: 0.42; SWITCH_SQUARES: 0.1
#    def reward(self, t):
#        return math.exp(-0.2878*(t-2.0))

    # SWITCH_HOLES: 1; SWITCH_PATCHES: 0.57; SWITCH_SQUARES: 0.23; SWITCH_GRID 0.09
    def reward(self, t):
        # return math.exp(-0.185*(t-2.0))
        return math.exp(-0.1*(t-1.0))

    def penalty(self, max_penalty=-1.0):
        penalty = 0.0
        current_idx = self.visited_holes[-1]
        for idx in reversed(self.visited_holes):
            hole = self.cryoEM_data.get_hole(idx)
            #consecutive holes on the same patch
            if self.cryoEM_data.is_patch_same(current_idx, idx) and hole.gt_ctf.value > self.ctf_thresh:
                penalty -= 0.2
                #penalty -= 0.1
                if penalty <= max_penalty:
                    break
            else:
                break

        return penalty

    def step(self, action):

        #ctf_value = input('Please enter....')
        #print (action, ctf_value)

        current_action = self.visited_holes[-1]
        self.take_action(action)

        switch_time = 0
        r = 0.0
        if self.cryoEM_data.is_patch_same(current_action, action): # same patch different holes
            switch_time = CryoEMConfig.SWITCH_HOLE_LOSS
            r = 1.0
        elif self.cryoEM_data.is_square_same(current_action, action): # same square different patches
            switch_time = CryoEMConfig.SWITCH_PATCH_LOSS
            # switch_time = CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
            # r = 0.57
            r = 0.90
        elif self.cryoEM_data.is_grid_same(current_action, action): # same grid different squares
            switch_time = CryoEMConfig.SWITCH_SQUARE_LOSS
            # switch_time = CryoEMConfig.SWITCH_SQUARE_LOSS + CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
            # r = 0.23
            r = 0.61
        else:
            switch_time = CryoEMConfig.SWITCH_GRID_LOSS
            # switch_time = CryoEMConfig.SWITCH_GRID_LOSS + CryoEMConfig.SWITCH_SQUARE_LOSS + \
            #           CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
            # r = 0.09
            r = 0.50

        self.time += switch_time

        next_hole = self.next_state(action)
        # use the OBSERVED ctf to compute reward
        if next_hole.gt_ctf.value <= self.ctf_thresh and not next_hole.status:
            #reward = self.reward(switch_time)
            reward = r
        else:
            reward = CryoEMConfig.HIGH_CTF_REWARD if not self.use_penalty else self.penalty(max_penalty=-1.0)
            if self.dynamic_reward:
                delta_ctf = (next_hole.gt_ctf.value - self.ctf_thresh) / 2.0
                reward = max(r * (1.2*math.exp(-1.7*delta_ctf) - 0.2), 0)
                #print (delta_ctf, r * (1.2*math.exp(-1.7*delta_ctf) - 0.2), r, reward)

        #print (switch_time, reward)
        self.total_rewards += reward
        self.total_visits += 1
        if next_hole.gt_ctf.value <= self.ctf_thresh and not next_hole.status: # true low ctf
            self.total_lctf += 1

        if self.evaluation and self.print_trajectory:
            current_hole = self.cryoEM_data.get_hole(current_action)
            current_idx = self.cryoEM_data.idx(current_action)
            next_idx = self.cryoEM_data.idx(action)
            current_gt_ctf = current_hole.gt_ctf.value
            current_ctf = current_hole.gt_ctf.value if self.prediction_type == CryoEMConfig.CLASSIFICATION else current_hole.ctf.value
            next_gt_ctf = next_hole.gt_ctf.value
            next_ctf = next_hole.gt_ctf.value if self.prediction_type == CryoEMConfig.CLASSIFICATION else next_hole.ctf.value
            print ("id {} CH {} ({} {} {} {}) [{} {}*] --> NH ({} {} {} {}) [{} {}*] Time {} Rew {}".format(self.id, current_hole.name, current_idx[0], current_idx[1], current_idx[2], current_idx[3], current_ctf, current_gt_ctf, \
                                                                                                next_idx[0], next_idx[1], next_idx[2], next_idx[3], next_ctf, next_gt_ctf, \
                                                                                                switch_time, reward), flush=True)

        self.cryoEM_data.set_hole_status(action, status=True)
        done = self.time > CryoEMConfig.Searching_Limit
        if not done:
            self.next_observation()

        if done and self.evaluation:
            self.hole_reward_list.append(self.total_rewards)
            self.hole_lctf_list.append(self.total_lctf)
            self.hole_visit_list.append(self.total_visits)
            if self.print_trajectory:
                trajectory_info = np.stack((self.hole_reward_list, self.hole_lctf_list, self.hole_visit_list), axis=1)
                #print (trajectory_info)
                hole_means = trajectory_info.mean(axis=0)
                hole_std = trajectory_info.std(axis=0)
                print ('End of Trajectory &{:.1f} $\pm$ {:.1f}&{:.1f} $\pm$ {:.1f}&{:.1f} $\pm$ {:.1f}\n'.format(
                   hole_means[0], hole_std[0], hole_means[1], hole_std[1], hole_means[2], hole_std[2]), flush=True)
        mask = np.array([True]*self.num_holes)
        mask[self.visited_holes] = False
        assert mask.shape[0] == self.num_holes
        obs={'obs': self.obs_reshaped(self.observation), 'mask': mask}

        info = {}
        if self.visual_sim is not None:
#            positive_idx = [ idx for idx in self.visited_holes if self.cryoEM_data.get_hole(idx).gt_ctf.value<=self.ctf_thresh]
#            negative_idx = [ idx for idx in self.visited_holes if self.cryoEM_data.get_hole(idx).gt_ctf.value>self.ctf_thresh]
            positive_idx = [action] if next_hole.gt_ctf.value <= self.ctf_thresh else []
            negative_idx = [action] if next_hole.gt_ctf.value > self.ctf_thresh else []

            visual_sim = self.visual_sim.compute_visual_similarity(self.cryoEM_visual_feature, positive_idx, negative_idx)
            #print ('xxx', np.sort(visual_sim)[-10:])
            info['visual_sim'] = visual_sim

        if self.planning:
            if self.use_prediction:
                info['planning'] = np.array([self.cryoEM_data.get_hole(idx).category.value[0] for idx in range(self.num_holes)])
            else:
                info['planning'] = np.array([self.cryoEM_data.get_hole(idx).gt_category.value[0] for idx in range(self.num_holes)])
        return obs, reward, done, info

    def take_action(self, action):
        # our action is taken to modify the ctf and the visited status
        #print (action, self.num_holes)
        self.visited_holes.append(action)

        # need to set the observed ctf value here
        # .....

    def reset(self):
        # compuate statistics
        if self.evaluation and self.print_trajectory:
            print ('--------- Start of Trajectory Evaluation ------------------', flush=True)

        # reset every hole as unvisited
        self.cryoEM_data.init_status()

        self.num_holes = self.cryoEM_data.num_holes()
        self.visited_holes = list()

        if self.visual_sim is not None:
            self.visual_sim.reset()

        # pick a random start
        self.time = 0
        self.random_start = np.random.randint(0, self.num_holes - 1)
        self.visited_holes.append(self.random_start)
        self.cryoEM_data.set_hole_status(self.random_start, status=True)
        self.initial_state()
        self.next_observation()

        self.total_rewards = 0
        self.total_lctf = 0
        self.total_visits = 0

        # mask out visited holes
        mask =np.array([True]*self.num_holes)
        mask[self.visited_holes] = False
        assert mask.shape[0] == self.num_holes
        obs={'obs':self.obs_reshaped(self.observation), 'mask':mask}
        return obs

    def obs_reshaped(self, obs):
        obs_new = np.transpose(obs, (1, 0, 2))  # history_size * num_holes * feature_dim
        obs_new = np.reshape(obs_new, (self.num_holes, -1))
        return obs_new

    def initial_state(self):

        # ????? need to double check
        self.observation = np.zeros((self.history_size, self.num_holes, self.base_feature_dim), dtype=np.float16)

    def next_observation(self):   #  shape [history_size * num_holes * feature_dim]

        current_hole = self.visited_holes[-1]
        feature = self.cryoEM_feature.compute_CryoEMdata_features(current_hole)

        # left shift the history and append the new one
        if len(self.visited_holes) == 1:
            self.observation[:, ...] = feature
        else:
            self.observation[:-1, ...] = self.observation[1:, ...]  # whether shift?
            self.observation[-1, ...] = feature

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return None

    def close(self):
        return None
