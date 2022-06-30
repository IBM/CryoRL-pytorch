import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from cryoEM_observer import CryoEMObserver
from cryoEM_config import *
from cryoEM_sampler import CryoEMSampler
import math

class CryoEMAction:
    def __init__(self, action_idx, hole_idx):
        self._action_idx = action_idx
        self._hole_idx = hole_idx

    @property
    def action_idx(self):
        return self._action_idx

    @property
    def hole_idx(self):
        return self._hole_idx

class CryoEMEnv(gym.Env):
    # ?
    metadata = {'render.modes': ['human']}

    def __init__(self, cryoEM_data,
                 id=-1,
                 ctf_thresh=6.0,
                 use_prediction=False,
                 use_penalty=False,
                 action_elimination=False,
                 dynamic_reward=False,
                 planning=False,
                 evaluation=False,
                 sample_method='sample_by_random',
                 print_trajectory=False):
        super(CryoEMEnv, self).__init__()
        self.cryoEM_data = cryoEM_data
        self.id = id
        self.use_prediction = use_prediction
        self.action_elimination = action_elimination
        self.use_penalty = use_penalty
        self.dynamic_reward = dynamic_reward
        # planning mode use a dummy DQN that only outputs classification results
        self.planning = planning
        self.print_trajectory = print_trajectory
        self.prediction_type = cryoEM_data.prediction_type
        self.evaluation = evaluation
        self.ctf_thresh = ctf_thresh
        self.sample_method = sample_method

        action_elimination_factor = CryoEMConfig.TRAIN_AE_FACTOR if not self.evaluation else CryoEMConfig.VAL_AE_FACTOR
        self.cryoEM_observer = CryoEMObserver(cryoEM_data,
                                              use_prediction=use_prediction,
                                              action_elimination=action_elimination,
                                              action_elimination_factor = action_elimination_factor,
                                              duration=CryoEMConfig.Searching_Limit)
        self.num_holes = self.cryoEM_observer.num_holes

        sample_method = 'sample_by_random' if not self.evaluation else self.sample_method
        self.cryoEM_sampler = CryoEMSampler(cryoEM_data,
                                            use_prediction=use_prediction,
                                            num_holes=self.cryoEM_observer.num_holes,
                                            sample_method=sample_method)

        #self.observation = None # keep the history of observations
        self.action_space = spaces.Discrete(self.num_holes)
        feature_dim = CryoEMConfig.FEATURE_DIM * CryoEMConfig.HISTORY_SIZE
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_holes, feature_dim), dtype=np.float16)

        self.visited_holes = list()  #a list of visited holes
        self.total_rewards = 0
        self.total_lctf = 0
        self.total_visits = 0
        self.hole_reward_list =[]
        self.hole_lctf_list = []
        self.hole_visit_list = []

#        self.index_list = self.cryoEM_data.index_list

    def action_to_hole_idx(self, action):
        return self.cryoEM_observer.actions[action]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def current_state(self):
        current_k = self.visited_holes[-1].hole_indx
        return self.cryoEM_data.get_hole(current_k)

    def next_state(self, action):
        hole_idx = self.action_to_hole_idx(action)
        return self.cryoEM_data.get_hole(hole_idx)

    # SWITCH_HOLES: 1; SWITCH_PATCHES: 0.42; SWITCH_SQUARES: 0.1
#    def reward(self, t):
#        return math.exp(-0.2878*(t-2.0))

    # SWITCH_HOLES: 1; SWITCH_PATCHES: 0.57; SWITCH_SQUARES: 0.23; SWITCH_GRID 0.09
    def reward(self, t):
        return math.exp(-0.185*(t-2.0))

    def penalty(self, max_penalty=-1.0):
        penalty = 0.0
        current_idx = self.visited_holes[-1].hole_idx
        for item in reversed(self.visited_holes):
            hole = self.cryoEM_data.get_hole(item.hole_idx)
            #consecutive holes on the same patch
            if self.cryoEM_data.is_patch_same(current_idx, item.hole_idx) and hole.gt_ctf.value > self.ctf_thresh:
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
        # don't move this line
        current_hole_idx = self.visited_holes[-1].hole_idx
        self.take_action(action)

        next_hole_idx = self.action_to_hole_idx(action)
        switch_time = 0
        r = 0.0
        if self.cryoEM_data.is_patch_same(current_hole_idx, next_hole_idx): # same patch different holes
            switch_time = CryoEMConfig.SWITCH_HOLE_LOSS
            r = 1.0
        elif self.cryoEM_data.is_square_same(current_hole_idx, next_hole_idx): # same square different patches
            #switch_time = CryoEMConfig.SWITCH_PATCH_LOSS
            switch_time = CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
            r = 0.57
        elif self.cryoEM_data.is_grid_same(current_hole_idx, next_hole_idx): # same grid different squares
            #switch_time = CryoEMConfig.SWITCH_SQUARE_LOSS
            switch_time = CryoEMConfig.SWITCH_SQUARE_LOSS + CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
            #r = 0.23
            r = 0.46
        else:
            #switch_time = CryoEMConfig.SWITCH_GRID_LOSS
            switch_time = CryoEMConfig.SWITCH_GRID_LOSS + CryoEMConfig.SWITCH_SQUARE_LOSS + \
                      CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
            r = 0.09

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
                # linear
                #reward = max(r * (1.0 - delta_ctf), 0.0)
                #print (delta_ctf, r * (1.2*math.exp(-1.7*delta_ctf) - 0.2), r, reward)

        #print (switch_time, reward)
        self.total_rewards += reward
        self.total_visits += 1
        if next_hole.gt_ctf.value <= self.ctf_thresh and not next_hole.status: # true low ctf
            self.total_lctf += 1

        if self.evaluation and self.print_trajectory:
            current_hole = self.cryoEM_data.get_hole(current_hole_idx)
            current_idx = self.cryoEM_data.idx(current_hole_idx)
            next_idx = self.cryoEM_data.idx(next_hole_idx)
            current_gt_ctf = current_hole.gt_ctf.value
            current_ctf = current_hole.gt_ctf.value if self.prediction_type == CryoEMConfig.CLASSIFICATION else current_hole.ctf.value
            next_gt_ctf = next_hole.gt_ctf.value
            next_ctf = next_hole.gt_ctf.value if self.prediction_type == CryoEMConfig.CLASSIFICATION else next_hole.ctf.value
            print ("id {} CH {} ({} {} {} {}) [{} {}*] --> NH ({} {} {} {}) [{} {}*] Time {} Rew {}".format(self.id, current_hole.name, current_idx[0], current_idx[1], current_idx[2], current_idx[3], current_ctf, current_gt_ctf, \
                                                                                                next_idx[0], next_idx[1], next_idx[2], next_idx[3], next_ctf, next_gt_ctf, \
                                                                                                switch_time, reward), flush=True)

        #self.cryoEM_data.set_hole_status(action, status=True)
        self.cryoEM_data.set_hole_status(next_hole_idx, status=True)
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

        '''
        mask = np.array([True]*self.num_holes)
        mask[self.visited_holes] = False
        assert mask.shape[0] == self.num_holes
        '''
        obs={'obs': self.obs_reshaped(self.cryoEM_observer.observation), 'mask': self.get_hole_mask()}

        info = {}
        if self.planning:
            if self.action_elimination:
                info['planning'] = np.array([self.cryoEM_data.get_hole(idx).category.value[0] for idx in self.cryoEM_observer.selection])
            else:
                info['planning'] = np.array([self.cryoEM_data.get_hole(idx).gt_category.value[0] for idx in self.cryoEM_observer.selection])
        return obs, reward, done, info

    def take_action(self, action):
        # our action is taken to modify the ctf and the visited status
        #print (action, self.num_holes)
        self.visited_holes.append(CryoEMAction(action, self.action_to_hole_idx(action)))
        
        # need to set the observed ctf value here
        # .....

    def reset(self):
        # compuate statistics
        if self.evaluation and self.print_trajectory:
            print ('--------- Start of Trajectory Evaluation ------------------', flush=True)

        # reset every hole as unvisited
        self.cryoEM_data.init_status()
        self.visited_holes = list()

        # pick a random start
        self.time = 0
        #self.random_start = np.random.randint(0, self.num_holes - 1)
        start_action_idx, start_hole_idx = self.random_start_action()
        self.visited_holes.append(CryoEMAction(start_action_idx, start_hole_idx))
        self.cryoEM_data.set_hole_status(start_hole_idx, status=True)
        self.initial_state()
        self.next_observation()

        self.total_rewards = 0
        self.total_lctf = 0
        self.total_visits = 0

        # mask out visited holes
        '''
        mask =np.array([True]*self.num_holes)
        action_idx_list = [item.action_idx for item in self.visited_holes]
        mask[action_idx_list] = False
        assert mask.shape[0] == self.num_holes
        '''
        obs={'obs':self.obs_reshaped(self.cryoEM_observer.observation), 'mask':self.get_hole_mask()}
        return obs

    def get_hole_mask(self):
        mask =np.array([True]*self.num_holes)
        action_idx_list = [item.action_idx for item in self.visited_holes]
        mask[action_idx_list] = False
        return mask

    def random_start_action(self):
#        r = np.random.randint(0, self.num_holes - 1)
        r = self.cryoEM_sampler.sample()
        return (r,r) if not self.action_elimination else (r, self.cryoEM_observer.actions[r])

    def obs_reshaped(self, obs):
        obs_new = np.transpose(obs, (1, 0, 2))  # history_size * num_holes * feature_dim
        obs_new = np.reshape(obs_new, (self.num_holes, -1))
        return obs_new

    def initial_state(self):
        self.cryoEM_observer.initialization()

    def next_observation(self):
        # no shift for the first visited holes
        is_shift = len(self.visited_holes) > 1
        self.cryoEM_observer.update_observation(self.visited_holes[-1].hole_idx, is_shift=is_shift)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return None

    def close(self):
        return None
