#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import argparse
import glob
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


# In[2]:


# Read Timestamps. In the timestamp csv file, there are two column: 1) file_name 2) time_stamp
TimeStamp_FILE = "cryo_em/timestamps.csv"
with open(TimeStamp_FILE, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    timestamps = [(row[0],row[1]) for row in csvreader] 
timeStamps = {}
names = []
for i in timestamps:
    timeStamps[i[0]] = i[1]
    names.append(i[0]) # name list 

# Read CTF values of ens
CTF_FILE = "cryo_em/target_CTF.csv"
with open(CTF_FILE, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    ctfs = [(row[0],row[3]) for row in csvreader if row[0].startswith('21mar02c')]   # find all 21mar02c data
CTFs = {}
for i in ctfs:
    CTFs[i[0]] = float(i[1])


# In[3]:


def splitDict(d):
    d1 = dict(list(d.items())[len(d)//2:])
    d2 = dict(list(d.items())[:len(d)//2])
    return d1,d2
trainCTFs, testCTFs = splitDict(CTFs)


# In[20]:


import tianshou as ts
import gym
from gym import spaces

class CryoEMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows, feature_dim, state_buff, hole_index, patch_index, patch_visited, square_visited):   
        
        super(CryoEMEnv, self).__init__()
        self.true_ctf = np.array(true_ctf)
        self.pred_ctf = pred_ctf
        self.patch_ctfs = patch_ctfs
        self.patch_ctfs_concat = np.concatenate( patch_ctfs, axis=0 )
        self.square_ctfs = np.array(square_ctfs)
        self.patch_lows = patch_lows
        self.patch_lows_concat = np.concatenate( patch_lows, axis = 0)
        self.square_lows = np.array(square_lows)
        self.hole_index = np.array(hole_index)
        self.patch_index = np.array(patch_index)
        self.patch_visited = patch_visited
        self.square_visited = square_visited
        # all holes available
        self.action_space = spaces.Discrete(len(true_ctf))
        # all observation following each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(true_ctf), feature_dim*state_buff), dtype=np.float16) 
        

    def step(self, action):
        one_step_start = time.time()
        # Execute one time step within the environment
        pt1 = time.time()
        self._take_action(action)
        pt2 = time.time()
#         print("take action:",pt2-pt1)
        if self.true_ctf[action] <6 and not self.visited:
            reward = 1
        else:
            reward = 0
        if self.changeSquare == 1:
#             reward -= 2  # penalize Square change  
            self.time += 10
        elif self.changePatch == 1:
#             reward -= 1  # penalize Patch change
            self.time += 5
        else:
            self.time += 2
        done = self.time > 100
        obs = self._next_observation()
        pt3 = time.time()
#         print("next observation:",pt3-pt2)
        log = {}
        log['time'] = self.time
#         print("reward,",reward)
#         print("visited?",self.visited,"pos?",self.square, self.patch, self.hole)
# #         print("one step time:", time.time()-one_step_start)
#         print()
        return obs, reward, done, log
        
    def reset(self):
#         print("reset!")
        # Reset the state of the environment to an initial state
        self.time = 0
        self.state = np.zeros(len(self.true_ctf)) # all empty, not visited
        self.random_start = random.randint(0,len(self.true_ctf)-1)
        self.state[self.random_start] = 1 # randomly starting from one position
        
        ##### init as zeros 4*23   ###########
        self._initial_state()
        
        return self._next_observation()
    
    
    def _initial_state(self):
        self.changeSquare = 1 # feature 1
        self.changeSquareList = np.array([0,0,0,self.changeSquare])
        self.changePatch = 1 # feature 2
        self.changePatchList = np.array([0,0,0,self.changePatch])
        self.gt_CTF = self.true_ctf[self.random_start]/np.max(self.true_ctf) # feature 3
        self.gt_CTFList = np.array([1,1,1,self.gt_CTF])
        self.est_CTF = self.true_ctf[self.random_start]/np.max(self.true_ctf) # feature 4
        self.est_CTFList = np.array([1,1,1,self.est_CTF])
        self.visited = 0 # feautre 5
        self.visitedList = np.array([1,1,1,self.visited])
        self.square, self.patch, self.hole = self.hole_index[self.random_start]
        self.patch_ctf_hist = self.patch_ctfs[self.square][self.patch]/np.sum(self.patch_ctfs[self.square][self.patch]) # feature 6-10
        self.patch_ctf_histList = np.append(np.array([0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),self.patch_ctf_hist)
        self.square_ctf_hist = self.square_ctfs[self.square]/np.sum(self.square_ctfs[self.square]) # feature 11-15
        self.square_ctf_histList = np.append(np.array([0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),self.square_ctf_hist)
        self.patch_visited_toUpdate = copy.deepcopy(self.patch_visited) 
        self.square_visited_toUpdate = np.array(copy.deepcopy(self.square_visited) )
        self.patch_visited_toUpdate[self.square][self.patch][0] += 1
        self.patch_visited_toUpdate[self.square][self.patch][1] -= 1
        self.square_visited_toUpdate[self.square][0] += 1
        self.square_visited_toUpdate[self.square][1] -= 1
        self.patch_visit = self.patch_visited_toUpdate[self.square][self.patch]/np.sum(self.patch_visited[self.square][self.patch]) # feature 16-17
        self.patch_visitList = np.append(np.array([0,1,0,1,0,1]),self.patch_visit)
        self.square_visit = self.square_visited_toUpdate[self.square]/np.sum(self.square_visited[self.square]) # feature 18-19
        self.square_visitList = np.append(np.array([0,1,0,1,0,1]),self.square_visit)
        self.patch_low = self.patch_lows[self.square][self.patch]/np.sum(self.patch_lows[self.square][self.patch]) # feature 20-21
        self.patch_lowList = np.append(np.array([0,1,0,1,0,1]),self.patch_low)
        self.square_low = self.square_lows[self.square]/np.sum(self.square_lows[self.square]) # feature 22-23
        self.square_lowList = np.append(np.array([0,1,0,1,0,1]),self.square_low)
        
    
    def _next_observation(self):
        # Append additional data and scale each value to between 0-1
        
        
        obs = self._history_state()
        obs_out = np.zeros([obs.shape[0],int(obs.shape[1]*4/3)])
        obs_out[:,0:obs.shape[1]] = obs
        
        obs_out[:,obs.shape[1]:] = self._get_obs()
        return obs_out
    
    
    
    def _get_obs(self):     # replace for into batch of actions.
        # change the code to def _get_observation(self, index)
        obs1_changeSquare = 1*np.array([self.hole_index[:,0] != self.square  ]).T # buffer the parameters 
        obs2_changePatch = 1*np.array([np.logical_and(self.hole_index[:,1] != self.patch , self.hole_index[:,0] == self.square)]).T
        obs3_gt_CTF = np.array([self.true_ctf/np.max(self.true_ctf) ]).T
        obs4_est_CTF = np.array([self.true_ctf/np.max(self.true_ctf)]).T
        obs5_visited = np.array([self.state]).T
        patch_hist = self.patch_ctfs_concat[self.patch_index].squeeze(axis = 1)
        obs6_patch_ctf_hist = self._norm_hist(patch_hist)
        obs11_square_ctf_hist = self.square_ctfs[self.hole_index[:,0]]
        obs11_square_ctf_hist = self._norm_hist(obs11_square_ctf_hist)
        self.patch_visit_concat = np.concatenate( self.patch_visited_toUpdate, axis=0 )
        patch_visit = self.patch_visit_concat[self.patch_index].squeeze(axis = 1)
        obs16_patch_visit = self._norm_hist(patch_visit)
        
        obs18_square_visit = self.square_visited_toUpdate[self.hole_index[:,0]]
        obs18_square_visit = self._norm_hist(obs18_square_visit)
        patch_low = self.patch_lows_concat[self.patch_index].squeeze(axis = 1)
        obs20_patch_low = self._norm_hist(patch_low)
        obs22_square_low = self.square_lows[self.hole_index[:,0]]
        obs22_square_low = self._norm_hist(obs22_square_low)
        obs = np.concatenate([obs1_changeSquare,obs2_changePatch,obs3_gt_CTF,obs4_est_CTF,obs5_visited,obs6_patch_ctf_hist,obs11_square_ctf_hist,obs16_patch_visit,obs18_square_visit,obs20_patch_low,obs22_square_low],axis=1)
        return obs  # 3500*23
    
    def _norm_hist(self,a):  # norm to [0,1] hist
        return (a.T/a.sum(axis=1)).T
    
#     def _next_observation(self):
#         new_observation = []
#         history = self._history_state()
#         for i in len(len(true_ctf)):
#             next_item = self._get_observation(i)
#             new_observation.append(torch.cat((history,next_item),dim=0))
#         new_observation = torch.concat(new_observation)
#         return new_observation
    
    def _history_state(self):
        obs1_changeSquare = self.changeSquareList[1:4]
        obs2_changePatch = self.changePatchList[1:4]
        obs3_gt_CTF = self.gt_CTFList[1:4]
        obs4_est_CTF = self.est_CTFList[1:4]
        obs5_visited = self.visitedList[1:4]
        obs6_patch_ctf_hist = self.patch_ctf_histList[1*5:4*5]
        obs11_square_ctf_hist = self.square_ctf_histList[1*5:4*5]
        obs16_patch_visit = self.patch_visitList[1*2:4*2]
        obs18_square_visit = self.square_visitList[1*2:4*2]
        obs20_patch_low = self.patch_lowList[1*2:4*2]
        obs22_square_low = self.square_lowList[1*2:4*2]
        obs = np.concatenate([obs1_changeSquare,obs2_changePatch,obs3_gt_CTF,obs4_est_CTF,obs5_visited,obs6_patch_ctf_hist,obs11_square_ctf_hist,obs16_patch_visit,obs18_square_visit,obs20_patch_low,obs22_square_low])
        obs = np.repeat(np.array([obs]),len(self.true_ctf),axis=0)
        return obs
        
#     def _get_observation(self, index):
#         square, patch, hole = hole_index[index] # record the index of square, patch, and hole.
#         obs = []
#         if square != self.square:
            
#         obs1_changeSquare = np.array([[int(i[0] != self.square ) for i in hole_index]]).T # buffer the parameters 
#         obs2_changePatch = np.array([[int(i[1] != self.patch ) for i in hole_index]]).T
#         obs3_gt_CTF = np.array([[i/np.max(self.true_ctf) for i in true_ctf]]).T
#         obs4_est_CTF = np.array([[i/np.max(self.true_ctf) for i in true_ctf]]).T
#         obs5_visited = np.array([[i for i in self.state]]).T
#         obs6_patch_ctf_hist = np.array([self.patch_ctfs[i[0]][i[1]]/np.sum(self.patch_ctfs[i[0]][i[1]]) for i in hole_index ])
#         obs11_square_ctf_hist = np.array([ self.square_ctfs[i[0]]/np.sum(self.square_ctfs[i[0]]) for i in hole_index ])
#         obs16_patch_visit = np.array([self.patch_visited[i[0]][i[1]]/np.sum(self.patch_visited[i[0]][i[1]]) for i in hole_index ])
#         obs18_square_visit = np.array([self.square_visited[i[0]]/np.sum(self.square_visited[i[0]]) for i in hole_index ])
#         obs20_patch_low = np.array([self.patch_lows[i[0]][i[1]]/np.sum(self.patch_lows[i[0]][i[1]]) for i in hole_index ])
#         obs22_square_low = np.array([self.square_lows[i[0]]/np.sum(self.square_lows[i[0]]) for i in hole_index ])
#         obs = np.concatenate([obs1_changeSquare,obs2_changePatch,obs3_gt_CTF,obs4_est_CTF,obs5_visited,obs6_patch_ctf_hist,obs11_square_ctf_hist,obs16_patch_visit,obs18_square_visit,obs20_patch_low,obs22_square_low],axis=1)
#         return obs  # 3500*23
    
    
    def _take_action(self, action):
        self.visited = self.state[action] #feature 5
        self.state[action] = 1 
        self.changeSquare = int(hole_index[action][0] != self.square) # feature 1
        self.changeSquareList = np.append(self.changeSquareList[1:4],self.changeSquare)
        self.changePatch = int(hole_index[action][1] != self.patch and hole_index[action][0] == self.square) # feature 2
        self.changePatchList = np.append(self.changePatchList[1:4],self.changePatch)
        self.gt_CTF = self.true_ctf[action]/np.max(self.true_ctf) # feature 3
        self.gt_CTFList = np.append(self.gt_CTFList[1:4],self.gt_CTF)
        self.est_CTF = self.true_ctf[action]/np.max(self.true_ctf) # feature 4
        self.est_CTFList = np.append(self.est_CTFList[1:4],self.est_CTF)
        self.visitedList = np.append(self.visitedList[1:4],self.visited)
        self.square, self.patch, self.hole = self.hole_index[action]
        self.patch_ctf_hist = self.patch_ctfs[self.square][self.patch]/np.sum(self.patch_ctfs[self.square][self.patch]) # feature 6-10
        self.patch_ctf_histList = np.append(self.patch_ctf_histList[1*5:4*5],self.patch_ctf_hist)
        self.square_ctf_hist = self.square_ctfs[self.square]/np.sum(self.square_ctfs[self.square]) # feature 11-15
        self.square_ctf_histList = np.append(self.patch_ctf_histList[1*5:4*5],self.square_ctf_hist)

        self.patch_visited_toUpdate[self.square][self.patch][0] += 1
        self.patch_visited_toUpdate[self.square][self.patch][1] -= 1  # def get_patch_visited_times(): to mention what the code is doing.
        self.square_visited_toUpdate[self.square][0] += 1
        self.square_visited_toUpdate[self.square][1] -= 1
        self.patch_visit = self.patch_visited_toUpdate[self.square][self.patch]/np.sum(self.patch_visited[self.square][self.patch]) # feature 16-17
        self.patch_visitList = np.append(self.patch_visitList[1*2:4*2],self.patch_visit)
        self.square_visit = self.square_visited_toUpdate[self.square]/np.sum(self.square_visited[self.square]) # feature 18-19
        self.square_visitList = np.append(self.square_visitList[1*2:4*2],self.square_visit)
        self.patch_low = self.patch_lows[self.square][self.patch]/np.sum(self.patch_lows[self.square][self.patch]) # feature 20-21
        self.patch_lowList = np.append(self.patch_lowList[1*2:4*2],self.patch_low)
        self.square_low = self.square_lows[self.square]/np.sum(self.square_lows[self.square]) # feature 22-23
        self.square_lowList = np.append(self.square_lowList[1*2:4*2],self.square_low)
        
        
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return None
    
    def close(self):
        return None
        
    def get_time(self):
        return self.time

# train_envs = CryoEMEnv(true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows,feature_dim,state_buff, hole_index,patch_index,patch_visited,square_visited)




# env = CryoEMEnv(true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows,feature_dim,state_buff, hole_index)
# env.reset()
# env.step(2)


# In[21]:


CTFs = trainCTFs
# Collect all squares and holes
squares = set()
holes = set()
for j in CTFs.keys(): 
    for i in names:
        if i.endswith('sq') and j.startswith(i):
            squares.add(i)
        if i.endswith('hl') and j.startswith(i):
            holes.add(i)   
squares = list(squares)
squares.sort()
holes = list(holes) 
holes.sort()
hl_en = {}
for i in holes:
    hl_en[i] = {en: CTFs[en] for en in CTFs.keys() if en.startswith(i)}
# sq,hl,en: ctf
sq_hl_en = {}
for i in squares:
    sq_hl_en[i] = {hl: hl_en[hl] for hl in holes if hl.startswith(i)}
    
# list of ctf
hole_index = []
patch_index = []
hole_ctf = []
square_ctf_histogram = []
square_low_ctf = []
patch_cnt = 0
for i_th, square in enumerate(sq_hl_en.values()):
    patch_ctf_histogram = []
    patch_low_ctf = []
    for j_th, patch in enumerate(square.values()):
        patch_j_ctf_hist = np.zeros(5)
        patch_j_low_ctf = np.zeros(2)
        
        for k_th, hole_CTF in enumerate(patch.values()):
            patch_index.append([patch_cnt])
            if hole_CTF > 20:  ############# we can also clip the large value into max value ################
                hole_CTF = 20
            hole_index.append((i_th,j_th,k_th)) # the index of each hole
            hole_ctf.append(hole_CTF) # the ctf of each hole
            if hole_CTF<3:
                patch_j_ctf_hist[0] += 1
            elif hole_CTF<5:
                patch_j_ctf_hist[1] += 1
            elif hole_CTF<7:
                patch_j_ctf_hist[2] += 1
            elif hole_CTF<9:
                patch_j_ctf_hist[3] += 1
            else:
                patch_j_ctf_hist[4] += 1
            if hole_CTF<6:
                patch_j_low_ctf[0] += 1
            else:
                patch_j_low_ctf[1] += 1
        patch_ctf_histogram.append(patch_j_ctf_hist)
        patch_low_ctf.append(patch_j_low_ctf)
        patch_cnt += 1
    square_ctf_histogram.append(patch_ctf_histogram)  
    square_low_ctf.append(patch_low_ctf)
    
# True CTF: 
true_ctf = hole_ctf
# Predicted CTF:
pred_ctf = hole_ctf # same as true ctf currently
# Histogram of CTF in patch, take 3,5,7,9 as division
patch_ctfs = square_ctf_histogram
# Histogram of CTF in square
square_ctfs = [np.sum(i,0) for i in square_ctf_histogram]
# How many low CTFs in patch, take 6 as cutoff
patch_lows = square_low_ctf
# How many low CTFs in square, take 6 as cutoff
square_lows = [np.sum(i,0) for i in square_low_ctf]
# Initial State of Patch visited:
patch_visited = copy.deepcopy(patch_lows)
for i,sq in enumerate(patch_visited):
    for j,pt in enumerate(sq):
        patch_visited[i][j] = np.array([0,np.sum(patch_visited[i][j])])
# Initial State of Square visited:       
square_visited = [np.sum(i,0) for i in patch_visited]
feature_dim = 23
state_buff = 4

# PS: How many holes are there?
train_envs = CryoEMEnv(true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows,feature_dim,state_buff, hole_index,patch_index,patch_visited,square_visited)


# In[15]:


CTFs = testCTFs

# Collect all squares and holes
squares = set()
holes = set()
for j in CTFs.keys(): 
    for i in names:
        if i.endswith('sq') and j.startswith(i):
            squares.add(i)
        if i.endswith('hl') and j.startswith(i):
            holes.add(i)   
squares = list(squares)
squares.sort()
holes = list(holes) 
holes.sort()
hl_en = {}
for i in holes:
    hl_en[i] = {en: CTFs[en] for en in CTFs.keys() if en.startswith(i)}
# sq,hl,en: ctf
sq_hl_en = {}
for i in squares:
    sq_hl_en[i] = {hl: hl_en[hl] for hl in holes if hl.startswith(i)}
    
# list of ctf
hole_index = []
patch_index = []
hole_ctf = []
square_ctf_histogram = []
square_low_ctf = []
patch_cnt = 0
for i_th, square in enumerate(sq_hl_en.values()):
    patch_ctf_histogram = []
    patch_low_ctf = []
    for j_th, patch in enumerate(square.values()):
        patch_j_ctf_hist = np.zeros(5)
        patch_j_low_ctf = np.zeros(2)
        
        for k_th, hole_CTF in enumerate(patch.values()):
            patch_index.append([patch_cnt])
            if hole_CTF > 20:  ############# we can also clip the large value into max value ################
                hole_CTF = 20
            hole_index.append((i_th,j_th,k_th)) # the index of each hole
            hole_ctf.append(hole_CTF) # the ctf of each hole
            if hole_CTF<3:
                patch_j_ctf_hist[0] += 1
            elif hole_CTF<5:
                patch_j_ctf_hist[1] += 1
            elif hole_CTF<7:
                patch_j_ctf_hist[2] += 1
            elif hole_CTF<9:
                patch_j_ctf_hist[3] += 1
            else:
                patch_j_ctf_hist[4] += 1
            if hole_CTF<6:
                patch_j_low_ctf[0] += 1
            else:
                patch_j_low_ctf[1] += 1
        patch_ctf_histogram.append(patch_j_ctf_hist)
        patch_low_ctf.append(patch_j_low_ctf)
        patch_cnt += 1
    square_ctf_histogram.append(patch_ctf_histogram)  
    square_low_ctf.append(patch_low_ctf)
    
# True CTF: 
true_ctf = hole_ctf
# Predicted CTF:
pred_ctf = hole_ctf # same as true ctf currently
# Histogram of CTF in patch, take 3,5,7,9 as division
patch_ctfs = square_ctf_histogram
# Histogram of CTF in square
square_ctfs = [np.sum(i,0) for i in square_ctf_histogram]
# How many low CTFs in patch, take 6 as cutoff
patch_lows = square_low_ctf
# How many low CTFs in square, take 6 as cutoff
square_lows = [np.sum(i,0) for i in square_low_ctf]
# Initial State of Patch visited:
patch_visited = copy.deepcopy(patch_lows)
for i,sq in enumerate(patch_visited):
    for j,pt in enumerate(sq):
        patch_visited[i][j] = np.array([0,np.sum(patch_visited[i][j])])
# Initial State of Square visited:       
square_visited = [np.sum(i,0) for i in patch_visited]
feature_dim = 23
state_buff = 4

# PS: How many holes are there?



test_envs = CryoEMEnv(true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows,feature_dim,state_buff, hole_index,patch_index,patch_visited,square_visited)


# In[7]:


# -*- coding: utf-8 -*-


# In[22]:


import torch, numpy as np
from torch import nn

# Policy transform from local feature space to action selection. 
# Feature 2*5*25 as our choice into which hole to be chosen. 




# input is a pair of 
# output is Qvalue not the action decision
class Net(nn.Module):
    def __init__(self, state_feature_dim):
        super().__init__() #23*4
        self.model = nn.Sequential(*[ 
            nn.Linear(state_feature_dim, 256), nn.ReLU(inplace=True), 
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 1) 
        ])
#         self.model = nn.Sequential(*[
#             nn.Linear(state_feature_dim, 1)
#         ])
    def forward(self, obs,  state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        action_shape = obs.shape[1]
        obs =  torch.reshape(obs, (batch*action_shape, -1))
#         m = nn.Softmax(dim=1)
        logits_output = self.model(obs)
        logits_output = torch.reshape(logits_output, (batch, -1))
#         logits_output = m(logits_output)
        return logits_output, state

state_shape = train_envs.observation_space.shape 
action_shape, state_feature_dim = state_shape

# print(env.action_space)
net = Net(state_feature_dim)
optim = torch.optim.Adam(net.parameters(), lr=1e-2)
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=10, target_update_freq=320)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, train_envs, exploration_noise=True)

result = ts.trainer.offpolicy_trainer(
    policy, train_collector, test_collector,
    max_epoch=10, step_per_epoch=10000, step_per_collect=10,
    update_per_step=0.1, episode_per_test=10, batch_size=10,
    train_fn=lambda epoch, env_step: policy.set_eps(0.1),
    test_fn=lambda epoch, env_step: policy.set_eps(0.05))
print(f'Finished training! Use {result["duration"]}')


# In[23]:


torch.save(policy.state_dict(), 'dqn.pth')
policy.load_state_dict(torch.load('dqn.pth'))


# In[31]:


import tianshou as ts
import gym
from gym import spaces

class CryoEMEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows, feature_dim, state_buff, hole_index, patch_index, patch_visited, square_visited,time_constraint):   
        
        super(CryoEMEnv, self).__init__()
        self.true_ctf = np.array(true_ctf)
        self.pred_ctf = pred_ctf
        self.patch_ctfs = patch_ctfs
        self.patch_ctfs_concat = np.concatenate( patch_ctfs, axis=0 )
        self.square_ctfs = np.array(square_ctfs)
        self.patch_lows = patch_lows
        self.patch_lows_concat = np.concatenate( patch_lows, axis = 0)
        self.square_lows = np.array(square_lows)
        self.hole_index = np.array(hole_index)
        self.patch_index = np.array(patch_index)
        self.patch_visited = patch_visited
        self.square_visited = square_visited
        self.time_constraint = time_constraint
        # all holes available
        self.action_space = spaces.Discrete(len(true_ctf))
        # all observation following each action
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(true_ctf), feature_dim*state_buff), dtype=np.float16) 
        

    def step(self, action):
        one_step_start = time.time()
        # Execute one time step within the environment
        pt1 = time.time()
        self._take_action(action)
        pt2 = time.time()
#         print("take action:",pt2-pt1)
        if self.true_ctf[action] <6 and not self.visited:
            reward = 1     # exp(-1/t)
        else: 
            reward = 0
        if self.changeSquare == 1:
#             reward -= 2  # penalize Square change  
            self.time += 10
        elif self.changePatch == 1:
#             reward -= 1  # penalize Patch change
            self.time += 5
        else:
            self.time += 2
        done = self.time > self.time_constraint
        obs = self._next_observation()
        pt3 = time.time()
#         print("next observation:",pt3-pt2)
        log = {}
        log['time'] = self.time
        print("reward,",reward)
        print("visited?",self.visited,"pos?",self.square, self.patch, self.hole,"CTF?", self.true_ctf[action])
# #         print("one step time:", time.time()-one_step_start)
        print()
        return obs, reward, done, log
        
    def reset(self):
#         print("reset!")
        # Reset the state of the environment to an initial state
        self.time = 0
        self.state = np.zeros(len(self.true_ctf)) # all empty, not visited
        self.random_start = random.randint(0,len(self.true_ctf)-1)
        self.state[self.random_start] = 1 # randomly starting from one position
        
        ##### init as zeros 4*23   ###########
        self._initial_state()
        
        return self._next_observation()
    
    
    def _initial_state(self):
        self.changeSquare = 1 # feature 1
        self.changeSquareList = np.array([0,0,0,self.changeSquare])
        self.changePatch = 1 # feature 2
        self.changePatchList = np.array([0,0,0,self.changePatch])
        self.gt_CTF = self.true_ctf[self.random_start]/np.max(self.true_ctf) # feature 3
        self.gt_CTFList = np.array([1,1,1,self.gt_CTF])
        self.est_CTF = self.true_ctf[self.random_start]/np.max(self.true_ctf) # feature 4
        self.est_CTFList = np.array([1,1,1,self.est_CTF])
        self.visited = 0 # feautre 5
        self.visitedList = np.array([1,1,1,self.visited])
        self.square, self.patch, self.hole = self.hole_index[self.random_start]
        self.patch_ctf_hist = self.patch_ctfs[self.square][self.patch]/np.sum(self.patch_ctfs[self.square][self.patch]) # feature 6-10
        self.patch_ctf_histList = np.append(np.array([0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),self.patch_ctf_hist)
        self.square_ctf_hist = self.square_ctfs[self.square]/np.sum(self.square_ctfs[self.square]) # feature 11-15
        self.square_ctf_histList = np.append(np.array([0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]),self.square_ctf_hist)
        self.patch_visited_toUpdate = copy.deepcopy(self.patch_visited) 
        self.square_visited_toUpdate = np.array(copy.deepcopy(self.square_visited) )
        self.patch_visited_toUpdate[self.square][self.patch][0] += 1
        self.patch_visited_toUpdate[self.square][self.patch][1] -= 1
        self.square_visited_toUpdate[self.square][0] += 1
        self.square_visited_toUpdate[self.square][1] -= 1
        self.patch_visit = self.patch_visited_toUpdate[self.square][self.patch]/np.sum(self.patch_visited[self.square][self.patch]) # feature 16-17
        self.patch_visitList = np.append(np.array([0,1,0,1,0,1]),self.patch_visit)
        self.square_visit = self.square_visited_toUpdate[self.square]/np.sum(self.square_visited[self.square]) # feature 18-19
        self.square_visitList = np.append(np.array([0,1,0,1,0,1]),self.square_visit)
        self.patch_low = self.patch_lows[self.square][self.patch]/np.sum(self.patch_lows[self.square][self.patch]) # feature 20-21
        self.patch_lowList = np.append(np.array([0,1,0,1,0,1]),self.patch_low)
        self.square_low = self.square_lows[self.square]/np.sum(self.square_lows[self.square]) # feature 22-23
        self.square_lowList = np.append(np.array([0,1,0,1,0,1]),self.square_low)
        print("visited?",self.visited,"pos?",self.square, self.patch, self.hole,"CTF?", self.true_ctf[self.random_start])
    
    def _next_observation(self):
        # Append additional data and scale each value to between 0-1
        
        
        obs = self._history_state()
        obs_out = np.zeros([obs.shape[0],int(obs.shape[1]*4/3)])
        obs_out[:,0:obs.shape[1]] = obs
        
        obs_out[:,obs.shape[1]:] = self._get_obs()
        return obs_out
    
    
    
    def _get_obs(self):     # replace for into batch of actions.
        # change the code to def _get_observation(self, index)
        obs1_changeSquare = 1*np.array([self.hole_index[:,0] != self.square  ]).T # buffer the parameters 
        obs2_changePatch = 1*np.array([np.logical_and(self.hole_index[:,1] != self.patch , self.hole_index[:,0] == self.square)]).T
        obs3_gt_CTF = np.array([self.true_ctf/np.max(self.true_ctf) ]).T
        obs4_est_CTF = np.array([self.true_ctf/np.max(self.true_ctf)]).T
        obs5_visited = np.array([self.state]).T
        patch_hist = self.patch_ctfs_concat[self.patch_index].squeeze(axis = 1)
        obs6_patch_ctf_hist = self._norm_hist(patch_hist)
        obs11_square_ctf_hist = self.square_ctfs[self.hole_index[:,0]]
        obs11_square_ctf_hist = self._norm_hist(obs11_square_ctf_hist)
        self.patch_visit_concat = np.concatenate( self.patch_visited_toUpdate, axis=0 )
        patch_visit = self.patch_visit_concat[self.patch_index].squeeze(axis = 1)
        obs16_patch_visit = self._norm_hist(patch_visit)
        
        obs18_square_visit = self.square_visited_toUpdate[self.hole_index[:,0]]
        obs18_square_visit = self._norm_hist(obs18_square_visit)
        patch_low = self.patch_lows_concat[self.patch_index].squeeze(axis = 1)
        obs20_patch_low = self._norm_hist(patch_low)
        obs22_square_low = self.square_lows[self.hole_index[:,0]]
        obs22_square_low = self._norm_hist(obs22_square_low)
        obs = np.concatenate([obs1_changeSquare,obs2_changePatch,obs3_gt_CTF,obs4_est_CTF,obs5_visited,obs6_patch_ctf_hist,obs11_square_ctf_hist,obs16_patch_visit,obs18_square_visit,obs20_patch_low,obs22_square_low],axis=1)
        return obs  # 3500*23
    
    def _norm_hist(self,a):  # norm to [0,1] hist
        return (a.T/a.sum(axis=1)).T
    
#     def _next_observation(self):
#         new_observation = []
#         history = self._history_state()
#         for i in len(len(true_ctf)):
#             next_item = self._get_observation(i)
#             new_observation.append(torch.cat((history,next_item),dim=0))
#         new_observation = torch.concat(new_observation)
#         return new_observation
    
    def _history_state(self):
        obs1_changeSquare = self.changeSquareList[1:4]
        obs2_changePatch = self.changePatchList[1:4]
        obs3_gt_CTF = self.gt_CTFList[1:4]
        obs4_est_CTF = self.est_CTFList[1:4]
        obs5_visited = self.visitedList[1:4]
        obs6_patch_ctf_hist = self.patch_ctf_histList[1*5:4*5]
        obs11_square_ctf_hist = self.square_ctf_histList[1*5:4*5]
        obs16_patch_visit = self.patch_visitList[1*2:4*2]
        obs18_square_visit = self.square_visitList[1*2:4*2]
        obs20_patch_low = self.patch_lowList[1*2:4*2]
        obs22_square_low = self.square_lowList[1*2:4*2]
        obs = np.concatenate([obs1_changeSquare,obs2_changePatch,obs3_gt_CTF,obs4_est_CTF,obs5_visited,obs6_patch_ctf_hist,obs11_square_ctf_hist,obs16_patch_visit,obs18_square_visit,obs20_patch_low,obs22_square_low])
        obs = np.repeat(np.array([obs]),len(self.true_ctf),axis=0)
        return obs
        
#     def _get_observation(self, index):
#         square, patch, hole = hole_index[index] # record the index of square, patch, and hole.
#         obs = []
#         if square != self.square:
            
#         obs1_changeSquare = np.array([[int(i[0] != self.square ) for i in hole_index]]).T # buffer the parameters 
#         obs2_changePatch = np.array([[int(i[1] != self.patch ) for i in hole_index]]).T
#         obs3_gt_CTF = np.array([[i/np.max(self.true_ctf) for i in true_ctf]]).T
#         obs4_est_CTF = np.array([[i/np.max(self.true_ctf) for i in true_ctf]]).T
#         obs5_visited = np.array([[i for i in self.state]]).T
#         obs6_patch_ctf_hist = np.array([self.patch_ctfs[i[0]][i[1]]/np.sum(self.patch_ctfs[i[0]][i[1]]) for i in hole_index ])
#         obs11_square_ctf_hist = np.array([ self.square_ctfs[i[0]]/np.sum(self.square_ctfs[i[0]]) for i in hole_index ])
#         obs16_patch_visit = np.array([self.patch_visited[i[0]][i[1]]/np.sum(self.patch_visited[i[0]][i[1]]) for i in hole_index ])
#         obs18_square_visit = np.array([self.square_visited[i[0]]/np.sum(self.square_visited[i[0]]) for i in hole_index ])
#         obs20_patch_low = np.array([self.patch_lows[i[0]][i[1]]/np.sum(self.patch_lows[i[0]][i[1]]) for i in hole_index ])
#         obs22_square_low = np.array([self.square_lows[i[0]]/np.sum(self.square_lows[i[0]]) for i in hole_index ])
#         obs = np.concatenate([obs1_changeSquare,obs2_changePatch,obs3_gt_CTF,obs4_est_CTF,obs5_visited,obs6_patch_ctf_hist,obs11_square_ctf_hist,obs16_patch_visit,obs18_square_visit,obs20_patch_low,obs22_square_low],axis=1)
#         return obs  # 3500*23
    
    
    def _take_action(self, action):
        self.visited = self.state[action] #feature 5
        self.state[action] = 1 
        self.changeSquare = int(hole_index[action][0] != self.square) # feature 1
        self.changeSquareList = np.append(self.changeSquareList[1:4],self.changeSquare)
        self.changePatch = int(hole_index[action][1] != self.patch and hole_index[action][0] == self.square) # feature 2
        self.changePatchList = np.append(self.changePatchList[1:4],self.changePatch)
        self.gt_CTF = self.true_ctf[action]/np.max(self.true_ctf) # feature 3
        self.gt_CTFList = np.append(self.gt_CTFList[1:4],self.gt_CTF)
        self.est_CTF = self.true_ctf[action]/np.max(self.true_ctf) # feature 4
        self.est_CTFList = np.append(self.est_CTFList[1:4],self.est_CTF)
        self.visitedList = np.append(self.visitedList[1:4],self.visited)
        self.square, self.patch, self.hole = self.hole_index[action]
        self.patch_ctf_hist = self.patch_ctfs[self.square][self.patch]/np.sum(self.patch_ctfs[self.square][self.patch]) # feature 6-10
        self.patch_ctf_histList = np.append(self.patch_ctf_histList[1*5:4*5],self.patch_ctf_hist)
        self.square_ctf_hist = self.square_ctfs[self.square]/np.sum(self.square_ctfs[self.square]) # feature 11-15
        self.square_ctf_histList = np.append(self.patch_ctf_histList[1*5:4*5],self.square_ctf_hist)

        self.patch_visited_toUpdate[self.square][self.patch][0] += 1
        self.patch_visited_toUpdate[self.square][self.patch][1] -= 1  # def get_patch_visited_times(): to mention what the code is doing.
        self.square_visited_toUpdate[self.square][0] += 1
        self.square_visited_toUpdate[self.square][1] -= 1
        self.patch_visit = self.patch_visited_toUpdate[self.square][self.patch]/np.sum(self.patch_visited[self.square][self.patch]) # feature 16-17
        self.patch_visitList = np.append(self.patch_visitList[1*2:4*2],self.patch_visit)
        self.square_visit = self.square_visited_toUpdate[self.square]/np.sum(self.square_visited[self.square]) # feature 18-19
        self.square_visitList = np.append(self.square_visitList[1*2:4*2],self.square_visit)
        self.patch_low = self.patch_lows[self.square][self.patch]/np.sum(self.patch_lows[self.square][self.patch]) # feature 20-21
        self.patch_lowList = np.append(self.patch_lowList[1*2:4*2],self.patch_low)
        self.square_low = self.square_lows[self.square]/np.sum(self.square_lows[self.square]) # feature 22-23
        self.square_lowList = np.append(self.square_lowList[1*2:4*2],self.square_low)
        
        
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return None
    
    def close(self):
        return None
        
    def get_time(self):
        return self.time

for time_constraint in [50,100,150,200,250,300,350,400]:    
    train_envs = CryoEMEnv(true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows,feature_dim,state_buff, hole_index,patch_index,patch_visited,square_visited,time_constraint)
    policy.eval()
    policy.set_eps(0.05)
    collector = ts.data.Collector(policy, train_envs, exploration_noise=False)
    print(collector.collect(n_episode=1, render=1 / 35))



# env = CryoEMEnv(true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows,feature_dim,state_buff, hole_index)
# env.reset()
# env.step(2)


# In[25]:


'''
# Task to do

1. Fast Convergence? Why the time-consuming?
Where is the bottleneck of the time cost? 
Adapt the code to the GPU. 

### Time bottleneck

a. shrink the network;
b. print the time slot of every code block; # data loading?; training?; step?
c. 

### test data env should be set up with 

'''
# env = CryoEMEnv(true_ctf,pred_ctf,patch_ctfs,square_ctfs,patch_lows,square_lows,feature_dim,state_buff, hole_index)

policy.eval()
policy.set_eps(0.05)
collector = ts.data.Collector(policy, train_envs, exploration_noise=False)
collector.collect(n_episode=1, render=1 / 35)


# In[ ]:


a = np.array(hole_index)
a[:,0]


# In[ ]:


obs6_patch_ctf_hist = np.array([patch_ctfs[i[0]][i[1]]/np.sum(patch_ctfs[i[0]][i[1]]) for i in hole_index ])
        


# In[ ]:


obs6_patch_ctf_hist.shape


# In[ ]:


obs6_patch_ctf_hist = np.array([patch_ctfs[a[:,0:2]]/np.sum(patch_ctfs[a[:,0:2]])])
obs6_patch_ctf_hist.shape


# In[ ]:


np.concatenate( patch_ctfs, axis=0 ).shape


# In[ ]:


np.array(hole_index).shape


# In[ ]:


a[:,0:2]


# In[ ]:


np.concatenate( square_ctfs, axis=0 ).shape


# In[ ]:


square_ctfs


# In[ ]:


# patch_ctfs_concat = np.concatenate( patch_ctfs, axis=0 )
patch_hist = patch_ctfs_concat[patch_index].squeeze(axis = 1)
(patch_hist.T/patch_hist.sum(axis=1)).T


# In[ ]:


patch_hist.shape


# In[ ]:


action = [(i,i) for i in range(1000000)] 
time1 = time.time()
feature = [i[0] == 1 for i in action]   # 10s
time2 = time.time()
feature2 = np.array(action)[:,0] != 1 # 1s
time3 = time.time()
print(time3-time2, time2-time1)


# In[ ]:


time1 = time.time()
obs6_patch_ctf_hist = np.array([patch_ctfs[i[0]][i[1]]/np.sum(patch_ctfs[i[0]][i[1]]) for i in hole_index ])
time2 = time.time()
patch_hist = patch_ctfs_concat[patch_index].squeeze(axis = 1)
obs6_patch_ctf_hist = _norm_hist(patch_hist)
time3 = time.time()
print(time3-time2, time2-time1)


# In[ ]:


patch_ctfs_concat = np.concatenate( patch_ctfs, axis=0 )
def _norm_hist(a):  # norm to [0,1] hist
    return (a.T/a.sum(axis=1)).T


# In[ ]:


# DS for CryoEMHole  # Clean up the code with the class-specific
class CryoEMHole:
    def __init__(self, idx=-1, parent_idx=-1, position=(-1, -1), ctf=-100):
        self.idx = idx
        self.parent_idx=parent_idx
        self.position = position
        self.visited = False
        self.ctf = ctf
        self.ctf_prediction = ctf
    @staticmethod
    def idx(self):
        return self.idx
    @staticmethod
    def parent_idx(self):
        return self.parent_idx
    @staticmethod
    def position(self):
        return self.position
    @staticmethod
    def is_visited(self):
        return self.visited
# DS for CryoEMPatch (a patch contains a list of holes)
class CryoEMPatch:
    def __init__(self, idx=-1, parent_idx=-1, position=(-1, -1), holeList=[]):
        self.idx = idx
        self.parent_idx=parent_idx
        self.position = position
        self.holeList=holeList
        # auxiliary data
    @staticmethod
    def idx(self):
        return self.idx
    @staticmethod
    def parent_idx(self):
        return self.parent_idx
    @staticmethod
    def position(self):
        return self.position
    @staticmethod
    def get_hole(self, idx):
        return self.hole_list[idx]
    def compute_feature(self): # if needed
# DS for CryoEMGrid (a grid contains a list of patches)
class CryoEMGrid:  # Square
    def __init__(self, idx=-1, position=(-1, -1), patchList=[]):
        self.idx = idx
        self.position = position
        self.patchList=patchList
        # auxiliary data
    @staticmethod
    def idx(self):
        return self.id
    @staticmethod
    def position(self):
        return self.pos
    @staticmethod
    def get_patch(self, idx):
        return self.patch_list[idx]
    @staticmethod
    def get_hole(self, idx):
        patch_idx, hole_idx = idx
        patch = self.get_patch(patch_idx)
        return patch.get_hole(hole_idx)
    def compute_feature(self): # if needed,
cryoEM_data = list() # list of CryoEMGrid
visited_holes = list() # list of tuples (grid_idx, patch_idx, hole_idx)


# In[ ]:


class CryoEMFeatures: # Concate  
    def __init__()

