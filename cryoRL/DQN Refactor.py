#!/usr/bin/env python
# coding: utf-8

# In[1]:


import argparse
import random
import csv
import matplotlib.pyplot as plt
import numpy as np
import copy
import time


# In[98]:


# Split the data into two parts
with open("cryo_em/target_CTF.csv") as csvfile:
    csvreader = csv.reader(csvfile)
    YilaiData = [row for row in csvreader if row[0].startswith('21mar02c')]
    for i, row in enumerate(YilaiData):
        if row[0].startswith('21mar02c_grid1c_00024gr_00018sq'):
            print(i)
            break
# print(YilaiData[1863])
# 0~1863: dataA
# 1864~end: dataB

filenameA = "cryo_em/target_CTF_A.csv"
filenameB = "cryo_em/target_CTF_B.csv"
mode = 'a' if os.path.exists(filenameA) else 'w'
with open(filenameA, mode) as csvfile:
    csvwriter = csv.writer(csvfile) 
    for i in range(1864):
        csvwriter.writerow(YilaiData[i])
mode = 'a' if os.path.exists(filenameB) else 'w'
with open(filenameB, mode) as csvfile:
    csvwriter = csv.writer(csvfile) 
    for i in range(1864,len(YilaiData)):
        csvwriter.writerow(YilaiData[i])       


# In[99]:


def CTF_loader(TimeStamp_FILE, CTF_FILE):
    with open(TimeStamp_FILE, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        timestamps = [(row[0],row[1]) for row in csvreader] 
    timeStamps = {}
    names = []
    for i in timestamps:
        timeStamps[i[0]] = i[1]
        names.append(i[0]) # name list 

    # Read CTF values of env
    with open(CTF_FILE, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        ctfs = [(row[0],row[3]) for row in csvreader if row[0].startswith('21mar02c')]   # find all 21mar02c data
    CTFs = {}
    for i in ctfs:
        CTFs[i[0]] = float(i[1])
    return names, CTFs
names, CTFs = CTF_loader("cryo_em/timestamps.csv","cryo_em/target_CTF_A.csv") # load dataA


# In[13]:


'''We have pre-split the data in the second block above'''
def splitDict(d):  # when splitting, we  same square
    d1 = dict(list(d.items())[len(d)//2:])
    d2 = dict(list(d.items())[:len(d)//2])
    return d1,d2
trainCTFs, testCTFs = splitDict(CTFs)   # split the data outside the program; into 2 csvs  
# split the data outside this script.


# In[134]:


# Data structure for CryoEMHole
class CryoEMHole:
    def __init__(self, idx=-1, parent_idx=-1, ctf=-100):
        self._idx = idx
        self._parent_idx = parent_idx
        self._visited = False
        self._ctf = ctf
        self._ctf_prediction = ctf
    @property
    def idx(self):
        return self._idx
    @property
    def parent_idx(self):
        return self._parent_idx
    @property
    def is_visited(self):
        return self._visited
    @property
    def ctf(self):
        return self._ctf
    @property
    def ctf_prediction(self):
        return self._ctf_prediction      
    def set_status(self, status=False):
        self._visited = status

class CryoEMPatch:
    def __init__(self, idx=-1, parent_idx=-1, holeList=[]):
        self._idx = idx
        self._parent_idx = parent_idx
        self._holeList = holeList
        self._visited_hole_count = 0
    @property
    def idx(self):
        return self._idx
    @property
    def parent_idx(self):
        return self._parent_idx
    
    def hole_count(self):
        self._hole_count = len(self._holeList) 
    
    def get_hole(self, idx):
        return self._holeList[idx]
    def CTF_histogram(self): # computing features 
        patch_CTF_histogram = np.zeros(5)
        for hole in self._holeList:
            if hole.ctf<3:
                patch_CTF_histogram[0] += 1
            elif hole.ctf<5:
                patch_CTF_histogram[1] += 1
            elif hole.ctf<7:
                patch_CTF_histogram[2] += 1
            elif hole.ctf<9:
                patch_CTF_histogram[3] += 1
            else:
                patch_CTF_histogram[4] += 1
        self.patch_CTF_histogram = patch_CTF_histogram
        
    def low_CTF_count(self):
        count = 0
        for hole in self._holeList:
            if hole.ctf<6:
                count += 1
        self._low_CTF_count = count
    def low_CTF_ratio(self):
        self._low_CTF_ratio = self._low_CTF_count/self._hole_count


    def visited_hole_ratio(self):
        self._visited_hole_ratio = self._visited_hole_count/self._hole_count
        
    def compute_features(self):
        self.hole_count()
        self.CTF_histogram()       
        self.low_CTF_count()
        self.low_CTF_ratio()
    @property    
    def patch_ctf_histogram(self):
        return self.patch_CTF_histogram
    @property
    def low_ctf_count(self):
        return self._low_CTF_count
    @property
    def low_ctf_ratio(self):
        return self._low_CTF_ratio
    @property
    def patch_hole_count(self):
        return self._hole_count
    @property
    def visited_hole_count(self):
        return self._visited_hole_count
    
    
    def set_status(self, status = False):
        for hole in self._holeList:
            hole.set_status(status)
    
class CryoEMSquare:
    def __init__(self, idx=-1, patchList=[]):
        self._idx = idx
        self._patchList = patchList
        self._visited_hole_count = 0
    @property
    def idx(self):
        return self._idx
    
    def hole_count(self):
        count = 0
        for patch in self._patchList:
            count += patch.patch_hole_count
        self._hole_count = count
    
    def get_patch(self, idx):
        return self._patchList[idx]
    
    def get_hole(self, idx):
        patch_idx, hole_idx = idx
        patch = self.get_patch(patch_idx)
        return patch.get_hole(hole_idx)
    
        
    
    def CTF_histogram(self):
        square_CTF_histogram = np.zeros(5)
        for patch in self._patchList:
            square_CTF_histogram += patch.patch_ctf_histogram
        self.square_CTF_histogram = square_CTF_histogram
    
    def low_CTF_count(self):
        count = 0
        for patch in self._patchList:
            count += patch.low_ctf_count
        self._low_CTF_count = count
        
    def low_CTF_ratio(self):
        self._low_CTF_ratio = self._low_CTF_count/self._hole_count
    
    @property
    def visited_hole_count(self):
        return self._visited_hole_count
    
    def visited_hole_ratio(self):
        return self._visited_hole_count/self._hole_count
    
    def compute_features(self):
        for patch in self._patchList:
            patch.compute_features()
        self.hole_count()
        self.CTF_histogram()
        self.low_CTF_count()
        self.low_CTF_ratio()
        
    
    def set_status(self, status = False):
        for patch in self._patchList:
            patch.set_status(status)
            
    @property    
    def square_ctf_histogram(self):
        return self.square_CTF_histogram
    @property
    def low_ctf_count(self):
        return self._low_CTF_count
    @property
    def low_ctf_ratio(self):
        return self._low_CTF_ratio
    @property
    def square_hole_count(self):
        return self._hole_count
    
class CryoEMGrid: # store all the features of each square # to be implemented
    def __init__(self, idx=-1, squareList=[]):
        pass
    
class CryoEMAtlas: # store all CryoEM data
    def __init__(self, squareList=[], indexList = []):
        self._squareList = squareList
        self._indexList = indexList
        
    @property
    def square_list(self):
        return self._squareList
    
    @property
    def index_list(self):
        return self._indexList
    
    def hole_count(self):
        count = 0
        for square in self._squareList:
            count += square.square_hole_count
        self._hole_count = count
    
    @property
    def atlas_hole_count(self):
        return self._hole_count
    
    def compute_features(self):
        for square in self._squareList:
            square.compute_features()
        self.hole_count()
    
    def set_status(self, status = False):
        for square in self._squareList:
            square.set_status(status)
            
    def get_square(self, idx):
        return self._squareList[idx]
        


# In[135]:


def CryoEM_loader(CTFs):   # CryoEM_data from the csv 
    squares = set()
    patches = set()
    for j in CTFs.keys(): 
        for i in names:
            if i.endswith('sq') and j.startswith(i):
                squares.add(i)
            if i.endswith('hl') and j.startswith(i):
                patches.add(i)   
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
    
    cryoEM_data = list()  #List of CryoEMSquares
    index_list = list()
    for i_th, square in enumerate(sq_hl_en.values()):     
        patchList = list()
        for j_th, patch in enumerate(square.values()):    
            holeList = list()
            for k_th, hole_CTF in enumerate(patch.values()):
                if hole_CTF > 20:
                    hole_CTF = 20
                hole_k = CryoEMHole(k_th, j_th, hole_CTF)
                holeList.append(hole_k)
                index_list.append((i_th,j_th,k_th))
            patch_j = CryoEMPatch(j_th, i_th, holeList)
            patchList.append(patch_j)
        square_i = CryoEMSquare(i_th, patchList)
        cryoEM_data.append(square_i)
    return cryoEM_data,index_list

cryoEM_data, index_list = CryoEM_loader(CTFs)
cryoEM_atlas = CryoEMAtlas(cryoEM_data, index_list)
cryoEM_atlas.compute_features()


# In[124]:


import gym
from gym import spaces

class CryoEMEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, cryoEM_atlas):
        super(CryoEMEnv,self).__init__()
        self.cryoEM_atlas = cryoEM_atlas
        self.visited_holes = list() # List of Tuples, which can save the sequence order of the visited holes.
        self.length = self.cryoEM_atalas.atlas_hole_count
        self.action_space = spaces.Discrete(self.length)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.length,19), dtype=np.float16)
        self.index_list = self.cryoEM_atlas.index_list
        
    def step(self, action):
        self._take_action(action)
        
    
    def _take_action(self, action):
        
    def reset(self):
        self.cryoEM_atlas.set_status(False)
        self.visited_holes = list()
        self.time = 0
        self.random_start = random.randint(0,self.length-1)
        self.square_index ,self.patch_index, self.hole_index = self.index_list[self.random_start]
        self.visited_holes.append(self.index_list[self.random_start])
        self._initial_state()
        return self._next_observation
    
    def _initial_state(self):
        self.changeSquare = np.array([0,0,0,1])
        self.changePatch = np.array([0,0,0,1])
        self.ctf = np.array([1,1,1,self.cryoEM_atalas.get_square[self.square_index].get_hole((self.patch_index,self.hole_index)).ctf/20])
        self.ctf_pred = np.array([1,1,1,self.cryoEM_atalas.get_square[self.square_index].get_hole((self.patch_index,self.hole_index)).ctf_prediction/20])
        self.visited = np.array([1,1,1,0])
        self.patch_ctf_hist = 
        self.cryoEM_atalas.get_square[self.square_index].get_hole((self.patch_index,self.hole_index)).is_visited = True
        self.cryoEM_atalas.get_square[self.square_index].get_patch(self.patch_index).visited_hole_count += 1
        self.cryoEM_atalas.get_square[self.square_index].visited_hole_count += 1
        
        
    def _next_observation(self):
        
    def _get_obs(self):
        
    def _norm_hist(self, a):
        
    def _history_state(self):
        
        
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        return None

    def close(self):
        return None
train_env = CryoEMEnv(cryoEM_atlas)


# In[ ]:


# change reward to 1-exp(-1/t)

