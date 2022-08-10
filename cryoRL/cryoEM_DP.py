import os
import argparse
import numpy as np
import math
import copy
import random

from cryoEM_data import get_dataset
from cryoEM_config import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CryoEM-5-5')
    parser.add_argument('--num-trials', type=int, default=50)
    parser.add_argument('--use-one-hot', action="store_true", default=False)
    parser.add_argument('--duration', type=float, default=120)
    parser.add_argument('--ctf-thresh', type=float, default=6.0)
    args = parser.parse_args()
    return args

def get_reward(data, from_idx, to_idx):
    '''
    switch_time = 0
    r = 0
    if data.is_patch_same(from_idx, to_idx):  # same patch different holes
        switch_time = CryoEMConfig.SWITCH_HOLE_LOSS
    elif data.is_square_same(from_idx, to_idx):  # same square different patches
        switch_time = CryoEMConfig.SWITCH_PATCH_LOSS
    elif data.is_grid_same(from_idx, to_idx):  # same grid different squares
        switch_time = CryoEMConfig.SWITCH_SQUARE_LOSS
    else:  # different squares
        switch_time = CryoEMConfig.SWITCH_GRID_LOSS
    '''
    switch_time = 0
    r = 0.0
    if data.is_patch_same(from_idx, to_idx): # same patch different holes
        switch_time = CryoEMConfig.SWITCH_HOLE_LOSS
        r = 1.0
    elif data.is_square_same(from_idx, to_idx): # same square different patches
        #switch_time = CryoEMConfig.SWITCH_PATCH_LOSS
        switch_time = CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
        r = 0.57
    elif data.is_grid_same(from_idx, to_idx): # same grid different squares
            #switch_time = CryoEMConfig.SWITCH_SQUARE_LOSS
        switch_time = CryoEMConfig.SWITCH_SQUARE_LOSS + CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
        r = 0.23
    else:
            #switch_time = CryoEMConfig.SWITCH_GRID_LOSS
        switch_time = CryoEMConfig.SWITCH_GRID_LOSS + CryoEMConfig.SWITCH_SQUARE_LOSS + \
                      CryoEMConfig.SWITCH_PATCH_LOSS + CryoEMConfig.SWITCH_HOLE_LOSS
        r = 0.09

    hole = data.get_hole(to_idx)
    gt_low_ctf = np.argmax(hole.gt_category.value) == 0
    predicted_low_ctf = np.argmax(hole.category.value) == 0

    '''
    reward = math.exp(-0.185*(switch_time - CryoEMConfig.SWITCH_HOLE_LOSS)) if predicted_low_ctf \
        else CryoEMConfig.HIGH_CTF_REWARD
    true_reward = math.exp(-0.185*(switch_time - CryoEMConfig.SWITCH_HOLE_LOSS)) if gt_low_ctf \
        else CryoEMConfig.HIGH_CTF_REWARD
    '''

    reward = r if predicted_low_ctf else CryoEMConfig.HIGH_CTF_REWARD
    true_reward = r if gt_low_ctf  else CryoEMConfig.HIGH_CTF_REWARD
    true_cnt = 1 if gt_low_ctf else 0

    return switch_time, reward, true_reward, true_cnt


def compute_all_rewards(data):
    num_holes = data.num_holes()
 #   num_holes = 10
    r = np.zeros((num_holes, num_holes, 4))
    for i in range(num_holes):
        for j in range(num_holes):
            if i != j:
                r[i,j,:] = get_reward(data, from_idx=i, to_idx=j)
    return r

def find_optimal_path(dataset, all_reward_info, start_idx=0, max_duration=100):
    num_holes = dataset.num_holes()
 #   num_holes = 10

    # initialize
    hole = dataset.get_hole(start_idx)
    total_rewards = [np.NINF] * num_holes
    total_duration =[0] * num_holes
    true_rewards = [0.0] * num_holes
    true_cnts = [0.0] * num_holes
    total_rewards[start_idx] = 1.0 if np.argmax(hole.category.value) == 0 else 0.0
    true_rewards[start_idx] = 1.0 if hole.gt_category.value[0] == 1 else 0.0
    true_cnts[start_idx] = 1.0 if hole.gt_category.value[0] == 1 else 0.0
    paths = [[] for _ in range(num_holes)]
    paths[start_idx] = [start_idx]
    #
    cnt = 0
    while True:
        tmp_total_rewards = total_rewards.copy()
        tmp_true_rewards = true_rewards.copy()
        tmp_true_cnts = true_cnts.copy()
        tmp_total_duration = total_duration.copy()
        tmp_paths = copy.deepcopy(paths)

        for current_idx in range(num_holes):
#            reward_info = [ get_reward(dataset, j, current_idx) for j in range(num_holes)]
            #print (reward_info)
#            next_rewards = [np.NINF] * num_holes
#            for idx in range(num_holes):
#                switch_time, reward, _ = reward_info[idx]
#                duration = tmp_total_duration[idx] + switch_time
                # filter out invalid candidates
               # print (idx, current_idx, tmp_paths[idx], duration)
#                if idx != current_idx and current_idx not in tmp_paths[idx] and duration <= max_duration:
#                    next_rewards[idx] = tmp_total_rewards[idx] + reward
                #    print (idx, tmp_total_rewards[idx], reward)
            t = all_reward_info[:,current_idx,0]
            r = all_reward_info[:,current_idx,1]
            t_reward = all_reward_info[:,current_idx,2]
            t_cnts = all_reward_info[:,current_idx,3]

            next_rewards = tmp_total_rewards + r
            next_duration = tmp_total_duration + t
            next_t_reward = tmp_true_rewards + t_reward
            next_t_cnts = tmp_true_cnts + t_cnts
            for idx in range(num_holes):
#                print (idx, len(tmp_paths[idx])>0 and tmp_paths[idx][-1]< 0)
                skip = 0
                if idx == current_idx or current_idx in tmp_paths[idx]:
                    next_rewards[idx] = np.NINF
                    skip = 1

                if tmp_paths[idx] and tmp_paths[idx][-1]< 0:
                    next_rewards[idx] = np.NINF
                    skip = 2

                if  next_duration[idx] > max_duration:
                    next_rewards[idx] = np.NINF
                    skip = 3
                #if skip:
                #    print (idx, tmp_paths[idx], current_idx, 'skip: ', skip)

            #print (current_idx, next_rewards)
            '''
            max_reward = max(next_rewards)
            #print (current_idx, next_rewards)
#            print (current_idx, max_reward)
            indices = [index for index, value in enumerate(next_rewards) if value == max_reward]
            if indices and max_reward != np.NINF:
            '''
            max_idx = np.argmax(next_rewards)
            max_reward = next_rewards[max_idx]
            if max_reward != np.NINF:
                total_rewards[current_idx] = max_reward
                true_rewards[current_idx] = next_t_reward[max_idx] # true reward
                true_cnts[current_idx] = next_t_cnts[max_idx] # true reward
                total_duration[current_idx] = next_duration[max_idx] # switch cost
                paths[current_idx] = tmp_paths[max_idx].copy()
                paths[current_idx].append(current_idx)
             #   print ('---', paths[current_idx])
            else:
#                total_rewards[current_idx] = tmp_total_rewards[current_idx]
##                true_rewards[current_idx] = tmp_true_rewards[current_idx]
#                total_duration[current_idx] = tmp_total_duration[current_idx] + t[current_idx]
 #               print (current_idx)
                if paths[current_idx] and paths[current_idx][-1] >= 0:
                    paths[current_idx].append(-1)


#        print (paths)
    #    print (total_duration)
        #print ('------ {} ------'.format(cnt))
    #    print (total_rewards)
        last_ids =[item[-1]for item in paths if len(item)>0]
#        print (last_ids)
#        if np.all(np.array(total_duration) >= max_duration) or cnt >= num_holes-1:
        if np.all(np.array(last_ids) < 0) or cnt >= num_holes-1:
            break

        cnt += 1

        # terminate?
    return total_rewards, true_rewards, true_cnts, total_duration, paths

if __name__ == '__main__':
    args = get_args()
    print(args)
    
    if 'ctf_thresh' in args:
        CryoEMConfig.LOW_CTF_THRESH = args.ctf_thresh
        print ('low CTF threshold', CryoEMConfig.LOW_CTF_THRESH)

    _, val_dataset, _, _,_,_ = get_dataset(args.dataset, prediction_type=CryoEMConfig.CLASSIFICATION, use_one_hot=args.use_one_hot)

 #   for i in range(10):
 #       print (val_dataset.get_hole(i))

    all_reward_info = compute_all_rewards(val_dataset)
    num_holes = val_dataset.num_holes()
    start_ids = random.sample(range(0, num_holes-1), args.num_trials)

    max_rewards=[]
    max_true_rewards = []
    max_true_cnts = []
    max_visits = []
    for k, sid in enumerate(start_ids):
        print ('xxxx', k, sid)
        total_rewards, true_rewards, true_cnts, total_duration, paths = find_optimal_path(val_dataset, all_reward_info, sid, max_duration=args.duration)
        max_id = np.argmax(np.array(total_rewards))
        visits = max([len(item) for item in paths])
        max_rewards.append(total_rewards[max_id])
        max_true_rewards.append(true_rewards[max_id])
        max_true_cnts.append(true_cnts[max_id])
        max_visits.append(len(paths[max_id]))

        print (k, sid, total_rewards[max_id], true_rewards[max_id], true_cnts[max_id], len(paths[max_id]))

    #print (max_rewards)
    #print (max_true_rewards)
    #print (max_visits)

    max_rewards = np.array(max_rewards)
    max_true_rewards = np.array(max_true_rewards)
    max_true_cnts = np.array(max_true_cnts)
    max_visits = np.array(max_visits)


    print (max_rewards.mean(), max_rewards.std(), max_true_rewards.mean(), max_true_rewards.std(), \
           max_true_cnts.mean(), max_true_cnts.std(), max_visits.mean(), max_visits.std())
