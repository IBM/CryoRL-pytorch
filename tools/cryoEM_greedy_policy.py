import argparse
import os
import numpy as np
from collections import OrderedDict

SWITCH_GRID_COST = 10
SWITCH_SQUARE_COST = 5
SWITCH_PATCH_COST = 3
SWITCH_HOLE_COST = 0
IMAGING_COST = 2


# counter the total number of low ctfs
def get_patch_value(classification):
    patches = list(set([item[0:43] for item in classification.keys()]))

    patch_counter = {item:0 for item in patches}
    for key, val in classification.items():
        patch_id = key[:43]
        if val >= 0.5:
            patch_counter[patch_id] += 1

    tuple_list = list(patch_counter.items())
    return sorted(tuple_list,key=lambda x:(-x[1],x[0]))

def sort_by_grid(patch_counter):
    grids = list(set([item[0][:23] for item in patch_counter]))
    grid_counter = {item:0 for item in grids}
    for key, val in patch_counter:
        grid_counter[key[0:23]] += val

    tuple_list = list(grid_counter.items())
    grid_list = sorted(tuple_list,key=lambda x:(-x[1],x[0]))
#    print('\n', grid_list, '\n')

    patch_dict = {grid:[] for grid, _ in grid_list}
    for key, val in patch_counter:
        patch_dict[key[0:23]].append((key,val))

    patch_list = []
    for grid, _ in grid_list:
        patch_list += patch_dict[grid]

    return patch_list

#grid_id = patch_id[:23]


def greedy_search(counter, classification, annotations, ctf=6.0, duration=120.0, start_id = 0):
    total_lctf = 0
    total_visit = 0
    total_rewards = 0
    t = 0

    holes = classification.keys()
    prev_h = None
    is_done = False

    # switch the first with the randomly picked patches
    counter_idx = [k for k in range(len(counter))]
    counter_idx[0], counter_idx[start_id] = counter_idx[start_id], counter_idx[0] # swap the first one out
    for idx in counter_idx:
        patch, val = counter[idx]
        patch_holes = [item for item in holes if item[0:43] == patch]
#        cnt = 0
        for h in patch_holes:
            if classification[h] == 0: # skip it
                continue

            # first hole
            if prev_h is None:
                prev_h = h

            r = 0.0
            if prev_h[:43] == h[:43]: # same patch
                t += IMAGING_COST
                r = 1.0
            elif prev_h[:31] == h[:31]: # same square
                t += (SWITCH_PATCH_COST + IMAGING_COST)
                r=0.57
            elif prev_h[:23] == h[:23]: # same grid
                t += (SWITCH_SQUARE_COST+SWITCH_PATCH_COST+IMAGING_COST)
                r = 0.23
            else:
                t += (SWITCH_GRID_COST+SWITCH_SQUARE_COST+SWITCH_PATCH_COST+IMAGING_COST)
                r = 0.09
                #print ('grid', r)
                #print  (prev_h, '--->',  h)

            if t >= duration:
                is_done = True
                break

#            print (classification[h], annotations[h], r)
            if annotations[h] <= ctf:
#                print ('---', r)
                total_rewards += r
                total_lctf += 1
                #print(r, total_rewards, total_lctf)
            #   cnt += 1

            total_visit += 1
            prev_h = h

        #print (patch, val, cnt, visit)


        if is_done:
            break

    return total_rewards, total_lctf, total_visit

def get_accuracy_by_patch(annotation, classification, patches, ctf_thresh):
    correct_counter = {patch:0 for patch in patches}
    total_lctf = {patch:0 for patch in patches}
    gt_lctf = {patch:0 for patch in patches}
    total_counter = {patch:0 for patch in patches}

    lctf_correct = 0
    lctf = 0
    hctf_correct = 0
    hctf = 0
    for hole, val in classification.items():
        patch = hole[:43]
        if val == 1.0 and annotation[hole] <= ctf_thresh:
            correct_counter[patch] +=1
            lctf_correct += 1
            #print (annotation[hole], ctf_thresh)

        if val == 0.0 and annotation[hole] > ctf_thresh:
            hctf_correct += 1

        if annotation[hole] <= ctf_thresh:
            gt_lctf[patch] += 1
            lctf += 1
        else:
            hctf += 1

        if val ==1.0:
            total_lctf[patch] += 1

        total_counter[patch] += 1

    print ('acc: lctf {:.2f} hctf {:.2f}'.format(lctf_correct/(lctf+1e-4), hctf_correct/(hctf+1e-4) ))
    return {patch: (item, total_lctf[patch], gt_lctf[patch], total_counter[patch]) for patch, item in correct_counter.items()}


def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch evaluation')
    parser.add_argument('--prediction', type=str, help="prediction results")
    parser.add_argument('--annotation', type=str, help="where is the data")
    parser.add_argument('--duration', type=float, default=120, help="time")
    parser.add_argument('--sort', type=str, default='patch')
    parser.add_argument('--ctf', type=float, default='6.0')
    parser.add_argument('--use-gt', action="store_true", default=False)
    parser.add_argument('--regress', action="store_true", default=False)
    parser.add_argument('--verbose', action="store_true", default=False)
    return parser

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    prediction = np.loadtxt(args.prediction, delimiter=' ', dtype=object)
    annotation = np.loadtxt(args.annotation, delimiter=',', dtype=object)

    np.random.seed(1)

    if args.use_gt:
        classification = {annotation[k][0]:1.0 if float(annotation[k][3])<=args.ctf else 0.0 for k in range(annotation.shape[0])}
    elif args.regress:
        classification = {prediction[k][0]:1.0 if float(prediction[k][1])<=args.ctf else 0.0 for k in range(prediction.shape[0])}
    else:
        max_id = np.argmax(prediction[:,1:].astype(float), axis=1)
        #print (max_id)
        classification = {prediction[k][0]: (1 if max_id[k]==0 else 0) for k in range(prediction.shape[0])}

    #print (classification)
    annotation = {annotation[k][0]:float(annotation[k][3]) for k in range(annotation.shape[0])}

    patch_counter = get_patch_value(classification)

    if args.sort == 'grid':
        patch_counter = sort_by_grid(patch_counter)

    #for item in patch_counter:
    #    print (item[0], item[1])

    if args.verbose:
        acc = get_accuracy_by_patch(annotation, classification, [patch for patch, _ in patch_counter], args.ctf)
        print ('True positives, Positives, Ground Truth, Total')
        for item in patch_counter:
            patch = item[0]
            cnts = acc[patch]
            assert cnts[1] == item[1]
            print ('{},{},{},{}, {}'.format(patch, cnts[0], cnts[1], cnts[2], cnts[3]))

    all_trials = []
    for _ in range(50):
        random_start = np.random.randint(0, len(patch_counter) - 1)
        all_trials.append(greedy_search(patch_counter, classification, annotation, args.ctf, args.duration, random_start))

    all_trials = np.array(all_trials)
    hole_means = all_trials.mean(axis=0)
    hole_std = all_trials.std(axis=0)
    print('{:3d} &{:.1f} $\pm$ {:.1f}&{:.1f} $\pm$ {:.1f}&{:.1f} $\pm$ {:.1f}'.format(int(args.duration),
    hole_means[0], hole_std[0], hole_means[1], hole_std[1], hole_means[2], hole_std[2]), flush=True)

if __name__ == '__main__':
    main()

