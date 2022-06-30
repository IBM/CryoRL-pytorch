import argparse
import os
import numpy as np
import json

def load_results(filename):
    with open(filename, 'r') as fp:
        results = json.load(fp)

    num = len(results)
#    if num == 50 or num == 100:
    if num == 100:
        return results

#    if num > 50 and num < 100:
 #       return results[:50]

 #   if num > 100:
 #       return results[:100]

    return []


def parse_results(results):
#    if len(results) < 100:
#        return []

    prev_hole = None
    path = []

    for hole in results:
        if prev_hole is None or hole['image_id'][:43] != prev_hole['image_id'][:43]:
            path.append(hole['image_id'])
        prev_hole = hole

    return path

    #return paths

def path_pairs(path, annotations):
    if len(path) <= 1:
        return []

    patches = [(item[0:43], annotations[item]) for item in path]
    counter = {}
    patch_path = []
    for patch, ctf in patches:
        if patch not in counter:
            counter[patch] = 0

        if ctf <= 6.0:
            counter[patch] += 1
        if patch not in patch_path:
            patch_path.append(patch)

    p_pairs = {}
    # skip the first pair
#    for k in range(2, len(patch_path)):
    for k in range(1, len(patch_path)):
        #p_pairs[(patch_path[k-1],patch_path[k])] = counter[patch_path[k-1]]
        p_pairs[(patch_path[k-1],patch_path[k])] = 1

    return p_pairs, patch_path

def compute_accuracy(results, annotations):
    total = 0
    for hole in results:
        if annotations[hole['image_id']] <= 6.0:
            total +=1
    return total

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch evaluation')
  #  parser.add_argument('--filename', type=str, help="where is the data")
    parser.add_argument('--annotation', type=str, help="where is the data")
    return parser

log_files = [
    '2021-11-11T15 58 38.573Z.json',
    '2021-11-11T20 30 30.906Z.json',
    '2021-11-11T20 40 07.930Z.json',
    '2021-11-12T16 43 44.989Z.json',
    '2021-11-12T19 03 49.670Z.json',
    '2021-11-12T21 05 30.494Z.json',
    '2021-11-12T22 11 55.111Z.json',
    '2021-11-12T22 16 43.124Z.json',
    '2021-11-12T22 27 38.219Z.json',
    'E.json',
    'L.json',
    'N.json',
    'S.json',
    'Z.json']

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()

    user_results = [load_results(os.path.join('user_study', item)) for item in log_files]
    user_results = [item for item in user_results if len(item) > 0]
    print([len(item) for item in user_results])
    paths =[parse_results(item) for item in user_results]
    paths =[ item for item in paths if len(item) > 0]
    print (paths)

    annotations = np.loadtxt(args.annotation, delimiter=',', dtype=str)

    ctfs = {}
    for k in range(annotations.shape[0]):
        ctfs[annotations[k][0]] = float(annotations[k][3])

    print ('------ path ------')
    results = {}
    for p in paths:
        r, trajectory = path_pairs(p, ctfs)
        for key, val in r.items():
            if key in results:
                results[key] += val
            else:
                results[key] = val
        print (trajectory)

    weight = 50.0 / len(paths)
    for key, val in results.items():
        results[key] = val * weight

    print ('\n------ path statistics------')
    print (results)

    #print (ctfs)
    cnts = [(compute_accuracy(item, ctfs), len(item)) for item in user_results]

    #print (cnts)
    cnts_50 = np.array([ c for c, total in cnts if total ==50 ])
    print (cnts_50)
    print (np.mean(cnts_50), np.std(cnts_50))
    cnts_100 = np.array([ c for c, total in cnts if total ==100 ])
    print (cnts_100)
    print (np.mean(cnts_100), np.std(cnts_100))

if __name__ == '__main__':
    main()

