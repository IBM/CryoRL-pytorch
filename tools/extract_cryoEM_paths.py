import argparse
import os
import numpy as np

def parse_results(filename):
    inputFile = open(filename, "r")
    lines = inputFile.readlines()

    paths = []
    for line in lines:
        if 'Start of Trajectory Evaluation' in line:
            path = list()

        if 'End of Trajectory' in line:
            if len(path) > 0:
                paths.append(path)

        if line[0:2] == 'id':
            hole_name = line.split(' ')[3]
            path.append(hole_name)

    return paths

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
    #for k in range(2, len(patch_path)):
    for k in range(1, len(patch_path)):
        #p_pairs[(patch_path[k-1],patch_path[k])] = counter[patch_path[k-1]]
        p_pairs[(patch_path[k-1],patch_path[k])] = 1

    return p_pairs, patch_path

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch evaluation')
    parser.add_argument('--filename', type=str, help="where is the data")
    parser.add_argument('--annotation', type=str, help="where is the data")
    return parser

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    paths = parse_results(args.filename)
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

    print ('\n------ path statistics------')
    print (results)

if __name__ == '__main__':
    main()

