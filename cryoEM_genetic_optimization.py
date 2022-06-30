import argparse
import pygad
import numpy as np
from cryoEM_dataset import get_dataset
from cryoEM_config import CryoEMConfig
from cryoEM_object import CryoEMPatch

SWITCH_GRID_COST = 10
SWITCH_SQUARE_COST = 5
SWITCH_PATCH_COST = 3
SWITCH_HOLE_COST = 0
IMAGING_COST = 2

SOLUTION_BY_PATCH=True
CryoEM_DATA=None
PATCH_DATA=[]
RUN_RESULTS=None
PATCH_PER_POPULATION=20
GENE_PRE_POPULATION=8

def fitness_func1(solution, solution_idx):
    global PATCH_DATA

#    print (solution, solution_idx)
    solution = list(set(solution))

#    print (solution)
#    print ('---')
    patch, _, _ = PATCH_DATA[solution[0]]

    prev_patch = patch
    r = 0.0
    t = 0.0
    duration = CryoEMConfig.Searching_Limit
    for k in solution:
        patch, good_holes, _ = PATCH_DATA[k]
        #assert good_holes > 0

        # first hole
        if patch.name[:43] == prev_patch.name[:43]:  # same patch
            t += IMAGING_COST
            r0 = 1.0
        elif patch.name[:31] == prev_patch.name[:31]:  # same square
            t += (SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.57
        elif patch.name[:23] == prev_patch.name[:23]:  # same grid
            t += (SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.23
        else:
            t += (SWITCH_GRID_COST + SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.09

        if t > CryoEMConfig.Searching_Limit:
            return r

        r += r0
        good_holes = max(good_holes - 1, 0)
        # cannot visit all good holes
        if t + good_holes * IMAGING_COST >= duration:
            return (r + np.round((duration - t) / IMAGING_COST))

        t += good_holes * IMAGING_COST
        r += good_holes
        prev_patch = patch

    return r

def sort_solution_by_grid(patch_list):
    patch_counter = {}
    patch_idx = {}
    for k, patch in enumerate(patch_list):
        patch_counter[patch[0].name] = patch[1]
        patch_idx[patch[0].name] = k
    grids = list(set([item[:23] for item in patch_counter]))
    grid_counter = {item:0 for item in grids}
    for key, val in patch_counter.items():
        grid_counter[key[0:23]] += val

    tuple_list = list(grid_counter.items())
    grid_list = sorted(tuple_list,key=lambda x:(-x[1],x[0]))
#    print('\n', grid_list, '\n')

    patch_dict = {grid:[] for grid, _ in grid_list}
    for key, val in patch_counter.items():
        patch_dict[key[0:23]].append((key,val))

    patch_list = []
    for grid, _ in grid_list:
        patch_list += patch_dict[grid]

    #print (patch_list)
    return [patch_idx[item[0]] for item in patch_list]

'''
def sort_solution_by_grid(solution):
    patch_counter = {}
    patch_idx = {}
    for k in solution:
        patch_counter[PATCH_DATA[k][0].name] = PATCH_DATA[k][1]
        patch_idx[PATCH_DATA[k][0].name] = k
    grids = list(set([item[:23] for item in patch_counter]))
    grid_counter = {item:0 for item in grids}
    for key, val in patch_counter.items():
        grid_counter[key[0:23]] += val

    tuple_list = list(grid_counter.items())
    grid_list = sorted(tuple_list,key=lambda x:(-x[1],x[0]))
#    print('\n', grid_list, '\n')

    patch_dict = {grid:[] for grid, _ in grid_list}
    for key, val in patch_counter.items():
        patch_dict[key[0:23]].append((key,val))

    patch_list = []
    for grid, _ in grid_list:
        patch_list += patch_dict[grid]

    #print (patch_list)
    return [patch_idx[item[0]] for item in patch_list]
'''

def fitness_func(solution, solution_idx):
    global PATCH_DATA, SOLUTION_BY_PATCH

#    print (solution, solution_idx)
    solution = list(set(solution))
    #print (solution, SOLUTION_BY_PATCH)
    patch_list = [PATCH_DATA[k] for k in solution] if SOLUTION_BY_PATCH else create_patch_list(solution)

    # sort patch by grid
    sort_idx = sort_solution_by_grid(patch_list)
    patch_list = [patch_list[k] for k in sort_idx]

    return patch_fitness(patch_list)

def create_patch_list(hole_idx_list):
    patch_dict = {}
    for idx in hole_idx_list:
        h = CryoEM_DATA.get_hole(idx)
        patch_name = (h.name)[:43]
        if patch_name not in patch_dict:
            patch_dict[patch_name] = []
        patch_dict[patch_name].append(h.copy())

    patch_list = []
    for key, val in patch_dict.items():
        patch = CryoEMPatch(key, hole_list=val)
        patch_list.append(patch(0, 30, 8))

    return patch_list

def hole_fitness(hole_idx_list):
    patch_list = create_patch_list(hole_idx_list)
    return patch_fitness(patch_list)

def patch_fitness(patch_list):
    prev_patch = None
    r = 0.0
    t = 0.0
    duration = CryoEMConfig.Searching_Limit
    for k, item in enumerate(patch_list):
        patch, good_holes, _ = item

        # first patch
        if prev_patch is None:
            prev_patch = patch

        # first hole
        if patch.name[:43] == prev_patch.name[:43]:  # same patch
            t += IMAGING_COST
            r0 = 1.0
        elif patch.name[:31] == prev_patch.name[:31]:  # same square
            t += (SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.57
        elif patch.name[:23] == prev_patch.name[:23]:  # same grid
            t += (SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.23
        else:
            t += (SWITCH_GRID_COST + SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.09

        if t > CryoEMConfig.Searching_Limit:
            return r

        r += r0
        good_holes = max(good_holes - 1, 0)
        # cannot visit all good holes
        if t + good_holes * IMAGING_COST >= duration:
            return (r + np.round((duration - t) / IMAGING_COST))

        t += good_holes * IMAGING_COST
        r += good_holes
        prev_patch = patch

    return r

def true_fitness(solution):
    global SOLUTION_BY_PATCH, PATCH_DATA

    solution = list(set(solution))
#    print (solution)

    # sort patch by the number of low ctfs predicted
#    predictions = np.array([ PATCH_DATA[k][1] for k in solution])
#    idx = np.argsort(predictions)
#    solution = [ solution[k] for k in idx]
#    print (solution)

    patch_list = [PATCH_DATA[k] for k in solution] if SOLUTION_BY_PATCH else create_patch_list(solution)

    solution = sort_solution_by_grid(patch_list)

    total_rewards = 0.0
    total_lctf = 0.0
    total_visit = 0.0
    t = 0.0
    prev_h_name = None
    is_done = False
    duration = CryoEMConfig.Searching_Limit
    ctf_thresh = CryoEMConfig.LOW_CTF_THRESH
    for k, (patch,_,_) in enumerate(patch_list):
#        patch, _, _ = PATCH_DATA[k]
        for h in patch:
            if h.category.value[0] < h.category.value[1]: # skip, negative prediction
                continue

            h_name = h.name
            # first hole
            if prev_h_name is None:
                prev_h_name = h_name

            r0 = 0.0
            if prev_h_name[:43] == h_name[:43]:  # same patch
                t += IMAGING_COST
                r0 = 1.0
            elif prev_h_name[:31] == h_name[:31]:  # same square
                t += (SWITCH_PATCH_COST + IMAGING_COST)
                r0 = 0.57
            elif prev_h_name[:23] == h_name[:23]:  # same grid
                t += (SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
                #r0 = 0.23
                r0 = 0.46
            else:
                t += (SWITCH_GRID_COST + SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
                r0 = 0.09
                # print ('grid', r)
                # print  (prev_h, '--->',  h)
            if t >= duration:
                print ('early exit')
                is_done = True
                break

            if h.gt_ctf.value <= ctf_thresh:
                total_rewards += r0
                total_lctf += 1
                # print(r, total_rewards, total_lctf)
            #   cnt += 1

            total_visit += 1
            prev_h_name = h_name

        if is_done:
            break

    return total_rewards, total_lctf, total_visit

def true_fitness1(solution):
    solution = list(set(solution))

    patch, _, _ = PATCH_DATA[solution[0]]

    prev_patch = patch
    r=0.0
    c = 0.0
    t = 0.0
    duration = CryoEMConfig.Searching_Limit
    for k in solution:
        patch, good_holes, true_holes = PATCH_DATA[k]

        # first hole
        if patch.name[:43] == prev_patch.name[:43]:  # same patch
            t += IMAGING_COST
            r0 = 1.0
        elif patch.name[:31] == prev_patch.name[:31]:  # same square
            t += (SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.57
        elif patch.name[:23] == prev_patch.name[:23]:  # same grid
            t += (SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.23
        else:
            t += (SWITCH_GRID_COST + SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.09

        if t > CryoEMConfig.Searching_Limit:
            print(r)
            return r, c

        r += r0
        c += 1
        good_holes = max(good_holes - 1, 0)
        # cannot visit all good holes
        if t + good_holes * IMAGING_COST >= CryoEMConfig.Searching_Limit:
            print(r)
            return (r + np.round((duration - t) / IMAGING_COST)), c+np.round((duration - t) / IMAGING_COST)

        t += good_holes * IMAGING_COST
        r += good_holes
        c += true_holes
        prev_patch = patch

    print (r)
    return r, c

def initial_population():
    good_hole_list = [item[2] for item in PATCH_DATA]
    I = np.argsort(good_hole_list)
    return [I[-PATCH_PER_POPULATION:] for i in range(GENE_PER_POPULATION)]


def get_patch_list(data):
    grid_list = data.grids
    patches = []
    for grid in grid_list:
        for square in grid:
            patches += [item for item in square]

    results = []
    for patch in patches:
        categories = patch.get_categories(prediction=True, visited=False)
        predictions = [item.value[0] >= item.value[1] for item in categories]
        true_categories = patch.get_categories(prediction=False, visited=False)
        ground_truth = [ item.value[0] >= item.value[1] for item in true_categories]
        true_positives = [True for p, t in zip(predictions, ground_truth) if t and p == t]
        low_predictions = [item for item in predictions if item is True]
#        print  (predictions)
#        print(ground_truth)
#        print (true_positives)
#        print ('-----\n')
        if len(low_predictions) > 0:
            results.append((patch, len(low_predictions), len(true_positives)))

    return results

def on_start(ga_instance):
    global RUN_RESULTS
    print("on_start()")
    RUN_RESULTS = None

def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_mutation(ga_instance, offspring_mutation):
    print("on_mutation()")

def on_generation(ga_instance):
    print("on_generation()")

def on_stop(ga_instance, last_population_fitness):
    global RUN_RESULTS

    print("on_stop()")
    best_solution = ga_instance.best_solutions[-1]
    #print (ga_instance.best_solutions_fitness)
    #print (fitness_func(best_solution, -1))

    for k in best_solution:
        print (PATCH_DATA[k][0].name, PATCH_DATA[k][1], PATCH_DATA[k][2])
    
    RUN_RESULTS = true_fitness(best_solution)

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch evaluation')
    #  parser.add_argument('--filename', type=str, help="where is the data")
    parser.add_argument('--dataset', type=str, help="where is the data")
    parser.add_argument('--duration', type=float, default=240.0, help="where is the data")
    parser.add_argument('--ctf-thresh', type=float, default=6.0, help="ctf threshold")
    parser.add_argument('--num-genes', type=int, default=8, help="number of genes")
    parser.add_argument('--num-patches', type=int, default=20, help="number of patches in search")
    parser.add_argument('--num-tries', type=int, default=10, help="number of tries")
    parser.add_argument('--solution-by-hole', action="store_true", default=False)
    return parser

def main():
    global args, PATCH_DATA, RUN_RESULTS, SOLUTION_BY_PATCH, CryoEM_DATA, GENE_PER_POPULATION, PATCH_PER_POPULATION
    parser = arg_parser()
    args = parser.parse_args()

    # update global variable
    CryoEMConfig.Searching_Limit = args.duration
    CryoEMConfig.LOW_CTF_THRESH = args.ctf_thresh

    _, val_dataset, _, _, _, _ = get_dataset(args.dataset, CryoEMConfig.CLASSIFICATION, use_one_hot=True)
    CryoEM_DATA = val_dataset
    PATCH_DATA = get_patch_list(val_dataset)
#    print (PATCH_DATA)
    SOLUTION_BY_PATCH = not args.solution_by_hole
    PATCH_PER_POPULATION = args.num_patches
    GENE_PER_POPULATION = args.num_genes
    print (SOLUTION_BY_PATCH)

    patch_len = len(PATCH_DATA) if SOLUTION_BY_PATCH else val_dataset.num_holes()
    fitness_function = fitness_func

    results = []
    for k in range(args.num_tries):
        print ('----- {} ------'.format(k))
        ga_instance = pygad.GA(num_generations=40,
                       num_parents_mating=3,
                       fitness_func=fitness_function,
                       #initial_population=initial_population(),
                       sol_per_pop=10,
                       num_genes=GENE_PER_POPULATION,
                       init_range_low = 0,
                       init_range_high = patch_len,
                       gene_type=int,
                       crossover_type='single_point',
                       on_start=on_start,
                       #on_fitness=on_fitness,
                       #on_parents=on_parents,
                       #on_crossover=on_crossover,
                       #on_mutation=on_mutation,
                       #on_generation=on_generation,
                        save_best_solutions=True,
                       on_stop=on_stop)

        ga_instance.run()
        print (RUN_RESULTS)
        results.append(list(RUN_RESULTS))

    results = np.array(results)
    print (np.mean(results, axis=0), np.std(results, axis=0))
if __name__ == '__main__':
    main()
