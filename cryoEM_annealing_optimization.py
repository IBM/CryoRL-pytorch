import math
import random
import numpy as np
import argparse

from cryoEM_dataset import get_dataset
from cryoEM_config import CryoEMConfig

SWITCH_GRID_COST = 10
SWITCH_SQUARE_COST = 5
SWITCH_PATCH_COST = 3
SWITCH_HOLE_COST = 0
IMAGING_COST = 2

PATCH_DATA=[]

class SimAnneal(object):
    def __init__(self, N, m, fitness_function, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.N = N
        self.T = math.sqrt(self.N) if T == -1 else T
        self.m = min(N, m)
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.fitness_function = fitness_func

        #self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

    '''
    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour).
        """
        cur_node = random.choice(self.nodes)  # start from a random node
        solution = [cur_node]

        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit
    '''

    def reset(self):
        self.iteration = 1
        self.T = self.T_save

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

    def initial_solution(self):
        solution = random.sample(range(0, self.N), self.m)
        fitness = self.fitness_function(solution)
        return solution, fitness

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        candidate_fitness = self.fitness_function(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.initial_solution()

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.m - 1)
            i = random.randint(0, self.m - l)
            candidate[i : (i + l)] = random.sample(range(0, self.N), l)
            self.accept(candidate)
            self.T *= self.alpha
            self.iteration += 1

            self.fitness_list.append(self.cur_fitness)

        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print(f"Iteration {i}/{times} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

    '''
    def visualize_routes(self):
        """
        Visualize the TSP route with matplotlib.
        """
        visualize_tsp.plotTSP([self.best_solution], self.coords)

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()
    '''

def fitness_func(solution):
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
        # assert good_holes > 0

        # first hole
        if patch.name[:43] == prev_patch.name[:43]:  # same patch
            t += IMAGING_COST
            r0 = 1.0
        elif patch.name[:31] == prev_patch.name[:31]:  # same square
            t += (SWITCH_PATCH_COST + IMAGING_COST)
            r0 = 0.57
        elif patch.name[:23] == prev_patch.name[:23]:  # same grid
            t += (SWITCH_SQUARE_COST + SWITCH_PATCH_COST + IMAGING_COST)
            #r0 = 0.23
            r0 = 0.46
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

    return -r

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

def true_fitness(solution):
    solution = list(set(solution))
#    print (solution)

    # sort patch by the number of low ctfs predicted
#    predictions = np.array([ PATCH_DATA[k][1] for k in solution])
#    idx = np.argsort(predictions)
#    solution = [ solution[k] for k in idx]
#    print (solution)
    solution = sort_solution_by_grid(solution)

    total_rewards = 0.0
    total_lctf = 0.0
    total_visit = 0.0
    t = 0.0
    prev_h_name = None
    is_done = False
    duration = CryoEMConfig.Searching_Limit
    ctf_thresh = CryoEMConfig.LOW_CTF_THRESH
    for k in solution:
        patch, _, _ = PATCH_DATA[k]
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
                r0 = 0.23
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
        ground_truth = [item.value[0] >= item.value[1] for item in true_categories]
        true_positives = [True for p, t in zip(predictions, ground_truth) if t and p == t]
        low_predictions = [item for item in predictions if item is True]
        #        print  (predictions)
        #        print(ground_truth)
        #        print (true_positives)
        #        print ('-----\n')
        if len(low_predictions) > 0:
            results.append((patch, len(low_predictions), len(true_positives)))

    return results

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch evaluation')
    #  parser.add_argument('--filename', type=str, help="where is the data")
    parser.add_argument('--dataset', type=str, help="where is the data")
    parser.add_argument('--duration', type=float, default=240.0, help="where is the data")
    parser.add_argument('--ctf-thresh', type=float, default=6.0, help="ctf threshold")
    parser.add_argument('--num-patches', type=int, default=8, help="number of genes")
    parser.add_argument('--num-tries', type=int, default=10, help="number of tries")
    return parser

def main():
    global args, PATCH_DATA, RUN_RESULTS
    parser = arg_parser()
    args = parser.parse_args()

    # update global variable
    CryoEMConfig.Searching_Limit = args.duration
    CryoEMConfig.LOW_CTF_THRESH = args.ctf_thresh

    _, val_dataset, _, _, _, _ = get_dataset(args.dataset, CryoEMConfig.CLASSIFICATION, use_one_hot=True)
    PATCH_DATA = get_patch_list(val_dataset)
    #    print (PATCH_DATA)

    patch_len = len(PATCH_DATA)
    annealer = SimAnneal(patch_len, args.num_patches, fitness_function=fitness_func, stopping_iter=1000)

    results = []
    for k in range(args.num_tries):
        print('----- {} ------'.format(k))
        annealer.reset()
        annealer.anneal()
        print (annealer.best_solution, annealer.best_fitness)
        best_solution = annealer.best_solution
        for k in best_solution:
            print(PATCH_DATA[k][0].name, PATCH_DATA[k][1], PATCH_DATA[k][2])

        results.append(list(true_fitness(best_solution)))

    results = np.array(results)
    print (results)
    print(np.mean(results, axis=0), np.std(results, axis=0))

if __name__ == '__main__':
    main()
