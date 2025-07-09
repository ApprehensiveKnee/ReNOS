'''
==================================================
File: ga_utils.py
Project: utils
File Created: Saturday, 28th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The ga_utils.py module contains the classes and functions used to implement the Genetic Algorithm,
especially the operators used in the pool extraction process.
"""

from dirs import *
import numpy as np
from utils.partitioner_utils import PE
from utils.aco_utils import manhattan_distance
import json

    
"""
# 1-POINT CROSSOVER
def cross_1point(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_point = np.random.choice(range(offspring_size[1]))

        parent1[random_split_point:] = parent2[random_split_point:]

        offspring.append(parent1)

        idx += 1

    return np.array(offspring)

# 2-POINT CROSSOVER
def cross_2point(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_split_points = np.random.choice(range(offspring_size[1]), 2)

        if random_split_points[0] > random_split_points[1]:
            random_split_points = random_split_points[::-1]
        parent1[random_split_points[0]:random_split_points[1]] = parent2[random_split_points[0]:random_split_points[1]]

        offspring.append(parent1)

        idx += 1

    return np.array(offspring)


# UNIFORM CROSSOVER
def cross_uniform(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_mask = np.random.choice([0, 1], size=offspring_size[1])

        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        offspring1[random_mask == 1] = parents[idx % parents.shape[0], random_mask == 1]
        offspring2[random_mask == 1] = parents[(idx + 1) % parents.shape[0], random_mask == 1]

        offspring.append(offspring1)
        offspring.append(offspring2)

        idx += 2

    return np.array(offspring)

"""

# IRC (ISOMORPHIC REPLACEMENT CROSSOVER)
# will be added to the pool if the optimal solution is not updated after a certain threshold
def create_isomorophic_chromosomes(parent):
    """
    Create an isomorphic chromosome: those are obtained using 
    a permutation of the genes of the parents, using the dimensions of the reference NoC.
    Available transformations are:
    - center symmetry
    - mirror (vertical and horizontal flip) symmetry
    - center rotation

    Args:
        parent (np.array): the parent chromosome.
        It represents a mapping of the task graph onto the NoC: the chromosome is a list of 
        indexes, each one representing the PE of the NoC where the corresponding task in th task
        list is mapped.

    Example:
        NoC size: 3x3
        parent = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        isomorphic_chromosomes :
            vertical flip: [6, 7, 8, 3, 4, 5, 0, 1, 2, 9]
            horizontal flip: [2, 1, 0, 5, 4, 3, 8, 7, 6, 9]
            center symmetry: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """
    pass



# REDUCED SURROGATE CROSSOVER 
def cross_rsc(parents, offspring_size, ga_instance):
    
    similarity_threshold = 1. # if the similarity is equal or greater than 1. (in percentage), the parents are considered similar

    offspring = []
    idx = 0

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        similarity = np.sum(parent1 == parent2) / len(parent1)

        if similarity >= similarity_threshold:
            continue
        else:
            # perform 1-point crossover
            random_split_point = np.random.choice(range(offspring_size[1]))
            parent1[random_split_point:] = parent2[random_split_point:]

            offspring.append(parent1)
        

        idx += 1

    return np.array(offspring)



# DISCRETE CROSSOVER
def cross_discrete(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        random_mask = np.random.choice([0, 1], size=offspring_size[1])

        parent1[random_mask == 1] = parent2[random_mask == 1]

        offspring.append(parent1)

        idx += 1

    return np.array(offspring)

# AVERAGE CROSSOVER
def cross_average(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        offspring.append((parent1 + parent2) // 2) # integer division

        idx += 1

    return np.array(offspring)

# MULTIPARENT CROSSOVER
def cross_multiparent(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        
        # parents subset
        parents_subset = [(idx + i) % parents.shape[0] for i in range(offspring_size[1])]
        offspring.append(np.array([parents[parents_subset[i], i] for i in range(offspring_size[1])]))

        idx += 1

    return np.array(offspring)

# FLAT CROSSOVER
def cross_flat(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        # pick a random number between min(parent1, parent2) and max(parent1, parent2) for each gene
        offspring.append(np.array([np.random.choice([parent1[i], parent2[i]]) for i in range(offspring_size[1])]))

        idx += 1

    return np.array(offspring)

# MULTIVARIATE CROSSOVER
def cross_multivariate(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    k = int(0.3 * offspring_size[1]) # number of subsegments
    # for each one, we draw a random number: if this number is higher than the
    # crossover probability, we swap the genes of the parents in the subsegment

    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        
        random_split_points = np.sort(np.random.choice(range(1, offspring_size[1]), k-1))
        print(random_split_points)
        # for each subsegment, we draw a random number: if this number is higher than the
        # crossover probability, we swap the genes of the parents in the subsegment

        for i in range(k):
            if i == 0:
                start = 0
            else:
                start = random_split_points[i-1]
            if i == k-1:
                end = offspring_size[1]
            else:
                end = random_split_points[i]

            roulette = np.random.rand()
            if roulette < ga_instance.crossover_probability:
                parent1[start:end], parent2[start:end] = parent2[start:end], parent1[start:end]

        offspring.append(parent1)
        offspring.append(parent2)

        idx += 2

    return np.array(offspring)

def  cross_shuffle(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    k = int(0.25 * offspring_size[1]) # number of genes to shuffle
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()

        shuffle_genes = np.zeros(parent1.shape[0])
        
        # find k points in the parents that are the same and shuffle them
        # if there are less than k points, we simply shuffle all the ones
        # we can find
        for i in range(offspring_size[1]):
            if parent1[i] == parent2[i]:
                shuffle_genes[i] = 1
            if sum(shuffle_genes) == k:
                break
        print(shuffle_genes)
        
        # shuffle the genes in the marked positions
        np.random.shuffle(parent1[shuffle_genes == 1])
        np.random.shuffle(parent2[shuffle_genes == 1])

        # perform 1-point crossover
        random_split_point = np.random.choice(range(offspring_size[1]))
        parent1[random_split_point:] = parent2[random_split_point:]

        offspring.append(parent1)

        idx += 1

    return np.array(offspring)


# def mutate_random(offspring, ga_instance):
#     for chromosome_idx in range(offspring.shape[0]):
#         random_gene_idx = np.random.choice(range(offspring.shape[1]))

#         offspring[chromosome_idx, random_gene_idx] = np.random.choice(range(ga_instance.domain.size))

#     return offspring

# def mutate_swap(offspring, ga_instance):

#     for chromosome_idx in range(offspring.shape[0]):
#         random_gene_idx1 = np.random.choice(range(offspring.shape[1]))
#         random_gene_idx2 = np.random.choice(range(offspring.shape[1]))

#         offspring[chromosome_idx, random_gene_idx1], offspring[chromosome_idx, random_gene_idx2] = offspring[chromosome_idx, random_gene_idx2], offspring[chromosome_idx, random_gene_idx1]

#     return offspring

# def mutate_inversion(offspring, ga_instance):
#     for chromosome_idx in range(offspring.shape[0]):
#         random_gene_idx1 = np.random.choice(range(offspring.shape[1]))
#         random_gene_idx2 = np.random.choice(range(offspring.shape[1]))

#         if random_gene_idx1 > random_gene_idx2:
#             random_gene_idx1, random_gene_idx2 = random_gene_idx2, random_gene_idx1

#         offspring[chromosome_idx, random_gene_idx1:random_gene_idx2] = offspring[chromosome_idx, random_gene_idx1:random_gene_idx2][::-1]

#     return offspring

# def mutate_scramble(offspring, ga_instance):
#     for chromosome_idx in range(offspring.shape[0]):
#         random_gene_idx1 = np.random.choice(range(offspring.shape[1]))
#         random_gene_idx2 = np.random.choice(range(offspring.shape[1]))

#         if random_gene_idx1 > random_gene_idx2:
#             random_gene_idx1, random_gene_idx2 = random_gene_idx2, random_gene_idx1

#         offspring[chromosome_idx, random_gene_idx1:random_gene_idx2] = np.random.permutation(offspring[chromosome_idx, random_gene_idx1:random_gene_idx2])

#     return offspring

MUTATION_OPERATORS = [
    "random",
    "swap",
    "scramble",
    "inversion",
    "adaptive"
]

CROSSOVER_OPERATORS = [
    "single_point",
    "two_points",
    "uniform",
    "scattered"
]

def mutation_selection( offspring, ga_instance, mutation_type):
    if (mutation_type == "random"):
        return ga_instance.random_mutation(offspring)
    elif (mutation_type == "swap"):
        return ga_instance.swap_mutation(offspring)
    elif (mutation_type == "scramble"):
        return ga_instance.scramble_mutation(offspring)
    elif (mutation_type == "inversion"):
        return ga_instance.inversion_mutation(offspring)
    elif (mutation_type == "adaptive"):
        return ga_instance.adaptive_mutation(offspring)
    else:
        raise ValueError("The mutation type is not valid")
    
def crossover_selection(parents, offspring_size, ga_instance, crossover_type):
    if (crossover_type == "single_point"):
        return ga_instance.single_point_crossover(parents, offspring_size)
    elif (crossover_type == "two_points"):
        return ga_instance.two_points_crossover(parents, offspring_size)
    elif (crossover_type == "uniform"):
        return ga_instance.uniform_crossover(parents, offspring_size)
    elif (crossover_type == "scattered"):
        return ga_instance.scattered_crossover(parents, offspring_size)
    else:
        raise ValueError("The crossover type is not valid")


class OperatorPool:
    """
    A class representing the pool of operators to be used in the Genetic Algorithm.
    A reward mechanism is used to select the operators to be used in the next generation.
    It uses two different metrics:
     - F1 records the previous perfomance of each operator:

            F1 = [ max(C_1^t, C_2^t, ..., C_n^t) - max(C_1^(t-1), C_2^(t-1), ..., C_n^(t-1)) ] / max (C_1^t, C_2^t, ..., C_n^t)

            where:
                C_i^t is the cost of the i-th individual at generation t
                n is the number of individuals in the population
                t is the current iteration

     - F2 records the last time the operator was used

            F2 = 1 / (t_op - t_op')

            where:
                t_op' represents the last time the operator was selected
                t_op represents the current time when the operator is selected

    The final reward funcion is defined as the following:


        F(n) = F(n-1) +  beta * e^(1/(N+1-n)) * F1 * F2   iff F1 > 0
             = F(n-1)                                     iff F1 <= 0

        where:
            beta is a constant ( = 100)
            N is the total number of itertions in the pool
            n is the index for the current iteration
    """

    def __init__(self, optimizer, beta = 10, evol_stagnation = 150,save_intermediate = False):

        self.mutation_pool = MUTATION_OPERATORS
        self.crossover_pool = CROSSOVER_OPERATORS
        self.optimizer = optimizer
        self.beta = beta
        self.evol_stagnation = evol_stagnation
        # IDEA BEHIND IT:
        # the parameter is used to monitor the optimal solution:
        # if the optimal solution is not updated after a certain threshold, the
        # IRC (isomorpic replacement crossover) is added to the pool, in order to 
        # increse the diversity of the population.
        #
        # APPLIED TO OUR CASE:
        # the isomorphich replacement crossover cannot be applied with guarantee of 
        # substituting invalid crossovers with isomorphic solutions (producing the same fitness):
        # therefore, it won't produce the same results as the original paper.

        self.num_operators = sum(self.__len__())
        self.last_chosen = [0 for _ in range(self.num_operators)] # used to keep track of the last time the operator was selected
        self.prev_pop_fit = np.zeros(optimizer.par.sol_per_pop)
        self.F = [0 for _ in range(self.num_operators)] # first len(crossover_pool) elements are for the crossover operators, the others are for the mutation operators
        self.F1 = [0 for _ in range(self.num_operators)]
        self.F2 = [0 for _ in range(self.num_operators)]

        self.cur_cross = CROSSOVER_OPERATORS[0] # used to keep track of the current crossover operator
        self.cur_mut = MUTATION_OPERATORS[0] # used to keep track of the current mutation operator

        self.statistics = {}
        self.statistics["mdn"] = []
        self.statistics["std"] = []
        self.statistics["best"] = []
        self.statistics["absolute_best"] = [np.inf] 
        self.statistics["mean_distance_absolute"] = []
        self.save_intermediate = save_intermediate

    def add_operator(self, operator, operator_type):
        if operator_type == "crossover":
            self.crossover_pool.append(operator)
        elif operator_type == "mutation":
            self.mutation_pool.append(operator)
        else:
            raise ValueError("The operator type is not valid")
        
    def remove_operator(self, operator, operator_type):
        if operator_type == "crossover":
            self.crossover_pool.remove(operator)
        elif operator_type == "mutation":
            self.mutation_pool.remove(operator)
        else:
            raise ValueError("The operator type is not valid")
    
    def update_rewards(self, pop_fit, pop_fit_prev, cur_it, total_it):
        max_pop_fit = max(pop_fit)
        max_pop_fit_prev = max(pop_fit_prev)
        if cur_it == 1:
            self.F = [max_pop_fit for _ in range(self.num_operators)]
            return
        
        for i in range(len(self.crossover_pool)):
            self.F1[i] = (max_pop_fit - max_pop_fit_prev) / max_pop_fit
            self.F2[i] = 1 / (cur_it - self.last_chosen[i])
            self.F[i] = self.F[i] + self.beta * np.exp(1 / (total_it + 1 - cur_it)) * self.F1[i] * self.F2[i] if self.F1[i] > 0 else self.F[i]

        for i in range(len(self.mutation_pool)):
            self.F1[i + len(self.crossover_pool)] = (max_pop_fit - max_pop_fit_prev) / max_pop_fit
            self.F2[i + len(self.crossover_pool)] = 1 / (cur_it - self.last_chosen[i + len(self.crossover_pool)])
            self.F[i + len(self.crossover_pool)] = self.F[i + len(self.crossover_pool)] + self.beta * np.exp(1 / (total_it + 1 - cur_it)) * self.F1[i + len(self.crossover_pool)] * self.F2[i + len(self.crossover_pool)] if self.F1[i + len(self.crossover_pool)] > 0 else self.F[i + len(self.crossover_pool)]

    def pick_operator(self, cur_it):
        
        # pick the crossover and mutation operators using roulette wheel selection:
        # the probability of selecting an operator is proportional to its reward
        # the rewards are normalized to sum to 1

        # compute the rewards for the CROSSOVER operators
        F_crossover = np.array(self.F[:len(self.crossover_pool)])
        F_crossover = F_crossover / sum(F_crossover)
        # compute the rewards for the MUTATION operators
        F_mutation = np.array(self.F[len(self.crossover_pool):])
        F_mutation = F_mutation / sum(F_mutation)

        # pick the crossover operator
        self.cur_cross = np.random.choice(self.crossover_pool, p = F_crossover)
        # pick the mutation operator
        self.cur_mut = np.random.choice(self.mutation_pool, p = F_mutation)

        print("The current crossover operator is: ", self.cur_cross)
        print("The current mutation operator is: ", self.cur_mut)
        # update the last time the operator was selected
        self.last_chosen[self.crossover_pool.index(self.cur_cross)] = cur_it
        self.last_chosen[self.mutation_pool.index(self.cur_mut) + len(self.crossover_pool)] = cur_it

        return (self.cur_cross, self.cur_mut)

    def on_generation(self, ga_instance):

        # compute the fitness of the current population
        pop_fit = ga_instance.last_generation_fitness
        # print("the fitness of the current population is: ", pop_fit)
        # udpate the rewards
        self.update_rewards(pop_fit, self.prev_pop_fit, ga_instance.generations_completed, ga_instance.num_generations)
        # print("the rewards for CROSSOVER operators are: ", self.F[:len(self.crossover_pool)])
        # print("the rewards for MUTATION operators are: ", self.F[len(self.crossover_pool):])
        self.pick_operator(ga_instance.generations_completed)
        # print("the current crossover operator is: ", self.cur_cross)s
        # print("the current mutation operator is: ", self.cur_mut)
        
        # udpate the fitness of the previous population
        self.prev_pop_fit = pop_fit

        # save the mean fitness, the standard deviation and the best fitness of the current population
        self.statistics["mdn"].append(np.mean(1/pop_fit))
        self.statistics["std"].append(np.std(1/pop_fit))
        self.statistics["best"].append(1/max(pop_fit))
        best_solution = ga_instance.best_solution()
        self.statistics["mean_distance_absolute"].append(self.compute_mean_distance(best_solution[0]))

        if (1/max(pop_fit)) < self.statistics["absolute_best"][-1]:
            self.statistics["absolute_best"].append(1/max(pop_fit))
            # save the dump file for the best solution
            os.makedirs(MAIN_DIR + "/data", exist_ok=True)
            os.makedirs(MAIN_DIR + "/data/GA", exist_ok=True)
            # move "dump_GA"+str(np.argmax(pop_fit))" from "config_files/dumps" to "data/GA"
            os.system("cp " + MAIN_DIR + "/config_files/dumps/dump_GA" + str(np.argmax(pop_fit)) + ".json " + MAIN_DIR + "/data/GA")
            # rename the file to "best_solution.json"
            os.system("mv " + MAIN_DIR + "/data/GA/dump_GA" + str(np.argmax(pop_fit)) + ".json " + MAIN_DIR + "/data/GA/best_solution.json")

        if self.save_intermediate:
            # if the simulation just started or in the middle of the tot iterations, save the json
            if (ga_instance.generations_completed == 1) or (ga_instance.generations_completed == ga_instance.num_generations // 2):
                #save the json in intermediate_results folder
                os.makedirs(MAIN_DIR + "/data/GA/intermediate_results", exist_ok=True)
                os.system("cp " + MAIN_DIR + "/data/GA/best_solution.json " + MAIN_DIR + "/data/GA/intermediate_results/best_solution_" + str(ga_instance.generations_completed) + ".json")
        print("=====================================================")
        print("The best latency of the generation n. {} is: {}".format(ga_instance.generations_completed, 1/max(pop_fit)))
        print("The mean latency of the generation n. {} is: {}".format(ga_instance.generations_completed, 1/np.mean(pop_fit)))

    def on_stop(self, ga_instance, last_generation_fitness):
        # at the end of the optimization process, save the statistics
        np.save(MAIN_DIR + "/data/GA/statistics.npy", self.statistics)

    
    def get_cross_func(self, parents, offspring_size, ga_instance):
        offspring = crossover_selection(parents, offspring_size, ga_instance, self.cur_cross)
        # check if the offspring is valid
        # if not, rerun the crossover operator until a valid offspring is obtained
        for i in range(len(offspring)):
            while not self.check_mem_constraints(offspring[i]):
                offspring[i] = crossover_selection(parents, (1, offspring_size[1]), ga_instance, self.cur_cross)[0]

        return offspring
        
    def get_mut_func(self, offspring, ga_instance):
        if self.cur_mut == "adaptive":
            ga_instance.mutation_probability = (0.35, 0.17)
        else:
            ga_instance.mutation_probability = self.optimizer.par.mutation_probability

        # check if the offspring is valid
        # if not, rerun the mutation operator until a valid offspring is obtained
        for i in range(len(offspring)):
            while not self.check_mem_constraints(offspring[i]):
                offspring[i] = mutation_selection(offspring[i], ga_instance, self.cur_mut)


        return mutation_selection(offspring, ga_instance, self.cur_mut)
    
    def check_mem_constraints(self, offspring):
        # the integer assigned to a certain position in the chromosome represents the PE of the NoC chosen
        # to map the corresponding task. The PE has a certain amount of memory, and the task has a certain
        # memory requirement. If total memory used by the tasks mapped on a certain PE is greater than the
        # memory available, the solution is invalid.
        # The function returns true is the offspring is valid, false otherwise.

        resources = [PE() for _ in range(self.optimizer.domain.size)]

        for gene in range(offspring.shape[0]):
            if resources[offspring[gene]].mem_used + self.optimizer.task_graph.get_node(self.optimizer.tasks[gene])["size"] > resources[offspring[gene]].mem_size:
                return False
            resources[offspring[gene]].mem_used += self.optimizer.task_graph.get_node(self.optimizer.tasks[gene])["size"]

        return True
    
    def compute_mean_distance(self, best_solution):
        """
        The function is used to compute the mean distance between dependent tasks.
        The distance metric used is the Manhattan distance.
        """

        mean_distance = 0
        tot_deps = 0

        for task_id in range(len(best_solution)):
            
            # get the node associated to the task_id
            node = best_solution[task_id]

            # get the dependencies of the task
            dependencies = self.optimizer.task_graph.get_dependencies(task_id)
            if len(dependencies) > 0:
                dependencies_pe = [best_solution[dep] for dep in dependencies]

                for dep in dependencies_pe:
                    mean_distance += manhattan_distance(node, dep, self.optimizer.domain)
                    tot_deps += 1

        return mean_distance / tot_deps

        


    def get_pool(self):
        return (self.crossover_pool, self.mutation_pool)

    def __len__(self):
        return (len(self.crossover_pool), len(self.mutation_pool))