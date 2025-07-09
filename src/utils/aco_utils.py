'''
==================================================
File: parallel_aco_utils.py
Project: utils
File Created: Thursday, 26th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The parallel_aco_utils.py module contains the classes and functions used to implement the parallel version of the Ant Colony Optimization algorithm.
"""

import ctypes as c
import numpy as np
import random
import time
import simulator_stub as ss
import mapper as mp
from dirs import *
from utils.partitioner_utils import PE
from domain import Grid, Topology


import matplotlib.pyplot as plt

# a utility function to compute manhattan distance
def manhattan_distance(n1, n2, domain: Grid):
    """
    Compute the manhattan distance between two nodes.
    """
    n1_x, n1_y = domain.get_node(n1)
    n2_x, n2_y = domain.get_node(n2)
    if domain.topology == Topology.MESH:
        return abs(n1_x - n2_x) + abs(n1_y - n2_y)
    elif domain.topology == Topology.TORUS:
        return min(abs(n1_x - n2_x), domain.K - abs(n1_x - n2_x)) + min(abs(n1_y - n2_y), domain.K - abs(n1_y - n2_y))
    else:
        raise ValueError("The topology is not supported.")

def random_heuristic_update(graph, domain):
    """
    Update the heuristic matrix randomly.
    """
    eta = np.random.rand(graph.n_nodes-1, domain.size, domain.size)
    return eta



# dictionary to store the shared variables (pheromone and heuristic matrices)
vardict = {}

class Ant:

    def __init__(self, id, graph, domain, tasks, alpha, beta):
        self.id = id
        self.graph = graph
        self.domain = domain
        self.tasks = tasks
        self.alpha = alpha
        self.beta = beta

        self.current_path = None


    def pick_move(self,task_id, d_level, current, resources, added_space, prev_path, random_heuristic = False):
        """
        Pick the next move of the ant, given the pheromone and heuristic matrices.
        """
        global vardict

        tau_start = np.frombuffer(vardict["tau_start"].get_obj())
        tau = np.frombuffer(vardict["tau"].get_obj()).reshape(vardict["tau.size"])
        eta = np.frombuffer(vardict["eta"].get_obj()).reshape(vardict["eta.size"])
            
        mask = np.array([0 if pe.mem_used + added_space > pe.mem_size else 1 for pe in resources])

        if d_level == 0:
            outbound_pheromones = tau_start
            outbound_heuristics = np.ones(self.domain.size)
        else:
            outbound_pheromones = tau[d_level-1,current,:]
            if random_heuristic:
                outbound_heuristics = eta[d_level-1, current, :]
            else:
                # find the id of the task on which task_id depends (may be multiple)
                dependencies = self.graph.get_dependencies(task_id)
                if len(dependencies) == 0:
                    outbound_heuristics = eta[d_level-1, current, :]
                else:
                    # find the PE on which the dependencies are mapped
                    dependencies_pe = [pe[1] for pe in prev_path if pe[0] in dependencies]
                    # generate the heuristics to favour the PEs near the  ones where the dependencies are mapped
                    outbound_heuristics = np.zeros(self.domain.size)
                    for pe in dependencies_pe:
                        for i in range(self.domain.size):
                            outbound_heuristics[i] += 1 / manhattan_distance(pe, i,self.domain) if manhattan_distance(pe, i, self.domain) != 0 else 1
                    outbound_heuristics = outbound_heuristics / np.sum(outbound_heuristics)

                
        row = (outbound_pheromones ** self.alpha) * (outbound_heuristics ** self.beta) * mask
        norm_row = (row / row.sum()).flatten()
        if norm_row is np.NaN:
            raise ValueError("The row is NaN")
        
        # if self.id == 0 and d_level == 0:
        #     fig, ax = plt.subplots()
        #     cmap = plt.get_cmap("magma")
        #     slice = tau[0]
        #     vmax = np.max(slice)
        #     print("Vmax:", vmax)
        #     ax.imshow(slice, cmap = cmap, vmax = vmax)
        #     plt.show()
        #     plt.close()

        return np.random.choice(range(self.domain.size), 1, p = norm_row)[0]
    
    def path_lenght(self, graph, domain, path):
        """
        Compute the lenght of the path.
        """

        # constuct the mapping form the path
        mapping = {task_id : int(pe) for task_id, pe, _ in path if task_id != "start" and task_id != "end"}

        mapper = mp.Mapper()
        mapper.init(graph, domain)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(CONFIG_DUMP_DIR + "/dump{}.json".format(self.id), file_to_append=ARCH_FILE)
        

        stub = ss.SimulatorStub()
        result, logger = stub.run_simulation(CONFIG_DUMP_DIR + "/dump{}.json".format(self.id))
        return result, logger
        
    def walk(self):
        """
        Perform a walk of the ant on the grid.
        """

        path = []
        resources = [PE() for _ in range(self.domain.size)]
        prev = -1
        for d_level, task_id in enumerate(self.tasks):
            current = prev
            added_space = self.graph.get_node(task_id)["size"] if task_id != "start" and task_id != "end" else 0
            move = self.pick_move(task_id, d_level, current, resources, added_space, path) if d_level != self.graph.n_nodes else np.inf
            # update the resources
            if task_id != "start" and task_id != "end" and move != np.inf:
                resources[move].mem_used += added_space
            path.append((task_id, current, move))
            prev = move
        
        path_lenght = self.path_lenght(self.graph, self.domain, path)
        self.current_path = (self.id, path, path_lenght[0])
        return self.current_path
    
    @staticmethod
    def update_pheromones(path):
        """
        Update the pheromone matrix in the shared memory.
        """
        tau_start = np.frombuffer(vardict["tau_start"].get_obj())
        tau = np.frombuffer(vardict["tau"].get_obj()).reshape(vardict["tau.size"])

        with vardict["tau_start"].get_lock():
            path_node = path[1][0]
            assert path_node[1] == -1
            tau_start[path_node[2]] += 1/path[2]
            # print("Tau start: ", tau_start)

        # print("Path", path)
        with vardict["tau"].get_lock():
            for d_level, path_node in enumerate(path[1][1:]):
                if d_level < vardict["tau.size"][0]:
                    # print("(first) tau[{},{},{}] = {}".format(d_level, path_node[1], path_node[2], tau[d_level, path_node[1], path_node[2]]))
                    tau[d_level, path_node[1], path_node[2]] += 1/path[2]
                    # print("(later) tau[{},{},{}] = {}".format(d_level, path_node[1], path_node[2], tau[d_level, path_node[1], path_node[2]]))

        
        # fig, ax = plt.subplots()
        # cmap = plt.get_cmap("magma")
        # slice = tau[0]
        # vmax = np.max(slice)
        # ax.imshow(slice, cmap = cmap, vmax = vmax)
        # plt.show()
        # plt.close()

    @staticmethod 
    def update_heuristics():
        """
        Update the heuristic matrix in the shared memory.
        """
        pass


def walk_batch(ants):
    """
    Perform a walk of the ants in the batch.
    """
    __seed = (os.getpid() * int(time.time())) % 123456789
    random.seed(__seed)
    np.random.seed(__seed)

    # pick a random number
    # print("Random number: ", random.randint(0, 100))

    paths = []
    for ant in ants:
        paths.append(ant.walk())
    return paths

def update_pheromones_batch(paths):
    """
    Update the pheromones of the ants in the batch.
    """
    for path in paths:
        Ant.update_pheromones(path)

    
        