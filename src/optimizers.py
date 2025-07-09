'''
==================================================
File: optimizer.py
Project: simopty
File Created: Wednesday, 18th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The optimizer.py module contains the classes used to perform the optimization of the mapping of a split-compiled NN onto a k-dimensional NoC grid.
"""

import os, sys
import logging
import enum
import numpy as np
from numpy.random import seed
import simulator_stub as ss
import mapper as ma
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import Union, ClassVar, Optional
from dirs import *
from utils.plotting_utils import plot_mapping_gif

import ctypes as c
from contextlib import closing
import multiprocessing as mp
from utils.aco_utils import Ant, vardict, walk_batch, update_pheromones_batch, manhattan_distance, random_heuristic_update
from utils.partitioner_utils import PE


class OptimizerType(enum.Enum):
    """
    Enum class representing the type of optimizer.
    """
    ACO = "ACO"
    GA = "GA"


@dataclass
class ACOParameters:
    """
    Dataclass representing the parameters of the optimization algorithm.

    The parameters of the optimization algorithm:
            - alpha : float
                The alpha parameter of the ACO algorithm.
            - beta : float
                The beta parameter of the ACO algorithm.
            - rho : float
                The evaporation rate of the pheromone.
            - n_ants : int
                The number of ants fro each colony
            - n_iterations : int
                The number of iterations of the algorithm.
            - n_best : int
                The number of best solutions to keep track of.
            - starting_rho : float
    """
    alpha : float = 1.0
    beta : float = 1.0
    rho : Union[float, None] = None
    n_ants : int = 10
    n_iterations : int = 100
    n_best : int = 10
    rho_start : Union[float, None] = None
    rho_end : Union[float, None] = None


class BaseOpt :

    optimizerType : ClassVar[Optional[Union[str, OptimizerType]]] = None

    def __str__(self):
        return self.optimizerType.value
    
    def run(self):
        pass
    

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- ACO -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
"""
class AntColony(BaseOpt):

    optimizerType : ClassVar[Union[str, OptimizerType]] = OptimizerType.ACO

    def __init__(self, optimization_parameters, domain, task_graph):
        """
        
        Parameters:
        -----------
        optimization_parameters : ACOParameters
            The parameters of the optimization algorithm.
        domain : Grid
            The domain of the optimization problem, the NoC grid.
        task_graph : TaskGraph
            The dependency graph of the tasks to be mapped onto the NoC grid.

        """

        super().__init__()
        self.par = optimization_parameters

        # --- Domain and Task Graph ---
        self.domain = domain
        self.task_graph = task_graph

        # --- Pheromone and Heuristic Information ---
        # The DAG on which to optimize is made out of implementation points, particular combinations
        # of tasks and PEs, representig to which PE a task is mapped to.
        #  the number of nodes in this new graph = #tasks * #PEs
        # self.tau_start = np.zeros((domain.size))
        # self.tau_start[4] = 1
        self.tau_start = np.ones((domain.size))/ domain.size
        self.tau = np.ones((task_graph.n_nodes-1, domain.size, domain.size))/ (task_graph.n_nodes * domain.size * domain.size) # Pheromone matrix
        self.eta = np.ones((task_graph.n_nodes-1, domain.size, domain.size)) # Heuristic matrix

        tasks = [task["id"] for task in self.task_graph.get_nodes() if task["id"] != "start"]
        tasks.insert(0, "start")
        self.tasks = tasks


    def run_and_show_traces(self, single_iteration_func, **kwargs):
        """
        This function is used to show the most likely paths taken by the ants throughout the 
        execution of the algorithm.
        It produces a gif file, evaluating the pheromone traces at a certain iteration and 
        finally plotting those, then stiching them up to create the animation
        """

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("magma")
        margin = 0.5
        ax.axis("off")
        ax.set_xlim( - 1 - margin, len(self.tasks) + margin)
        ax.set_ylim( - margin , self.domain.size + margin)

        a = 0.7 # ellipse width
        b = 0.5 # ellipse height    

        #implementation_points = []
        for i,task in enumerate(self.tasks[1:]):
            for j in range(self.domain.size):
                #implementation_points.append((i, j))
                ax.add_patch(patches.Ellipse((i, j), a, b, alpha = 1, facecolor = 'navajowhite', edgecolor = 'black', linewidth = 2, zorder = 2))
                ax.text(i, j, "(%s, %d)" % (task, j), ha = 'center', va = 'center', color = 'black')
        
        # Add starting and ending points
        ax.add_patch(patches.Ellipse((-1, int(self.domain.size/2)), a, b, alpha = 1, facecolor = 'palegreen', edgecolor = 'black', linewidth = 2, zorder = 2))
        ax.text(-1, int(self.domain.size/2), "start", ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        ax.add_patch(patches.Ellipse((len(self.tasks)-1, int(self.domain.size/2)), a, b, alpha = 1, facecolor = 'lightcoral', edgecolor = 'black', linewidth = 2, zorder = 2))
        ax.text(len(self.tasks)-1, int(self.domain.size/2), "end", ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        
        # Draw the connections (fully connected layers style)
        for i in range(len(self.tasks)-2):
            for j in range(self.domain.size):
                for k in range(self.domain.size):
                    ax.plot([i, i+1], [j, k], color = 'lightgrey', zorder = -2)

        for j in range(self.domain.size):
            ax.plot([-1, 0], [int(self.domain.size/2),j], color = 'lightgrey', zorder = -2)
            ax.plot([len(self.tasks)-2, len(self.tasks)-1], [j, int(self.domain.size/2)], color = 'lightgrey', zorder = -2)

        all_time_shortest_path = []
        edges = []

        def update(frame, edges = edges, all_time_shortest_path = all_time_shortest_path):
            # Each time the update function is called, an iteration of the algorithm is performed
            # and, if the frame is a multiple of the once_every parameter, the plot is actually updated

            all_time_shortest_path.append(single_iteration_func(frame, kwargs["once_every"], kwargs["rho_step"]))
            if len(all_time_shortest_path) > 1:
                assert len(all_time_shortest_path) == 2
                if all_time_shortest_path[1][1] < all_time_shortest_path[0][1]:
                    all_time_shortest_path[0] = all_time_shortest_path[1]
                all_time_shortest_path.pop(1)


            if frame % kwargs["once_every"] == 0:
                # extract the pheromone values and plot them on the corresponding edges
                

                for edge in edges:
                    edge.remove()
                edges.clear()

                for d_level in range(self.tau.shape[0]):
                    vmax = np.max(self.tau[d_level])
                    for i in range(self.tau.shape[1]):
                        for j in range(self.tau.shape[2]):
                            # find the maximum value out of the pheromones of che considered level
                            if self.tau[d_level, i, j] > 0:
                                edges.append(ax.plot([d_level, d_level+1], [i, j], color = cmap(self.tau[d_level, i, j]/vmax), zorder = self.tau[d_level, i, j]/vmax)[0])
                
                vmax_start = np.max(self.tau_start)
                for i in range(self.tau_start.shape[0]):
                    if self.tau_start[i] > 0:
                        edges.append(ax.plot([-1, 0], [int(self.domain.size/2), i], color = cmap(self.tau_start[i]/vmax_start), zorder = self.tau[d_level, i, j]/vmax)[0])

                return edges

            return []
                

        ani = animation.FuncAnimation(fig, update, frames = kwargs["n_iterations"], repeat = False) 
        plt.show()
        return all_time_shortest_path[0]

    def run(self, once_every = 10, show_traces = False):
        """
        Run the algorithm
        """

        def single_iteration(i, once_every, rho_step = 0):
            all_paths = self.generate_colony_paths()
            self.update_pheromones(all_paths)
            # print(self.tau)
            # plot a heatmap of the pheromonesa at dlevel 0 and save it
            # fig, ax = plt.subplots()
            # cmap = plt.get_cmap("magma")
            # slice = self.tau[0]
            # vmax = np.max(slice)
            # ax.imshow(slice, cmap = cmap, vmax = vmax)
            # plt.show()
            # plt.close()

            self.update_heuristics()
            shortest_path = min(all_paths, key=lambda x: x[1])
            moving_average = np.mean([path[1] for path in all_paths])
            if once_every is not None and i%once_every == 0:
                print("Iteration #", i, ", chosen path is:", shortest_path)
                print("Moving average for the path lenght is:", moving_average)
            
            self.evaporate_pheromones(rho_step)
            return shortest_path


        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)

        if self.par.rho_start is not None and self.par.rho_end is not None:
            self.rho = self.par.rho_start
            rho_step = (self.par.rho_end - self.par.rho_start) / self.par.n_iterations
        else:
            self.rho = self.par.rho
            rho_step = 0

        if show_traces:
            all_time_shortest_path = self.run_and_show_traces(single_iteration, once_every = once_every, n_iterations = self.par.n_iterations, rho_step = rho_step)
        else:
            for i in range(self.par.n_iterations):
                shortest_path = single_iteration(i, once_every, rho_step)
                if shortest_path[1] < all_time_shortest_path[1]:
                    all_time_shortest_path = shortest_path 

        return all_time_shortest_path


    def pick_move(self,task_id, d_level, current, resources, added_space, prev_path, random_heuristic = False):
        """
        Pick the next move of the ant, given the pheromone and heuristic matrices.
        """

        # compute a mask to filter out the resources that are already used
        mask = np.array([0 if pe.mem_used + added_space > pe.mem_size else 1 for pe in resources])

        if d_level == 0:
            outbound_pheromones = self.tau_start
            outbound_heuristics = np.ones(self.domain.size)
        else:
            outbound_pheromones = self.tau[d_level-1,current,:]
            if random_heuristic:
                outbound_heuristics = self.eta[d_level-1, current, :]
            else:
                # find the id of the task on which task_id depends (may be multiple)
                dependencies = self.task_graph.get_dependencies(task_id)
                # print("The dependencies of the task are:", dependencies)
                if len(dependencies) == 0:
                    outbound_heuristics = self.eta[d_level-1, current, :]
                else:
                    # find the PE on which the dependencies are mapped
                    dependencies_pe = [pe[2] for pe in prev_path if pe[0] in dependencies]
                    # print("The PEs on which the dependencies are mapped are:", dependencies_pe)
                    # generate the heuristics to favour the PEs near the  ones where the dependencies are mapped
                    outbound_heuristics = np.zeros(self.domain.size)
                    for pe in dependencies_pe:
                        for i in range(self.domain.size):
                            outbound_heuristics[i] += 1 / manhattan_distance(pe, i, self.domain) if manhattan_distance(pe, i, self.domain) != 0 else 1
                    outbound_heuristics = outbound_heuristics / np.sum(outbound_heuristics)
                

        row = (outbound_pheromones ** self.par.alpha) * (outbound_heuristics ** self.par.beta) * mask
        norm_row = (row / row.sum()).flatten()

        # if there is a NaN in the row, raise an error
        if np.isnan(norm_row).any():
            # print the row value that caused the error
            print("The row is:", row)
            raise ValueError("The row is NaN")
        
        return np.random.choice(range(self.domain.size), 1, p = norm_row)[0]


    def generate_ant_path(self, verbose = False):
        # No need to specify the start node, all the ants start from the "start" node
        path = []
        # A list of the available resources for each PE
        resources = [PE() for _ in range(self.domain.size)]
        prev = -1
        for d_level, task_id in enumerate(self.tasks):
            current = prev
            added_space = self.task_graph.get_node(task_id)["size"] if task_id != "start" and task_id != "end" else 0
            move = self.pick_move(task_id, d_level, current, resources, added_space, path) if d_level != self.task_graph.n_nodes else np.inf
            # udpate the resources
            if task_id != "start" and task_id != "end" and move != np.inf:
                resources[move].mem_used += added_space
            path.append((task_id, current, move))
            prev = move

        if verbose:
            print("The path found by the ant is:", path)
        return path


    def path_length(self, path, verbose = False):
        """
        Compute the "length" of the path using the NoC simulator.
        """

        # constuct the mapping form the path
        mapping = {task_id : int(pe) for task_id, pe, _ in path if task_id != "start" and task_id != "end"}

        mapper = ma.Mapper()
        mapper.init(self.task_graph, self.domain)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(CONFIG_DUMP_DIR + "/dump.json", file_to_append=ARCH_FILE)
        
        if verbose:
            plot_mapping_gif(mapper, "../visual/solution_mapping.gif")

        stub = ss.SimulatorStub()
        result, logger = stub.run_simulation(CONFIG_DUMP_DIR + "/dump.json", dwrap=True)
        return result, logger


    def generate_colony_paths(self):
        colony_paths = []
        for _ in range(self.par.n_ants):
            ant_path = self.generate_ant_path()
            path_length = self.path_length(ant_path)
            colony_paths.append((ant_path, path_length[0]))
        return colony_paths
    
    def evaporate_pheromones(self, step):
        if self.par.rho is not None:
            self.par.rho += step
        else:
            raise ValueError("The evaporation rate is not set")
        self.tau_start = (1 - self.rho) * self.tau_start
        self.tau = (1 - self.rho) * self.tau

    def update_pheromones(self, colony_paths):
        if self.par.n_best is None:
            self.par.n_best = len(colony_paths)
        sorted_paths = sorted(colony_paths, key = lambda x : x[1])
        best_paths = sorted_paths[:self.par.n_best]
        for path, path_length in best_paths:
            for d_level, path_node in enumerate(path):
                if path_node[1] == -1: # starting decision level
                    self.tau_start[path_node[2]] += 1 / path_length
                elif d_level-1 < self.tau.shape[0]:
                    self.tau[d_level-1, path_node[1], path_node[2]] += 1 / path_length

    def update_heuristics(self):
        """
        Update the heuristic matrix.
        """
        # RANDOM HEURISTIC UPDATE
        self.eta = random_heuristic_update(self.task_graph, self.domain)

    def compute_mean_distance(self, best_path: list[tuple[Union[str, int], int, int]]) -> float:
        """
        The function is used to compute the mean distance between dependent tasks.
        The distance metric used is the Manhattan distance.
        """

        mean_distance = 0
        tot_deps = 0
        
        # for each task in the best path
        for imp in best_path:
            task_id, node, _ = imp
            if task_id == "start":
                continue

            # if the task is not the start, find the distance between the task and its dependencies
            dependencies = self.task_graph.get_dependencies(task_id)
            if len(dependencies) > 0:
                dependencies_pe = [pe[1] for pe in best_path if pe[0] in dependencies]
                for pe in dependencies_pe:
                    # compute the distance between the dependent task and the task
                    mean_distance += manhattan_distance(pe, node, self.domain)
                    tot_deps += 1
                # print("Mean distance for task", task_id, "is:", manhattan_distance(injection_pe, node, self.domain))
            else:
                # if the task has no dependencies, it may depend from the start node
                # compute the distance between the task and the start node
                injection_pe = self.task_graph.SOURCE_POINT
                mean_distance += manhattan_distance(injection_pe, node, self.domain)
                tot_deps += 1
                
        return mean_distance / tot_deps
            
        
    

class ParallelAntColony(AntColony):

    optimizerType : ClassVar[Union[str, OptimizerType]] = OptimizerType.ACO


    def __init__(self, number_of_processes, optimization_parameters, domain, task_graph, save_intermediate = True):
        """
        
        Parameters:
        -----------
        optimization_parameters : ACOParameters
            The parameters of the optimization algorithm.
        domain : Grid
            The domain of the optimization problem, the NoC grid.
        task_graph : TaskGraph
            The dependency graph of the tasks to be mapped onto the NoC grid.

        """

        super().__init__(optimization_parameters, domain, task_graph)

        # The number of executors that will be used to run the algorithm
        self.n_processes = number_of_processes

        # self.logger = mp.log_to_stderr()
        # self.logger.setLevel(logging.INFO)

        self.ants = [Ant(i, self.task_graph, self.domain, self.tasks, self.par.alpha, self.par.beta) for i in range(self.par.n_ants)]

        # --- Pheromone and Heuristic Information ---
        # The pheromone and heuristic matrices are shared arrays among the ants

        self.tau_start = mp.Array(c.c_double, self.domain.size)
        tau_start_np = np.frombuffer(self.tau_start.get_obj())
        #initialize the tau_start vector
        tau_start_np[:] = 1 / self.domain.size

        self.tau = mp.Array(c.c_double, (self.task_graph.n_nodes-1) * self.domain.size * self.domain.size)
        tau_np = np.frombuffer(self.tau.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        #initialize the tau tensor
        tau_np[:] = 1 / (self.domain.size * self.domain.size * self.task_graph.n_nodes)

        self.eta = mp.Array(c.c_double, (self.task_graph.n_nodes-1) * self.domain.size * self.domain.size)
        eta_np = np.frombuffer(self.eta.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        #initialize the eta tensor
        eta_np[:] = 1

        self.statistics = {}
        self.statistics["mdn"] = []
        self.statistics["std"] = []
        self.statistics["best"] = []
        self.statistics["absolute_best"] = [np.inf]
        self.statistics["mean_distance"] = []
        self.statistics["mean_distance_absolute"] = []
        self.save_intermediate = save_intermediate

        #Calculate and store intervals for parallel processing
        
        self.intervals = [ (i, i + self.par.n_ants//self.n_processes + min(i, self.par.n_ants % self.n_processes)) for i in range(0, self.par.n_ants, self.par.n_ants//self.n_processes)]
        if self.par.n_best >= self.n_processes:
            self.best_intervals = [ (i, i + self.par.n_best//self.n_processes + min(i, self.par.n_best % self.n_processes)) for i in range(0, self.par.n_best, self.par.n_best//self.n_processes)]

    
    def run(self, once_every = 10, show_traces = False):
        """
        Run the algorithm
        """

        def single_iteration(i, once_every, rho_step = 0):
            all_paths = self.generate_colony_paths()
            self.update_pheromones(all_paths)

            # # plot a heatmap of the pheromonesa at dlevel 0
            # fig, ax = plt.subplots()
            # cmap = plt.get_cmap("magma")
            # slice = np.frombuffer(self.tau.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)[0]
            # vmax = np.max(slice)
            # print("Vmax:", vmax)
            # ax.imshow(slice, cmap = cmap, vmax = vmax)
            # plt.show()
            # plt.close()

            self.update_heuristics()
            shortest_path = min(all_paths, key=lambda x: x[2])
            moving_average = np.mean([path[2] for path in all_paths])
            moving_std = np.std([path[2] for path in all_paths])
            if once_every is not None and i%once_every == 0:
                print("Iteration #", i, ", chosen path is:", shortest_path)
                print("Moving average for the path lenght is:", moving_average)
            
            self.evaporate_pheromones(rho_step)
            self.statistics["mdn"].append(moving_average)
            self.statistics["std"].append(moving_std)
            self.statistics["best"].append(shortest_path[2])
            if shortest_path[2] < self.statistics["absolute_best"][-1]:
                self.statistics["absolute_best"].append(shortest_path[2])
            else:
                self.statistics["absolute_best"].append(self.statistics["absolute_best"][-1])

            return shortest_path


        shortest_path = None
        all_time_shortest_path = (np.inf, "placeholder", np.inf)

        if self.par.rho_start is not None and self.par.rho_end is not None:
            self.rho = self.par.rho_start
            rho_step = (self.par.rho_end - self.par.rho_start) / self.par.n_iterations
        else:
            self.rho = self.par.rho
            rho_step = 0

        if show_traces:
            all_time_shortest_path = self.run_and_show_traces(single_iteration, once_every = once_every, n_iterations = self.par.n_iterations, rho_step = rho_step)
        else:
            for i in range(self.par.n_iterations):
                shortest_path = single_iteration(i, once_every, rho_step)
                if shortest_path[2] < all_time_shortest_path[2]:
                    all_time_shortest_path = shortest_path 
                    # save the dump of the best solution in data
                    # 1. get the id of the ant that found the best solution
                    ant_id = shortest_path[0]
                    # save the corresponding dump file into data
                    dump_file = CONFIG_DUMP_DIR + "/dump" + str(ant_id) + ".json"
                    print("Saving the best solution found by ant", ant_id, "in data/best_solution.json")
                    os.makedirs("data", exist_ok = True)
                    os.makedirs("data/ACO", exist_ok = True)
                    os.system("cp " + dump_file + " data/ACO")
                    os.system("mv data/ACO/dump" + str(ant_id) + ".json data/ACO/best_solution.json")
                

                self.statistics["mean_distance"].append(self.compute_mean_distance(shortest_path[1]))
                self.statistics["mean_distance_absolute"].append(self.compute_mean_distance(all_time_shortest_path[1]))
                # save intermediate results
                if self.save_intermediate:
                    if i == 0 or i == self.par.n_iterations//2:
                        # copy the best solution json file into the intermediate results folder
                        os.makedirs("data/ACO/intermediate_results", exist_ok = True)
                        os.system("cp data/ACO/best_solution.json data/ACO/intermediate_results/best_solution_" + str(i) + ".json")
            # Finalize the simulation: save the data
            np.save("data/ACO/statistics.npy", self.statistics)
            

        return all_time_shortest_path

    @staticmethod
    def init(tau_start_, tau_, eta_, tau_shape, eta_shape, **kwargs):
        global vardict
        vardict["tau_start"] = tau_start_
        vardict["tau"] = tau_
        vardict["eta"] = eta_
        vardict["tau.size"] = tau_shape
        vardict["eta.size"] = eta_shape
        

    def generate_colony_paths(self):
        # generate the colony of ants (parallel workers)
        
        with closing(mp.Pool(processes = self.n_processes, initializer = ParallelAntColony.init, initargs = (self.tau_start, self.tau, self.eta, (self.task_graph.n_nodes-1, self.domain.size, self.domain.size), (self.task_graph.n_nodes-1, self.domain.size, self.domain.size)))) as pool:
            # generate the paths in parallel: each process is assigned to a subsed of the ants
            # evenly distributed
            colony_paths = pool.map_async(walk_batch,[self.ants[start:end] for start, end in self.intervals])
            colony_paths = colony_paths.get()
        pool.join()
        # unpack the batches of paths
        colony_paths = [path for batch in colony_paths for path in batch]
        return colony_paths
    
    def evaporate_pheromones(self, step):
        if self.par.rho is not None:
            self.par.rho += step
        else:
            raise ValueError("The evaporation rate is not set")
        tau_start = np.frombuffer(self.tau_start.get_obj())
        tau_start[:] = (1 - self.rho) * tau_start
        tau = np.frombuffer(self.tau.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        tau[:] = (1 - self.rho) * tau


    def update_pheromones(self, colony_paths):
        if self.par.n_best is None:
            self.par.n_best = len(colony_paths)
        sorted_paths = sorted(colony_paths, key = lambda x : x[2])
        best_paths = sorted_paths[:self.par.n_best]
        if self.par.n_best < self.n_processes:
            # update the pheromones in parallel
            with closing(mp.Pool(processes = self.par.n_best, initializer = ParallelAntColony.init, initargs = (self.tau_start, self.tau, self.eta, (self.task_graph.n_nodes-1, self.domain.size, self.domain.size), (self.task_graph.n_nodes-1, self.domain.size, self.domain.size)))) as pool:
                pool.map(update_pheromones_batch, [[best_paths[i]] for i in range(self.par.n_best)])
            pool.join()
        else:
            # update the pheromones in parallel
            with closing(mp.Pool(processes = self.n_processes, initializer = ParallelAntColony.init, initargs = (self.tau_start, self.tau, self.eta, (self.task_graph.n_nodes-1, self.domain.size, self.domain.size), (self.task_graph.n_nodes-1, self.domain.size, self.domain.size)))) as pool:
                pool.map(update_pheromones_batch, [best_paths[start:end] for start, end in self.best_intervals])
            pool.join()

    def update_heuristics(self): 
        # introduce stochasticity in the heuristic matrix
        eta = np.frombuffer(self.eta.get_obj()).reshape(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)
        eta[:] = np.random.rand(self.task_graph.n_nodes-1, self.domain.size, self.domain.size)

"""
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- GA -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
The following python classes for the optimization using  a Genetic Algorithm
will be primarly be a wrapper for the pyGAD library.
"""

import pygad # type: ignore
from utils.ga_utils import *

@dataclass
class GAParameters:
    """
    Dataclass representing the parameters of the optimization algorithm.

    The parameters of the optimization algorithm:
            - n_generations : int
                The number of generations of the algorithm.
            - n_parents_mating : int
                The number of solutions to be selected as parents for the next generation
            - sol_per_pop : int
                The number of solutions in the population.
            - parent_selection_type : str
                The type of parent selection.
            - num_genes:
                The number of genes in the chromosome.
            - init_range_low : float
                The lower bound of the initial range of the solutions.
            - init_range_high : float
                The upper bound of the initial range of the solutions.
            - keep_parents : int
                The number of parents to keep in the next generation.
            - gene_type : type = int
                The type of the genes of the solutions.
            - mutation_probability : float
                The probability of mutation
            - crossover_probability : float
                The probability of crossover

    """

    n_generations : int = 100
    n_parents_mating : int = 20
    sol_per_pop : int = 20
    parent_selection_type : str = "sss"
    num_genes : int = None
    init_range_low : float = None
    init_range_high : float = None
    keep_parents : int = 1
    gene_type : type = int
    mutation_probability : float = 0.2
    crossover_probability : float = 0.8


class GeneticAlgorithm(BaseOpt):

    def __init__(self, optimization_parameters, domain, task_graph, save_intermediate = True):
        """
        
        Parameters:
        -----------
        optimization_parameters : ACOParameters
            The parameters of the optimization algorithm.
        domain : Grid
            The domain of the optimization problem, the NoC grid.
        task_graph : TaskGraph
            The dependency graph of the tasks to be mapped onto the NoC grid.

        """

        super().__init__()
        self.par = optimization_parameters

        if self.par.num_genes is None:
            # the default value for the number of genes is the number of tasks in the graph
            self.par.num_genes = task_graph.n_nodes

        if self.par.init_range_low is None:
            # the default value for the lower bound of the initial range is 0
            self.par.init_range_low = 0

        if self.par.init_range_high is None:
            # the default value for the upper bound of the initial range is the size of the domain
            self.par.init_range_high = domain.size

        # --- Domain and Task Graph ---
        self.domain = domain
        self.task_graph = task_graph

        self.tasks = [task["id"] for task in self.task_graph.get_nodes() if task["id"] != "start"]

        # --- Pool of Operatros ---
        self.pool = OperatorPool(self, save_intermediate=save_intermediate)

        # --- Initialize the GA object of pyGAD ---

        self.ga_instance = pygad.GA(num_generations = self.par.n_generations,
                                    num_parents_mating = self.par.n_parents_mating,
                                    sol_per_pop = self.par.sol_per_pop,
                                    num_genes = self.par.num_genes,
                                    init_range_low = self.par.init_range_low,
                                    init_range_high = self.par.init_range_high,
                                    parent_selection_type = self.par.parent_selection_type,
                                    keep_parents = self.par.keep_parents,
                                    gene_type = self.par.gene_type,
                                    fitness_func = self.fitness_func,
                                    crossover_type = self.pool.get_cross_func,
                                    mutation_type = self.pool.get_mut_func,
                                    mutation_probability = self.par.mutation_probability,
                                    crossover_probability = self.par.crossover_probability,
                                    on_generation = self.pool.on_generation,
                                    on_stop = self.pool.on_stop,
        )



    def fitness_func(self, ga_instance, solution, solution_idx):

        # fitness function is computed using the NoC simulator:
        # 1. construct the mapping from the solution
        mapping = {}
        for task_idx, task in enumerate(self.tasks):
            mapping[task] = int(solution[task_idx])

        # 2. apply the mapping to the task graph
        mapper = ma.Mapper()
        mapper.init(self.task_graph, self.domain)
        mapper.set_mapping(mapping)
        mapper.mapping_to_json(CONFIG_DUMP_DIR + "/dump_GA.json", file_to_append=ARCH_FILE)

        # 3. run the simulation
        stub = ss.SimulatorStub()
        result, _ = stub.run_simulation(CONFIG_DUMP_DIR + "/dump_GA.json", dwrap=True)
    
        return 1 / result
    
    def run(self):

        self.ga_instance.run()

        return self.ga_instance.best_solution()
    

class ParallelGA(GeneticAlgorithm):

    def __init__(self, n_procs, optimization_parameters, domain, task_graph, save_intermediate = True):
        super().__init__(optimization_parameters, domain, task_graph, save_intermediate)

        self.ga_instance = pygad.GA(num_generations = self.par.n_generations,
                                    num_parents_mating = self.par.n_parents_mating,
                                    sol_per_pop = self.par.sol_per_pop,
                                    num_genes = self.par.num_genes,
                                    init_range_low = self.par.init_range_low,
                                    init_range_high = self.par.init_range_high,
                                    parent_selection_type = self.par.parent_selection_type,
                                    keep_parents = self.par.keep_parents,
                                    gene_type = self.par.gene_type,
                                    fitness_func = self.fitness_func,
                                    crossover_type = self.pool.get_cross_func,
                                    mutation_type = self.pool.get_mut_func,
                                    mutation_probability = self.par.mutation_probability,
                                    crossover_probability = self.par.crossover_probability,
                                    on_generation = self.pool.on_generation,
                                    on_stop = self.pool.on_stop,
                                    parallel_processing=["process", n_procs],
        )

    def fitness_func(self, ga_instance, solution, solution_idx):

        # fitness function is computed using the NoC simulator:
        # 1. construct the mapping from the solution
        mapping = {}
        for task_idx, task in enumerate(self.tasks):
            mapping[task] = int(solution[task_idx])

        # 2. apply the mapping to the task graph
        mapper = ma.Mapper()
        mapper.init(self.task_graph, self.domain)
        mapper.set_mapping(mapping)

        # 3. determine which process is running the simulation
        mapper.mapping_to_json(CONFIG_DUMP_DIR + "/dump_GA"+ str(solution_idx)+".json", file_to_append=ARCH_FILE)

        # 3. run the simulation
        stub = ss.SimulatorStub()
        result, _ = stub.run_simulation(CONFIG_DUMP_DIR + "/dump_GA"+ str(solution_idx)+".json")

        return 1 / result