'''
==================================================
File: domain.py
Project: simopty
File Created: Monday, 9th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The domain.py module contains the classes used to represent the domain (NoC search space) for the optimization problem.
"""

import os
from enum import Enum
import numpy as np

class Topology(Enum):
    MESH = "mesh"
    TORUS = "torus"

class Grid:
    """
    The class is used for a simplified representation of a k-dimensional NoC grid. Two different topologies are supported: mesh and torus.
    """

    def init(self, k, n, topology):
        """
        Initializes the grid with the given parameters.

        Parameters
        ----------
        k : int
            The number of elements for each dimension of the grid.
        n : int
            The number of dimensions of the grid.
        topology : Topology
            The topology of the grid (mesh or torus).

        Returns
        -------
        None
        """
        self.K = k
        self.N = n
        self.topology = topology
        self.grid = self.create_grid()
    

    def create_grid(self):
        """
        Creates the grid according to the given parameters.

        Returns
        -------
        np.array
            A numpy array representing the grid. Each element of the array represents the coordinate of the node in the grid.
        
        """
        self.size = self.K ** self.N
        dim_offset = [self.K ** i for i in range(self.N)]
        grid = np.zeros(self.size, dtype = object)
        for i in range(self.size):
            grid[i]  = tuple([(i // dim_offset[j]) % self.K for j in range(self.N)])
        return grid
    
    def get_node(self, index):
        """
        Returns the node corresponding to the given index.

        Parameters
        ----------
        index : int
            The index of the node.
        
        Returns
        -------
        tuple
            The coordinates of the node.
        """
        return self.grid[index]
    
    def get_neighbors(self, node):
        """
        Returns the neighbors of the given node.

        Parameters
        ----------
        node : tuple
            The coordinates of the node.
        
        Returns
        -------
        list
            A list of tuples representing the coordinates of the neighbors of the given node.
        """
        neighbors = []
        for i in range(self.N):
            for j in [-1, 1]:
                new_node = list(node)
                if Topology.MESH == self.topology:
                    new_node[i] += j
                    if new_node[i] >= 0 and new_node[i] < self.K:
                        neighbors.append(tuple(new_node))
                elif Topology.TORUS == self.topology:
                    new_node[i] = (new_node[i] + j) % self.K
                    neighbors.append(tuple(new_node))
        return neighbors
    

    def get_neighbor(self,node, dimension = 0, direction = -1):
        """
        Returns the neighbor of the given node in the given dimension and direction.

        Parameters
        ----------
        node : tuple
            The coordinates of the node.
        dimension : int
            The dimension in which the neighbor is located.
        direction : int
            The direction in which the neighbor is located.

        Returns
        -------
        tuple
            The coordinates of the neighbor of the given node in the given dimension and direction.
        """
        new_node = list(node)
        if Topology.MESH == self.topology:
            new_node[dimension] += direction
            if new_node[dimension] >= 0 and new_node[dimension] < self.K:
                return tuple(new_node)
        elif Topology.TORUS == self.topology:
            new_node[dimension] = (new_node[dimension] + direction) % self.K
            return tuple(new_node)
        return None
        
    def get_neighbors_index(self, node):
        """
        Returns the indexes of the neighbors of the given node.

        Parameters
        ----------
        node : tuple
            The coordinates of the node.
        
        Returns
        -------
        list
            A list of indexes representing the neighbors of the given node.
        """
        neighbors = []
        dim_offset = [self.K ** i for i in range(self.N)]
        for i in range(self.N):
            for j in [-1, 1]:
                new_node = list(node)
                if Topology.MESH == self.topology:
                    new_node[i] += j
                    if new_node[i] >= 0 and new_node[i] < self.K:
                        neighbors.append(np.sum([new_node[h] * dim_offset[h] for h in range(self.N)]))
                elif Topology.TORUS == self.topology:
                    new_node[i] = (new_node[i] + j) % self.K
                    neighbors.append(np.sum([new_node[h] * dim_offset[h] for h in range(self.N)]))
        return neighbors
    
    def get_neighbor_index(self,node, dimension = 0, direction = -1):
        """
        Returns the index of the neighbor of the given node in the given dimension and direction.

        Parameters
        ----------
        node : tuple
            The coordinates of the node.
        dimension : int
            The dimension in which the neighbor is located.
        direction : int
            The direction in which the neighbor is located.

        Returns
        -------
        int
            The index of the neighbor of the given node in the given dimension and direction.
        """
        new_node = list(node)
        dim_offset = [self.K ** i for i in range(self.N)]
        if Topology.MESH == self.topology:
            new_node[dimension] += direction
            if new_node[dimension] >= 0 and new_node[dimension] < self.K:
                return np.sum([new_node[h] * dim_offset[h] for h in range(self.N)])
        elif Topology.TORUS == self.topology:
            new_node[dimension] = (new_node[dimension] + direction) % self.K
            return np.sum([new_node[h] * dim_offset[h] for h in range(self.N)])
        return None
    
    def get_grid(self):
        """
        Returns the grid.

        Returns
        -------
        np.array
            A numpy array representing the grid. Each element of the array represents the coordinate of the node in the grid.
        """
        return self.grid
    
        
        
        