'''
==================================================
File: mapper.py
Project: simopty
File Created: Monday, 9th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The mapper.py module contains the classes used to perform the mapping of a split-compiled neural gridwork onto a k-dimensional NoC grid.
The mapper takes as input a list of elements, representing to which PE of the NoC a certain task of the dependecy graph gets mapped to.
It then produces a JSON-like list of elements, representing the mapping of the tasks onto the NoC and the communication between them.
It also produces a graphical representation of the mapping.
"""

import os
import json
import numpy as np

class Mapper:

    def init(self, dep_graph, grid):
        """
        Initializes the Mapper object with the given parameters.

        Parameters
        ----------
        dep_graph : TaskGraph
            The dependency graph to be mapped.
        grid : Grid
            The NoC grid on which the mapping is to be performed.

        Returns
        -------
        None
        """
        self.dep_graph = dep_graph
        self.grid = grid
        self.mapping = None

    def set_mapping(self, mapping):
        """
        Sets the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        mapping : dict
            A dictionary representing the mapping of the tasks onto the NoC grid:
            mapping = {job_id : node_id}

        Example
        -------
        mapping = {0 : 0, 1 : 3, 2 : 4, ...}

        Returns
        -------
        None
        """
        self.mapping = mapping


    def mapping_to_json(self, file_path, file_to_append = None):
        """
        Produces a JSON-like list of elements representing the mapped dependency graph on the NoC grid
        and writes it to a file.

        Parameters
        ----------
        file_path : str
            The path of the file where the JSON-like list is to be saved.

        Returns
        -------
        None
        """

        # Clear the previous mapping and set the new one
        self.dep_graph.clear_mapping()
        self.dep_graph.apply_mapping(self.mapping)
        # Write the mapping to a file, using the JSON format
        structure = self.dep_graph.graph_to_json()
        path = os.path.join(os.path.dirname(__file__), file_path)
        # create the directory if it does not exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # write the JSON-like list to a file
        if not file_to_append:
            with open(path, "w") as f:
                json.dump(structure, f, indent = 4)
        else:
            with open(file_to_append, "r") as f:
                data = json.load(f)

            data["arch"]["n"] = self.grid.N
            data["arch"]["k"] = self.grid.K
            data["arch"]["topology"] = self.grid.topology.value
            
            data["workload"] = structure
            with open(path, "w") as f:
                json.dump(data, f, indent = 4)


        




        

        



