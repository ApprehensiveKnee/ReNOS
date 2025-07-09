'''
==================================================
File: graph.py
Project: simopty
File Created: Sunday, 8th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The graph.py module defines the class TaskGraph, which is used to represent a dependency graph.
Specifically, we use it as a wrapper for the networkx.DiGraph class, to represent a directed graph. 
The TaskGraph class can be initialized direcly from a JSON-like list of elements, or progressively by adding tasks and dependencies.
representing the computation and communication operations of a split-compiled neural network.
"""

import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import tensorflow.keras as keras
import keras.layers as layers
from utils.partitioner_utils import build_partitions, compute_partition_statistics
from utils.plotting_utils import plot_partitions


class TaskGraph:

    NODE_INFO = "Node ID: {0}\nType: {1}\nLayer ID: {2}\n Data Size : {3}\nComputing Time Required: {4}\nDependencies: {5}"
    EDGE_INFO = "Edge ID: {0}\nType: {1}\nData Size: {2}\nProcessing Time Required: {3}\nDependencies: {4}"


    def __init__(self, source = 0, drain = 20):
        self.graph = nx.DiGraph()
        self.SOURCE_POINT = source
        self.DRAIN_POINT = drain
        # Starting and ending nodes to group starting and ending dependencies
        self.graph.add_node("start", node = self.SOURCE_POINT, type = "START", layer = -1, color = 'lightgreen')
        self.graph.add_node("end", node = self.DRAIN_POINT, type = "END", layer = 999999, color = 'indianred')
        self.nodes = {}
        self.nodes_mapping = {}
        self.edges = {}
        self.edges_mapping = {}
        self.id = 0

    @property
    def n_nodes(self):
        """
        Returns the number of nodes (tasks) in the graph.
        """
        return len(self.nodes)
    
    @property
    def n_edges(self):
        """
        Returns the number of edges (dependencies) in the graph.
        """
        return len(self.edges)

    def clear(self):
        """
        Clears the graph.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        self.graph.clear()
        self.graph.add_node("start", node = 0, type = "START", layer = -1, color = 'lightgreen')
        self.graph.add_node("end", node = 8, type = "END", layer = np.inf, color = 'indianred')
        self.nodes = {}
        self.nodes_mapping = {}
        self.edges = {}
        self.edges_mapping = {}
        self.id = 0

    def clear_mapping(self):
        """
        Clears the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """

        self.nodes_mapping = {}
        self.edges_mapping = {}

    def apply_mapping(self, mapping):
        """
        Sets the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        mapping : dict
            A dictionary representing the mapping of the tasks onto the NoC grid:
            mapping = {job_id : node_id}
        
        Returns
        -------
        self.nodes_mapping : dict
            A dictionary representing the mapping of the nodes onto the NoC grid:
            nodes_mapping = {node_id : node}
        self.edges_mapping : dict
            A dictionary representing the mapping of the edges onto the NoC grid:
            edges_mapping = {edge_id : {"src": src, "dst": dst}}
        """

        self.nodes_mapping = mapping
        for edge_id in self.edges.keys():
            self.edges_mapping[edge_id] = {"src": None, "dst": None}
        # Check that all the nodes are in the mapping
        assert set(self.nodes.keys()).issubset(set(self.nodes_mapping.keys())), "Some nodes are missing in the mapping."
        
        for node_id, node in self.nodes.items():
            # if any of the node's dependencies are in the edges list, set the "dst" attribute of the edge
            # to the node's "node" attribute
            for dep in node["dep"]:
                if dep in self.edges.keys():
                    self.edges_mapping[dep]["dst"] = self.nodes_mapping[node_id]
        for edge_id, edge in self.edges.items():
            # if any of the edge's dependencies are in the nodes list, set the "src" attribute of the edge
            # to the node's "node" attribute
            for dep in edge["dep"]:
                if dep == -1:
                    edge["src"] = self.SOURCE_POINT
                elif dep in self.nodes.keys():
                    self.edges_mapping[edge_id]["src"] = self.nodes_mapping[dep]

        # If, at the end of the mapping, there are still some edges without a "src" or "dst" attribute,
        # set them to "start" and "end" respectively
        for edge in self.edges_mapping.values():
            if edge["src"] is None:
                edge["src"] = self.SOURCE_POINT
            if edge["dst"] is None:
                edge["dst"] = self.DRAIN_POINT

        return self.nodes_mapping, self.edges_mapping

        

    def add_task_fully_qualified(self, id, type, layer_id, size, weight_size, input_range, output_range, ct_required, dep, color = "lightblue"):
        """
        Adds a task (node) to the graph. Differently from the add_task method, this method allows to 
        specify all the parameters of the task that will be used during the simulation as a dictionary,
        that is then appended to the list of nodes.

        Parameters
        ----------
        id : int
            The id of the task. If None, the id is automatically assigned.
        layer_id : int
            The layer id of the task.
        type : str
            A string representing the task type.
        size : float
            The size of the task.
        weight_size : float
            The size of the weights of the task.
        input_range : dictionary
            The range of the input data (both spatial and channel-wise).
        output_range : dictionary
            The range of the output data (both spatial and channel-wise).
        ct_required : float
            The computing time required by the task.
        dep : list
            The list of dependencies of the task.
        color : str (optional)  
            The color of the node in the graph.

        """
        if id is None:
            id = self.id
            self.id += 1
        
        elem = {"id": id, "type": type, "layer_id": layer_id, "size": size, "weight_size": weight_size, "input_range": input_range, "output_range": output_range , "ct_required": ct_required, "dep": dep}
        self.graph.add_node(id, layer = layer_id, color = color)
        self.nodes[id] = elem



    def add_dependency_fully_qualified(self, task_id1, task_id2, id, type, size, pt_required, dep, cl = 0):
        """
        Adds a dependency (edge) to the graph. Differently from the add_dependency method, this method allows to
        specify all the parameters of the dependency that will be used during the simulation as a dictionary,
        that is then appended to the list of edges.

        Parameters
        ----------
        task_id1 : int
            The id of the source task.
        task_id2 : int
            The id of the destination task.
        id: int
            The id of the dependency. If None, the id is automatically assigned.
        type : str
            The type of the dependency.
        src : int
            The source PE of the dependency.
        dst : int
            The destination PE of the dependency.
        size : float
            The size of the data to be transferred.
        pt_required : float 
            The processing time required by the dependency.
        dep : list
            The list of dependencies of the dependency.
        """
        elem = {"id": id, "size": size, "dep": dep,"type": type, "cl" : cl, "pt_required": pt_required }
        self.graph.add_edge(task_id1, task_id2, id = id)
        self.edges[id] = elem


    def trim_graph(self):
        """
        The function to trim the graph from superfluous nodes: 
        - pairs of nodes that share deependencies only between each other are merged
        """ 

        for node1 in self.nodes.values():
            if len(node1["dep"]) == 1:
                node1_id = node1["id"]
                node2_id = node1["dep"][0]
                node2 = self.nodes[node2_id]
                if len(self.nodes[node2_id]["dep"]) == 1 and self.nodes[node2_id]["dep"][0] == node1_id:
                    # save the in edges of the first node and the out edges of the second node
                    in_edges = [(edge, self.graph.get_edge_data(edge, node1_id)["id"]) for edge in self.graph.in_edges(node1_id)]
                    out_edges = [(edge, self.graph.get_edge_data(node2_id, edge)["id"]) for edge in self.graph.out_edges(node2_id)]

                    # remove edges and nodes, substitute the nodes with a new one
                    # having as index the one of the first node, layer id of the first node,
                    # dep of the first node, ct_required and size of the sum of the two nodes
                    # and type of the first node
                    self.graph.remove_edge(node1_id, node2_id)
                    edge_id = self.get_edge_id(node1_id, node2_id)
                    self.edges.pop(edge_id)
                    self.graph.remove_node(node1_id)
                    self.nodes.pop(node1_id)
                    self.graph.remove_node(node2_id)
                    self.nodes.pop(node2_id)
                    new_node = {"id": node1_id, "type": node1["type"], "layer_id": node1["layer_id"], "size": node1["size"] + node2["size"], "ct_required": node1["ct_required"] + node2["ct_required"], "dep": node1["dep"]}
                    self.graph.add_node(node1_id, layer = node1["layer_id"], color = 'lightblue')
                    self.nodes[node1_id] = new_node

                    # stitch together the dependencies of the two nodes: the tasks depending on the second node
                    # now depends on the first node
                    for node in self.nodes.values():
                        if node["id"] != node1_id and node["id"] != node2_id:
                            node["dep"].remove(node2_id)
                            node["dep"].append(node1_id)
                    
                    # reconstruct the edges for the graph
                    for edge in in_edges:
                        self.graph.add_edge(edge[0][0], node1_id, id = edge[1])
                    for edge in out_edges:
                        self.graph.add_edge(node1_id, edge[0][1], id = self.get_edge_id(node1_id, edge[1]))

    def json_to_graph(self, structure):
        """
        Initializes the graph from a JSON-like list of elements, representing both
        the nodes and edges of the graph. Those elements are obtained as the output of
        a compile, whose job is to analyze the structure of the NN and perform the slicing
        of the layers.

        Parameters
        ----------
        structure : list
            A JSON-like list of elements, representing the nodes and edges of the graph.
        
        Returns
        -------
        None
        """

        NODES_TYPES = ["COMP_OP"]
        EDGES_TYPES = ["WRITE", "READ", "READ_REQ", "WRITE_REQ"]

        c_to_m = {}
        m_to_c = {}

        for elem in structure:
            if elem["type"] in NODES_TYPES:
                # append the node to the list
                self.nodes[elem["id"]] = elem 
                # delete mapping info if present
                if elem.get("node") is not None:
                    del elem["node"]
                self.graph.add_node(elem["id"], layer = elem["layer_id"], color = 'lightblue')
                for dep in elem["dep"]:
                    if m_to_c.get(dep) is None:
                        m_to_c[dep] = []
                    m_to_c[dep].append(elem["id"])
                

            elif elem["type"] in EDGES_TYPES:
                # append the edge to the list
                self.edges[elem["id"]] = elem
                # delete mapping info if present
                if elem.get("src") is not None:
                    del elem["src"]
                if elem.get("dst") is not None:
                    del elem["dst"]

                if elem["dep"][0] == -1:
                    assert len(elem["dep"]) == 1
                    if c_to_m.get("start") is None:
                        c_to_m["start"] = []
                    c_to_m["start"].append(elem["id"])
                else:
                    for dep in elem["dep"]:
                        if c_to_m.get(dep) is None:
                            c_to_m[dep] = []
                        c_to_m[dep].append(elem["id"])

        for node1 in c_to_m.keys():
            for edge in c_to_m[node1]:
                if  edge not in m_to_c.keys():
                    node2 = "end"
                    self.graph.add_edge(node1, node2, id = edge)
                else:
                    for node2 in m_to_c[edge]:
                        self.graph.add_edge(node1, node2, id = edge)

    def graph_to_json(self, verbose = False):
        """
        Returns the graph as a JSON-like list of elements.

        Parameters
        ----------
        nodes_mapping_info : dict
            A dictionary representing the mapping of the nodes onto the NoC grid:
            nodes_mapping_info = {node_id : node}
        edges_mapping_info : dict
            A dictionary representing the mapping of the edges onto the NoC grid:
            edges_mapping_info = {edge_id : {"src": src, "dst": dst}}
        verbose : bool
            If True, print the nodes and edges of the graph when constructing the list.
        
        Returns
        -------
        list
            A JSON-like list of elements, representing the nodes and edges of the graph.
        """
        
        structure = []
        packet_dependencies = [-1]
        workload_dependencies = []
        # Remove elements whose element appearing more than once
        nodes = self.get_nodes()
        edges = self.get_edges()

        # start from the edges and look for the ones whose dependencies are satisfied
        # then append them to the structure. If no more edges are found with
        # satisfied dependencies, look for the nodes.
        # The process is repeated until all the nodes and edges are appended to the structure

        already_appended_packets = []
        already_appended_workloads = []

        while (len(nodes)>0 or len(edges)>0):
            for edge in edges:
                if set(edge["dep"]).issubset(packet_dependencies):
                    if self.edges_mapping:
                        edge["src"] = self.edges_mapping[edge["id"]]["src"]
                        edge["dst"] = self.edges_mapping[edge["id"]]["dst"]
                    structure.append(edge)
                    if verbose:
                        print("\t\t\t Appending edge: ", edge["id"])
                        print("====================================")
                        print(self.EDGE_INFO.format(edge["id"], edge["type"], edge["size"], edge["pt_required"], edge["dep"]))
                        if self.edges_mapping:
                            print("Mapping info: ")
                            print("src: ", self.edges_mapping[edge["id"]]["src"])
                            print("dst: ", self.edges_mapping[edge["id"]]["dst"])
                        print("====================================")
                    workload_dependencies.append(edge["id"])
                    already_appended_packets.append(edge["id"])
            for node in nodes:
                # check that all the dependeices are satisfied
                if set(node["dep"]).issubset(workload_dependencies):
                    if self.nodes_mapping:
                        node["node"] = self.nodes_mapping[node["id"]]
                    structure.append(node)
                    if verbose:
                        print("\t\t\t Appending node: ", node["id"])
                        print("====================================")
                        print(self.NODE_INFO.format(node["id"], node["type"], node["layer_id"], node["size"], node["ct_required"], node["dep"]))
                        if self.nodes_mapping:
                            print("Mapping info: ")
                            print("node: ", self.nodes_mapping[node["id"]])
                        print("====================================")
                    packet_dependencies.append(node["id"])
                    already_appended_workloads.append(node["id"])
            
            # remove the already appended nodes and edges
            for edge_id in already_appended_packets:
                edges.remove(self.edges[edge_id])
            for node_id in already_appended_workloads:
                nodes.remove(self.nodes[node_id])
            
            # reset the already appended lists
            already_appended_packets = []
            already_appended_workloads = []


        assert len(nodes) == 0 and len(edges) == 0
        return structure
    
    def add_task_dep(self, id, dep):
        """
        Adds a dependency to a task.

        Parameters
        ----------
        id : int
            The id of the task.
        dep : int or list
            The id of the dependency or a list of dependencies.
        
        Returns
        -------
        None
        """

        # find the node in the graph corresponding to the id
        node = self.nodes.get(id)
        if node is None:
            return
        # if the dependency is a single integer
        if isinstance(dep, int):
            node["dep"].append(dep)
        # if the dependency is a list
        elif isinstance(dep, list):
            node["dep"].extend(dep)

                     
    def get_node(self, id,  verbose = False):
        """
        Returns the node with the given id.

        Parameters
        ----------
        id : int
            The id of the node to be returned.
        
        Returns
        -------
        dict
            The node with the given id.
        """

        node = self.nodes.get(id)

        if verbose:
            print("====================================")
            print(self.NODE_INFO.format(node["id"], node["type"], node["layer_id"],node["size"], node["ct_required"], node["dep"]))
            print("====================================")

        return node
            
        return None
    
    def get_edge(self, id, verbose = False):
        """
        Returns the edge with the given id.

        Parameters
        ----------
        id : int
            The id of the edge to be returned.
        
        Returns
        -------
        dict
            The edge with the given id.
        """

        edge = self.edges.get(id)

        if verbose:
            print("====================================")
            print(self.EDGE_INFO.format(edge["id"], edge["type"], edge["size"], edge["pt_required"], edge["dep"]))
            print("====================================")
        return edge
            
        return None
    

    def get_edge_id(self, src, dst):
        """
        Checks if an edge exists between two nodes. 
        If so, returns the id of the edge.

        Parameters
        ----------
        src : int
            The source node.
        dst : int
            The destination node.
        
        Returns
        -------
        bool
            the id of the edge if it exists, False otherwise.
        """

        return self.graph.get_edge_data(src, dst)["id"] if self.graph.has_edge(src, dst) else False
    

    def get_nodes(self, verbose = False):
        """
        Returns the list of nodes.

        Parameters
        ----------
        None
        
        Returns
        -------
        list
            The list of nodes.
        """
        
        if verbose:
            print("------------------------------------")
            print("\t\t\tNODES (", len(self.nodes), ")")
            print("------------------------------------")
            for node in self.nodes.values():
                print("====================================")
                print(self.NODE_INFO.format(node["id"], node["type"], node["layer_id"], node["size"], node["ct_required"], node["dep"]))
                print("====================================")


        return list(self.nodes.values())
    
    def get_edges(self, verbose = False):
        """
        Returns the list of edges.

        Parameters
        ----------
        None
        
        Returns
        -------
        dict
            The list of edges.
        """

        if verbose:
            print("------------------------------------")
            print("\t\t\tEDGES (", len(self.edges), ")")
            print("------------------------------------")
            for edge in self.edges.values():
                print("====================================")
                print(self.EDGE_INFO.format(edge["id"], edge["type"], edge["size"], edge["pt_required"], edge["dep"]))
                print("====================================")   


        return list(self.edges.values())
    
    def get_dependencies(self, node_id):
        """
        Returns the tasks on which the given task depends.

        Parameters
        ----------
        node_id : int
            The id of the task.

        Returns
        -------
        list
            The list of tasks on which the given task depends.
        """

        messages =  self.nodes[node_id]["dep"]
        dep_tasks = []
        for message in messages:
            if self.edges[message]["dep"][0] != -1:
                dep_tasks.append(self.edges[message]["dep"][0])
        return dep_tasks
            


def model_to_graph(model, verbose = False):
        """
        A function to create the depencency graph of the model that will be used for the simulation on the NoC.

        Args:
        - model : the model for which to create the dependency graph

        Returns:
        - a dependency graph of the model
        """

        dep_graph = TaskGraph()
        parts, deps = build_partitions(model)

        if verbose:
            print("Plotting the partitions and dependencies of the model...")
            plot_partitions(parts, deps)
            print("Done!")

        task_id = 0
        # dep_id starts from the closest power of 10 that is greater than the number of tasks
        dep_id = 10**math.ceil(math.log10(len(parts) + 1)) 
        layer_id = 0
        # DIRTY FIX: the scaling factors should be included in restart rather than simopty
        # mem_scaling_factor = 64 # byte/flit
        # comp_scaling_factor = 100 # FLOPs per cycle

        

        # assign task ids to the partitions
        for layer, partitions in parts.items():
            for partition in partitions:
                if isinstance(partition.layer, layers.InputLayer):
                    # task_id += 1
                    continue
                partition.task_id = task_id
                partition.layer_id = layer_id

                # Define computing time and task size
                computing_time = int(partition.FLOPs)
                computing_time = 1 if computing_time == 0 else computing_time
                task_size = int(partition.tot_size)
                task_size = 1 if task_size == 0 else task_size
                weight_size = 0
                for weight in partition.weights_shape:
                    weight_size += np.prod(weight)
                weight_size = int(weight_size)
                # create input_range and output_range:
                # input_range = [ "spatial_min": [min_x, min_y], "spatial_max": [max_x, max_y], "ch_bounds": [min_c, max_c]]
                # output_range = [ "spatial_min": [min_x, min_y], "spatial_max": [max_x, max_y], "ch_bounds": [min_c, max_c]]
                input_range = {"spatial_min": [min_bond for min_bond in partition.in_bounds[0]],
                                "spatial_max": [max_bond for max_bond in partition.in_bounds[1]],
                                "ch_bounds": [ch for ch in partition.in_ch] if partition.in_ch is not None else []}
                output_range = {"spatial_min": [min_bond for min_bond in partition.out_bounds[0]],
                                "spatial_max": [max_bond for max_bond in partition.out_bounds[1]],
                                "ch_bounds": [ch for ch in partition.out_ch] if partition.out_ch is not None else []}


                if task_size > 0 and computing_time > 0:
                    dep_graph.add_task_fully_qualified(id=partition.task_id, type = "COMP_OP", layer_id = partition.layer_id, size = task_size, weight_size= weight_size,input_range=input_range,output_range=output_range,ct_required= computing_time, dep = [])
                    task_id += 1
            layer_id += 1

        for key, value in deps.items():
            partitions1 = parts[key[0]]
            partitions2 = parts[key[1]]
            for dep, weight in value.items(): 
                partition1_match = (partition for partition in partitions1 if partition.id == dep[0])
                partition2_match = (partition for partition in partitions2 if partition.id == dep[1])
                try:
                    partition1 = next(partition1_match)
                except StopIteration:
                    raise ValueError(f"Partition with id {dep[0]} not found in partitions1")

                try:
                    partition2 = next(partition2_match)
                except StopIteration:
                    raise ValueError(f"Partition with id {dep[1]} not found in partitions2")
            
                first_node ="start" if isinstance(partition1.layer, layers.InputLayer) else partition1.task_id
                second_node = partition2.task_id

                # Define communication type and size
                comm_type = "WRITE" if isinstance(partition1.layer, layers.InputLayer) else "WRITE_REQ"
                comm_size = int(weight) 
                if comm_size == 0:
                    comm_size = 0 if weight == 0 else 1
                processing_time =int(comm_size* 0.3)

                if comm_size > 0:
                    dep_graph.add_dependency_fully_qualified(first_node, second_node , id = dep_id, type = comm_type , size = comm_size, pt_required= processing_time , cl = 0, dep= [-1] if isinstance(partition1.layer, layers.InputLayer) else [partition1.task_id])
                    # fetch the second partition from the graph and add the dependency of the communication
                    second_task = dep_graph.add_task_dep(second_node, dep_id)
                    dep_id += 1

        #find the last partitions of the model:
        # i.e. partitions contained in the list that belong to the furthest layer
        # down the model

        # print(list(parts.keys()))
        last_partitions = parts[list(parts.keys())[-1]]
        for layer in model.layers:
            if layer.name in [partition for partition in list(parts.keys())] and parts[layer.name] != []:
                last_partitions = parts[layer.name]
            
        # print("Last partitions: ", last_partitions)
        # Finally, connect the last partitions to the end node
        for partition in last_partitions:
            results = int(np.prod(partition.out_bounds))
            results = 1 if results == 0 else results
            dep_graph.add_dependency_fully_qualified(partition.task_id, "end", id = dep_id, type = "WRITE", size = results, pt_required = processing_time, cl = 0, dep = [partition.task_id])
            dep_id += 1

        # print(deps)
        compute_partition_statistics(model, parts, deps)
        

        return dep_graph
