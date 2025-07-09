'''
==================================================
File: plotting_utils.py
Project: utils
File Created: Thursday, 26th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""

The plotting_utils.py module contains the functions used to plot the results of the simulation.
Those where originally distributed in the different classes of the project.

"""

import os
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from domain import Grid
from mapper import Mapper
import tensorflow.keras as keras
import pydot
from utils.partitioner_utils import PartitionInfo


"""
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = TASKGRAPH PLOTTING = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
"""

def plot_graph(graph, file_path = None):
    """
    Plots the nodes and edges in a top-down fashion.

    Parameters
    ----------
    file_path : str
        The path where to save the plot.
    
    Returns
    -------
    None
    """

    scale = 3
    pos = nx.multipartite_layout(graph.graph, subset_key="layer", scale = scale)
    colors = [graph.graph.nodes[node]["color"] for node in graph.graph.nodes()]
    labels = {node: node for node in graph.graph.nodes()}
    node_size = 1500
    node_shape = "o"
    fig, ax = plt.subplots(1,1, figsize = (13, 27))
    for node_id, node in enumerate(graph.graph.nodes()):
        nx.draw_networkx_nodes(graph.graph, pos, nodelist = [node], node_color = colors[node_id], node_size = node_size, node_shape = node_shape, ax = ax)
        nx.draw_networkx_labels(graph.graph, pos, labels = labels, font_size = 20, ax=ax)
    nx.draw_networkx_edges(graph.graph, pos, node_size = node_size, node_shape = node_shape, ax = ax, width = 2)
    # ax.axis("off")
    # plt.tight_layout()
    ax.set_ylim(-scale-0.1, scale+0.1)
    
    
    if file_path is not None:
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        fig.savefig(file_path, dpi = 300)
    # plt.show()


"""
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = GRID PLOTTING = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
"""

def plot_grid_2D(domain : Grid, file_path = None):
        """
        Plots the grid
        
        Parameters
        ----------
        file_path : str
            The path of the file where the plot is to be saved.

        Returns
        -------
        None
        """
        assert domain.N == 2
        fig, ax = plt.subplots()
        box_size = 0.30
        ax.set_aspect('equal')
        for i in range(domain.size):
            ax.text(domain.grid[i][0], domain.grid[i][1], str(i), ha='center', va='center')
            ax.plot(domain.grid[i][0], domain.grid[i][1], "s", markerfacecolor = 'lightblue', markeredgecolor = 'black', markersize = box_size*100, markeredgebox_size = 2)
        plt.xlim(-box_size-0.5, domain.K-0.5 + box_size)
        plt.ylim(-box_size-0.5, domain.K-0.5 + box_size)
        plt.axis("off")
        
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            plt.savefig(file_path)
        plt.show()

"""
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = MAPPING PLOTTING = = = = = = = = = = = = = = = = =
 = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
"""

def plot_mapping_2D(mapper: Mapper, file_path = None):
        """
        Plots the mapping of the tasks onto the NoC grid.

        Parameters
        ----------
        file_path : str
            The path of the file where the plot is to be saved.

        Returns
        -------
        None
        """
        assert mapper.grid.N == 2, "The plot_mapping_2D method is only available for 2D grids."
        fig,ax = plt.subplots()
        box_size = 0.50
        dim_offset = [mapper.grid.K ** i for i in range(mapper.grid.N)]
        # choose a discrete colormap (pastel): a color for each layer
        cmap = plt.cm.get_cmap('Pastel1', len(mapper.dep_graph.nodes.keys()))


        ax.set_aspect('equal')
        for i in range(mapper.grid.K):
            for j in range(mapper.grid.K):
                ax.add_patch(plt.Rectangle((i-box_size/2, j-box_size/2), box_size, box_size, facecolor = 'white', edgecolor = 'black', linewidth = 2, zorder = 0))
                ax.text(i, j - box_size*3/5, f"{i, j}", ha = 'center', va = 'top')
        for i in mapper.dep_graph.nodes.keys():
            mapped_node = mapper.mapping[i]
            layer_id = mapper.dep_graph.nodes[i]["layer_id"]
            color = cmap(layer_id)
            mapped_coords = [mapped_node // dim_offset[h] % mapper.grid.K for h in range(mapper.grid.N)]
            ax.add_patch(plt.Rectangle((mapped_coords[0]-box_size/2, mapped_coords[1]-box_size/2), box_size, box_size, facecolor = color, edgecolor = 'black', linewidth = 2, zorder = 1))
            ax.text(mapped_coords[0], mapped_coords[1], str(i), ha = 'center', va = 'center', color = 'black', fontweight = 'bold')
        plt.xlim(-box_size-0.5, mapper.grid.K-0.5 + box_size)
        plt.ylim(-box_size-0.5, mapper.grid.K-0.5 + box_size)
        plt.axis("off")
        
        if file_path is not None:
            file_path = os.path.join(os.path.dirname(__file__), file_path)
            plt.savefig(file_path)
        plt.show()


def plot_mapping_gif(mapper: Mapper, file_path = None):
    """
    The function is used to create a GIF of the mapping of the tasks onto the NoC grid.
    Each frame in the GIF is used for a different layer (level in the dependency graph): the
    tasks will appear assigned to the PEs of the NoC grid in the order in which they are
    defined in the depency graph (parallel tasks will appear in the same frame).
    """
    assert mapper.grid.N == 2, "The plot_mapping_gif method is only available for 2D grids."
    box_size = 1.0
    scale = 2
    dim_offset = [mapper.grid.K ** i for i in range(mapper.grid.N)]
    layers = set([mapper.dep_graph.nodes[task]["layer_id"] for task in mapper.dep_graph.nodes.keys()])
    cmap = plt.cm.get_cmap('tab20c', len(layers))

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.xlim(-box_size-0.5, scale*mapper.grid.K-0.5 + box_size)
    plt.ylim(-box_size-0.5, scale*mapper.grid.K-0.5 + box_size)
    plt.axis("off")
    plt.tight_layout()


    global patches, texts
    patches = []
    texts = []

    for i in range(mapper.grid.K):
            for j in range(mapper.grid.K):
                patches.append(plt.Rectangle((scale*i-box_size/2, scale*j-box_size/2), box_size, box_size, facecolor = 'white', edgecolor = 'black', linewidth = 2, zorder = 0))
                texts.append(plt.text(scale*i, scale*j - box_size*3/5, f"{i, j}", ha = 'center', va = 'top', fontdict={'fontsize': 7}))

    def init():
        global patches, texts
        for patch in patches[mapper.grid.K ** 2:]:
            patch.remove()
        for text in texts[mapper.grid.K ** 2:]:
            text.set_text("")
        patches = patches[:mapper.grid.K ** 2]
        texts = texts[:mapper.grid.K ** 2]

        for patch in patches:
            ax.add_patch(patch)
        for text in texts:
            ax.add_artist(text)
        return patches + texts

    def update(frame):
        global patches, texts
        #clear the previous frame
        for patch in patches[mapper.grid.K ** 2:]:
            patch.remove()
        for text in texts[mapper.grid.K ** 2:]:
            text.set_text("")

        patches = patches[:mapper.grid.K ** 2]
        texts = texts[:mapper.grid.K ** 2]

        
        # get the tasks of the current layer
        current_layer = [(task,mapper.dep_graph.nodes[task]["layer_id"]) for task in mapper.dep_graph.nodes.keys() if mapper.dep_graph.nodes[task]["layer_id"] == frame]
        
        for n,(task,layer_id) in enumerate(current_layer):
            mapped_node = mapper.mapping[task]
            color = cmap(frame)
            mapped_coords = [mapped_node // dim_offset[h] % mapper.grid.K for h in range(mapper.grid.N)]
            patches.append(plt.Rectangle((scale*mapped_coords[0]-box_size/2, scale*mapped_coords[1]-box_size/2), box_size, box_size, facecolor = color, edgecolor = 'black', linewidth = 2, zorder = 1))
            ax.add_patch(patches[-1])
            texts.append(plt.text(scale*mapped_coords[0], scale*mapped_coords[1], "L{}-{}".format(layer_id, n), ha = 'center', va = 'center', color = 'black', fontweight = 'bold', fontdict={'fontsize': 6}))
            ax.add_artist(texts[-1])
        
        return patches + texts
        
    
    ani = animation.FuncAnimation(fig, update, frames = layers, init_func = init, interval = 2000, blit = False, repeat = True)
    if file_path is not None:
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        ani.save(file_path, writer = 'magick', fps = 1)
    plt.show()


def plot_mapping_timeline(model, json_file, namefile = 'visual/mapping_timeline.gif'):
    """
    A function to plot the mapping of the layers of a model on the NoC nodes.
    The y axis represents the node IDs, the x axis instead represents the single layers of the model.

    The function reads the JSON file containing the mapping of the layers on the NoC nodes and plots
    the mapping of the layers on the NoC nodes.
    """

    # read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    arch = data["arch"]
    n_nodes = arch["k"] ** arch["n"]
    workload_list = data["workload"]
    layer_list = model.layers

    # remove input and flatten layers
    layer_list = [layer for layer in layer_list if not isinstance(layer, keras.layers.InputLayer) and not isinstance(layer, keras.layers.Flatten)]

    layer_to_nodes = {layer.name: {i:[] for i in range(n_nodes)} for layer in layer_list}

    # loop over the list of workloads: if the layer is a COMP_OP type, 
    # then we check the the layer_id -> the corresponding element in the list of layers
    # in the model is the corresponing layer
    for workload in workload_list:
        if workload["type"] == "COMP_OP":
            layer_id = workload["layer_id"]-1
            layer = layer_list[layer_id]
            layer_str = layer.name
            node_id = workload["node"]
            layer_to_nodes[layer_str][node_id].append(workload["id"])
    
    # create the plot
    fig, ax = plt.subplots(1,1,figsize = (8,8))
    bars = {i: 0 for i in range(n_nodes)}
    layer_intervals = []
    for layer_str, node_dict in layer_to_nodes.items():
        n_tasks = sum([len(tasks) for tasks in node_dict.values()])
        # find the lowest task id for the current layer
        t_low = min([task for tasks in node_dict.values() for task in tasks])
        t_id = 0
        colors = plt.cm.magma(np.linspace(0, 1, n_tasks))

        if len(layer_intervals) > 0:
            left_bound = layer_intervals[-1][1]
        else:
            left_bound = 0

        for node_id, workload in node_dict.items():
            # for each workload of the current layer assigned to the node,
            # add a 1 by 1 rectangle to the plot
            for task in workload:
                ax.barh(node_id, 1, left = bars[node_id], color = colors[task-t_low], edgecolor = 'black')
                bars[node_id] += 1
                t_id += 1
                # Annotate the block with the task id (color of the text 
                # is dependent on the color of the block)
                ax.text(bars[node_id]-0.5, node_id, str(task), color = 'white' if colors[task-t_low][0] < 0.5 else 'black', ha = 'center', va = 'center')

        
        right_bound = max(bars.values())
        for i in range(n_nodes):
            bars[i] = right_bound
        layer_intervals.append((left_bound, right_bound))

    ax.set_yticks(list(range(n_nodes)))
    ax.set_yticklabels(list(range(n_nodes)))
   
    ax.set_xlim(0, right_bound)
    # set the xticks at the middle of the intervals
    ax.set_xticks([(interval[0] + interval[1])/2 for interval in layer_intervals])
    # ax.set_xticklabels([layer_str for layer_str in layer_to_nodes.keys()], rotation = 30, ha="right", rotation_mode="anchor")
    ax.set_xticklabels(["Conv1", "MaxPool1", "Conv2", "MaxPool2", "FC1", "FC2"], rotation = 30, ha="right", rotation_mode="anchor")
    # mark the bounds of the intervals
    for interval in layer_intervals:
        ax.axvline(interval[0], color = 'black', linestyle = '--')
        ax.axvline(interval[1], color = 'black', linestyle = '--')
    
    # ax.set_xlabel("Layers")
    ax.set_ylabel("Nodes", fontsize = 15)
    ax.set_yticklabels(list(range(n_nodes)), fontsize = 13)

    # save the plot
    plt.savefig(namefile, dpi = 300)
            

def plot_partitions(partitions, partitions_deps, namefile = 'visual/task_graph.png'):
    """
    A function to plot the partitions of a layer using pydot package.

    Args:
    - partitions : a dictionary of partitions of the layers

    Returns:
    - a plot representing the task graph that will be deployed on the NoC
    """
    task_id = -1

    def format_node(partition: PartitionInfo):
    
        # we divide the node horizontally in 3 parts: the first part contains the partition id,
        # the second contains the input, output bounds, the number of input and output channels and the weights shape
        # the third part contains the MACs and FLOPs
        struct = f"{partition.id}\ntask_id:{partition.task_id} | layer_type:{type(partition.layer).__name__}\ninput bounds:{partition.in_bounds}\noutput bounds:{partition.out_bounds}\ninput channels:{partition.in_ch}\noutput channels:{partition.out_ch}\nweights shape:{partition.weights_shape} | MACs:{partition.MACs}\nFLOPs:{partition.FLOPs}\ntot_size:{partition.tot_size}"
        if partition.additional_data is not None:
            struct = "{{" + struct
            struct += "}| merged tasks: \n"
            for keys in partition.additional_data.keys():
                struct += f"{keys} \n"
            struct += "}"
        return struct

    # get the type of keras layer


    graph = pydot.Dot(graph_type='digraph')
    for layer, partition_list in partitions.items():
        for partition in partition_list:
            partition.task_id = task_id
            task_id += 1
            if partition.FLOPs > 0:
                node = pydot.Node(partition.id,label = format_node(partition), shape = "Mrecord")
                graph.add_node(node)

    for key, value in partitions_deps.items():
        for dep, weight in value.items():
            if weight > 0:
                edge = pydot.Edge(dep[0], dep[1], label = weight)
                graph.add_edge(edge)

    graph.write_png(namefile)


