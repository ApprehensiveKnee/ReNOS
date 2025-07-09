'''
==================================================
File: visualizer.py
Project: simopty
File Created: Friday, 7th March 2025
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

import os
import graph as dg
from graph import model_to_graph
import domain as dm
import mapper as mp
import simulator_stub as ss
from dirs import *
import optimizers as op
from utils.plotting_utils import *
from utils.ga_utils import *
from utils.partitioner_utils import *
from utils.ani_utils import *
import time
import nocsim # type: ignore

# Thanks to J. Jastrzebski
def create_logger(path_to_json = "/test.json", verbose = False):
    
    # Create a SimulatorStub object
    stub = ss.SimulatorStub()

    # Run the simulation in parallel
    
    # processors = list(range(6))
    # config_files = [os.path.join(RUN_FILES_DIR, f) for f in os.listdir(RUN_FILES_DIR) if f.endswith('.json')]
    # results, logger = stub.run_simulations_in_parallel(config_files=config_files, processors=processors, verbose=True)
    # results, logger = stub.run_simulation("config_files/dumps/dump.json", verbose = True)

    path_data = CONFIG_DUMP_DIR + path_to_json
    os.makedirs(os.path.dirname(path_data), exist_ok=True)
    # append "logger" : 1, to the arch field of the json file while 
    # leaving the rest of the file unchanged
    
    with open(path_data, "r") as f:
        data = json.load(f)
        data["arch"]["logger"] = 1
    with open(path_data, "w") as f:
        json.dump(data, f, indent = 4)



    results, logger = stub.run_simulation(path_data, verbose = False)
    
    if verbose: 
        for event in logger.events:
            print(event)
            print(f"Event ID: {event.id}, Type: {event.type}, Cycle: {event.cycle}, Additional info: {event.additional_info}," 
                    f"Info: {event.info}")
        #     if event.type == nocsim.EventType.START_COMPUTATION:
        #         print(f"Type: {event.type}, Event info: {event.info}")
        #         print(f"Node ID: {event.info.node} , Add_info Node ID {event.additional_info}")
        #     elif event.type == nocsim.EventType.END_COMPUTATION:
        #         print(f"Type: {event.type}, Event info: {event.info}")
        #         print(f"Node ID: {event.info.node}, Add_info Node ID {event.additional_info}")
        #     elif event.type == nocsim.EventType.OUT_TRAFFIC:
        #         print(f"Type: {event.type}, Event info: {event.info}")
        #         print(f"History: {event.info.history}")
        #     elif event.type == nocsim.EventType.IN_TRAFFIC:
        #         print(f"Type: {event.type}, Event info: {event.info}")
        #         print(f"History: {event.info.history}")
        #     elif event.type == nocsim.EventType.START_RECONFIGURATION:
        #         print(f"Type: {event.type}, Event info: {event.info}")
        #         print(f"Node ID: {event.additional_info}")
        #     elif event.type == nocsim.EventType.END_RECONFIGURATION:
        #         print(f"Type: {event.type}, Event info: {event.info}")
        #         print(f"Add_info Node ID: {event.additional_info}")
        #     else:
        #         pass
        #         print(f"I don't know how to handle this event: {event.type}")
    
    return logger, path_data

def plot_3d_animation(path_to_json = "/test.json", fps = 2, gif_path = "visual/test.gif"):
    
    logger, path_data = create_logger(path_to_json)
    
    # Initialize 3D plotter
    plotter_3d_animation = NoCPlotter()
    
    start_time = time.time()
    print("Plotting 3D animation...")
    plotter_3d_animation.plot(logger, fps, path_data, gif_path, verbose = False)  # Original 3D plot
    end_time = time.time()
    print(f"3D animation plotting took {end_time - start_time:.2f} seconds")

def plot_timeline(path_to_json = "/test.json", timeline_path = "visual/test.png", verbose = False):
    
    logger, path_data = create_logger(path_to_json)
    
    # Initialize timeline plotter
    plotter_timeline = NoCTimelinePlotter()

    print("Plotting timeline...")
    # Generate 2D timeline
    plotter_timeline.setup_timeline(logger, path_data)
    plotter_timeline.plot_timeline(timeline_path)
    
    if verbose:
        plotter_timeline._print_node_events()
    print("Timeline plotting done!")

def generate_animation_timeline_3d_plot(path_to_json = "/test.json", fps = 2, animation_path = "visual/combined_animation.gif"):
    
    logger, path_data = create_logger(path_to_json)
    
    plotter_3d_animation = NoCPlotter()
    plotter_timeline = NoCTimelinePlotter()
    
    start_time = time.time()
    print("Plotting combined visualisation...")
    #Create synchronized animation
    animator = SynchronizedNoCAnimator(plotter_3d_animation, plotter_timeline, logger, path_data)
    animator.create_animation(animation_path, fps=fps)
    end_time = time.time()
    print(f"Combined animation plotting took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    # Create a SimulatorStub object
    stub = ss.SimulatorStub()
    path_data = "/../../data/ACO_1/best_solution.json"
    
    #plot_timeline("/test.json", "visual/test.png", verbose = False)
    #plot_3d_animaiton("/test.json", fps = 2, gif_path = "visual/test.gif")
    generate_animation_timeline_3d_plot(path_data, fps = 100, animation_path = "data/GA_0batch_spatial/events.mp4")
    print("Done!")
    