'''
==================================================
File: simulator_stub.py
Project: simopty
File Created: Tuesday, 10th December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''


"""
The simulator_stub.py module contains the classes used to interface with the simulator estimate latency and energy consumption
for a given mapping of the tasks onto the NoC grid.
In particular, to perform a simulation, the simulator requires a configuration (.JSON) file, which contains:
- the architectural parameters of the NoC grid (e.g., the number of PEs, the topology of the grid, the communication latency, etc.);
- the workload and corresponding mapping of the tasks onto the NoC grid.
The first is predefined and kept fixed, while the second can be synthsized by the mapper.py module, given the specific mapping.
that has been considered.
The Stub class should simply provide and interface from the main.py module to the simulator.
"""

import os
import sys
import platform
import multiprocessing as mpr
from concurrent.futures import ThreadPoolExecutor
import time
from dirs import PYTHON_MODULE_DIR

# import the cpython module contained in the directory specified by PYTHON_MODULE_DIR
sys.path.append(PYTHON_MODULE_DIR)
import nocsim # type: ignore

def wrapper(func, q, event, *args):
    q.put(func(*args))
    event.set()

def dangerwrap(func, *args):
    event = mpr.Event()
    q = mpr.Queue()

    f_process = mpr.Process(target=wrapper, args=(func, q, event, *args))
    f_process.start()
    try:
        event.wait()
    except KeyboardInterrupt:
        f_process.terminate()
        f_process.join()
        print("Caught in dangerwrap")
        return None

    print("Exiting normally")
    return q.get()

def run_nocsim_simulation(path_to_config_file):
    return nocsim.simulate(path_to_config_file, "")


class SimulatorStub:

    def __init__(self, path_to_executable = None):
        self.path_to_executable = path_to_executable

    def run_simulation(self, path_to_config_file, verbose=False, dwrap=False):
        """
        Runs the simulation with the given configuration file.

        Parameters
        ----------
        path_to_config_file : str
            The path to the configuration file.

        Returns
        -------
        None
        """
        
        if verbose:
            print("Running simulation with configuration file: " + path_to_config_file)
        
        start = time.time()
        # use multiprocessing to allow for response to keyboard interrupts
        results, logger = dangerwrap(run_nocsim_simulation, path_to_config_file) if dwrap else run_nocsim_simulation(path_to_config_file)
        end = time.time()
        if verbose:
            print(f"Simulation completed in {end - start:.2f} seconds.")
        return results, logger

    def run_simulation_on_processor(self, path_to_config_file, processor, verbose=False):
        """
        Runs the simulation with the given configuration file on the specified processor.

        Parameters
        ----------
        path_to_config_file : str
            The path to the configuration file.
        processor : int
            The processor on which the simulation is to be run.

        Returns
        -------
        None
        """
        
        if verbose:
            print(f"Running simulation on processor {processor} with configuration file: {path_to_config_file}")

        if platform.system().lower() != "linux":
            if verbose:
                print("Setting processor affinity is not supported on this OS.")
        else:
            def set_affinity():
                os.system(f"taskset -p -c {processor} {os.getpid()}")
            p = mpr.Process(target=set_affinity)
            p.start()
            p.join()
            
        start = time.time()
        # os.system(self.path_to_executable + " " + path_to_config_file)
        results, logger = nocsim.simulate(path_to_config_file, "")
        end = time.time()
        if verbose:
            print(f"Simulation completed in {end - start:.2f} seconds.")

        return results, logger

    def run_simulations_in_parallel(self, config_files, processors, verbose=False):
        """
        Runs simulations in parallel on different processors.

        Parameters
        ----------
        config_files : list of str
            List of paths to configuration files.
        processors : list of str
            List of processors to run the simulations on.
        verbose : bool
            If True, prints verbose output.

        Returns
        -------
        None
        """
        if verbose:
            print("Running batch of simulations in parallel.")
        start = time.time()
        if processors is None:
            processors = list(range(os.cpu_count()))
        with ThreadPoolExecutor(max_workers=len(processors)) as executor:
            futures = []
            for config_file, processor in zip(config_files, processors):
                futures.append(executor.submit(self.run_simulation_on_processor, config_file, processor, False))
            results_loggers = [future.result() for future in futures]  # Wait for all futures to complete
        end = time.time()
        if verbose:
            print(f"Simulations completed in {end - start:.2f} seconds.")
        
        # Separate results and loggers
        results = [result for result, logger in results_loggers]
        loggers = [logger for result, logger in results_loggers]
        
        return results, loggers
                


