# the file should loop over all the files specified
# and execute them with the simulator stub to check
# if one of the files is not working properly

import simulator_stub as ss
from dirs import *


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
import visualizer
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.keras.utils import plot_model

def test_model(input_shape):
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='linear') (inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(32, kernel_size=(5, 5))(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(10, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=x)
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    summary = model.summary()
    return model


def load_model(model_name):
    available_models = ["ResNet50", "MobileNetV2", "MobileNet", "ResNet18"]
    if model_name not in available_models:
        raise ValueError(f"Model not available. Please choose from the following: {', '.join(available_models)}")
    
    # Load the model
    model = keras.applications.__getattribute__(model_name)(weights='imagenet')
    return model

FILES_TO_RUN = ["config_files/dumps/dump{}.json".format(i) for i in range(0, 100)]

def test_files():
    # Create a SimulatorStub object
    stub = ss.SimulatorStub()

    for file in FILES_TO_RUN:
        print("File: ", file)
        results, logger = stub.run_simulation(file, verbose = False)
        print("Results: ", results)
    
if __name__ == "__main__":
    # test_files()

    model = test_model((28, 28, 1))
    # task_graph = model_to_graph(model, verbose=True)

    data_path = "/data/ACO_0/"
    plot_mapping_timeline(model, MAIN_DIR+data_path+"best_solution.json", MAIN_DIR+data_path+"mapping_timeline.png")
    
    # stub = ss.SimulatorStub()
    # results, logger = stub.run_simulation(MAIN_DIR+data_path+"best_solution.json", verbose = True)
    # NoCPlotter().plot(logger, 10 ,MAIN_DIR+data_path+"best_solution.json", MAIN_DIR+data_path+"NoC_simulation_result.gif", verbose = True)