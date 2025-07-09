'''
==================================================
File: partitioner.py
Project: simopty
File Created: Tuesday, 31st December 2024
Author: Edoardo Cabiati (edoardo.cabiati@mail.polimi.it)
Under the supervision of: Politecnico di Milano
==================================================
'''

"""
The partitioner.py module contains the classes and functions needed to create a partition/task graph out of a 
given DNN model. The chosen library for this purpose is TensorFlow.




"""

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.activations as activations


from dataclasses import dataclass
from utils.partitioner_utils import *










