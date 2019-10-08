import os
#Comment this out to use your gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import metrics
from tensorflow.keras import optimizers

from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow.keras
import numpy

import sys
import h5py

import argparse
import random
import time
import subprocess

########
# INIT #
########

numpy.random.seed( 0 )

num_input_values = 4 * 5 # 5 loops * (temp + 3 rates)

# Layers
input = Input(shape=(num_input_values,), name="in", dtype="float32" )
dense1 = Dense( name="dense1", units=100, activation="relu" )( input )
dense2 = Dense( name="dense2", units=100, activation="relu" )( dense1 )
dense3 = Dense( name="dense3", units=100, activation="relu" )( dense2 )
output = Dense( name="output", units=1, activation='sigmoid' )( dense3 ) # final value is between 0 and 1

model = Model(inputs=input, outputs=output )

metrics_to_output=[ 'binary_accuracy' ]
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )
model.save( "model.h5" )
model.summary()
