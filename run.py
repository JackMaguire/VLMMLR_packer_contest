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
import pandas as pd

numpy.random.seed( 0 )

##############
# PARAMETERS #
##############

window_size = 5 # consider 5 loops of data (columns) at once
channels = 3 # temp + 3 rates

extra_values = 0 # number of loops remainint

num_input_values = (window_size * channels) + extra_values

################
# CREATE MODEL #
################

# Layers
input = Input(shape=(num_input_values,), name="in", dtype="float32" )
dense1 = Dense( name="dense1", units=100, activation="relu" )( input )
dense2 = Dense( name="dense2", units=100, activation="relu" )( dense1 )
dense3 = Dense( name="dense3", units=100, activation="relu" )( dense2 )
output = Dense( name="output", units=1, activation='sigmoid' )( dense3 ) # final value is between 0 and 1

model = Model(inputs=input, outputs=output )

metrics_to_output=[ 'binary_accuracy' ]
model.compile( loss='binary_crossentropy', optimizer='adam', metrics=metrics_to_output )
model.summary()


#############
# LOAD DATA #
#############

def read_from_file( filename ):
    data = pd.read_csv( filename, header=None ).values
    amino_acids = data[:,0:1]
    input_data = data[:,1:21]
    output = data[:,81:82]
    return input_data, output

input,output = read_from_file( "data/training_data.100000.csv" )
test_input,test_output = read_from_file( "data/validation_data.10000.csv" )

#############
# CALLBACKS #
#############

csv_logger = tensorflow.keras.callbacks.CSVLogger( "training_log.csv", separator=',', append=False )
# Many fun options: https://keras.io/callbacks/
callbacks=[csv_logger]

#########
# TRAIN #
#########

class_weight = {0: 1.,
                1: 20.}

model.fit( x=input, y=output, batch_size=64, epochs=1, verbose=1, callbacks=callbacks, validation_data=(test_input,test_output), shuffle=True, class_weight=class_weight )


#############
# SPIN DOWN #
#############

model.save( "model.h5" )
