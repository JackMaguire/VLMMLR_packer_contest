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
import argparse

from read_vlmmlr_data import *

numpy.random.seed( 0 )

#############
# ARGUMENTS #
#############
parser = argparse.ArgumentParser()
parser.add_argument( "--model", help="filename for output file", default="model.h5", required=False )
parser.add_argument( "--training_data", help="csv file to train from", default="data/training_data.first_block.100000.csv", required=False )
args = parser.parse_args()


##############
# PARAMETERS #
##############

window_size = 5 # consider 5 loops of data (columns) at once
channels = 3

#make this 0 if you turn off onehot encoding
extra_values = 0

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
input,output = read_from_file( args.training_data )

test_input,test_output = read_from_file( "data/validation_data.first_block.100000.csv" )


#############
# CALLBACKS #
#############

csv_logger = tensorflow.keras.callbacks.CSVLogger( "training_log.csv", separator=',', append=False )
# Many fun options: https://keras.io/callbacks/
def schedule( epoch, lr ):
    if lr < 0.0001:
        return lr * 2
    return lr * 0.9
lrs = tensorflow.keras.callbacks.LearningRateScheduler(schedule, verbose=0)

chkpt = tensorflow.keras.callbacks.ModelCheckpoint("checkpoint.{epoch:02d}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


callbacks=[csv_logger,lrs,chkpt]

#########
# TRAIN #
#########

class_weight = {0: 1.,
                1: 200.}

model.fit( x=input, y=output, batch_size=64, epochs=10, verbose=1, callbacks=callbacks, validation_data=(test_input,test_output), shuffle=True, class_weight=class_weight )


#############
# SPIN DOWN #
#############

model.save( args.model )
