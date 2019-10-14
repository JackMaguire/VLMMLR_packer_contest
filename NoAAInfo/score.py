import os
#Comment this out to use your gpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow.keras.models import load_model
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

################
# CREATE MODEL #
################

parser = argparse.ArgumentParser()
parser.add_argument( "--model", help="filename for model", default="model.h5", required=False )
args = parser.parse_args()

model = load_model( args.model )


#########
# TRAIN #
#########

def determine_cutoff_for_percentile( output, predictions, perc ):
    true_values = []
    for i in range( 0, len( predictions) ):
        if output[ i ] == 1:
            true_values.append( predictions[ i ] )
    true_values.sort()
    index = int( len(true_values) * perc )
    return true_values[ index ]

def measure_cutoff( predictions, output, cutoff ):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    
    for i in range( 0, len( predictions ) ):
        if predictions[i] < cutoff: #predict negative
            if output[i] == 0: # true negative
                true_neg += 1
            else:
                false_neg += 1
        else: #predict positive
            if output[i] == 0: # false positive
                false_pos += 1
            else:
                true_pos += 1
    return true_pos,true_neg,false_pos,false_neg

input,output = read_from_file_NOAA( "data/final_test_data.first_block.500000.csv" )
model.evaluate( x=input, y=output, batch_size=32 )

predictions = model.predict( x=input )

first_cutoff = determine_cutoff_for_percentile( output, predictions, 0.01 )
second_cutoff = determine_cutoff_for_percentile( output, predictions, 0.001 )

print( "cutoff, frac work eliminated, frac good AAs lost" )
true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, first_cutoff )
#print( '{:02.2f}'.format(first_cutoff), true_pos, true_neg, false_pos, false_neg, 0.75*((true_neg+false_neg)/(0.0+true_pos+true_neg+false_pos+false_neg)), (false_neg / (0.0+true_pos+false_neg)) )
print( first_cutoff[ 0 ], 0.75*((true_neg+false_neg)/(0.0+true_pos+true_neg+false_pos+false_neg)), (false_neg / (0.0+true_pos+false_neg)) )


true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, second_cutoff )
print( second_cutoff[ 0 ], 0.75*((true_neg+false_neg)/(0.0+true_pos+true_neg+false_pos+false_neg)), (false_neg / (0.0+true_pos+false_neg)) )



