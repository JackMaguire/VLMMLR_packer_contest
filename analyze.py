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

numpy.random.seed( 0 )

##############
# PARAMETERS #
##############

window_size = 5 # consider 5 loops of data (columns) at once
channels = 4 # temp + 3 rates

extra_values = 1 # number of loops remainint

num_input_values = (window_size * channels) + extra_values

################
# CREATE MODEL #
################

model = load_model( "model.h5" )

#############
# LOAD DATA #
#############

def read_from_file_raw( filename ):
    data = pd.read_csv( filename, header=None ).values
    amino_acids = data[:,0:1]
    input_data = data[:,1:81]
    output = data[:,81:82]
    return input_data, output

def read_from_file( filename, x ):
    input_data,output=read_from_file_raw( filename )
    inp = []
    out = []
    for i in range( 0, len( input_data ) ):
        ncol = int( len( input_data[i] ) / channels )
        number_of_columns_to_measure = ncol - window_size #exclude final window, why would we care about that?
        for j in range( 0, number_of_columns_to_measure ):
            number_of_cols_remaining = number_of_columns_to_measure - j
            if number_of_cols_remaining != x:
                continue
            row_data = []
            starting_element = j * channels
            for k in range( 0, window_size * channels ):
                row_data.append( input_data[ i ][ starting_element + k ] )
            row_data.append( number_of_cols_remaining )
            inp.append( row_data )
            out.append( output[ i ] )
    inp=numpy.asarray( inp )
    out=numpy.asarray( out )
    return inp,out

#############
# CALLBACKS #
#############

#csv_logger = tensorflow.keras.callbacks.CSVLogger( "training_log.csv", separator=',', append=False )
# Many fun options: https://keras.io/callbacks/
#callbacks=[csv_logger]

#########
# TRAIN #
#########

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

class_weight = {0: 1.,
                1: 20.}

for x in range( 0, 16 ):
    input,output = read_from_file( "data/final_test_data.csv", x )
    if len(input) == 0:
        continue
    model.evaluate( x=input, y=output, batch_size=32 )

    predictions = model.predict( x=input )


    cutoff = 0.5
    print( "x, cutoff, true_pos, true_neg, false_pos, false_neg" )
    while cutoff > 0.09:
        true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, cutoff )
        print( x, '{:02.1f}'.format(cutoff), true_pos, true_neg, false_pos, false_neg )
        cutoff -= 0.1

    golden_cutoff = 1.0
    for i in range( 0, len( predictions ) ):
        if output[i] == 1 and predictions[i] < golden_cutoff:
            golden_cutoff = predictions[i]
    true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, golden_cutoff )
    print( x, golden_cutoff, true_pos, true_neg, false_pos, false_neg, (true_neg/(0.0+true_pos+true_neg+false_pos+false_neg)))

    mod = golden_cutoff + 0.05
    true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, mod )
    print( x, mod, true_pos, true_neg, false_pos, false_neg, (true_neg/(0.0+true_pos+true_neg+false_pos+false_neg)))
    
    mod = golden_cutoff * 1.25
    true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, mod )
    print( x, mod, true_pos, true_neg, false_pos, false_neg, (true_neg/(0.0+true_pos+true_neg+false_pos+false_neg)))

#############
# SPIN DOWN #
#############

model.save( "model.h5" )
