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

aa_indices = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y"
]

def onehot_to_name1(aa_in1):
    for i in range( 0, 20 ):
        if aa_in1[ i ] == 1:
            return aa_indices[ i ]


def measure_cutoff( predictions, output, cutoff, aa_in ):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    
    for i in range( 0, len( predictions ) ):
        if predictions[i] < cutoff: #predict negative
            if output[i] == 0: # true negative
                true_neg += 1
            else:
                name1=onehot_to_name1(aa_in[ i ])
                if name1 == "W" or name1 == "F" or name1 == "G" or name1 == "P" or name1 == "A":
                    pass
                    #we'll just let these slide
                    #print( onehot_to_name1(aa_in[ i ]), predictions[i] )
                else:
                    false_neg += 1
        else: #predict positive
            if output[i] == 0: # false positive
                false_pos += 1
            else:
                true_pos += 1
    return true_pos,true_neg,false_pos,false_neg

input, aa_in, output = read_from_file( "data/final_test_data.first_block.500000.csv" )
model.evaluate( x=[input,aa_in], y=output, batch_size=32 )

predictions = model.predict( x=[input,aa_in] )

cutoff = 0.2
print( "cutoff, true_pos, true_neg, false_pos, false_neg, fraction of work prevented, fraction of good AAs lost" )
true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, cutoff, aa_in )
print( '{:02.2f}'.format(cutoff), true_pos, true_neg, false_pos, false_neg, 0.75*((true_neg+false_neg)/(0.0+true_pos+true_neg+false_pos+false_neg)), (false_neg / (0.0+true_pos+false_neg)) )

'''
golden_cutoff = 1.0
for i in range( 0, len( predictions ) ):
    if output[i] == 1 and predictions[i] < golden_cutoff:
        golden_cutoff = predictions[i]
true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, golden_cutoff )
print( golden_cutoff, true_pos, true_neg, false_pos, false_neg, 0.75*((true_neg+false_neg)/(0.0+true_pos+true_neg+false_pos+false_neg)), (false_neg / (0.0+true_pos+false_neg)) )

mod = golden_cutoff + 0.05
true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, mod )
print( mod, true_pos, true_neg, false_pos, false_neg, 0.75*((true_neg+false_neg)/(0.0+true_pos+true_neg+false_pos+false_neg)), (false_neg / (0.0+true_pos+false_neg)) )

mod = golden_cutoff * 1.25
true_pos,true_neg,false_pos,false_neg = measure_cutoff( predictions, output, mod )
print( mod, true_pos, true_neg, false_pos, false_neg, 0.75*((true_neg+false_neg)/(0.0+true_pos+true_neg+false_pos+false_neg)), (false_neg / (0.0+true_pos+false_neg)) )
'''

#############
# SPIN DOWN #
#############

model.save( "model.h5" )
