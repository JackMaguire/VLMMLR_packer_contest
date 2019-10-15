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
parser.add_argument( "--data", help="data to run on", default="test_data.csv", required=False )
args = parser.parse_args()

model = load_model( args.model )


#########
# TRAIN #
#########

input,output = read_from_file( args.data )
predictions = model.predict( x=input )
print( predictions )
