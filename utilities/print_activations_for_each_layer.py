import jack_mouse_test

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from random import shuffle

from matplotlib import pyplot as plt
from matplotlib.pyplot import *

#from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics

import tensorflow.keras.losses

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow.keras.callbacks
import tensorflow.keras
import numpy

from tensorflow.keras import models

import sys
#sys.path.append("/nas/longleaf/home/jackmag")#for h5py
import h5py

import pandas as pd
import gzip
import math

import argparse
import random

#import threading
import time
import subprocess

import scipy
import scipy.stats

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

import tensorflow as tf
#from tensorflow.keras.backend.tensorflow_backend import set_session

'''
#Only use part of the GPU, from https://github.com/keras-team/keras/issues/4161
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.3
#config.gpu_options.visible_device_list = "0"
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

'''

########
# INIT #
########

num_input_dimensions = 18494
num_source_residue_inputs = 26
num_ray_inputs = 18494 - 26
WIDTH = 36
HEIGHT = 19
CHANNELS = 27
if( WIDTH * HEIGHT * CHANNELS != num_ray_inputs ):
    print( "WIDTH * HEIGHT * CHANNELS != num_ray_inputs" )
    exit( 1 )
num_output_dimensions = 1

numpy.random.seed( 0 )

parser = argparse.ArgumentParser()

parser.add_argument( "--model", help="Most recent model file", required=True )

parser.add_argument( "--prefix", help="Prefix for PDFs", default="" )

parser.add_argument( "--data", help="CSV where each line has two elements. First element is the absolute path to the input csv file, second element is the absolute path to the corresponding output csv file.", required=True )
# Example: "--data foo.csv" where foo.csv looks like:
# /home/jack/input.1.csv,/home/jack/output.1.csv
# /home/jack/input.2.csv,/home/jack/output.2.csv
# /home/jack/input.3.csv,/home/jack/output.3.csv
# ...

args = parser.parse_args()

def get_vmin():
    return None

def get_vmax():
    return None

#########
# FUNCS #
#########

def hello_world():
    x1 = [ 1., 2., 4. ]
    y1 = [ 2., 4., 6. ]

    x2 = [ 1., 2., 4. ]
    y2 = [ 0., 3., 3. ]

    print( scipy.stats.pearsonr( x1, y1 )[ 0 ] )
    print( scipy.stats.pearsonr( x2, y2 )[ 0 ] )

    print( scipy.stats.spearmanr( x1, y1 ).correlation )
    print( scipy.stats.spearmanr( x2, y2 ).correlation )

    print( scipy.stats.kendalltau( x1, y1 ).correlation )
    print( scipy.stats.kendalltau( x2, y2 ).correlation )


def my_assert_equals( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        exit( 1 )

class AssertError(Exception):
    pass

def my_assert_equals_thrower( name, actual, theoretical ):
    if actual != theoretical:
        print( str( name ) + " is equal to " + str( actual ) + " instead of " + str( theoretical ) )
        raise AssertError

#https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

def assert_vecs_line_up( input, output ):
    #Each line starts with "RESID: XXX,"
    my_assert_equals_thrower( "input file length", len( input ), len( output ) )
    for i in range( 0, len( input ) ):
        in_elem = input[ i ][ 0 ]
        in_resid = int( in_elem.split( " " )[ 1 ] )
        out_elem = output[ i ][ 0 ]
        out_resid = int( out_elem.split( " " )[ 1 ] )
        my_assert_equals_thrower( "out_resid", out_resid, in_resid )

def generate_data_from_files( filenames_csv, six_bin ):
    #dataset = numpy.genfromtxt( filename, delimiter=",", skip_header=0 )
    split = filenames_csv.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    #t0 = time.time()

    # Both of these elements lead with a dummy
    if split[ 0 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = pd.read_csv( f, header=None ).values
        f.close()
    elif split[ 0 ].endswith( ".csv" ):
        input = pd.read_csv( split[ 0 ], header=None ).values
    elif split[ 0 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 0 ], "r" )
        input = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 0 ].endswith( ".npy" ):
        input = numpy.load( split[ 0 ], allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + split[ 0 ] )
        exit( 1 )

    if split[ 1 ].endswith( ".csv.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = pd.read_csv( f, header=None ).values
        f.close()
    elif split[ 1 ].endswith( ".csv" ):
        output = pd.read_csv( split[ 1 ], header=None ).values
    elif split[ 1 ].endswith( ".npy.gz" ):
        f = gzip.GzipFile( split[ 1 ], "r" )
        output = numpy.load( f, allow_pickle=True )
        f.close()
    elif split[ 1 ].endswith( ".npy" ):
        output = numpy.load( split[ 1 ], allow_pickle=True )
    else:
        print ( "We cannot open this file format: " + split[ 1 ] )
        exit( 1 )

    assert_vecs_line_up( input, output )

    source_input_no_resid = input[:,1:27]
    ray_input_no_resid = input[:,27:]

    #print( "output.shape:" )
    #print( output.shape )
    output_no_resid = output[:,1:2]
    #print( "output_no_resid.shape:" )
    #print( output_no_resid.shape )

    my_assert_equals_thrower( "len( source_input_no_resid[ 0 ] )", len( source_input_no_resid[ 0 ] ), num_source_residue_inputs );
    my_assert_equals_thrower( "len( ray_input_no_resid[ 0 ] )",    len( ray_input_no_resid[ 0 ] ), num_ray_inputs );

    #https://www.kaggle.com/vishwasgpai/guide-for-creating-cnn-model-using-csv-file        

    if six_bin:
        six_bin_output_no_resid = output_no_resid.copy()
        new_shape = ( output_no_resid.shape[ 0 ], num_output_dimensions )
        six_bin_output_no_resid.resize( new_shape )
        for x in range( 0, len( output_no_resid ) ):
            my_assert_equals_thrower( "len(six_bin_output_no_resid[ x ])", len( six_bin_output_no_resid[ x ] ), num_output_dimensions )
            six_bin_output_no_resid[ x ][ 0 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -7.0 else 0.0
            six_bin_output_no_resid[ x ][ 1 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -5.0 else 0.0
            six_bin_output_no_resid[ x ][ 2 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -3.0 else 0.0
            six_bin_output_no_resid[ x ][ 3 ] = 1.0 if output_no_resid[ x ][ 0 ] <= -1.0 else 0.0
            six_bin_output_no_resid[ x ][ 4 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 1.0  else 0.0
            six_bin_output_no_resid[ x ][ 5 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 3.0  else 0.0
            six_bin_output_no_resid[ x ][ 6 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 5.0  else 0.0
            six_bin_output_no_resid[ x ][ 7 ] = 1.0 if output_no_resid[ x ][ 0 ] <= 7.0  else 0.0
        return source_input_no_resid, ray_input_no_resid, six_bin_output_no_resid
    else:
        my_assert_equals_thrower( "len( output_no_resid[ 0 ] )", len( output_no_resid[ 0 ] ), num_output_dimensions );
        for x in range( 0, len( output_no_resid ) ):
            my_assert_equals_thrower( "len(output_no_resid[x])", len(output_no_resid[x]), 1 )
            val = output_no_resid[x][0]
            #Stunt large values
            if( val > 1 ):
                val = val**0.75
            #subtract mean of -2:
            val += 2.0
            #divide by span of 3:
            val /= 3.0
        return source_input_no_resid, ray_input_no_resid, output_no_resid

def denormalize_val( val ):
    #print( "denromalizing ", val, " to ", (math.exp( math.exp( val + 1 ) ) - 10) )
    return math.exp( math.exp( val + 1 ) ) - 10;

#########
# START #
#########

if os.path.isfile( args.model ):
    model = load_model( args.model )
else:
    print( "Model " + args.model + " is not a file" )
    exit( 1 )

def display4( pred, filename ):
    plt.close('all')
    plt.axis('off')
    images_per_row = 5
    n_features = pred.shape[-1]
    cols=int(5)
    rows = int( int( n_features + cols - 1 ) / int( cols ) )
    figure()
    print( n_features, cols, rows )
    for i in range( 0, n ):
        ax = subplot( rows, cols, i + 1)
        ax.axis('off')
        matshow( pred[0, :, :, i], cmap='viridis', fignum=False, vmin=get_vmin(), vmax=get_vmax())
    #Good for 2,4,6,8
    #plt.subplots_adjust(wspace=-0.8, hspace=0.1)
    #Good for 10, 13
    #plt.subplots_adjust(wspace=-0.2, hspace=0.1)
    #Good for 12
    #plt.subplots_adjust(wspace=-0.9, hspace=0.1)

    #layer 6 movies hefty
    plt.subplots_adjust(wspace=-0.95, hspace=0.1)
    plt.savefig( filename, bbox_inches='tight' )

# 4) Fit Model

with open( args.data, "r" ) as f:
    file_lines = f.readlines()

layer_outputs = [layer.output for layer in model.layers]
activation_model = models.Model( inputs=model.input, outputs=layer_outputs )

layer_names = []
for layer in model.layers[:12]:
    layer_names.append( layer.name )

for line in file_lines:
    #print( "reading " + str( line ) )
    split = line.split( "\n" )[ 0 ].split( "," );
    my_assert_equals_thrower( "split.length", len( split ), 2 );

    try:
        cpp_structs = jack_mouse_test.read_mouse_data( split[ 0 ], split[ 1 ] )
        source_input = cpp_structs[ 0 ]
        ray_input = cpp_structs[ 1 ]
        output = cpp_structs[ 2 ]
        #source_input, ray_input, output = generate_data_from_files( line, False )
    except AssertError:
        continue


    print( source_input.shape )
    print( ray_input.shape )

    #source_input = source_input[0:1]
    #ray_input = ray_input[0:1]

    #source_input = source_input[24:25]
    #ray_input = ray_input[24:25]

    #51 is good
    guy = 51
    guy2 = guy+1

    source_input = source_input[guy:guy2]
    ray_input = ray_input[guy:guy2]

    '''
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][0] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][1] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][2] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][3] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][4] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][5] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][6] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][7] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][8] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][9] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][10] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][11] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][12] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][13] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][14] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][15] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][16] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][17] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][18] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][19] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][20] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][21] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][22] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][23] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][24] = 0.0
    print( model.predict( x=[source_input,ray_input] ) )
    source_input[0][25] = 0.0
    '''

    print( source_input.shape )
    print( ray_input.shape )
    #exit( 0 )

    #if len( source_input ) > 1:
    #    print( "Let's only submit one line at a time for now. Sorry buddy." )

    #https://towardsdatascience.com/visualizing-intermediate-activation-in-convolutional-neural-networks-with-keras-260b36d60d0

    #predictions = model.predict( x=[source_input,ray_input] )
    predictions = activation_model.predict( x=[source_input,ray_input] )
    print( len( layer_names ), len( predictions ) )

    for i in range( 0, len( predictions ) ):
        #name = layer_names[ i ]
        name = "?"
        plt.close('all')
        pred = predictions[ i ]
        length1 = len( pred.shape )
        if length1 == 4:
            x = pred.shape[ 1 ]
            y = pred.shape[ 2 ]
            n = pred.shape[ 3 ]
            print( name, i, pred.shape, x, y, n )
            '''
            for j in range( 0, n ):
                plt.close('all')
                plt.matshow( pred[0, :, :, j], cmap='viridis', vmin=0.0, vmax=0.1)
                plt.savefig( str(i) + '_' + str(j) + '.pdf' )
            '''
            plt.close('all')
            if len( args.prefix ) > 0:
                display4( pred, args.prefix + "." + str(i) + '.pdf' )
            else:
                display4( pred, str(i) + '.pdf' )
            pass
        elif length1 == 2:
            x = pred.shape[ 1 ]
            y = 1
            n = 1
            print( name, i, pred.shape, x, y, n )
            test = np.zeros(( 1, x ))
            for j in range( 0, x ):
                test[0][j] = pred[0][j]
            plt.matshow( test, cmap='viridis', vmin=get_vmin(), vmax=get_vmax())
            plt.axis('off')
            if len( args.prefix ) > 0:
                plt.savefig( args.prefix + "." + str(i) + '.pdf' )
            else:
                plt.savefig( str(i) + '.pdf' )
        else:
            print( "length1 == ", length1, ", not printing" )
            exit( 1 )
            #print( pred.shape, length1 )

    # print( predictions[ len( predictions ) - 2 ] )
    '''
    np.set_printoptions(threshold=sys.maxsize)
    print( predictions[ 7 ] )
    print( predictions[ 9 ] )
    print( predictions[ 11 ] )
    '''

    '''
    with tf.GradientTape() as tape:
        #tape.watch( model.output )
        tape.watch( model.input )
        a = model( inputs=[source_input,ray_input] );
        print( a )
        #gradients = tape.gradient( model.output, model.input )
        #gradients = tape.gradient( a, model.variables )
        gradients = tape.gradient( a, model.layers[0] )
        #gradients = tape.jacobian( a, model.input )
        
        for i in range( 0, len( gradients ) ):
            grad = gradients[ i ]
            num = 1
            for n in grad.shape:
                num *= n
            print( num, i, grad.shape )
    '''

    #a = model( inputs=[source_input,ray_input] );
    '''
    a = model.predict( x=[source_input,ray_input] );
    print( tf.gradients(model.output, model.input) )
    '''
    
