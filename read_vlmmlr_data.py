import numpy
import pandas as pd

numpy.random.seed( 0 )

aa_indices = {
    "A":0,
    "C":1,
    "D":2,
    "E":3,
    "F":4,
    "G":5,
    "H":6,
    "I":7,
    "K":8,
    "L":9,
    "M":10,
    "N":11,
    "P":12,
    "Q":13,
    "R":14,
    "S":15,
    "T":16,
    "V":17,
    "W":18,
    "Y":19
}

def create_one_hot_input( raw_input, amino_acids ):#DIFF
    aa_input = [];
    for aa in amino_acids:
        values = numpy.zeros( 20 )
        index=int(aa_indices[ aa[0] ])
        values[ index ] = 1.0
        aa_input.append( values )
    aa_input_array = numpy.asarray( aa_input )
    result = numpy.append( raw_input, aa_input_array, axis=1 )
    return result

#normal version:
def read_from_file_NOAA( filename ):
    data = pd.read_csv( filename, header=None ).values
    amino_acids = data[:,0:1]
    input_data = data[:,1:16]
    output = data[:,16:17]
    return input_data, output


#onehot version
def read_from_file( filename ):
    data = pd.read_csv( filename, header=None ).values
    amino_acids = data[:,0:1]
    input_data = data[:,1:16]
    output = data[:,16:17]
    return create_one_hot_input( input_data, amino_acids ), output

def read_from_file_SPLIT( filename ):
    data = pd.read_csv( filename, header=None ).values
    amino_acids = data[:,0:1]
    input_data = data[:,1:16]
    output = data[:,16:17]
    return input_data, create_one_hot_input( input_data, amino_acids ), output
