import numpy
import pandas as pd

numpy.random.seed( 0 )

def read_from_file( filename ):
    data = pd.read_csv( filename, header=None ).values
    amino_acids = data[:,0:1]
    input_data = data[:,1:16]
    output = data[:,16:17]
    return input_data, output

