import numpy as np
import os
from scipy.stats import norm
#import ipdb; ipdb.set_trace()
if 'LowDiscrepancy.so' in os.listdir(os.getcwd()):
    pass
else: 
    import numpy.f2py
    numpy.f2py.compile(open('LowDiscrepancy.f', 'rb').read(), modulename='LowDiscrepancy')

import LowDiscrepancy

def test_if_uniform(QN):
    """
    function that tests whether the array is really in 0,1
    return false if any of the values is outside (0,1)
    """
    return((QN<1.).all() and (QN>0.).all())

def error_dim(DIMEN):
    """
    due to the fortran code, the dimension cannot exceed 1111
    """
    if DIMEN > 1111:
        raise ValueError('the dimension cannot exceed 1111')
    else: 
        pass

def sobol_sequence(N, DIMEN, IFLAG=0, iSEED=0, INIT=1, TRANSFORM=0):
    """
    Input: 
        N         - NUMBERS OF POINTS TO GENERATE
        DIMEN     - DIMENSION OF THE SEQUENCY
        IFLAG     - INITIALIZATION FLAG
                   0 - NO SCRAMBLING
                   1 - OWEN TYPE SCRAMBLING
                   2 - FAURE-TEZUKA TYPE SCRAMBLING
                   3 - OWEN + FAURE-TEZUKA TYPE SCRAMBLING
        iSEED     - SCRAMBLING iSEED
        INIT      - INITIALIZAYION FLAG, 0 NEXT, 1 RE-INITIALZE
        TRANSFORM - FLAG, 0 FOR UNIFORM, 1 FOR NORMAL DISTRIBUTION
    Output:
        QN        - QUASI NUMBERS, A "N" BY "DIMEN" ARRAY
    """
    error_dim(DIMEN)
    QN = LowDiscrepancy.sobol(N, DIMEN, IFLAG, iSEED, INIT, 0)
    while not test_if_uniform(QN):
        iSEED += 1
        QN = LowDiscrepancy.sobol(N, DIMEN, IFLAG, iSEED, INIT, 0)
        print("error in scrambling, increment seed")
    if TRANSFORM==1:
        return(norm.ppf(QN))
    else: 
        return(QN)


def halton_sequence(N, DIMEN, INIT=1, TRANSFORM=0):
    """
    Input: 
        N         - NUMBERS OF POINTS TO GENERATE
        DIMEN     - DIMENSION OF THE SEQUENCY
        INIT      - INITIALIZAYION FLAG, 0 NEXT, 1 RE-INITIALZE
        TRANSFORM - FLAG, 0 FOR UNIFORM, 1 FOR NORMAL DISTRIBUTION
    Output:
        QN        - QUASI NUMBERS, A "N" BY "DIMEN" ARRAY
    """
    error_dim(DIMEN)
    QN = LowDiscrepancy.halton(N, DIMEN, INIT, 0)
    if TRANSFORM==1:
        return(norm.ppf(QN))
    else: 
        return(QN)



if __name__ == '__main__':
    import matplotlib 
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    N, DIMEN = 200, 2

    sobol_array = sobol_sequence(N, DIMEN, TRANSFORM=0)
    halton_array = halton_sequence(N, DIMEN, TRANSFORM=0)
    
    plt.scatter(x=sobol_array[:,0], y=sobol_array[:,1])
    plt.scatter(x=halton_array[:,0], y=halton_array[:,1])
    plt.show()
    #import ipdb; ipdb.set_trace()

