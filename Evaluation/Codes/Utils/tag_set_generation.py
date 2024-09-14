import numpy as np
import math
from scipy.special import comb

# Encoding capacity of Polaris
def comb_num(k,m):
    return comb(k*k-4,m-3,exact=True)
def capacity(p, k, m):
    return pow(p, m) * comb_num(k, m)

# Build the tag set
def tag_generation(P, K, M, d):
    # P: the number of polarity orientations
    # K: the number of tag order
    # M: the number of passive magnet
    # d: the distance between two adjacent magnets
    
    cap = capacity(P, K, M)
    tag_set = []
    for i in range(cap):
        
        tag_set.append(np.zeros(K*K))

    return