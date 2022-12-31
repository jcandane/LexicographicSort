import numpy as np 
import time
from numba import njit

π = np.pi 

@njit ## for numba (comment out if not needed)
def L_binarysearch(A, T, L=0, R=None):
    """
    np.searchsorted has no contraints, i.e. L nor R
    GIVEN:  A (1d sorted numpy.array)
            T (searched for entry)
            *L (lowest  index to search for)
            *R (highest index to search for)
    GET:    L (left-(lowest) most index of entry)
    """
    if R is None:
        R = len(A)
    while L < R:
        m = (L + R) // 2
        if A[m] < T:
            L = m + 1
        else:
            R = m
    return L

@njit ## for numba (comment out if not needed)
def R_binarysearch(A, T, L=0, R=None):
    """
    np.searchsorted has no contraints, i.e. L nor R
    GIVEN:  A (1d sorted numpy.array)
            T (searched for entry)
            *L (lowest  index to search for)
            *R (highest index to search for)
    GET:    R (right-(highest) most index of entry)
    """
    if R is None:
        R = len(A)
    while L < R:
        m = (L + R) // 2
        if A[m] > T:
            R = m
        else:
            L = m + 1
    return R

@njit ## for numba (comment out if not needed)
def interval_binarysearch(A, value, L=0, R=None):
    return L_binarysearch(A, value, L=L, R=R), R_binarysearch(A, value, L=L, R=R)

@njit ## for numba (comment out if not needed)
def tuplebsearch_interval(B, value, L=0, R=None):
    for columns in range(B.shape[0]): ### over entries/columns of the tuple
        L, R = interval_binarysearch(B[columns,:], value[columns], L=L, R=R)
    return L, R

def linear_domain(A):
    """
    GIVEN:  A (sorted 1d numpy array)
    GET:    domains (sorted 1d numpy array, ranges of unique elements)
    """
    domain = np.where( np.diff( A , prepend=-1, append=len(A)) != 0)[0]
    return domain

def log_domain(A):
    """
    GIVEN:  A (sorted 1d numpy array)
    GET:    domains (sorted 1d numpy array, ranges of unique elements)
    """
    domain = [0]
    while domain[-1]!=len(A):
        domain.append( R_binarysearch(A, A[domain[-1]], L=domain[-1], R=None) )
    return np.asarray(domain)

def domain_finder(A):
    """
    GIVEN:  A (sorted 1d numpy array)
    GET:    domains (sorted 1d numpy array, ranges of unique elements)
    """
    if (A[-1]-A[0]) < len(A).bit_length() - 1:
        return log_domain(A) # log_domain_finder(A)
    else:
        return linear_domain(A) # linear_domain_finder(A)

def ssort(A, return_unique=False):
    """
    GIVEN:  A (2d numpy array of shape (tuple, list))
            *return_unique (boolean, choice to output indices and uniques)
    GET:    A lexicographically-sorted along 2nd (list) index.
    """

    ARG = np.argsort(A[0] , kind="mergesort")
    A   = A[:,ARG]
    
    ikk = np.array([], dtype=np.int32)
    for k in range(1,len(A)):
        ik = np.where( np.diff(A[k-1], prepend=A[k-1,0]-1, append=A[k-1,-1]+1) != 0)[0]
        ik = np.union1d(ikk, ik)
        for l in range(len(ik)-1): ## For
            argssort = np.argsort(A[k,ik[l]:ik[l+1]] , kind="mergesort")
            A[:,ik[l]:ik[l+1]] = (A[:,ik[l]:ik[l+1]])[:,argssort]
            ARG[ik[l]:ik[l+1]] = (ARG[ik[l]:ik[l+1]])[argssort]
        ikk = 1*ik

    k = len(A)-1 ## 8/30
    if return_unique:
        return A, ARG, ik
    return A

@njit
def tuple_comparison(A,B): ## A < B?
    for ba in (B-A):
        if ba > 0: ## find the first nonzero positive entry
            return True
    return False


