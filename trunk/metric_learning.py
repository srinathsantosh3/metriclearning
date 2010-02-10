
from numpy import *
from math import *

import operator
import scipy.linalg


def boosted_dist(classifier):
    def distC(x,y):
        return -classifier.predict([x-y])[0]
    return distC


def boosted_dist(classifier):
    def distC(X): # X: N-by-dim
        return -classifier.predict(X) # classifiers accept N-by-dim matrix
    return distC


import cascade
def cascaded_dist(cas):
    MAX_DIST = 9999
    def distC(X): # X: N-by-dim
        dist = ones(X.shape[0])*MAX_DIST
        o = cascade.Predict(cas,X)
        if o: dist[o[0]] = o[1]
        return dist
    return distC



"""
X = loadtxt('trn_X.txt')
y = loadtxt('trn_y.txt',dtype=type(1))
"""


def random_ints(n):
    import random
    for i in range(0, n):
        yield random.randint(0, 10000)



def evaluate(dist, trunk, thres):
    trunk = array(trunk)
    o = sorted(enumerate(dist),key=operator.itemgetter(1))
    sorted_ind = array(o)[:,0]
    sorted_dist = array(o)[:,1]
    trunk = trunk[sorted_ind]
    neg_gt = where(trunk[:,2]==0)[0]
    pos_gt = where(trunk[:,2]==1)[0]
    pos_pred = where(sorted_dist<thres)[0]
    neg_pred = where(sorted_dist>=thres)[0]
    FP = set(pos_pred).intersection(set(neg_gt))
    FN = set(neg_pred).intersection(set(pos_gt))
    FP, FN = array(list(FP)), array(list(FN))
    """
    assert(max(dist[FP])<thres)
    assert(sum(trunk[FP][:,2])==0)
    assert(min(dist[FN])>=thres)
    assert(min(trunk[FN][:,2])==1)
    """
    return FP, FN


def search_threshold(dist, trunk, sensitivity):  # sensitivity = recall = 1-FNr
    trunk = array(trunk)
    o = sorted(enumerate(dist),key=operator.itemgetter(1))
    sorted_ind = array(array(o)[:,0],int32)
    sorted_dist = array(o)[:,1]
    trunk = trunk[sorted_ind]
    neg_gt = where(trunk[:,2]==0)[0]
    pos_gt = where(trunk[:,2]==1)[0]
    n_pos = len(pos_gt)
    expected_miss = n_pos * (1-sensitivity)
    print expected_miss

    l, r = 0, len(sorted_dist)-1
    print l, r
    while l<r-1:
        m = (l+r)/2
        thres = sorted_dist[m]

        pos_pred = where(sorted_dist<thres)[0]
        neg_pred = where(sorted_dist>=thres)[0]

        FP = set(pos_pred).intersection(set(neg_gt))
        FN = set(neg_pred).intersection(set(pos_gt))
        TP = set(pos_pred).intersection(set(pos_gt))

        FP, FN = array(list(FP)), array(list(FN))
        TP = array(list(TP))
        print thres, len(FN), len(FP), (l,r,m)
        if len(FN) > expected_miss: # raise the threshold
            l = m
        else: # otherwise, lower the threshold
            r = m

    if len(TP)+len(FP)==0: return thres, ([],[]), []
    if len(TP)==0: return thres, ([],trunk[FP][:,0:2]), []
    if len(FP)==0: return thres, (trunk[TP][:,0:2],[]), []
    """
    assert(max(dist[FP])<thres)
    assert(sum(trunk[FP][:,2])==0)
    assert(min(dist[FN])>=thres)
    assert(min(trunk[FN][:,2])==1)
    """
    print max(sorted_dist[FP]), min(sorted_dist[FN])
    #return thres, FP, FN
    pos, neg = trunk[TP][:,0:2], trunk[FP][:,0:2]
    trunk_p = c_[pos,ones(len(pos),dtype=int)]
    trunk_n = c_[neg,zeros(len(neg),dtype=int)]
    new_trunk = vstack([trunk_p,trunk_n])
    return thres, (pos,neg), new_trunk


"""
X[hstack([pos[:,0],neg[:,0]])] - X[hstack([pos[:,1],neg[:,1]])]


"""

