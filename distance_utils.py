#!/usr/bin/env python
#coding=utf-8

"""
Calculate the distance matrix for n-dimensional point array
http://code.activestate.com/recipes/498246/
"""
import operator

from numpy import array, dot, sum, zeros, where, ndarray, int32
from numpy.matlib import repmat, repeat
from scipy import reshape, sqrt, identity
import scipy.linalg



def euclidean_dist(x,y):
    return linalg.norm(x-y)


def weighted_dist(A):
    def distA(x,y):
        return dot((x-y),dot(A,(x-y)))
    return distA


def projected_dist(L):
    def distL(x,y):
        return euclidean_dist(dot(L,x), dot(L,y))
    return distL

def projected_dist(L):
    distFunc = lambda delta: sqrt(sum( dot(delta,L)**2, axis=1))
    return distFunc


import unittest

# nDimPoints: list of n-dim tuples
# distFunc: calculates the distance based on the differences
# Ex: Manhatten would be: distFunc=sum(deltaPoint[d] for d in xrange(len(deltaPoint)
def calcDistanceMatrix(nDimPoints,
                       distFunc=lambda deltaPoint: sqrt(sum(deltaPoint[d]**2 for d in xrange(len(deltaPoint))))):
    nDimPoints = array(nDimPoints)
    dim = len(nDimPoints[0])
    delta = [None]*dim
    for d in xrange(dim):
        data = nDimPoints[:,d]
        delta[d] = data - reshape(data,(len(data),1)) # computes all possible combinations

    dist = distFunc(delta)
    dist = dist + identity(len(data))*dist.max() # eliminate self matching
    # dist is the matrix of distances from one coordinate to any other
    return dist

def calcDistanceMatrix2(AB,
                       distFunc=lambda delta: sqrt(sum(delta**2,axis=1))):
    assert(len(AB) in [1,2] and type(AB)!=ndarray)
    if len(AB)==2:
        A,B = AB
        #if (A==B).all(): return calcDistanceMatrix2([A],distFunc)
        #A = array(A)
        #B = array(B)
        nA,dim = A.shape
        assert(B.shape[1]==dim)
        nB = B.shape[0]
        print A.shape,nB,B.shape,nA
        delta = repeat(A,nB,0) - repmat(B,nA,1)
        dist = distFunc(delta).reshape(nA,nB)  # dist[i,j] = d(A[i],B[j])
        del delta
        return dist
    else: # elif len(AB)==1:
        A = array(AB[0])
        nA,dim = A.shape #max nA <= 800
        rows = repeat(range(nA),nA) # 0,0,0,...,n-1,n-1
        cols = array(range(nA)*nA) # 0,1,2
        upper_ind = where(cols>rows)[0]
        # nA == (1+sqrt(1+8*len(upper_ind))/2
        ##lower_ind = where(cols<rows)[0]
        delta = A[rows[upper_ind],:]- A[cols[upper_ind],:]
        del rows
        del cols
        # computes all possible combinations
        #dist = zeros(nA*nA)
        #partial_delta = delta[:,upper_ind]
        partial_dist = distFunc(delta)
        del delta
        partial_dist.setfield(upper_ind, dtype=int32)
        #dist[upper_ind] = partial_dist
        #dist = dist.reshape(nA, nA) # dist[i,j] = d(A[i],A[j]) for i<j
        #dist = dist + dist.T # make it symmetric
        return partial_dist




def calcDistanceMatrixFastEuclidean(points):
    numPoints = len(points)
    dist = (repmat(points, numPoints, 1) - repeat(points, numPoints, axis=0))**2
    distMat = sqrt(sum(dist, axis=1))
    return distMat.reshape((numPoints,numPoints))

from numpy import mat, zeros, newaxis
def calcDistanceMatrixFastEuclidean2(nDimPoints):
    nDimPoints = array(nDimPoints)
    n,m = nDimPoints.shape
    delta = zeros((n,n),'d')
    for d in xrange(m):
        data = nDimPoints[:,d]
        delta += (data - data[:,newaxis])**2
    return sqrt(delta)

#################
# Unittest
#################
class CalcDistanceMatrixTestCase(unittest.TestCase):
    def setUp(self):
        self.distanceMatrixFunc = "calcDistanceMatrix"

    def test_2D(self):
        points = [[0, 0], [1, 1], [4, 5]]
        dm = eval("%s(points)"%self.distanceMatrixFunc)
        self.assertAlmostEqual(1.414213562373095049, dm[0][1])
        self.assertAlmostEqual(6.403124237432848686, dm[0][2])
        self.assertAlmostEqual(5, dm[1][2])
        self._testSymmetry(dm)

    def test_3D(self):
        points = [[0, 0, 0], [1.0, 1, 1], [4, 5, 6], [10,10,10]]
        dm = eval("%s(points)"%self.distanceMatrixFunc)
        self.assertAlmostEqual(1.732050807568877294, dm[0][1])
        self.assertAlmostEqual(8.77496438739212206, dm[0][2])
        self.assertAlmostEqual(17.32050807568877294, dm[0][3])
        self.assertAlmostEqual(7.071067811865475244, dm[1][2])
        self.assertAlmostEqual(15.58845726811989564, dm[1][3])
        self.assertAlmostEqual(8.77496438739212206, dm[2][3])
        self._testSymmetry(dm)

    def _testSymmetry(self, dm):
        for i in range(len(dm)):
            for j in range(len(dm)):
                self.assertEqual(dm[i][j], dm[j][i])

class CalcDistanceMatrixFastTestCase(CalcDistanceMatrixTestCase):
    def setUp(self):
        self.distanceMatrixFunc = "calcDistanceMatrixFastEuclidean"

class CalcDistanceMatrixFast2TestCase(CalcDistanceMatrixTestCase):
    def setUp(self):
        self.distanceMatrixFunc = "calcDistanceMatrixFastEuclidean2"


