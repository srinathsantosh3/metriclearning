#!/usr/bin/env python
#coding=utf-8

import math, copy, random
import operator


#from numpy import array, int16, bool, int32, sort, where, hstack
from numpy import *
from numpy.matlib import repmat, repeat

import scipy.linalg

import distance_utils

import heapq

def create_equivalence(y):
    n = len(y)
    return [(i,j,y[i]==y[j]) for i in xrange(n) for j in xrange(i+1,n)]







class TrainingData(object):
    def __init__(self, X, y):
        X = array(X)
        self.X = X
        self.y = y
        N,d = X.shape
        assert(len(y)==N)
        self.N = N
        self.dim = d
        #self.compute_distance_matrix()

        self.trunk = create_equivalence(y)
        self.positive_pool = [(i,j) for i,j,z in self.trunk if z]
        self.negative_pool = [(i,j) for i,j,z in self.trunk if not z]

        """
        self.dim = X.shape[1]

        self.training_pool = [(X[i]-X[j],2*z-1) for i,j,z in self.trunk]
        pool = self.training_pool
        self.positive_pool = [(x,y) for x,y in pool if y]
        self.negative_pool = [(x,y) for x,y in pool if not y]
        """

        """
        N = len(self.Y)
        weights = (1.0/N)*numpy.ones(N)
        self.weights = weights
        """

    def compute_distance_matrix(self):
        X = self.X
        y = self.y
        self.EDM = distance_utils.calcDistanceMatrixFastEuclidean(X)
        N,d = X.shape
        ind = repmat(array(range(N)),N,1)
        self.EDM.setfield(ind, dtype=int16)
        Y = repmat(y, N, 1)
        flag = (Y==Y.T)
        self.EDM.setfield(flag, dtype=bool, offset=2)

    def create_pairs(self, k):
        #if self.EDM is None:
            #todo: The truth value of an array with more than one element is ambiguous.
            #Use a.any() or a.all()
            #self.compute_distance_matrix()

        o = sort(self.EDM) # sort along rows
        ind = o.getfield(int16)
        flag = o.getfield(bool,offset=2)
        k_nearest_ind = ind[:,1:k+1]
        k_nearest_flag = flag[:,1:k+1]
        k_nearest = (k_nearest_ind, k_nearest_flag)

        k_farest_ind = ind[:,-1:-k-1:-1]
        k_farest_flag = flag[:,-1:-k-1:-1]
        k_farest = (k_farest_ind, k_farest_flag)

        return k_nearest, k_farest

    def select_samples(self, k_nearest, k_farest):
        # todo: remove duplicate samples
        k_nearest_ind, k_nearest_flag = k_nearest
        ind = where(k_nearest_flag==False)
        ii = ind[0]
        kk = k_nearest_ind[ind]
        self.negative_samples = zip(ii,kk)

        k_farest_ind, k_farest_flag = k_farest
        ind = where(k_farest_flag==True)
        ii = ind[0]
        jj = k_farest_ind[ind]
        self.positive_samples = zip(ii,jj)

    def filter_samples(self, k_nearest, k_farest):
        self.positive_samples = self.positive_pool
        self.negative_samples = self.negative_pool

        k_nearest_ind, k_nearest_flag = k_nearest
        N,k = k_nearest_ind.shape
        ind = where(k_nearest_flag==True)
        ii = ind[0]
        jj = k_nearest_ind[ind]
        for i,j in zip(ii,jj):
            if sum(ii==i)==k and (i,j) in self.positive_samples:
                self.positive_samples.remove((i,j))

        k_farest_ind, k_farest_flag = k_farest
        ind = where(k_farest_flag==False)
        ii = ind[0]
        jj = k_farest_ind[ind]
        for i,j in zip(ii,jj):
            if (i,j) in self.negative_samples:
                self.negative_samples.remove((i,j))


    """
    def create_triplets(self, k_nearest, k_flag):
        N,k = k_nearest.shape
        triplets = []
        for i in range(N):
            flags = k_flag[i]
            jj = k_nearest[i][where(flags==True)[0]]
            kk = k_nearest[i][where(flags==False)[0]]
            triplets += [(i,j,k) for j in jj for k in kk]
        return triplets
    """
    def get_training_sample(self):
        pairs = self.positive_pool, self.negative_pool
        return create_training_sample(self.X, pairs)


def create_training_sample(X, pairs):
    pos, neg = pairs
    D = array([X[i]-X[j] for i,j in pos] + [X[i]-X[j] for i,j in neg])
    y = hstack([ones(len(pos),dtype=int8),-1*ones(len(neg),dtype=int8)])
    return D,y  #todo: from pairs to distance matrix and new trunk






