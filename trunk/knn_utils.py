#!/usr/bin/env python
#coding=utf-8

import sys, heapq
import operator
import numpy


import distance_utils
"""
def top_n(l, n):
    http://stackoverflow.com/questions/1602998/
"""

def score_func(knn):
    def score(y):
        s = 0
        for i,l in enumerate(knn):
            if y==l: s += 1+1.0/(10+i)
        return s
    return score


class KNN(object):
    def __init__(self, k=3, dist_func=''):
        self.k = k
        self.dist_func = dist_func
        pass

    def set_training_data(self, X, y):
        self.training_data = zip(X,y)
        self.training_X = numpy.array(X)
        self.training_y = y

    def compute_distance(self, X):
        f = self.dist_func
        X = numpy.array(X)
        training_X = self.training_X
        # compute the distance matrix from X to training_X
        dist = distance_utils.calcDistanceMatrix2((X, training_X), distFunc = f)
        ## 'dist' is reshaped s.t. dist[i,j] = d(X[i],training_X[j])
        labels = numpy.matlib.repmat(self.training_y, len(X), 1)
        dist.setfield(labels, dtype=labels.dtype)
        # for each x in X, sort the training samples w.r.t the distance to x
        self.sorted_dist = numpy.sort(dist)
        self.sorted_labels = self.sorted_dist.getfield(labels.dtype)

    def classify(self, is_training = False):
        res = []
        ind = int(is_training)
        k = self.k + ind
        for yy in self.sorted_labels[:,ind:k]:
            o = sorted(numpy.unique(yy),key=score_func(yy))[-1]
            res.append(o)
        return res #, sorted_labels







