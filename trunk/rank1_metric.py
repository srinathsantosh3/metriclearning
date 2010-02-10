#!/usr/bin/env python
#coding=utf-8

import math
import numpy
import operator
import scipy.linalg

from abstract_classifier import Classifier
#from metric_learning import *
from decision_stump import *


class Rank1_Metric(Classifier):
    def __init__(self):
        Classifier.__init__(self)

    def train(self):
        X = self.X
        Y = self.Y
        w = self.weights
        self.dim = len(X[0])
        MD = numpy.zeros((self.dim,self.dim))
        MS = numpy.zeros((self.dim,self.dim))
        M = {1:MS, -1:MD}
        for v, z, ww in zip(X,Y,w):
            M[z]+=ww*numpy.outer(v,v)
        ew, ev = scipy.linalg.eig(MD,MS)

        sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
        selected_ind = sorted_pairs[0][0]
        #print sorted_pairs[0][1],sorted_pairs[1][1]
        self.discriminant_vector = ev[:,selected_ind]
        p = self.discriminant_vector
        x = [numpy.dot(p, v)**2 for v in X]
        self.stump = build_stump_1d(x,Y,w)


    def predict(self,X):
        X = numpy.array(X)
        N,d = X.shape
        threshold = self.stump.threshold
        s = self.stump.s
        Y = numpy.zeros(N)
        p = self.discriminant_vector
        pX = numpy.dot(X,p)**2
        Y[numpy.where(pX<threshold)[0]] = -1*s
        Y[numpy.where(pX>=threshold)[0]] = 1*s
        return Y






