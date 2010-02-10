
from numpy import *

import training_utils, distance_utils
import adaboost, rank1_metric
import metric_learning
import sys

import pickle


def Predict(casc, X):
    N=len(X)
    pind = array(range(N))
    for c, thres, tpfp in casc:
        sys.stdout.write('.')
        d = -c.predict(X)
        #if d>=thres: return -1
        ind = where(d<thres)[0]
        if len(ind)==0: return [] # all classified as negative
        X = X[ind]
        #print pind
        pind = pind[ind]
    return pind,d[ind]


def split_data_label(XY):
    if len(XY)>2:
        X = array([x for x,y in XY])
        Y = array([y for x,y in XY])
    else:
        X,Y = XY
    return X,Y


def predict_tst(casc, tst, trn):
    X,Y = split_data_label(trn)
    tst_X, tst_Y = split_data_label(tst)

    dim = len(X[0])
    labels = list(set(Y))
    print labels
    o = []
    for tx in tst_X:
        sys.stdout.write('.')
        vv = [tx-x for x in X]
        match = [Predict(casc, v.reshape(1,dim)) for v in vv]
        candidate = [y for i,y in zip(match,Y) if i>0]
        o.append(labels[argmax(array([candidate.count(l) for l in labels]))])
    return o




def Cascade(trn, max_level):
    if len(trn)>2:
        X = array([x for x,y in trn])
        Y = array([y for x,y in trn])
    else:
        X,Y = trn
    T = 512
    classifiers = []

    td = training_utils.TrainingData(X,Y)
    SD = distance_utils.calcDistanceMatrix2([X])
    trunk = td.trunk
    sensitivity = 0.99
    o = metric_learning.search_threshold(SD, trunk, sensitivity)
    thres = o[0]
    TP,FP = o[1]
    num_FN = (1-sensitivity)*(len(TP)/sensitivity)
    pairs = TP,FP
    MAX_NUM_POS = 2000
    MAX_NUM_NEG = 4000
    num_pos = min(len(pairs[0]),MAX_NUM_POS)
    num_neg = min(len(pairs[1]),MAX_NUM_NEG)
    print "TP vs FP: %d vs %d"%(len(TP),len(FP))
    trunk = o[2]
    classifiers.append((thres, (TP,FP)))
    FPvsFN = len(FP)/num_FN
    print "#FP/#FN = ",FPvsFN
    #return classifiers

    for level in range(1,max_level+1):
        print "level = %d"%level
        random.shuffle(TP)
        random.shuffle(FP)
        pos = TP[-1:-num_pos-1:-1]
        neg = FP[0:num_neg]
        pairs = pos, neg
        print len(pairs[0]), len(pairs[1])
        XX,yy = training_utils.create_training_sample(X, pairs)
        classifier = adaboost.AdaBoost(rank1_metric.Rank1_Metric)
        classifier.set_training_sample(XX,yy)
        classifier.train(T, 1)
        #classifier

        VD, vy = training_utils.create_training_sample(X, (TP,FP)) # for validation
        SD = -classifier.predict(VD)  # be careful about the sign!!!

        sensitivity = 0.99
        o = metric_learning.search_threshold(SD, trunk, sensitivity)
        thres = o[0]
        TP,FP = o[1]
        num_FN = (1-sensitivity)*(len(TP)/sensitivity)
        print "TP vs FP: %d vs %d"%(len(TP),len(FP))
        trunk = o[2]
        classifiers.append((classifier, thres, (TP,FP)))
        FPvsFN = len(FP)/num_FN
        print "#FP/#FN = ",FPvsFN
        if len(TP)==0 or len(FP)==0 or FPvsFN < 1.2 :  break
    return classifiers


