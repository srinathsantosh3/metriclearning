import rca
import knn_utils, distance_utils, metric_learning, training_utils
import numpy

def load_wine_data():
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    trn_X = numpy.loadtxt('trn_X.txt')
    trn_y = numpy.loadtxt('trn_y.txt',dtype=type(1))

    tst_X = numpy.loadtxt('tst_X.txt')
    tst_y = numpy.loadtxt('tst_y.txt',dtype=type(1))

    trn = trn_X, trn_y
    tst = tst_X, tst_y

    return trn, tst


def compute_accuracy(y1, y2):
    assert(len(y1)==len(y2))
    return len(numpy.where(y1==y2)[0])*1.0/len(y1)

import unittest
class RCA_KNN_TestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testWineData(self):
        trn, tst = load_wine_data()
        trn_X, trn_y = trn
        tst_X, tst_y = tst
        P = rca.compute_RCA(trn_X, trn_y)
        knn = knn_utils.KNN(3)
        knn.dist_func = distance_utils.projected_dist(P)
        knn.set_training_data(trn_X, trn_y)
        # test on training
        knn.compute_distance(trn_X)
        predicted_y = numpy.array(knn.classify(True))
        accuracy_trn = compute_accuracy(predicted_y, trn_y)
        # test on testing
        knn.compute_distance(tst_X)
        predicted_y = numpy.array(knn.classify(False))
        accuracy_tst = compute_accuracy(predicted_y, tst_y)

        print accuracy_trn, accuracy_tst
        self.failUnless(min(accuracy_trn, accuracy_tst) > .95)


import rank1_metric, adaboost
class Rank1AdaBoost_KNN_TestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testWineData(self):
        trn, tst = load_wine_data()
        trn_X, trn_y = trn
        tst_X, tst_y = tst

        td = training_utils.TrainingData(trn_X, trn_y)
        XX,yy = td.get_training_sample()
        classifier = adaboost.AdaBoost(rank1_metric.Rank1_Metric)
        classifier.set_training_sample(XX,yy)
        T = 20
        classifier.train(T,1)
        f = metric_learning.boosted_dist(classifier)

        knn = knn_utils.KNN(3)
        knn.dist_func = f
        knn.set_training_data(trn_X, trn_y)
        # test on training
        knn.compute_distance(trn_X)
        predicted_y = numpy.array(knn.classify(True))
        accuracy_trn = compute_accuracy(predicted_y, trn_y)
        # test on testing
        knn.compute_distance(tst_X)
        predicted_y = numpy.array(knn.classify(False))
        accuracy_tst = compute_accuracy(predicted_y, tst_y)

        print accuracy_trn, accuracy_tst
        self.failUnless(min(accuracy_trn, accuracy_tst) > .95)


def main():
    suite = unittest.TestLoader().loadTestsFromTestCase(RCA_KNN_TestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(Rank1AdaBoost_KNN_TestCase)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    main()






