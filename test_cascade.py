"""
class CascadeClassifier(Classifier):
    def __init__(self, base_classifier_type):
        self.base_classifier_type = base_classifier_type

    def new_classifier(self, *args):
        return self.base_classifier_type(*args)

    def config(self, max_level):
        pass

    def set_training_pool(self, X, Y):
        #self.X = copy.copy(X)
        #self.Y = copy.copy(Y)
        self.training_data = shuffle(zip(X,Y))
        pool = self.training_data
        self.positive_pool = [(x,y) for x,y in pool if y]
        self.negative_pool = [(x,y) for x,y in pool if not y]
        self.pos_start_ind = 0
        self.pos_remain = len(self.positive_pool)
        self.neg_start_ind = 0
        self.neg_remain = len(self.negative_pool)

    def sample_with_no_repl(self, n_pos, n_neg):
        if self.pos_remain > 0:
            pool = self.positive_pool
            ind = self.pos_start_ind
            pos = pool[ind:ind+n_pos]
            self.pos_start_ind += len(pos)
            self.pos_remain -= len(pos)
        else:
            pos = []
        if self.

        self.pos_start_ind
        X = self.positive_pool[0:min(n_pos,n)]

    def train(self, *args):
        pass





def cascade(weak_classifier_type, training_pool, max_level, T, k):
    level = 0
    classifiers = []
    X,Y = [],[]
    while (True):
        sample (with no replacement) from training pool -> X,Y
        Pos = set(numpy.where(Y==1)[0])
        Neg = set(numpy.where(Y==-1)[0])
        classifier = Adaclassifier_type(weak_classifier_type)
        classifier.set_training_sample(X,Y)
        classifier.train(T, k)
        Y_pred = classifier.predict(X)
        for thres in numpy.linspace(1,-1,10): #todo
            FP, FN = classifier.measure_accuracy(Y,o,thres)
            if len(FN)/len(Pos) < min_FNr:
                break
        classifiers.append(classifier, thres)
        if len(FP)/len(FN) < ratio: break
        level += 1
        if level >= max_level: break
        FP, FN -> X, Y (w_FN > w_FP)
"""
