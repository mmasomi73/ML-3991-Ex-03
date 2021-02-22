import datetime

import numpy as np
from dtw import dtw
from sklearn.neighbors import KNeighborsClassifier as KNNClassifier

from KnnDtw import KnnDtw


class DTWHandler:
    train_x = []
    train_y = []

    test_x = []
    test_y = []

    clf = object

    st_tr_time = 0
    en_tr_time = 0
    st_te_time = 0
    en_te_time = 0

    currect = 0

    labels_predict = []

    knnDwt = object

    def __init__(self, model, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
        self.model = model
        self.extract_XY()
        self.test_x = self.filler(test_x)
        n_neighbors = 20
        self.knnDwt = KnnDtw()

        def DTW_Distance(data_1, data_2):
            data_1 = np.delete(data_1, [-1])
            data_2 = np.delete(data_2, [-1])
            manhattan_distance = lambda a, b: abs(a - b)
            d, cost_matrix, acc_cost_matrix, path = dtw(data_1, data_2, dist=manhattan_distance)
            return d

        self.clf = KNNClassifier(n_neighbors=n_neighbors,
                                 algorithm='auto',
                                 metric=DTW_Distance
                                 # metric_params={"func": DTW_Distance}
                                 )

    def extract_XY(self):
        self.train_x = []
        for data in self.model:
            self.train_x.append(data[0])
            self.train_y.append(data[1])

        row_lengths = []
        for row in self.train_x:
            row_lengths.append(len(row))
        for row in self.test_x:
            row_lengths.append(len(row))
        max_length = max(row_lengths)

        for row in self.train_x:
            while len(row) < max_length:
                row.append(-1)
        balanced_array = np.array(self.train_x)
        self.train_x = balanced_array

    def filler(self, test_x):
        self.test_x = []
        for data in test_x:
            self.test_x.append(data)

        row_lengths = []
        for row in self.test_x:
            row_lengths.append(len(row))
        for row in self.train_x:
            row_lengths.append(len(row))
        max_length = max(row_lengths)

        for row in self.test_x:
            while len(row) < max_length:
                row.append(-1)
        balanced_array = np.array(self.test_x)
        self.test_x = balanced_array
        return self.test_x

    def fit(self):
        print('\tFit Start ...')
        self.st_tr_time = datetime.datetime.now().timestamp()
        self.clf.fit(self.train_x, self.train_y)
        # self.knnDwt.fit(self.train_x, self.train_y)
        self.en_tr_time = datetime.datetime.now().timestamp()
        print('\tFit Finished ...')

    def predict(self):
        print('\tPredict Start ...')
        self.st_te_time = datetime.datetime.now().timestamp()
        self.labels_predict = self.clf.predict(self.test_x)
        # self.labels_predict = self.knnDwt.predict(self.test_x)
        self.en_te_time = datetime.datetime.now().timestamp()
        print('\tPredict Finished ...')
        print(self.labels_predict)
        return self.labels_predict

    def getAccuracy(self):
        if len(self.labels_predict) == 0:
            self.predict()
        for i in range(len(self.labels_predict)):
            if self.labels_predict[i] == self.test_y[i]:
                self.currect += 1
        return (self.currect / len(self.test_y)) * 100

    def trainTime(self):
        return self.en_tr_time - self.st_tr_time

    def testTime(self):
        return self.en_te_time - self.st_te_time

    def printResult(self):
        if len(self.labels_predict) == 0:
            self.fit()
            self.predict()
        print("\t+-----=[ DTW Result ]=----- ")
        print("\t| Train Time         : {}".format(self.trainTime()))
        print("\t| Test  Time         : {}".format(self.testTime()))
        print("\t| Accuracy           : {}".format((self.currect / len(self.test_y)) * 100))
        print("\t| MissClassification : {}".format(((len(self.test_y) - self.currect) / len(self.test_y)) * 100))
        print("\t------------------------------------------- ")
