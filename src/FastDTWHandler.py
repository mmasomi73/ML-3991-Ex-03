import datetime

import numpy as np
from dtw import *
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNNClassifier


class FastDTWHandler:
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

    misclassify = []

    def __init__(self, model, test_x, test_y):
        self.test_x = test_x
        self.test_y = test_y
        self.model = model
        self.test_x = self.filler(test_x)
        self.extract_XY()
        self.test_x = self.filler(test_x)
        n_neighbors = 20

        def DTW_Distance(data_1, data_2):
            data_1 = np.delete(data_1, [-1])
            data_2 = np.delete(data_2, [-1])
            alignment = dtw(data_1, data_2, keep_internals=True)
            return alignment.distance

        self.clf = KNNClassifier(n_neighbors=n_neighbors,
                                 algorithm='auto',
                                 metric=DTW_Distance
                                 # metric_params={"func": DTW_Distance}
                                 )
        data = self.trainDataSeparator()
        self.modelCreator(data)
        data, labels = self.modelExtractor(1)
        self.train_x = data
        self.train_y = labels
        # for key, value in data.items():
        #     print(' len of {} is : {}'.format(key, len(value)))

    def trainDataSeparator(self):
        data = dict()
        for i in range(len(self.train_x)):
            data_list = []
            i_key = str(int(self.train_y[i]))
            if i_key in data:
                data_list = data.get(i_key)
            data_list.append(self.train_x[i])
            data[i_key] = data_list
        return data

    def modelCreator(self, data):
        first_layer = {}
        second_layer = {}

        for key, value in data.items():
            total = np.array(value)
            total[total == -1] = 0

            # --------= first Layer
            first_separation_coefficient = 100
            first = np.split(total, first_separation_coefficient, axis=0)
            first = np.array(first)
            sub_sec_arr = []
            for i in range(first_separation_coefficient):
                # temp_arr = first[i].mean(0)
                index = np.random.choice(first[i].shape[0], 1, replace=False)[0]
                temp_arr = first[i][index]
                temp_arr[temp_arr == 0] = -1
                sub_sec_arr.append(temp_arr)
            first_layer[key] = sub_sec_arr
            # --------= Second Layer
            second_separation_coefficient = 1000
            second = np.split(total, second_separation_coefficient, axis=0)
            second = np.array(second)
            sub_sec_arr = []
            for i in range(second_separation_coefficient):
                # temp_arr = second[i].mean(0)
                index = np.random.choice(second[i].shape[0], 1, replace=False)[0]
                temp_arr = second[i][index]
                temp_arr[temp_arr == 0] = -1
                sub_sec_arr.append(temp_arr)
            second_layer[key] = sub_sec_arr
        self.model = {
            'first': first_layer,
            'second': second_layer
        }

    def modelExtractor(self, layer=1):
        labels = np.array([])
        data = np.array([])
        if layer == 1:
            origin_data = self.model.get('first')
        else:
            origin_data = self.model.get('second')

        for key, value in origin_data.items():
            value = np.array(value)
            partial_labels = [key] * value.shape[0]
            partial_labels = np.array(partial_labels)

            if len(labels) <= 0:
                labels = partial_labels
            if len(data) <= 0:
                data = value

            data = np.concatenate((data, value), axis=0)
            labels = np.concatenate((labels, partial_labels), axis=0)

        return data, labels

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

    def fit(self, train_x, train_y):
        self.st_tr_time = datetime.datetime.now().timestamp()
        self.clf.fit(train_x, train_y)
        self.en_tr_time = datetime.datetime.now().timestamp()

    def predict(self, test_x):
        self.st_te_time = datetime.datetime.now().timestamp()
        self.labels_predict = self.clf.predict(test_x)
        self.en_te_time = datetime.datetime.now().timestamp()
        return self.labels_predict

    def getAccuracy(self):
        if len(self.labels_predict) == 0:
            self.predict(self.test_x)
        for i in range(len(self.labels_predict)):
            if int(self.labels_predict[i]) == int(self.test_y[i]):
                self.currect += 1
            else:
                self.misclassify.append([int(self.labels_predict[i]), int(self.test_y[i])])
        return (self.currect / len(self.test_y)) * 100

    def trainTime(self):
        return self.en_tr_time - self.st_tr_time

    def testTime(self):
        return self.en_te_time - self.st_te_time

    def printResult(self):
        if len(self.labels_predict) == 0:
            self.fit(self.train_x, self.train_y)
            self.predict(self.test_x)
            self.getAccuracy()
        misclassify = pd.DataFrame(self.misclassify)
        misclassify = misclassify.groupby(misclassify.columns.tolist(), as_index=False).size()
        misclassify = misclassify.to_numpy()
        print("\t+-----=[ DTW Result ]=----- ")
        print("\t| Train Data Size    : {}".format(len(self.train_x)))
        print("\t| Test  Data Size    : {}".format(len(self.test_x)))
        print("\t| Train Time         : {}".format(self.trainTime()))
        print("\t| Test  Time         : {}".format(self.testTime()))
        print("\t| Accuracy           : {}".format((self.currect / len(self.test_y)) * 100))
        print("\t| MisClassification : {}".format(((len(self.test_y) - self.currect) / len(self.test_y)) * 100))
        if misclassify.shape[0] > 0:
            print("\t| Most MisClassification ")
            print("\t|\t [Predict - Actual] -> [ # ] ")
            for i in range(misclassify.shape[0]):
                print("\t|\t [   {}   -   {}   ] -> [ {} ] ".format(misclassify[i, 0],
                                                                    misclassify[i, 1],
                                                                    misclassify[i, 2]))
        print("\t------------------------------------------- ")

        file_name = '../outs/outputs.csv'
        file = open(file_name, "w+")
        file.write('Train Data Size,{}\n'.format(len(self.train_x)))
        file.write('Test  Data Size,{}\n'.format(len(self.test_x)))
        file.write('Train Time,{}\n'.format(self.trainTime()))
        file.write('Test  Time,{}\n'.format(self.testTime()))
        file.write('Accuracy,{}\n'.format((self.currect / len(self.test_y)) * 100))
        file.write('MisClassification,{}\n'.format(((len(self.test_y) - self.currect) / len(self.test_y)) * 100))
        if misclassify.shape[0] > 0:
            file.write(",Predict, Actual,#\n")
            for i in range(misclassify.shape[0]):
                file.write(",{},{},{}".format(misclassify[i, 0],
                                              misclassify[i, 1],
                                              misclassify[i, 2]))
        file.close()
