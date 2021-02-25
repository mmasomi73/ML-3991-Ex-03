import datetime
import os

from DTWHandler import DTWHandler
from Extractor import Extractor
import pandas as pd
import matplotlib.pyplot as plt
from FastDTWHandler import FastDTWHandler


class Evaluator:
    def __init__(self, dc, log=False):
        self.dc = dc
        self.log = log
        self.path = '../outs/Evaluator'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def preprocessTimer(self):
        times = []

        # --------= Canny Filter
        if self.log:
            print('\t+----------------------------------------------')
            print('\t| Pre-processing Timer for [Canny] Filter')
        extactor = Extractor(self.dc, 'canny')
        st_te_time = datetime.datetime.now().timestamp()
        extracted_test = extactor.getXTestEdgeVector()
        ed_te_time = datetime.datetime.now().timestamp()
        st_tr_time = datetime.datetime.now().timestamp()
        extracted_train = extactor.getXTrainEdgeVector()
        ed_tr_time = datetime.datetime.now().timestamp()
        times.append([
            'Canny',
            round(ed_tr_time - st_tr_time, 2),
            round(ed_te_time - st_te_time, 2)])
        if self.log:
            print('\t| [Canny] Train Data Pre-Process Time: {}s'.format(round(ed_tr_time - st_tr_time, 2)))
            print('\t| [Canny] Test  Data Pre-Process Time: {}s'.format(round(ed_te_time - st_te_time, 2)))

        # --------= Bounded Tracing Filter
        if self.log:
            print('\t+----------------------------------------------')
            print('\t| Pre-processing Timer for [Bounded Tracing] Filter')
        extactor = Extractor(self.dc, 'bnd_tracer')
        st_te_time = datetime.datetime.now().timestamp()
        extracted_test = extactor.getXTestEdgeVector()
        ed_te_time = datetime.datetime.now().timestamp()
        st_tr_time = datetime.datetime.now().timestamp()
        extracted_train = extactor.getXTrainEdgeVector()
        ed_tr_time = datetime.datetime.now().timestamp()
        times.append([
            'Bounded Tracing',
            round(ed_tr_time - st_tr_time, 2),
            round(ed_te_time - st_te_time, 2)])
        if self.log:
            print('\t| [Bounded Tracing] Train Data Pre-Process Time: {}s'.format(round(ed_tr_time - st_tr_time, 2)))
            print('\t| [Bounded Tracing] Test  Data Pre-Process Time: {}s'.format(round(ed_te_time - st_te_time, 2)))

        # --------= Erode Filter
        if self.log:
            print('\t+----------------------------------------------')
            print('\t| Pre-processing Timer for [Erode] Filter')
        extactor = Extractor(self.dc, 'erode')
        st_te_time = datetime.datetime.now().timestamp()
        extracted_test = extactor.getXTestEdgeVector()
        ed_te_time = datetime.datetime.now().timestamp()
        st_tr_time = datetime.datetime.now().timestamp()
        extracted_train = extactor.getXTrainEdgeVector()
        ed_tr_time = datetime.datetime.now().timestamp()
        times.append([
            'Erode',
            round(ed_tr_time - st_tr_time, 2),
            round(ed_te_time - st_te_time, 2)])
        if self.log:
            print('\t| [Erode] Train Data Pre-Process Time: {}s'.format(round(ed_tr_time - st_tr_time, 2)))
            print('\t| [Erode] Test  Data Pre-Process Time: {}s'.format(round(ed_te_time - st_te_time, 2)))

        # --------------= Write Output
        file_name = self.path + '/preprocessTiming.csv'
        file = open(file_name, "w+")
        for row in times:
            file.write('Filter: ,{}\n'.format(row[0]))
            file.write(',Train Data Pre-Process (s): ,{}\n'.format(row[1]))
            file.write(',Test  Data Pre-Process (s): ,{}\n'.format(row[2]))
        file.close()

    def timeEstimator(self):

        times = []
        train_size = len(self.dc.X_train)
        test_size = len(self.dc.X_test)

        self.dc.X_train = self.dc.X_train[0:100]
        self.dc.Y_train = self.dc.Y_train[0:100]
        self.dc.X_test = self.dc.X_test[0:1]
        self.dc.Y_test = self.dc.Y_test[0:1]
        extactor = Extractor(self.dc, filter)
        extracted_test = extactor.getXTestEdgeVector()
        extracted_train = extactor.getXTrainEdgeVector()
        prepared_model = self.modelCreator(extracted_train, self.dc.Y_train)

        test_data = self.dataCreator(extracted_test)
        test_data = test_data
        test_label = self.dc.Y_test

        # --------------= dtwHandler
        dtwHandler = DTWHandler(prepared_model, test_data, test_label)
        dtwHandler.fit()
        dtwHandler.predict()
        train = dtwHandler.trainTime()
        test = dtwHandler.testTime()
        sub_total_time = train + test

        full_knn_dtw_times = []
        fast_knn_dtw_times = []
        number_of_ins = []

        for i in range(100):
            full_knn_dtw_time = train_size * (test_size * (i + 1) / 100) * sub_total_time / 100
            fast_knn_dtw_time = 1100 * (test_size * (i + 1) / 100) * sub_total_time / 100
            full_knn_dtw_times.append(full_knn_dtw_time)
            fast_knn_dtw_times.append(fast_knn_dtw_time)
            number_of_ins.append((test_size * (i + 1) / 100))

        full_knn_dtw_series = pd.Series(full_knn_dtw_times)
        fast_knn_dtw_series = pd.Series(fast_knn_dtw_times)
        plt.rcParams["font.family"] = "Times New Roman"
        csfont = {'fontname': 'Times New Roman'}

        plt.figure(figsize=(8, 6))

        plt.plot(number_of_ins, full_knn_dtw_times, label="Full KNN DTW")
        plt.plot(number_of_ins, fast_knn_dtw_times, label="Fast KNN DTW")

        plt.xlabel(' Number of Test Data', **csfont)
        plt.ylabel('Time (s)', **csfont)
        plt.title('Time Estimation (s) by # of Instance', **csfont)
        plt.legend(loc='upper left')
        plt.savefig(self.path + '/time-esti.png', format='png')
        plt.close('all')
        print("\t+----------------------------------------------")
        print("\t| Final Time Estimation ")
        print("\t| Full KNN DTW Time is {} s  ".format(round(full_knn_dtw_series.iloc[-1], 2)))
        print("\t| Fast KNN DTW Time is {} s  ".format(round(fast_knn_dtw_series.iloc[-1], 2)))

    def modelCreator(self, extracted_data, labels):
        row_wise_data = []
        for i in range(len(extracted_data)):
            data = extracted_data[i]
            row_wise = []
            for itm in data:
                row_wise.append(itm[0] * 32 + itm[1])
            row_wise_data.append([row_wise, labels[i]])
        return row_wise_data

    def dataCreator(self, extracted_data):
        row_wise_data = []
        for i in range(len(extracted_data)):
            data = extracted_data[i]
            row_wise = []
            for itm in data:
                row_wise.append(itm[0] * 32 + itm[1])
            row_wise_data.append(row_wise)
        return row_wise_data
