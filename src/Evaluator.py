import datetime
import os

from Extractor import Extractor


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
