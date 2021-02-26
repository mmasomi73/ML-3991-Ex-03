# Persian Handwritten Number Recognition by Sequences Matching -> PHNbSM
# first need to read dataset
import os

import matplotlib.pyplot as plt
import numpy as np
from DTWHandler import DTWHandler
from DataCollector import DataCollector
from Evaluator import Evaluator
from Extractor import Extractor
from FastDTWHandler import FastDTWHandler
from OutputWriter import OutputWriter
from Visualizer import Visualizer
from colorama import Fore


class PHNbSM:
    log = False
    create_output_image = False

    def __init__(self, log=False):
        self.dc = DataCollector(log)
        self.log = log



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

    def executor(self, option, filter='canny'):
        self.create_output_image = True
        if option == 1:
            img_writer = OutputWriter()
            # -----= Train images
            for i in range(len(self.dc.X_train)):
                img_writer.imageWriter(self.dc.X_train[i], self.dc.Y_train[i], 'train')
                if i > 100:
                    break
            # -----= Test images
            for i in range(len(self.dc.X_test)):
                img_writer.imageWriter(self.dc.X_test[i], self.dc.Y_test[i], 'test')
                if i > 100:
                    break
            # -----= Renaming images
            for i in range(len(self.dc.X_naming)):
                img_writer.imageWriter(self.dc.X_naming[i], self.dc.Y_naming[i], 'naming')
                if i > 100:
                    break
        # TODO: Uncomment Train And Namely Lines in Data Collector
        # TODO: Change here Test With Train
        extactor = Extractor(self.dc, filter)
        # extactor = Extractor(self.dc)
        extracted_test = extactor.getXTestEdgeVector()
        extracted_train = extactor.getXTrainEdgeVector()
        prepared_model = self.modelCreator(extracted_train, self.dc.Y_train)

        test_data = self.dataCreator(extracted_test)
        test_data = test_data
        test_label = self.dc.Y_test

        if option == 2:
            dtwHandler = DTWHandler(prepared_model, test_data, test_label)
            dtwHandler.printResult()

        if option == 3:
            dtwHandler = FastDTWHandler(prepared_model, test_data, test_label)
            dtwHandler.printResult()

        if option == 4:
            visualizer = Visualizer(self.dc.X_test, self.dc.Y_test, self.dc)
            visualizer.drawSequences()

        if option == 5:
            visualizer = Visualizer(self.dc.X_test, self.dc.Y_test, self.dc)
            visualizer.drawCompares()

        if option == 6:
            evaluator = Evaluator(self.dc, True)
            evaluator.preprocessTimer()

        if option == 7:
            evaluator = Evaluator(self.dc, True)
            evaluator.timeEstimator()


if __name__ == '__main__':
    phnbsm = PHNbSM(True)
    print(Fore.GREEN + '\t+------------------------------------+')
    print(Fore.GREEN + '\t|' + Fore.BLUE + ' Persian Handwritten Number         ' + Fore.GREEN + '|')
    print(Fore.GREEN + '\t|' + Fore.BLUE + ' Recognition by Sequences Matching  ' + Fore.GREEN + '|')
    print(Fore.GREEN + '\t+------------------------------------+')
    print(Fore.GREEN + '\t| Options : (Press Option Key pls)   |')
    print(Fore.GREEN + '\t|    [1] \x1B[3mCreate Output Image\x1B[23m         |')
    print(Fore.GREEN + '\t|    [2] \x1B[3mFull KNN DTW \x1B[23m               |')
    print(Fore.GREEN + '\t|    [3] \x1B[3mFast KNN DTW \x1B[23m               |')
    print(Fore.GREEN + '\t|    [4] \x1B[3mDraw Sequences Chart \x1B[23m       |')
    print(Fore.GREEN + '\t|    [5] \x1B[3mDraw Compares Chart \x1B[23m        |')
    print(Fore.GREEN + '\t|    [6] \x1B[3mPre-processing Timer \x1B[23m       |')
    print(Fore.GREEN + '\t|    [7] \x1B[3mTime Estimator \x1B[23m             |')

    print(Fore.GREEN + '\t+------------------------------------+')
    option = input('\t  Please enter Option Number: ')
    print(Fore.GREEN + '\t+------------------------------------+')
    print(Fore.GREEN + '\t| Please enter Filter Number:        |')
    print(Fore.GREEN + '\t|    [1] \x1B[3mCanny\x1B[23m                       |')
    print(Fore.GREEN + '\t|    [2] \x1B[3mBoundary Tracing\x1B[23m            |')
    print(Fore.GREEN + '\t|    [3] \x1B[3mErode\x1B[23m                       |')
    print(Fore.GREEN + '\t+------------------------------------+')
    filter_op = input('\t  Please enter Filter Number: ')
    print(Fore.GREEN + '\t+------------------------------------+')
    filter_op = int(filter_op)
    filter = ''
    if filter_op == 2:
        filter = 'bnd_tracer'
    elif filter_op == 3:
        filter = 'erode'
    else:
        filter = 'canny'

    option = int(option)
    phnbsm.executor(option, filter)
