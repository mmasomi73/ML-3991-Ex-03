# Persian Handwritten Number Recognition by Sequences Matching -> PHNbSM
# first need to read dataset

import matplotlib.pyplot as plt
import numpy as np
from DTWHandler import DTWHandler
from DataCollector import DataCollector
from Evaluator import Evaluator
from Extractor import Extractor
from FastDTWHandler import FastDTWHandler
from OutputWriter import OutputWriter
from Visualizer import Visualizer


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

    def executor(self, create_output_image, filter='canny'):
        self.create_output_image = create_output_image
        if self.create_output_image:
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
        # extactor = Extractor(self.dc, filter)
        # # extactor = Extractor(self.dc)
        # extracted_test = extactor.getXTestEdgeVector()
        # extracted_train = extactor.getXTrainEdgeVector()
        # prepared_model = self.modelCreator(extracted_train, self.dc.Y_train)
        #
        # test_data = self.dataCreator(extracted_test)
        # test_data = test_data[0:10]
        # test_label = self.dc.Y_test[0:10]

        # dtwHandler = DTWHandler(prepared_model, test_data, test_label)
        # dtwHandler.printResult()

        # dtwHandler = FastDTWHandler(prepared_model, test_data, test_label)
        # dtwHandler.printResult()

        # visualizer = Visualizer(self.dc.X_test, self.dc.Y_test, self.dc)
        # visualizer.drawSequences()
        # visualizer.drawCompares()
        #
        evaluator = Evaluator(self.dc, True)
        # evaluator.preprocessTimer()
        evaluator.timeEstimator()


if __name__ == '__main__':
    phnbsm = PHNbSM(True)
    phnbsm.executor(False)
