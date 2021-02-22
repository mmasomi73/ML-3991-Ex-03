# Persian Handwritten Number Recognition by Sequences Matching -> PHNbSM
# first need to read dataset
import cv2
import numpy as np
import pandas
from dtw import dtw
from DataCollector import DataCollector
from Extractor import Extractor
from OutputWriter import OutputWriter
import matplotlib.pyplot as plt

class PHNbSM:
    log = False
    create_output_image = False

    def __init__(self, log=False):
        self.dc = DataCollector(log)
        # self.log = log

        if self.create_output_image:
            img_writer = OutputWriter()
            # -----= Train images
            for i in range(len(self.dc.X_train)):
                # print(self.dc.Y_train[i])
                img_writer.imageWriter(self.dc.X_train[i], self.dc.Y_train[i], 'train')
                if i > 100:
                    break
            # -----= Test images
            for i in range(len(self.dc.X_test)):
                # print(self.dc.Y_train[i])
                img_writer.imageWriter(self.dc.X_test[i], self.dc.Y_test[i], 'test')
                if i > 100:
                    break
            # -----= Renaming images
            for i in range(len(self.dc.X_naming)):
                # print(self.dc.Y_train[i])
                img_writer.imageWriter(self.dc.X_naming[i], self.dc.Y_naming[i], 'naming')
                if i > 100:
                    break

        extactor = Extractor(self.dc)
        aida = extactor.getXTestEdgeVector()
        # alignment = dtw(aida[0], aida[1], keep_internals=True)
        # alignment.plot(type="threeway")
        mim = []
        for data in aida:
            row_wise = []
            for itm in data:
                row_wise.append(itm[0] * 32 + itm[1])
            mim.append(row_wise)
        print(mim[0])
        aida_1 = pandas.Series(mim[0])
        aida_2 = pandas.Series(mim[0])
        manhattan_distance = lambda x, y: np.abs(x - y)


        d, cost_matrix, acc_cost_matrix, path = dtw(aida_1,aida_2, dist=manhattan_distance)

        # fig, axs = plt.subplots(3)
        plt.imshow(self.dc.X_test[0].reshape([32, 32]), cmap='gray')
        plt.show()
        plt.imshow(self.dc.X_test[0].reshape([32, 32]), cmap='gray')
        plt.show()
        plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        # plt.show()
        plt.plot(path[0], path[1], 'w')
        plt.show()




if __name__ == '__main__':
    aida = PHNbSM(True)
