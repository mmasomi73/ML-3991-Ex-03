# Persian Handwritten Number Recognition by Sequences Matching -> PHNbSM
# first need to read dataset

from DTWHandler import DTWHandler
from DataCollector import DataCollector
from Extractor import Extractor
from OutputWriter import OutputWriter


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
        # TODO: Uncomment Train And Namely Lines in Data Collector
        # TODO: Change here Test With Train
        extactor = Extractor(self.dc)
        extracted_test = extactor.getXTestEdgeVector()
        prepared_model = self.modelCreator(extracted_test, self.dc.Y_test)
        # print(len(prepared_model))

        test_data = self.dataCreator(extracted_test)
        test_data = test_data[0:1]
        test_label = self.dc.Y_test[0:1]
        dtwHandler = DTWHandler(prepared_model, test_data, test_label)
        dtwHandler.fit()
        # dtwHandler.predict()
        dtwHandler.printResult()

        # aida_1 = pandas.Series(mim[4500])
        # aida_2 = pandas.Series(mim[4600])
        # # manhattan_distance = lambda a, b: (abs(a[0] - b[0]) + abs(a[1] - b[1]))
        # manhattan_distance = lambda a, b: abs(a - b)
        #
        # d, cost_matrix, acc_cost_matrix, path = dtw(aida_1, aida_2, dist=manhattan_distance)
        #
        # # fig, axs = plt.subplots(3)
        # print(d)
        # plt.imshow(self.dc.X_test[4500].reshape([32, 32]), cmap='gray')
        # plt.show()
        # plt.imshow(self.dc.X_test[4600].reshape([32, 32]), cmap='gray')
        # plt.show()
        # plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
        # # plt.show()
        # plt.plot(path[0], path[1], 'w')
        # plt.show()
        # plt.close('all')

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


if __name__ == '__main__':
    aida = PHNbSM(True)
