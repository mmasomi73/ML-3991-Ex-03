import cv2
import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    total_list = []

    def __init__(self, test_x, test_y):
        self.X_test = test_x
        self.Y_test = test_y

    def drawCompares(self):
        data = self.dataSeparator()
        indexes = []
        for key, value in data.items():
            value = np.array(value)
            # print(key, value.shape)
            indexes.append(np.random.choice(value.shape[0], 2, replace=False))
        self.getEdgeVector(indexes, data)
        fig, axis = plt.subplots(10, 4, figsize=(4, 10))
        plt.rcParams["font.family"] = "Times New Roman"
        csfont = {'fontname': 'Times New Roman'}
        fig.suptitle('HODA Digits With Canny Filter ', fontsize=12)
        for i in range(10):
            for j in range(4):
                axis[i, j].axis('off')
            axis[i, 0].imshow(np.array(self.total_list[i][1]).reshape([32, 32]), cmap='gray')
            axis[i, 1].imshow(np.array(self.total_list[i][2]).reshape([32, 32]), cmap='gray')
            axis[i, 2].imshow(np.array(self.total_list[i][3]).reshape([32, 32]), cmap='gray')
            axis[i, 3].imshow(np.array(self.total_list[i][4]).reshape([32, 32]), cmap='gray')

        plt.savefig('../outs/Digits-Canny.png', format='png')
        plt.close('all')

    def dataSeparator(self):
        data = dict()
        for i in range(len(self.X_test)):
            data_list = []
            i_key = str(int(self.Y_test[i]))
            if i_key in data:
                data_list = data.get(i_key)
            data_list.append(self.X_test[i])
            data[i_key] = data_list
        return data

    def getEdgeVector(self, indexes, data):
        for key, value in data.items():
            index = indexes[int(key)]
            label = key

            data_1 = value[index[0]]
            img = data_1.reshape([32, 32])
            img = np.uint8(img)
            edges = cv2.Canny(img, 1, 1)
            edges_1 = np.array(edges).copy()

            data_2 = value[index[1]]
            img = data_2.reshape([32, 32])
            img = np.uint8(img)
            edges = cv2.Canny(img, 1, 1)
            edges_2 = np.array(edges).copy()
            self.total_list.append([
                label,
                data_1,
                edges_1,
                data_2,
                edges_2])
