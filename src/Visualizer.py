import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
import matplotlib.cm as cmap
from skimage.draw import polygon_perimeter
from BoundaryTracer import trace_boundary
from Extractor import Extractor
import pandas as pd


class Visualizer:
    total_list = []

    def __init__(self, test_x, test_y, dc):
        self.dc = dc
        self.X_test = test_x
        self.Y_test = test_y

    #     TODO: Put Choice in init

    def drawCanny(self):
        data = self.dataSeparator()
        indexes = []
        for key, value in data.items():
            value = np.array(value)
            # print(key, value.shape)
            indexes.append(np.random.choice(value.shape[0], 2, replace=False))
        self.getFilters(indexes, data)
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

        plt.savefig('../outs/Digits-Canny-1.png', format='png')
        plt.close('all')

    def drawCompares(self):
        data = self.dataSeparator()
        indexes = []
        for key, value in data.items():
            value = np.array(value)
            # print(key, value.shape)
            indexes.append(np.random.choice(value.shape[0], 2, replace=False))
        self.getFilters(indexes, data)
        fig, axis = plt.subplots(10, 4, figsize=(8, 20))
        plt.rcParams["font.family"] = "Times New Roman"
        csfont = {'fontname': 'Times New Roman'}
        fig.suptitle('HODA Digits With Different Filter', fontsize=16)
        filters_list = ['Orginal', 'Canny', 'BTrace', 'Erode']
        for i in range(10):
            for j in range(4):
                if i == 0:
                    axis[i, j].title.set_text(filters_list[j])
                axis[i, j].axis('off')
            axis[i, 0].imshow(np.array(self.total_list[i][1]).reshape([32, 32]), cmap='gray')
            axis[i, 1].imshow(np.array(self.total_list[i][2]).reshape([32, 32]), cmap='gray')
            axis[i, 2].imshow(np.array(self.total_list[i][3]))
            axis[i, 3].imshow(np.array(self.total_list[i][4]).reshape([32, 32]), cmap='gray')

        plt.savefig('../outs/Digits-Filter-Compare.png', format='png')
        plt.close('all')

    def drawSequences(self):
        data = self.dataSeparator()
        indexes = []
        for key, value in data.items():
            value = np.array(value)
            indexes.append(np.random.choice(value.shape[0], 2, replace=False))
        sequences_list = self.getSequences(indexes, data)
        for i in range(10):
            plt.rcParams["font.family"] = "Times New Roman"
            csfont = {'fontname': 'Times New Roman'}
            filters_list = ['Canny', 'BTrace', 'Erode']

            plt.figure(figsize=(8, 6))
            for j in range(3):
                plt.plot(sequences_list[i][j+1], label=filters_list[j])
            plt.xlabel('Order', **csfont)
            plt.ylabel('Pixel Number', **csfont)
            plt.title('Extracted Sequences for [{}]'.format(i), **csfont)
            plt.legend(loc='upper right')
            plt.savefig('../outs/Seq-Comp-{}.png'.format(str(i)), format='png')
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
        self.total_list = []
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

    def getFilters(self, indexes, data):
        self.total_list = []
        for key, value in data.items():
            index = indexes[int(key)]
            label = key

            data = value[index[0]]
            img = data.reshape([32, 32]).copy()
            img = np.uint8(img)
            edges = cv2.Canny(img, 1, 1)
            edges_1 = np.array(edges).copy()

            img = data.reshape([32, 32]).copy()
            img = np.uint8(img)
            thresh = threshold_otsu(img)
            bw = closing(img > thresh, square(3))
            borders = trace_boundary(bw)
            img = np.stack([img] * 3, axis=-1)
            for idx, border in enumerate(borders):
                # demonstrate that pixels are in order
                rr_bord, cc_bord = border
                rr, cc = polygon_perimeter(rr_bord, cc_bord)
                # rainbow *_*
                col_idx = idx * (1337 + 42000) % len(borders)

                color = np.array(cmap.gist_rainbow(1. * col_idx / len(borders))) * 255
                img[rr, cc, :] = color[:-1]
            # img[np.where((img <= [5, 5, 5]).all(axis=2))] = [255, 255, 255]
            edges_2 = np.array(img).copy()

            img = data.reshape([32, 32]).copy()
            img = np.uint8(img)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.erode(img, kernel, iterations=1)
            edges_3 = np.array(edges).copy()

            self.total_list.append([
                label,
                data,
                edges_1,
                edges_2,
                edges_3
            ])

    def getSequences(self, indexes, data):
        self.total_list = []
        sequences_list = []
        self.getFilters(indexes, data)
        for key, value in data.items():
            index = indexes[int(key)]
            label = str(int(key))
            data = value[index[0]]
            extactor = Extractor(self.dc)
            canny_data = extactor.getEdgeVector([data], 'canny')[0]
            btrace_data = extactor.getEdgeVector([data], 'bnd_tracer')[0]
            erode_data = extactor.getEdgeVector([data], 'erode')[0]

            canny_data = pd.Series(self.rowWiser(canny_data))
            btrace_data = pd.Series(self.rowWiser(btrace_data))
            erode_data = pd.Series(self.rowWiser(erode_data))

            sequences_list.append([
                label,
                canny_data,
                btrace_data,
                erode_data])
        return sequences_list

    def rowWiser(self, data):
        row_wise_data = []
        for i in range(len(data)):
            data_point = data[i]
            row_wise_data.append(data_point[0] * 32 + data_point[1])
        return row_wise_data
