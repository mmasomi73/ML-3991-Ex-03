import cv2
import numpy as np
from BoundaryTracer import trace_boundary
from skimage.filters import threshold_otsu
from skimage.morphology import closing, square
from OutputWriter import OutputWriter
import pandas as pd


class Extractor:
    dc = object

    def __init__(self, dc, filter='canny'):
        self.dc = dc
        self.filter = filter
        # img_writer = OutputWriter()
        # for i in range(len(dc.X_test)):
        #     img = dc.X_test[i].reshape([32, 32])
        #     img = np.uint8(img)
        #     edges = cv2.Canny(img, 1, 1)
        #     img_writer.imageWriter(edges, dc.Y_test[i], 'aida')
        #     if i >= 1:
        #         break

    def getXTrainEdgeVector(self):
        return self.getEdgeVector(self.dc.X_train)

    def getXTestEdgeVector(self):
        return self.getEdgeVector(self.dc.X_test)

    def getXNamingEdgeVector(self):
        return self.getEdgeVector(self.dc.X_naming)

    def getEdgeVector(self, all_data, filter='none'):
        if filter != 'none':
            self.filter = filter
        if self.filter == 'canny':
            return self.filterCanny(all_data)
        elif self.filter == 'bnd_tracer':
            return self.filterBoundaryTracer(all_data)
        elif self.filter == 'erode':
            return self.filterErode(all_data)
        return self.filterCanny(all_data)

    def filterBoundaryTracer(self, all_data):
        data_bnd_tracer = []
        for data in all_data:
            img = data.reshape([32, 32])
            img = np.uint8(img)

            thresh = threshold_otsu(img)
            bw = closing(img > thresh, square(3))
            borders = trace_boundary(bw)
            edge_tuple_vector = []
            for idx, border in enumerate(borders):
                rr_bord, cc_bord = border
                for j in range(len(rr_bord)):
                    edge_tuple_vector.append((rr_bord[j], cc_bord[j]))
            data_bnd_tracer.append(edge_tuple_vector)
        return data_bnd_tracer

    def filterCanny(self, all_data):
        data_canny = []
        for data in all_data:
            img = data.reshape([32, 32])
            img = np.uint8(img)
            edges = cv2.Canny(img, 1, 1)
            edges = np.array(edges)
            vec_edge = np.where(edges == 255)
            edge_tuple_vector = []
            for j in range(len(vec_edge[0])):
                edge_tuple_vector.append((vec_edge[0][j], vec_edge[1][j]))
            # print(edge_tuple_vector)
            # print(vec_edge)
            data_canny.append(edge_tuple_vector)
        return data_canny

    def filterErode(self, all_data):
        data_erode = []
        for data in all_data:
            img = data.reshape([32, 32])
            img = np.uint8(img)
            kernel = np.ones((2, 2), np.uint8)
            edges = cv2.erode(img, kernel, iterations=1)
            edges = np.array(edges)
            vec_edge = np.where(edges == 1)
            edge_tuple_vector = []
            for j in range(len(vec_edge[0])):
                edge_tuple_vector.append((vec_edge[0][j], vec_edge[1][j]))
            data_erode.append(edge_tuple_vector)
        return data_erode
