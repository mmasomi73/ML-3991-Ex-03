import cv2
import numpy as np

from OutputWriter import OutputWriter


class Extractor:
    dc = object

    def __init__(self, dc):
        self.dc = dc
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

    def getEdgeVector(self, all_data):
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
