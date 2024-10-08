__authors__ = ['1667799', '1688916', '1607129']
__group__ = '150'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #Cambiar dimensiones y convertir los valores a float
        self.train_data = np.reshape(train_data, (train_data.shape[0], -1)).astype(float)

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        test_data = np.reshape(test_data, (test_data.shape[0], -1)).astype(float)
        X = cdist(test_data, self.train_data, "euclidean")  #cogemos las distancias euclidianas
        Y = np.argsort(X, axis = 1) #ordenamos
        Z = Y[:,0:k] #cogemos los K valores
        self.neighbors = Z.astype(str)
        for i in range(len(Z)):
            for j in range(len(Z[i])):
                self.neighbors[i][j] = self.labels[Z[i][j]]  #para cada id que tenemos ponemos su equivalente en prenda

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #Pasar por elementos y contar votos
        masRepetidos = []
        votos={}
        for element in self.neighbors:
            for item in element:
                item=str(item)
                if item not in votos:
                    votos[item] = 1
                else: 
                    votos[item] += 1
            maxim=0
            for item in votos: #cogemos el que mas sale
                if votos[item]>maxim:
                    maxim=votos[item]
                    ganador=item
            votos.clear()
            masRepetidos.append(ganador) 
        return np.array(masRepetidos)
    
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()
