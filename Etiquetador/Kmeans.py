__authors__ = ['1667799', '1688916', '1607129']
__group__ = 'TO_BE_FILLED'

import time

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        self.old_centroids = np.empty((self.K, self.X.shape[1]), dtype=self.X.dtype)
        self._init_centroids()

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """

        data_type = X.dtype

        if data_type != np.float64 or data_type != np.float32:
            X_float = X.astype(float)

        dimensions = X_float.shape
        self.N = dimensions[0] * dimensions[1]
        self.X = np.reshape(X_float, (dimensions[0] * dimensions[1], dimensions[2]))
        print()

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.
        if 'threshhold' not in options:
            options['threshold'] = 20

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.old_centroids = np.empty((self.K, self.X.shape[1]), dtype=self.X.dtype)
        if self.options['km_init'].lower() == 'first':
            unique_points = set()
            i = 0
            k_count = 0
            dim = self.X.shape
            while k_count < self.K and i < dim[0]:
                point = tuple(self.X[i])
                if point not in unique_points:
                    self.old_centroids[k_count] = self.X[i]
                    unique_points.add(point)
                    k_count += 1
                i += 1
        elif self.options['km_init'].lower() == 'random':
            """
            indices = np.random.choice(np.arange(len(self.X)), self.K, replace=False)
            self.old_centroids = self.X[indices]
            self.centroids = self.X[indices]
            """
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])

        self.centroids = self.old_centroids.copy()

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        self.labels = np.argmin(distance(self.X, self.centroids), axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids = self.centroids.copy()
        for i in range(self.K):
            points = self.X[self.labels == i]
            if len(points) > 0:
                self.centroids[i] = np.mean(points, axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # Calculamos la diferencia entre los centroides actuales y los anteriores
        diff = np.sum(np.abs(self.centroids - self.old_centroids))
        # Comprobamos si la diferencia es menor que la tolerancia especificada
        return diff <= self.options['tolerance']

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        converged = False
        i = 0

        while not converged and i < self.options['max_iter']:
            # Asignar cada punto al centroide más cercano
            self.get_labels()

            # Calcular nuevos centroides
            self.get_centroids()

            # Aumentar el número de iteraciones
            i += 1

            converged = self.converges()

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # calculamos la distancia mínima al centroide
        dist = distance(self.X, self.centroids).min(axis=1)
        # sumamos las distancias
        dist = np.sum(np.power(dist, 2))
        # devolvemos dist/N
        return dist / len(self.X)

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        previous_WCD = None
        for K in range(2, max_K + 1):
            self.K = K
            self.fit()
            WCD = self.withinClassDistance()

            if previous_WCD is not None:
                percent_decrease = 100 * (WCD / previous_WCD) #porcentaje de bajada
                if 100 - percent_decrease < self.options['threshold']:
                    self.K = K - 1 #agafem la k de la iteracio abans de que sigui d'un 20% que es la nostra ideal
                    return self.K
                if previous_WCD == WCD:
                    return self.K

            previous_WCD = WCD

        return max_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        distancia: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    sizeC = len(C)
    distancia = np.zeros((len(X), sizeC))
    for x in range(sizeC):
        distancia[:, x] = np.sqrt(np.sum(((X - C[x]) ** 2), axis=1))
    return distancia


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    color_prob = utils.get_color_prob(centroids)
    cp_size = len(color_prob)
    labels = []
    for x in range(cp_size):
        labels.append(utils.colors[np.argmax(color_prob[x])])
    return labels
