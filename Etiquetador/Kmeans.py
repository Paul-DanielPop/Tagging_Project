__authors__ = ['1667799', '1688916', '1607129']
__group__ = 'noneyet'

import numpy as np
from numpy.linalg import norm
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
        self.WCD = None
        self.labels = None
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
        self._init_centroids() #inicialitzacio dels centroids
        
        while(self.num_iter < self.options['max_iter']):
            self.get_labels() #busquem els centroids mes propers
            
            self.get_centroids() #calcuem els nous centroids
            
            self.num_iter += 1 #aument d'iteracio en 1
            
            if (self.converges()):  #quan convergeix fem q surti del bucle
                break    

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # calculamos la distancia mínima al centroide
        dist = distance(self.X, self.centroids)

        # Calculamos la distancia mínima al cuadrado para cada punto
        min_distances_sq = dist.min(axis=1) ** 2

        # Sumamos las distancias mínimas al cuadrado
        sum_min_distances_sq = np.sum(min_distances_sq)

        self.WCD = sum_min_distances_sq / len(self.X)
        return self.WCD
    
    def inter_class_distance(self):
        """
        returns the inter class distance of the current clustering
        """
        suma = 0
        for i in range(len(self.centroids)):
            for j in range(i+1, len(self.centroids)):
                inter_class_distance = norm(np.array(self.centroids[i]) - np.array(self.centroids[j])) ** 2
                suma += inter_class_distance
        self.ICD =  suma
    
    def Fisher_coefficient(self):
        """
        returns the Fisher's coefficient of the current clustering
        """    
        self.FISHER =  self.WCD/self.ICD


    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self.K = 2 #primer fem la k=2, aixi ens enstalviarem algunes iteracions en el bucle o condicions extres
        self.fit()
        self.withinClassDistance()
        WCD_anterior = self.WCD #calculem el WCD de k=2

        for K in range(3, max_K + 1):
            
            self.K = K
            self.fit()
            self.withinClassDistance()
            WCD = self.WCD

            percent_decrease = 100 * (WCD / WCD_anterior) #porcentaje de bajada
            if 100 - percent_decrease < self.options['threshold']:
                self.K = K - 1 #agafem la k de la iteracio abans de que sigui d'un 20% que es la nostra ideal
                return self.K #fem el return per tal de finalitzar la funcio

            WCD_anterior = self.WCD #guardem el WCD actual en la variable anterior

        return max_K #en cas de arribar al max establert es retornara aquest


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

    sizeC = len(C) #calculem la longitud de C per tal d'estalviar calcularla dos cops mes, aixi la tenim guardada en una variable
    
    distancia = np.zeros((len(X), sizeC)) #creem la matriu de 0
    
    for x in range(sizeC):
        distancia[:, x] = np.sqrt(np.sum(((X - C[x]) ** 2), axis=1)) #calcul de la distancia
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
    tamany_colorsp = len(color_prob)
    labels = []
    for x in range(tamany_colorsp):
        labels.append(utils.colors[np.argmax(color_prob[x])])
    return labels
