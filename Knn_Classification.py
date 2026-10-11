# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-13

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-13
import numpy as np

"""KNN Classification Class"""
class Knn_Classification:
    # Constructor
    def __init__(self, x_train, y_train):
        """ Constructs a KNN classification object.
            :param x_train: The standardized training attributes.
            :type x_train: numpy.ndarray
            :param y_train: The training labels.
            :type y_train: numpy.ndarray
            :return: None
        """
        # Data fields
        self._x_train = x_train
        self._y_train = y_train

    # Getters
    def get_x_train(self):
        """ Returns the training attributes.
            :return: The training attributes.
            :rtype: numpy.ndarray
        """
        return self._x_train

    def get_y_train(self):
        """ Returns the training labels.
            :return: The training labels.
            :rtype: numpy.ndarray
        """
        return self._y_train

    # Methods
    def distance(self, x1, x2):
        """ Computes the Euclidean distance between two points.
            :param x1: The first point.
            :type x1: numpy.ndarray
            :param x2: The second point.
            :type x2: numpy.ndarray
            :return: The Euclidean distance between the two points.
            :rtype: float
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def get_k_nearest_neighbors(self, x_test, k):
        """ Finds the k nearest neighbors of a test point.
            :param x_test: The test point.
            :type x_test: numpy.ndarray
            :param k: The number of nearest neighbors.
            :type k: int
            :return: list of tuples [(distance, label)]
            :rtype: list of tuples
        """
        neighbors = []

        # Compute the distance from the test point to each training point
        differences = self.get_x_train() - x_test
        distances = np.sqrt(np.sum(differences ** 2, axis=1))

        # Sort the distance from smallest to largest
        sorted_indices = np.argsort(distances, kind='stable')[:k]

        neighbors = []

        # Stores the distance and label of the k nearest neighbors
        for index in sorted_indices:
            class_label = self.get_y_train()[index]
            neighbors.append((distances[index], class_label))

        # Return the first k neighbors
        return neighbors
    def predict(self, x_test, k):
        """ Predicts the class of a test point based on majority vote.
            :param x_test: The test point.
            :type x_test: numpy.ndarray
            :param k: The number of nearest neighbors.
            :type k: int
            :return: The predicted class label.
            :rtype: int
        """
        neighbors = self.get_k_nearest_neighbors(x_test, k)

        class_counts = {}

        # Count the class labels of the k nearest neighbors
        for (_, label) in neighbors:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Find the class label with the maximum count
        predicted_class = max(class_counts, key=class_counts.get)

        return predicted_class
