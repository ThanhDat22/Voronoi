# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-14

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-17

from Knn_Classification import Knn_Classification

import numpy as np
import matplotlib.pyplot as plt

"""Voronoi KNN Classification Class"""

class Voronoi_Knn_Classification(Knn_Classification):
    # Construction
    def __init__(self, blue_points, red_points, resolution):
        """ Constructs a Voronoi Knn classification object.
            :param blue_points: Coordinates of blue training points
            :type blue_points: list of tuples
            :param red_points: Coordinates of red training points
            :type red_points: list of tuples
            :param resolution: Resolution of the query grid
            :type resolution: float
            :return: None
        """
        # Data fields
        self._blue_points = np.array(blue_points, dtype=float)
        self._red_points = np.array(red_points, dtype=float)
        self.resolution = resolution
        self._x_grid = None
        self._y_grid = None
        self.predictions = None

        # Prepare the training data
        x_train, y_train = self.prepare_dataset()

        # Initialize the Knn_Classification class
        Knn_Classification.__init__(self, x_train, y_train)

    # Getters
    def get_blue_points(self):
        """ Returns the coordinates of blue training points.
            :return: Coordinates of blue training points
            :rtype: numpy.ndarray
        """
        return self._blue_points
    def get_red_points(self):
        """ Returns the coordinates of red training points.
            :return: Coordinates of red training points
            :rtype: numpy.ndarray
        """
        return self._red_points
    def get_x_grid(self):
        """ Returns the x-coordinates of the grid points.
            :return: x-coordinates of the grid points
            :rtype: numpy.ndarray
        """
        return self._x_grid
    def get_y_grid(self):
        """ Returns the y-coordinates of the grid points.
            :return: y-coordinates of the grid points
            :rtype: numpy.ndarray
        """
        return self._y_grid
    def get_predictions(self):
        """ Returns the predictions.
            :return: Predictions
            :rtype: numpy.ndarray
        """
        return self.predictions
    # Setters
    def set_x_grid(self, x_grid):
        """ Sets the x-coordinates of the grid points.
            :param x_grid: x-coordinates of the grid points
            :type x_grid: numpy.ndarray
            :return: None
        """
        self._x_grid = x_grid
    def set_y_grid(self, y_grid):
        """ Sets the y-coordinates of the grid points.
            :param y_grid: y-coordinates of the grid points
            :type y_grid: numpy.ndarray
            :return: None
        """
        self._y_grid = y_grid
    def set_predictions(self, predictions):
        """ Sets the predictions.
            :param predictions: Predictions
            :type predictions: numpy.ndarray
            :return: None
        """
        self.predictions = predictions
    # Methods
    def prepare_dataset(self):
        """ Prepares the training dataset by combining blue and red points.
            :return: Tuple of x_train and y_train
            :rtype: tuple of numpy.ndarray
        """
        # Combine blue and red points
        x_train = np.vstack((self._blue_points, self._red_points))
        # Create labels: 1 for blue, 0 for red
        blue_labels = np.ones(len(self.get_blue_points()),dtype=int)
        red_labels = np.zeros(len(self.get_red_points()),dtype=int)
        # Combine labels
        y_train = np.concatenate((blue_labels, red_labels))
        return x_train, y_train
    def create_query_grid(self, x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0):
        """ Creates a grid of query points for classification.
            :param x_min: Minimum x-coordinate
            :type x_min: float
            :param x_max: Maximum x-coordinate
            :type x_max: float
            :param y_min: Minimum y-coordinate
            :type y_min: float
            :param y_max: Maximum y-coordinate
            :type y_max: float
            :return: None      
        """
        x_values = np.linspace(x_min, x_max, self.resolution)
        y_values = np.linspace(y_min, y_max, self.resolution)

        # Creates the two-dimensional grid
        x_grid, y_grid = np.meshgrid(x_values, y_values)

        self.set_x_grid(x_grid)
        self.set_y_grid(y_grid)
    def classify_query_points(self):
        """ Classifies the query points in the grid using KNN.
            :return: None
        """
        predictions = np.zeros(self.get_x_grid().shape, dtype=int)

        # Classify each point in the grid
        for row in range(self.get_x_grid().shape[0]):
            for column in range(self.get_x_grid().shape[1]):
                # Current query point
                x_test = np.array([
                    self.get_x_grid()[row, column], 
                    self.get_y_grid()[row, column]
                ])
                # K = 1 for the Voronoi classification
                predicted_class = self.predict(x_test, k=1)
                predictions[row, column] = predicted_class

        self.set_predictions(predictions)
    def print_points_info(self):
        """ Prints the coordinates of blue and red training points.
            :return: None
        """
        print("Blue training points:", self.get_blue_points().tolist())
        print("Red training points:", self.get_red_points().tolist())
    def line_equation(self, slope, intercept):
        """ Returns a line quation as y = ax + b
            :param slope: The slope of the line.
            :type slope: float
            :param intercept: The y-intercept of the line.
            :type intercept: float
            :return: A function representing the line equation.
            :rtype: function
        """
        if abs(slope) < 1e-10:
            slope = 0.0
        if abs(intercept) < 1e-10:
            intercept = 0.0
        if intercept >= 0:
            return f"y = {slope:.2f}x + {intercept:.2f}"
        else:
            return f"y = {slope:.2f}x - {abs(intercept):.2f}"
    def get_midpoint(self, point1, point2):
        """ Computes the midpoint of two points.
            :param point1: The first point
            :type point1: numpy.ndarray
            :param point2: The second point
            :type point2: numpy.ndarray
            :return: The midpoint of the two points.
            :rtype: numpy.ndarray
        """
        return (point1 + point2) / 2.0
    def get_segment_direction(self, point1, point2):
        """ Computes the direction vector of the line segment connecting two points.
            :param point1: The first point
            :type point1: numpy.ndarray
            :param point2: The second point
            :type point2: numpy.ndarray
            :return: The direction vector [dx, dy]of the line segment.
            :rtype: numpy.ndarray
        """
        x1, y1 = point1
        x2, y2 = point2
        dx = x2 - x1
        dy = y2 - y1
        return np.array([dx, dy])
    def get_perpendicular_direction(self, point1, point2):
        """ Computes the direction vector of the perpendicular bisector of the line segment connecting two points.
            :param point1: The first point
            :type point1: numpy.ndarray
            :param point2: The second point
            :type point2: numpy.ndarray
            :return: The direction vector [dx, dy] of the perpendicular bisector.
            :rtype: numpy.ndarray
        """
        segment_direction = self.get_segment_direction(point1, point2)
        # Perpendicular direction = [-dy, dx]
        dx = segment_direction[0]
        dy = segment_direction[1]
        return np.array([-dy, dx])
    def get_perpendicular_equation(self, point1, point2):
        """ Computes the equation of the perpendicular bisector of the line segment connecting two points.
            :param point1: The first point
            :type point1: numpy.ndarray
            :param point2: The second point
            :type point2: numpy.ndarray
            :return: Equation of the perpendicular bisector
            :rtype: str
        """
        midpoint = self.get_midpoint(point1, point2)
        direction = self.get_perpendicular_direction(point1, point2)

        dx = direction[0]
        dy = direction[1]

        # Original segment is vertical
        if abs(dx) < 1e-10:
            slope = 0.0
            return f"y = {midpoint[1]:.2f}"
        # Original segment is horizontal
        elif abs(dy) < 1e-10:
            return f"x = {midpoint[0]:.2f}"
        # General case
        else:
        
            perpendicular_slope = dy / dx
            intercept = midpoint[1] - perpendicular_slope * midpoint[0]
            return self.line_equation(perpendicular_slope, intercept)
    def get_perpendicular_bisector(self, point1, point2):
        """ Computes the perpendicular bisector of the line segment connecting two points.
            :param point1: The first point
            :type point1: numpy.ndarray
            :param point2: The second point
            :type point2: numpy.ndarray
            :return: Midpoint, direction vector, and equation of the perpendicular bisector.
            :rtype: tuple of numpy.ndarray
        """
        midpoint = self.get_midpoint(point1, point2)
        direction = self.get_perpendicular_direction(point1, point2)
        equation = self.get_perpendicular_equation(point1, point2)
        return midpoint, direction, equation
    def update_t_interval(self, coordinate, direction, minnimum, maximum, t_min, t_max):
        """ Updates the interval so a line remains inside plotting area.
            :param coordinate: The coordinate of the point
            :type coordinate: float
            :param direction: The direction of the line
            :type direction: float
            :param minnimum: Minimum value of the plotting area
            :type minnimum: float
            :param maximum: Maximum value of the plotting area
            :type maximum: float
            :param t_min: Current minimum t value
            :type t_min: float
            :param t_max: Current maximum t value
            :type t_max: float
            :return: Updated minimum and maximum t values and validity
            :rtype: tuple of float
        """
        if abs(direction) < 1e-10:
            if coordinate < min or coordinate > max:
                return t_min, t_max, False
            else:
                return t_min, t_max, True

        t1 = (minnimum - coordinate) / direction
        t2 = (maximum - coordinate) / direction

        lower_t = min(t1, t2)
        upper_t = max(t1, t2)

        t_min = max(t_min, lower_t)
        t_max = min(t_max, upper_t)

        return t_min, t_max, t_min <= t_max
    def calculate_voronoi_edge(self, point_index1, point_index2, x_min=0.0, x_max=5.0, y_min=0.0, y_max=5.0):
        """ Calculates the Voronoi edge between two points.
            :param point_index1: Index of the first point
            :type point_index1: int
            :param point_index2: Index of the second point
            :type point_index2: int
            :param x_min: Minimum x-coordinate of the plotting area
            :type x_min: float
            :param x_max: Maximum x-coordinate of the plotting area
            :type x_max: float
            :param y_min: Minimum y-coordinate of the plotting area
            :type y_min: float
            :param y_max: Maximum y-coordinate of the plotting area
            :type y_max: float
            :return: Coordinates of the Voronoi edge endpoints or None if not valid.
            :rtype: tuple of numpy.ndarray or None
        """
        points = self.get_x_train()

        point1 = points[point_index1]
        point2 = points[point_index2]
        midpoint, direction, equation = (self.get_perpendicular_bisector(point1, point2))

        # Start with an unlimited line
        t_min = -np.inf
        t_max = np.inf

        # Limit the line to the plotting area
        t_min, t_max, valid = self.update_t_interval(midpoint[0], direction[0], x_min, x_max, t_min, t_max)

        if not valid:
            return None

        t_min, t_max, valid = self.update_t_interval(midpoint[1], direction[1], y_min, y_max, t_min, t_max)

        if not valid:
            return None

        # Check all other training points
        for k in range(len(points)):
            if k == point_index1 or k == point_index2:
                continue
            other_point = points[k]
            difference = other_point - point1
            a = 2.0 * np.dot(direction, difference)
            b = (np.dot(other_point, other_point) - np.dot(point1, point1) - 2.0 * np.dot(midpoint, difference))

            if abs(a) < 1e-10:
                if b < -1e-10:
                    return None
            elif a > 0:
                t_max = min(t_max, b / a)
            else:
                t_min = max(t_min, b / a)
            if t_min > t_max:
                return None
        # If the valid portion is only a single point
        if abs(t_min - t_max) < 1e-8:
            return None
        start_point = midpoint + t_min * direction
        end_point = midpoint + t_max * direction

        return {
            'point1': point1,
            'point2': point2,
            'midpoint': midpoint,
            'start_point': start_point,
            'end_point': end_point,
            'equation': equation
        }
    def calculate_voronoi_boundaries(self):
        """ Calculates the Voronoi boundaries for all pairs of training points.
            :return: List of Voronoi edges
            :rtype: list of dict
        """
        boundaries = []
        number_of_points = len(self.get_x_train())

        # Checks every pair of training points
        for i in range(number_of_points):
            for j in range(i + 1, number_of_points):
                boundary = self.calculate_voronoi_edge(i, j)
                if boundary is not None:
                    boundaries.append(boundary)
        return boundaries
    def print_boundary_info(self):
        """ Prints the information of Voronoi boundaries.
            :return: None
        """
        
        boundaries = self.calculate_voronoi_boundaries()
        for i, boundary in enumerate(boundaries, start=1):
            point1 = boundary['point1']
            point2 = boundary['point2']
            midpoint = boundary['midpoint']

            print(f"Boundary {i}:")
            print(f"  Point 1: {point1[0]:.1f}, {point1[1]:.1f}")
            print(f"  Point 2: {point2[0]:.1f}, {point2[1]:.1f}")
            print(f"  Midpoint: {midpoint[0]:.1f}, {midpoint[1]:.1f}")
            print(f"  Perpendicular bisector equation: {boundary['equation']}")
        print('' + '-' * 60)
    def plot_voronoi_boundaries(self):
        """ Plots the Voronoi boundaries.
            :return: None
        """
        boundaries = self.calculate_voronoi_boundaries()

        midpoint_lable_added = False

        for boundary in boundaries:
            start_point = boundary['start_point']
            end_point = boundary['end_point']
            midpoint = boundary['midpoint']

            # Draw the actual Voironoi edge
            plt.plot(
                [start_point[0], end_point[0]],
                [start_point[1], end_point[1]],
                color='black',
                linestyle='--',
                linewidth=2
            )
            # Draw midpoint
            plt.scatter(
                midpoint[0],
                midpoint[1],
                color='green',
                marker='x',
                s=70,
                zorder=6,
                label='Midpoint' if not midpoint_lable_added else None
            )

            midpoint_lable_added = True

            # Label midpoint coordinates
            plt.annotate(
                f"({midpoint[0]:.2f}, {midpoint[1]:.2f})",
                (midpoint[0], midpoint[1]),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=8,
            )
    def plot_voronoi(self, case_number=None):
        """ Plots the Voronoi diagram based on the classified query points.
            :param case_number: Number of the current case
            :type case_number: int
            :return: None
        """
        plt.figure(num=case_number, figsize=(7, 7))

        # Shade only the blue classification region
        plt.contourf(
            self.get_x_grid(), 
            self.get_y_grid(), 
            self.get_predictions(), 
            levels=[0.5, 1.5], 
            colors=['blue'], 
            alpha=0.6
        )

        # Draw all calculated Voronoi boundaries
        self.plot_voronoi_boundaries()
        # Plot the blue training points
        blue_points = self.get_blue_points()
        plt.scatter(
            blue_points[:, 0], 
            blue_points[:, 1], 
            color='blue', 
            marker='s',
            s=80,
            label='Blue'
        )

        # Plot the red training points
        red_points = self.get_red_points()
        plt.scatter(
            red_points[:, 0], 
            red_points[:, 1], 
            color='red', 
            marker='o',
            s=80,
            label='Red'
        )
        plt.title(
            f'Case {case_number}: 1-NN Voronoi KNN Classification' + 
            f'\nBlue points: {self.get_blue_points().tolist()}, '
            f'Red points: {self.get_red_points().tolist()}'
        )
        plt.xlabel('x')
        plt.ylabel('y')

        plt.xlim(self.get_x_grid().min(), self.get_x_grid().max())
        plt.ylim(self.get_y_grid().min(), self.get_y_grid().max())

        plt.xticks(np.arange(self.get_x_grid().min(), self.get_x_grid().max() + 1, 1))
        plt.yticks(np.arange(self.get_y_grid().min(), self.get_y_grid().max() + 1, 1))

        plt.grid()
        plt.legend()

        plt.gca().set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Save the figure
        plt.savefig(f'Voronoi_KNN_Classification_{case_number}.png', dpi=300, bbox_inches='tight')
        
    def run_voronoi(self, case_number=None):
        """ Runs the Voronoi KNN classification process.
            :param case_number: Number of the current case
            :type case_number: int
            :return: None
        """
        # Print the training points
        self.print_points_info()

        # Print the decision boundary information
        self.print_boundary_info()

        # Creates the query points
        self.create_query_grid()

        # Classifies the query points
        self.classify_query_points()

        # Plots the Voronoi diagram
        self.plot_voronoi(case_number)

def main():
    # Variable declaration
    resolution = 400

    case_1 = Voronoi_Knn_Classification(
        blue_points=[(1, 1)],
        red_points=[(4, 3)],
        resolution=resolution
    )
    case_2 = Voronoi_Knn_Classification(
        blue_points=[(1, 1), (2, 4)],
        red_points=[(4, 3)],
        resolution=resolution
    )
    case_3 = Voronoi_Knn_Classification(
        blue_points=[(2, 1), (3, 4)],
        red_points=[(1, 2), (4, 3)],
        resolution=resolution
    )

    case_1.run_voronoi(1)
    case_2.run_voronoi(2)
    case_3.run_voronoi(3)

    plt.show()

if __name__ == "__main__":
    main()