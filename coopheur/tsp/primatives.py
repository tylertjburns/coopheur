import numpy as np


class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    def __str__(self):
        return str(self._x) + ';' + str(self._y)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def dist(self, other):
        if not type(other) == Point:
            raise TypeError()

        # intializing points in
        # numpy arrays
        point1 = np.array((self.x, self.y))
        point2 = np.array((other.x, other.y))

        # calculating Euclidean distance
        # using linalg.norm()
        dist = np.linalg.norm(point1 - point2)
        return dist
