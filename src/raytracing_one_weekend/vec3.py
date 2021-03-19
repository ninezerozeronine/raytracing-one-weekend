"""
Vector of 3 values.

Used for colours, positions and normals
"""
import numbers
import math


class Vec3():
    """
    A vector of 3 floats.

    Can be used for points, vectors and RGB colours
    """
    def __init__(self, e0, e1, e2):
        """
        Initialise a vector from 3 values.

        Args:
            e0 (float): The first value.
            e1 (float): The first value.
            e2 (float): The first value.
        """

        self._validate_value(e0)
        self._validate_value(e1)
        self._validate_value(e2)

        self._e0 = e0
        self._e1 = e1
        self._e2 = e2

    def __getitem__(self, key):
        """
        Allow indexing into the 3 values.

        Args:
            key (int): The index of the value to get.

        Returns:
            int: The value for the element at the given index.
        """
        self._validate_key(key)

        if key == 0:
            return self._e0
        elif key == 1:
            return self._e1
        else:
            return self._e2

    def __setitem__(self, key, value):
        """
        Allow setting one of the 3 values by indexing

        Args:
            key (int): The index of the value to set.
            value (float): The value to set.
        """
        self._validate_key(key)
        self._validate_value(value)

        if key == 0:
            self._e0 = value
        elif key == 1:
            self._e1 = value
        else:
            self._e2 = value

    def __add__(self, other):
        """

        """
        if isinstance(other, Vec3):
            return Vec3(
                self._e0 + other._e0,
                self._e1 + other._e1,
                self._e2 + other._e2
            )
        else:
            raise TypeError("Cannot add {other_type} to Vec3".format(
                other_type=type(other)
            ))

    def __iadd__(self, other):
        """

        """
        if isinstance(other, Vec3):
            self._e0 += other._e0
            self._e1 += other._e1
            self._e2 += other._e2
            return self
        else:
            raise TypeError("Cannot add {other_type} to Vec3".format(
                other_type=type(other)
            ))

    def __eq__(self, other):
        """

        """
        if isinstance(other, Vec3):
            tolerance = 0.0001
            return (
                abs(self._e0 - other._e0) < tolerance
                and abs(self._e1 - other._e1) < tolerance
                and abs(self._e2 - other._e2) < tolerance
            )
        else:
            raise TypeError("Cannot compare {other_type} to Vec3".format(
                other_type=type(other)
            ))

    def _validate_key(self, key):
        """

        """
        if not isinstance(key, int):
            raise(TypeError("Key is not an int"))

        if key not in (0, 1, 2):
            raise(IndexError("Key is out of range"))

    def _validate_value(self, value):
        """

        """
        if not isinstance(value, numbers.Number):
            raise ValueError("Value passed is not a number.")

    @property
    def x(self):
        return self._e0

    @x.setter
    def x(self, value):
        self._validate_value()
        self._e0 = value

    @property
    def y(self):
        return self._e1

    @x.setter
    def y(self, value):
        self._validate_value()
        self._e1 = value

    @property
    def z(self):
        return self._e2

    @x.setter
    def z(self, value):
        self._validate_value()
        self._e2 = value

    def length(self):
        """

        """
        return math.sqrt(self.length_squared())

    def length_squared(self):
        """

        """
        return (self._e0 ** 2) + (self._e1 ** 2) + (self._e2 ** 2)
