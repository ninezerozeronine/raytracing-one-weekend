"""
Vector of 3 values.

Used for colours, positions and normals
"""
import numbers
import math


class Vec3():
    def __init__(self, e0, e1, e2):
        """

        """
        self._e0 = e0
        self._e1 = e1
        self._e2 = e2

    @classmethod
    def from_vals(cls, v1, v2, v3):
        return cls(v1, v2, v3)

    @classmethod
    def from_vec3(cls, other):
        if not isinstance(other, cls):
            raise ValueError("Not a Vec3!")
        return cls(other._e0, other._e1, other._e2)

    def __getitem__(self, key):
        """

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

        """
        self._validate_key(key)
        self._validate_value(value)

        if key == 0:
            self._e0 = value
        elif key == 1:
            self._e1 = value
        else:
            self._e2 = value

    def _validate_key(self, key):
        """

        """
        if not isinstance(key, int):
            raise(TypeError("Key is not an int"))

        if key not in (0, 1, 2):
            raise(IndexError("Key is out of range"))

    def _validate_value():
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
