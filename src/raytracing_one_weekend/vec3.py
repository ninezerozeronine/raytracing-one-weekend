"""
Vector of 3 values.

Used for colours, positions and normals
"""
import numbers
import math

_TOLERANCE = 0.00001


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
            e1 (float): The second value.
            e2 (float): The third value.
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
        Add two Vec3s together.

        E.g.::
            vec3 = vec1 + vec2
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
        Add another Vec3 to the current one in place.

        E.g.::
            vec1 += vec2

        Implementing __iadd__ as well as __add__ because I think it
        makes it more efficient, a new object isn't assigned in memory,
        whereas if only __add__ is implemented, a new object gets
        created.
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

    def __sub__(self, other):
        """
        Subtract one Vec3 from another.

        E.g.::
            vec3 = vec1 - vec2
        """
        if isinstance(other, Vec3):
            return Vec3(
                self._e0 - other._e0,
                self._e1 - other._e1,
                self._e2 - other._e2
            )
        else:
            raise TypeError("Cannot add {other_type} to Vec3".format(
                other_type=type(other)
            ))

    def __isub__(self, other):
        """
        Subtract another Vec3 from the current one in place.

        E.g.::
            vec1 -= vec2

        Implementing __isub__ as well as __sub__ because I think it
        makes it more efficient, a new object isn't assigned in memory,
        whereas if only __sub__ is implemented, a new object gets
        created.
        """
        if isinstance(other, Vec3):
            self._e0 -= other._e0
            self._e1 -= other._e1
            self._e2 -= other._e2
            return self
        else:
            raise TypeError("Cannot add {other_type} to Vec3".format(
                other_type=type(other)
            ))

    def __eq__(self, other):
        """
        Check for equality between two Vec3s.

        E.g.::
            vec1 == vec2
        """
        if isinstance(other, Vec3):
            return (
                abs(self._e0 - other._e0) < _TOLERANCE
                and abs(self._e1 - other._e1) < _TOLERANCE
                and abs(self._e2 - other._e2) < _TOLERANCE
            )
        else:
            raise TypeError("Cannot compare {other_type} to Vec3".format(
                other_type=type(other)
            ))

    def _validate_key(self, key):
        """
        Check if key is valid.

        The key is used when indexing into a Vec3. E.g.::
            first_elem = my_vec3[0]

        Args:
            key (int): The key to check.

        Raises:
            TypeError: If the key isn't an int.
            IndexError: If the index is out of range.
        """
        if not isinstance(key, int):
            raise(TypeError("Key is not an int"))

        if key not in (0, 1, 2):
            raise(IndexError("Key is out of range"))

    def _validate_value(self, value):
        """
        Check if a value is valid for a Vec3.

        Raises:
            TypeError: If the value isn't a number.
        """
        if not isinstance(value, numbers.Number):
            raise TypeError("Value passed is not a number.")

    @property
    def x(self):
        """
        Access the first element as x.
        """
        return self._e0

    @x.setter
    def x(self, value):
        self._validate_value(value)
        self._e0 = value

    @property
    def y(self):
        """
        Access the second element as y.
        """
        return self._e1

    @y.setter
    def y(self, value):
        self._validate_value(value)
        self._e1 = value

    @property
    def z(self):
        """
        Access the third element as z.
        """
        return self._e2

    @z.setter
    def z(self, value):
        self._validate_value(value)
        self._e2 = value

    @property
    def r(self):
        """
        Access the first element as r.
        """
        return self._e0

    @r.setter
    def r(self, value):
        self._validate_value(value)
        self._e0 = value

    @property
    def g(self):
        """
        Access the second element as g.
        """
        return self._e1

    @g.setter
    def g(self, value):
        self._validate_value(value)
        self._e1 = value

    @property
    def b(self):
        """
        Access the third element as b.
        """
        return self._e2

    @b.setter
    def b(self, value):
        self._validate_value(value)
        self._e2 = value

    def length(self):
        """
        The length of the Vec3.

        This assumes that the x, y and z values are in an orthonormal
        coordinate system. (I.e. all 3 axes are unit length and at right
        angles to each other.)

        Returns:
            float: The length of the Vec3.
        """
        return math.sqrt(self.length_squared())

    def length_squared(self):
        """
        The length of the Vec3 squared.

        This assumes that the x, y and z values are in an orthonormal
        coordinate system. (I.e. all 3 axes are unit length and at right
        angles to each other.)

        This is useful as it's faster to compare lengths of vectors
        because you don't have to calculate the square root.

        Returns:
            float: The length of the Vec3 squared.
        """
        return (self._e0 ** 2) + (self._e1 ** 2) + (self._e2 ** 2)
