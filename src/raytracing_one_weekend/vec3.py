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

        # self._validate_value(e0)
        # self._validate_value(e1)
        # self._validate_value(e2)

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
        # self._validate_key(key)

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
        # self._validate_key(key)
        # self._validate_value(value)

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

    def __mul__(self, other):
        """
        Multiply the components of the Vec3 by a scalar value.

        E.g.::
            newvec = vec * 0.3
        """

        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Cannot multiply a Vec3 by a non scalar value."
            )

        return Vec3(
            self._e0 * other,
            self._e1 * other,
            self._e2 * other,
        )

    def __rmul__(self, other):
        """
        Multiply the components of the Vec3 by a scalar value.

        E.g.::
            newvec = 0.3 * vec
        """

        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Cannot multiply a Vec3 by a non scalar value."
            )

        return Vec3(
            self._e0 * other,
            self._e1 * other,
            self._e2 * other,
        )

    def __imul__(self, other):
        """
        Multiply the components of the Vec3 in place by a scalar value.

        E.g.::
            vec *= 0.3

        Implementing __imul__ as well as __mul__ because I think it
        makes it more efficient, a new object isn't assigned in memory,
        whereas if only __mul__ is implemented, a new object gets
        created.
        """

        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Cannot multiply a Vec3 by a non scalar value."
            )

        self._e0 *= other
        self._e1 *= other
        self._e2 *= other
        return self

    def __truediv__(self, other):
        """
        Divide the components of the Vec3 by a scalar value.

        E.g.::
            newvec = vec / 0.3
        """

        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Cannot divide a Vec3 by a non scalar value."
            )

        return Vec3(
            self._e0 / other,
            self._e1 / other,
            self._e2 / other,
        )

    def __itruediv__(self, other):
        """
        Divide the components of the Vec3 in place by a scalar value.

        E.g.::
            vec /= 0.3

        Implementing __idiv__ as well as __div__ because I think it
        makes it more efficient, a new object isn't assigned in memory,
        whereas if only __div__ is implemented, a new object gets
        created.
        """

        if not isinstance(other, numbers.Number):
            raise TypeError(
                "Cannot multiply a Vec3 by a non scalar value."
            )

        self._e0 /= other
        self._e1 /= other
        self._e2 /= other
        return self

    def __repr__(self):
        """

        """
        return "'Vec3({e0}, {e1}, {e2})'".format(
            e0=self._e0,
            e1=self._e1,
            e2=self._e2,
        )

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

    def dot(self, other):
        """
        Calculate dot product between this and another Vec3.

        The result of the dot product is equal to:
         - The length of the projection of this Vec3 onto the other
           Vec3.
         - The product of:
         -- The length of this Vec3
         -- The length of the other Vec3
         -- The Cosine of the angle between them

        https://www.falstad.com/dotproduct/
        https://mathworld.wolfram.com/DotProduct.html

        Args:
            other (Vec3): The other Vec3 to calculate the dot product
                with.
        Raises:
            TypeError: If other is not a Vec3.
        Returns:
            float: The dot product
        """

        if isinstance(other, Vec3):
            return (
                self._e0 * other._e0
                + self._e1 * other._e1
                + self._e2 * other._e2
            )
        else:
            raise TypeError(
                "Cannot calculate dot product with non Vec3 object "
                "({other_type})".format(other_type=type(other))
            )

    def cross(self, other):
        """
        Calculate cross product between this and another Vec3.

        The result of the cross product is equal to:
         - A Vec3 perpendicular to the plane formed by the two Vec3s,
           pointing in the direction governed by the right hand rule. If
           C = A x B then:
         -- If A is your thumb, and B is your index finger, then C will
            point in the direction of your middle finger.
         -- If A is your index finger, and B is your middle finger,
            then C will point in the direction of your thumb.
         - A Vec3 with a magnitude equal to the area of the
           parallelogram formed by the two vectors.

        https://en.wikipedia.org/wiki/Cross_product

        Args:
            other (Vec3): The other Vec3 to calculate the cross product
                with.
        Raises:
            TypeError: If other is not a Vec3.
        Returns:
            Vec3: The cross product Vec3.
        """

        if isinstance(other, Vec3):
            return Vec3(
                (self._e1 * other._e2) - (self._e2 * other._e1),
                (self._e2 * other._e0) - (self._e0 * other._e2),
                (self._e0 * other._e1) - (self._e1 * other._e0),
            )
        else:
            raise TypeError(
                "Cannot calculate cross product with non Vec3 object "
                "({other_type})".format(other_type=type(other))
            )

    def normalise(self):
        """
        Normalise the Vec3 in place so it is unit length.

        This is done by dividing all the elements by the length of the
        Vec3.
        """

        length = self.length()
        self._e0 /= length
        self._e1 /= length
        self._e2 /= length

    def normalised(self):
        """
        Get a normalised version of this Vec3.

        This is done by dividing all the elements by the length of the
        Vec3.

        Returns:
            Vec3: This Vec3, but normalised.
        """
        length = self.length()
        return Vec3(
            self._e0 / length,
            self._e1 / length,
            self._e2 / length,
        )
