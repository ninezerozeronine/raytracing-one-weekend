import numpy


class Ray():
    """
    A ray starting from a point in space and going in a direction from
    there.
    """
    def __init__(self, origin, direction):
        """
        Initialise the object.

        Args:
            origin (numpy.array): The origin or starting point of the
                ray.
            direction (numpy.array): The direction that the ray is
                pointed in.
        """

        self.origin = origin
        # Normalise the direction on creation - this was causing all
        # sorts of wierdness in the ray-sphere intersection code.
        self.direction = direction / numpy.sqrt(direction.dot(direction))

    def at(self, t):
        """
        Get a position along the ray.

        Args:
            t (float): How far along the ray to travel to get to the
                desired position.

                If t is 0, the resultant position will be the origin
                of the ray. If t is 2, the resultant position will be
                the origin, plus moving twice in the direction that
                direction is pointing.
        """

        return self.origin + self.direction * t
