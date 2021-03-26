from .vec3 import Vec3


class Ray():
    """
    A ray starting from a point in space and going in a direction from
    there.
    """
    def __init__(self, origin, direction):
        """
        Initialise the object.

        Args:
            origin (Vec3): The origin or starting point of the ray.
            direction (Vec3): The direction that the ray is pointed in.
        """

        self.origin = origin
        self.direction = direction

    def at(self, t):
        """
        Get a position along the ray.

        Args:
            t (float): How far along the ray to travel to get to the
                desired position.

                If t is 0, the resultant position will be the origin
                of the ray. If t is 2, the resultant position will be
                the origin, plus moving two units in the direction
                direction is pointing.
        """

        return self.origin + self.direction * t
