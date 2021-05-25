from enum import Enum, auto


class Side(Enum):
    """
    Whether we hit the front or back of an object.
    """
    FRONT = auto()
    BACK = auto()


class HitRecord():
    """
    A record of information about a ray hitting an object.
    """

    def __init__(self, hit_point, normal, t, side, material):
        """
        Initialise the object.

        Args:
            hit_point (numpy.array): Where in space the ray hit the
                object.
            normal (numpy.array): The normal of the surface at the hit
                point (this is always facing the ray).
            t (float): How far along the ray the collision occured.
            side (Side): Whether we hit the front or the back of the
                surface.
            material(object): The material of the object at the hit
                point.
        """
        self.hit_point = hit_point
        self.normal = normal
        self.t = t
        self.side = side
        self.material = material


class World():
    """
    All the objects, and an iterface to see them.
    """

    def __init__(self):
        """
        Initialise the object
        """

        self.renderables = []

    def hit(self, ray, t_min, t_max):
        """

        """

        hit_anything = False
        closest_hit_t = t_max
        closest_hit_record = None

        for renderable in self.renderables:
            hit, hit_record = renderable.hit_test(ray, t_min, closest_hit_t)
            if hit:
                hit_anything = True
                closest_hit_t = hit_record.t
                closest_hit_record = hit_record

        return hit_anything, closest_hit_record
