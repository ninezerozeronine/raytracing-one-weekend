import abc

from enum import Enum, auto


class Side(Enum):
    """
    Whether we hit the front or back of an object.
    """
    FRONT = auto()
    BACK = auto()


class Renderable(abc.ABC):
    """
    Base class for all things that the renderer can "see".
    """

    @abc.abstractmethod
    def hit_test(ray, t_min, t_max):
        """
        Test whether the object was hit by the ray

        Args:
            ray (Ray): The ray being tested against.
            t_min (float): The smallest value of t along the ray that
                is considered a valid hit.
            t_max (float): The largest value of t along the ray that
                is considered a valid hit.
        Returns:
            Tuple(Bool, HitRecord): Whether the ray hit the object, and
            Information about the hit.
        """
        pass


class HitRecord():
    """
    A record of information about a ray hitting an object.
    """

    def __init__(self, hit_point, normal, t, side):
        """
        Initialise the object.

        Args:
            hit_point (Vec3): Where in space the ray hit the object.
            normal (Vec3): The normal of the surface at the hit point
                (this is always facing the ray).
            t (float): How far along the ray the collision occured.
            side (Side): Whether we hit the front or the back of the
                surface.
        """
        self.hit_point = hit_point
        self.normal = normal
        self.t = t
        self.side = side


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
