import abc


class Hittable(abc.ABC):
    """
    Base clas for all things that the renderer can "see".
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
