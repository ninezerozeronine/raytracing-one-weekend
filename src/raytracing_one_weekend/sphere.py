from . import hittable


class Sphere(Hittable):
    def hit_test(ray, t_min, t_max):
        return False, None
