import math

import numpy

from . import renderable


class MTTriangle():
    """

    This is a direct implementation of The Muller-Tumbore Algoritm as descibed
    in https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection    

    Note that we don't use a base renderable class because that slows
    things down.
    """
    def __init__(self, pt0, pt1, pt2, material):
        """
        Initialise the object

        The points are defined in a counter clockwise order:

          2
          |\
        B | \
          |  \
          0---1
            A

        A = 0 -> 1
        B = 0 -> 2
        C = AxB

        A is X-like, B is Y-like and C is Z-like from a right handed
        coordinate system.
        """

        self.pt0 = pt0
        self.pt1 = pt1
        self.pt2 = pt2
        self.A = pt1 - pt0
        self.B = pt2 - pt0
        self.normal = numpy.cross(self.A, self.B)
        self.normal = self.normal / numpy.linalg.norm(self.normal)
        self.material = material

    def hit_test(self, ray, t_min, t_max):
        """
        Check if a ray intesects the triangle
        """

        # This is more or less magic which I don't understand - there's
        # lots of detail in the scratachapixel article about how this
        # relates to transforming the triangle into barycentric
        # coordinate space and coalculating matrix determinants - all
        # very fancy.
        p_vec = numpy.cross(ray.direction, self.B)
        determinant = self.A.dot(p_vec)

        # Check if the ray and triangle are parallel
        if abs(determinant) < 0.00001:
            return False, None

        inverse_determinant = 1/determinant

        t_vec = ray.origin - self.pt0

        u = t_vec.dot(p_vec) * inverse_determinant
        if (u > 1) or (u < 0):
            return False, None

        q_vec = numpy.cross(t_vec, self.A)
        v = ray.direction.dot(q_vec) * inverse_determinant
        if (v < 0) or ((u + v) > 1):
            return False, None

        t = self.B.dot(q_vec) * inverse_determinant
        # Check if t is within range (this also culls triangles behind
        # the camera)
        if t < t_min or t > t_max:
            return False, None

        hit_point = ray.at(t)
        side = renderable.Side.FRONT
        normal = self.normal
        # Check if the triangle is back facing
        if determinant < 0:
            side = renderable.Side.BACK
            normal = self.normal * -1.0

        return True, renderable.HitRecord(
            hit_point,
            normal,
            t,
            side,
            self.material
        )
