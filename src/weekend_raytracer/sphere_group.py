import math

import numpy

from . import renderable


class SphereGroup():
    """
    Note that we don't use a base class because that slows things down.
    """
    def __init__(self):
        """
        Initialise the object   
        """

        self.centres = None
        self.radii = None
        self.materials = []

    def add_sphere(self, centre, radius, material):
        if self.centres is None:
            self.centres = numpy.array([centre])
            self.radii = numpy.array([radius])
        else:
            self.centres = numpy.append(
                self.centres, numpy.array([centre]), axis=0
            )
            self.radii = numpy.append(
                self.radii, numpy.array([radius]), axis=0
            )

        self.materials.append(material)

    def hit_test(self, ray, t_min, t_max):
        C_to_Os = ray.origin - self.centres
        Hs = numpy.einsum("ij,j->i", C_to_Os, ray.direction)
        Cs = numpy.einsum("ij,ij->i", C_to_Os, C_to_Os) - numpy.square(self.radii)
        discriminants = numpy.square(Hs) - Cs

        # https://stackoverflow.com/questions/52622172/numpy-where-function-can-not-avoid-evaluate-sqrtnegative
        smaller_ts = numpy.full_like(discriminants, t_max + 1)
        mask = discriminants > 0.00001
        smaller_ts[mask] = -Hs[mask] - numpy.sqrt(discriminants[mask])

        larger_ts = numpy.full_like(discriminants, t_max + 1)
        mask = discriminants > 0.00001
        larger_ts[mask] = -Hs[mask] + numpy.sqrt(discriminants[mask])

        smaller_ts[smaller_ts < t_min] = t_max + 1
        larger_ts[larger_ts < t_min] = t_max + 1
        smallest_ts = numpy.minimum(smaller_ts, larger_ts)
        index = numpy.argmin(smallest_ts)
        t = smallest_ts[index]
        if t > t_max:
            return False, None

        hit_point = ray.at(t)
        # Dividing by the radius is a quick way to normalise!
        normal = (hit_point - self.centres[index]) / self.radii[index]
        # hit_point += normal * 0.0001
        side = renderable.Side.FRONT
        # In the typical case the ray is outside the sphere, and
        # the normal is facing "toward" the ray. This means the
        # Cosine of the angle between them will be < 0 (i.e.
        # between 90 and 180).
        #
        # However if the ray is inside the sphere the normal will
        # be facing "away" from the ray and the Cosine of the angle
        # between them will be > 0 (i.e between 0 and 90)
        #
        # If we're inside the sphere we flip the normal so that
        # the normal always faces the camera, and make a note
        # that this is a back facing surface.
        if ray.direction.dot(normal) > 0.0:
            normal *= -1.0
            side = renderable.Side.BACK

        return True, renderable.HitRecord(
            hit_point,
            normal,
            t,
            side,
            self.materials[index]
        )
