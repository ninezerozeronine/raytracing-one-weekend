import math

import numpy

from . import renderable


class Sphere():
    def __init__(self, centre, radius):
        """
        Initialise the object   
        """

        self.centre = centre
        self.radius = radius

    def hit_test(self, ray, t_min, t_max):
        """
        Check if a ray intersects this sphere.

        Note that the direction of the ray needs to be normalised - not for
        the hit detection, but for the correct calculation of the hit point.

        A point is on a sphere if the length of the vector from the centre
        of the sphere to a point on the sphere is equal to the radius of
        the sphere.

        If you take the dot product of a vector with itself, this is the
        length squared.

        If R is the vector from the center to a point on the sphere, and r
        is the radius then if R.R = r^2 the point is on the sphere.

        R is the vector from the centre (C) to a point (P) this means
        R = P - C (i.e. centre to origin, then origin to point).

        So if (P - C).(P - C) = r^2, the point is on the sphere.

        We want to find if a point on a ray intersects the sphere. We can
        plug in the point on ray representation (Pt on Ray = O + tD, or
        origin + multiplier of the direction) to the equation above:

        (O + tD - C).(O + tD - C) = r^2

        Exapnd this out and we get:

        O.O + t(D.O) - C.O

        + t(D.O) + t^2(D.D) - t(C.D)

        - C.O - t(C.D) + C.C = r^2

        Collecting terms we get:

        O.O + 2t(D.O) - 2(C.O) + t^2(D.D) - 2t(C.D) + C.C = r^2

        Note that (O - C).(O - C) = O.O - 2(C.O) + C.C so we can simplify
        to:

        2t(D.O) + t^2(D.D) -2t(C.D) + (O - C).(O - C) = r^2

        Collapsing the 2t factors and re-arranging a bit:

        t^2(D.D) + 2tD.(O - C) + (O - C).(O - C) = r^2

        Re arrange to equal zero and we have a quadratic in terms of t!

        t^2(D.D) + 2tD.(O - C) + (O - C).(O - C) - r^2 = 0

        Where:

        A = D.D
        B = 2D.(O - C)
        C = (O - C).(O - C) - r^2

        (O - C) is the vector from the centre of the sphere to the ray
        origin - C to O

        Using our old friend the quadratic equation::

            x = (-B +/- sqrt(B^2 - 4AC)) / (2A)

        We know that if B^2 - 4AC is less than 0 the equation has no
        roots - or - the ray doesn't intersect the sphere!

        Note that B has a factor of 2 in it, so if we consider that B = 2H
        we can factor out the 2 and simplify the calculation a bit:

        (-B +/- sqrt(B^2 - 4AC)) / 2A

        (-2H +/- sqrt((2H)^2 - 4AC)) / 2A

        (-2H +/- 2 x (sqrt(H^2 - AC))) / 2A

        (-H +/- sqrt(H^2 - AC)) / A

        As the direction is normalised, and dotting something with itself
        is the length squared, A is 1 so it can be ignored/removed from
        the last equation above.

        -H +/- sqrt(H^2 - C)
        """

        C_to_O = ray.origin - self.centre

        H = ray.direction.dot(C_to_O)
        C = C_to_O.dot(C_to_O) - self.radius**2
        discriminant = H**2 - C

        # This rules out very tiny discriminants - or glancing
        # hits on the sphere as well as complete misses.
        # If the glancing hits are left in, they can cause precision
        # issues and normals can flip when they're not meant to.
        #
        # Seeing as it's a glancing hit anyway, just call it a miss.
        if discriminant < 0.00001:
            # The ray didn't intersect the sphere - no hit and no hit
            # record.
            return False, None
        else:
            # The ray did intersect the sphere, calculate the t value
            # where the hit occured.
            #
            # We calculate the smaller of the two first (i.e. the one
            # closer to the camera) by using the - of the +/- option in
            # the quadratic root
            #
            # We also need it to be within the range of t_min and t_max.
            t = -H - math.sqrt(discriminant)
            if t < t_min or t > t_max:
                t = -H + math.sqrt(discriminant)
                if t < t_min or t > t_max:
                    # Neither root was suitable.
                    return False, None

            hit_point = ray.at(t)
            # Dividing by the radius is a quick way to normalise!
            normal = (hit_point - self.centre) / self.radius
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
                side
            )
