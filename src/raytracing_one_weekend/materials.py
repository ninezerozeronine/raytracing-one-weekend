"""
Materials for the objects in a scene.

(As per the renderable we don't use a base class because that slows down execution)
"""
import numpy

from .ray import Ray

RNG = numpy.random.default_rng()
AXIS_COLOUR_PAIRS = [
    # +X : Red
    (numpy.array([1.0, 0.0, 0.0]), numpy.array([1.0, 0.0, 0.0])),

    # +Y : Green
    (numpy.array([0.0, 1.0, 0.0]), numpy.array([0.0, 1.0, 0.0])),

    # +Z : Blue
    (numpy.array([0.0, 0.0, 1.0]), numpy.array([0.0, 0.0, 1.0])),

    # -X : Pink
    (numpy.array([-1.0, 0.0, 0.0]), numpy.array([1.0, 0.0, 1.0])),

    # -Y : Yellow
    (numpy.array([0.0, -1.0, 0.0]), numpy.array([1.0, 1.0, 0.0])),

    # -Z : Cyan
    (numpy.array([0.0, 0.0, -1.0]), numpy.array([0.0, 1.0, 1.0])),
]



class PointOnHemiSphereMaterial():
    """
    Scatter rays towards points on a hemisphere at the hit point.

    This provides a good approximation to the lambert shading model.

    This comes from https://raytracing.github.io/books/RayTracingInOneWeekend.html#diffusematerials/analternativediffuseformulation.

    A scattered ray bounces off the hitpoint, aiming toward a
    random point on the surface of a hemisphere with the centre of it's
    flat side at the hit point, and the centre/top of the dome pointing
    in the direction of the normal at that point of the surface.
    """

    def __init__(self, colour):
        """
        Initialise the object.

        Args:
            colour (numpy.array): An RGB 0-1 array representing the
                colour of the material.
        """
        self.colour = colour

    def scatter(self, in_ray, hit_record):
        """
        Scatter (or absorb) the incoming ray.

        Args:
            in_ray (Ray): The ray that hit the surface.
            hit_record (HitRecord): Details about the hit between the
                ray and the surface.

        Returns:
            (tuple): tuple containing:
                absorbed (bool): Whether the ray was absorbed or not.
                surface_colour (numpy.array): RGB 0-1 array representing
                    the colour of the surface at the hit point
                scattered_ray (Ray): The ray that bounced off the
                    surface.
        """

        absorbed = False
        scatter_direction = random_vec_in_unit_hemisphere(hit_record.normal)
        # Catch if our randomly generated direction vector is very close to
        # zero
        if scatter_direction.dot(scatter_direction) < 0.000001:
            scatter_direction = hit_record.normal
        scattered_ray = Ray(
            hit_record.hit_point,
            scatter_direction
        )

        return (
            absorbed,
            self.colour,
            scattered_ray,
        )


class PointInUnitSphereMaterial():
    """
    Scatter rays towards points in a unit sphere which it's base at the
    hit point.

    This comes from https://raytracing.github.io/books/RayTracingInOneWeekend.html#diffusematerials/asimplediffusematerial

    A scattered ray bounces from the hit point, toward a random point
    inside a unit sphere, with the centre of the sphere at the tip of
    the unit normal at the hit point.
    """

    def __init__(self, colour):
        """
        Initialise the object.

        Args:
            colour (numpy.array): An RGB 0-1 array representing the
                colour of the material.
        """
        self.colour = colour

    def scatter(self, in_ray, hit_record):
        """
        Scatter (or absorb) the incoming ray.

        The ray starts at the hit point. The direction the ray should
        point in is determined by generating a target point. The
        target point is a random point in a unit sphere, with it's
        centre at the tip of the unit normal at the hit point.

        Args:
            in_ray (Ray): The ray that hit the surface.
            hit_record (HitRecord): Details about the hit between the
                ray and the surface.

        Returns:
            (tuple): tuple containing:
                absorbed (bool): Whether the ray was absorbed or not.
                surface_colour (numpy.array): RGB 0-1 array representing
                    the colour of the surface at the hit point
                scattered_ray (Ray): The ray that bounced off the
                    surface.
        """

        absorbed = False

        dir_target = (
            hit_record.hit_point
            + hit_record.normal
            + random_vec_in_unit_sphere()
        )

        scattered_ray = Ray(
            hit_record.hit_point,
            dir_target - hit_record.hit_point
        )

        return (
            absorbed,
            self.colour,
            scattered_ray,
        )


class PointOnUnitSphereMaterial():
    """
    Scatter rays towards points on a unit sphere which it's base at the
    hit point.

    This comes from https://raytracing.github.io/books/RayTracingInOneWeekend.html#diffusematerials/truelambertianreflection

    A scattered ray bounces from the hit point, toward a random point
    on the surface of a unit sphere, with the centre of the sphere at
    the tip of the unit normal at the hit point.
    """

    def __init__(self, colour):
        """
        Initialise the object.

        Args:
            colour (numpy.array): An RGB 0-1 array representing the
                colour of the material.
        """
        self.colour = colour

    def scatter(self, in_ray, hit_record):
        """
        Scatter (or absorb) the incoming ray.

        The ray starts at the hit point. The direction the ray should
        point in is determined by generating a target point. The
        target point is a random point on a unit sphere with it's 
        centre  at the tip of the unit normal at the hit point.

        Args:
            in_ray (Ray): The ray that hit the surface.
            hit_record (HitRecord): Details about the hit between the
                ray and the surface.

        Returns:
            (tuple): tuple containing:
                absorbed (bool): Whether the ray was absorbed or not.
                surface_colour (numpy.array): RGB 0-1 array representing
                    the colour of the surface at the hit point
                scattered_ray (Ray): The ray that bounced off the
                    surface.
        """

        absorbed = False

        dir_target = (
            hit_record.hit_point
            + hit_record.normal
            + random_unit_vec() 
        )

        bounce_ray = Ray(
            hit_record.hit_point,
            dir_target - hit_record.hit_point
        )

        return (
            absorbed,
            self.colour,
            scattered_ray,
        )


class NormalToRGBMaterial():
    """
    Colour the surface based on the world normal at that point.

    Based on :class:`~PointOnHemiSphereMaterial`.

    Need to check if adding a get_colour method and inheriting from
    the base class comes with a speed penalty.
    """

    def scatter(self, in_ray, hit_record):
        """
        Scatter (or absorb) the incoming ray.

        Args:
            in_ray (Ray): The ray that hit the surface.
            hit_record (HitRecord): Details about the hit between the
                ray and the surface.

        Returns:
            (tuple): tuple containing:
                absorbed (bool): Whether the ray was absorbed or not.
                surface_colour (numpy.array): RGB 0-1 array representing
                    the colour of the surface at the hit point
                scattered_ray (Ray): The ray that bounced off the
                    surface.
        """

        absorbed = False
        scatter_direction = random_vec_in_unit_hemisphere(hit_record.normal)
        # Catch if our randomly generated direction vector is very close to
        # zero
        if scatter_direction.dot(scatter_direction) < 0.000001:
            scatter_direction = hit_record.normal
        scattered_ray = Ray(
            hit_record.hit_point,
            scatter_direction
        )

        colour = numpy.array([
            hit_record.normal[0] + 1.0,
            hit_record.normal[1] + 1.0,
            hit_record.normal[2] + 1.0,
        ]) * 0.5

        return (
            absorbed,
            colour,
            scattered_ray,
        )


class NormalToDiscreteRGBMaterial():
    """
    Colour the surface based on the discretised world normal at that
    point.

    Based on :class:`~PointOnHemiSphereMaterial`.

    Need to check if adding a get_colour method and inheriting from
    the base class comes with a speed penalty.
    """

    def scatter(self, in_ray, hit_record):
        """
        Scatter (or absorb) the incoming ray.

        Args:
            in_ray (Ray): The ray that hit the surface.
            hit_record (HitRecord): Details about the hit between the
                ray and the surface.

        Returns:
            (tuple): tuple containing:
                absorbed (bool): Whether the ray was absorbed or not.
                surface_colour (numpy.array): RGB 0-1 array representing
                    the colour of the surface at the hit point
                scattered_ray (Ray): The ray that bounced off the
                    surface.
        """

        absorbed = False
        scatter_direction = random_vec_in_unit_hemisphere(hit_record.normal)
        # Catch if our randomly generated direction vector is very close to
        # zero
        if scatter_direction.dot(scatter_direction) < 0.000001:
            scatter_direction = hit_record.normal
        scattered_ray = Ray(
            hit_record.hit_point,
            scatter_direction
        )

        ret_colour = numpy.array([1.0, 0.0, 1.0])
        for axis, colour in AXIS_COLOUR_PAIRS:
            cos_angle = axis.dot(hit_record.normal)
            if cos_angle > 0.8:
                ret_colour = colour
                break
        else:
            ret_colour = numpy.array([0.4, 0.4, 0.4])

        return (
            absorbed,
            ret_colour,
            scattered_ray,
        )


class MetalMaterial():
    """
    Reflect rays that hit the material.

    This comes from https://raytracing.github.io/books/RayTracingInOneWeekend.html#metal/mirroredlightreflection
    """

    def __init__(self, colour, fuzziness):
        """
        Initialise the object.

        Args:
            colour (numpy.array): An RGB 0-1 array representing the
                colour of the material.
            fuziness (float): The fuzziness of the reflections of the
                material. Must be greater than or equal to 0, preferably
                less than 1.
        """
        self.colour = colour
        self.fuzziness = fuzziness

    def scatter(self, in_ray, hit_record):
        """
        Scatter (or absorb) the incoming ray.

        The following pieces make up the system in which the reflected
        ray is calcluated:
         * A hit point P.
         * An incoming unit length vector V - the incoming ray
           that has hit the surface.
         * A unit length normal N which is the normal at the hit point.
         * An offset vector B, which is V projected onto N, then
           reversed (so it points in the direction of the normal).
         * The reflected vector R.

        We can consider R = V + 2B by thinking of the incoming vector, V
        starting at P, continuing into the surface, then moving "out" by
        B twice to come back out of the surface.

        As X.Y is the length of X projected onto Y (if Y is unit length)
        we can find B by calculating V.N, multiplying N by the result,
        then multiply again -1 to reverse it.

        To simulate the fuziness the end point of the reflected ray is
        moved to a random location in a sphere centered at the tip of
        the reflected ray. I _believe_ this gives a non uniform result,
        instead picking points on a disk perpendicular to the ray may
        give more uniform results.

        Args:
            in_ray (Ray): The ray that hit the surface.
            hit_record (HitRecord): Details about the hit between the
                ray and the surface.

        Returns:
            (tuple): tuple containing:
                absorbed (bool): Whether the ray was absorbed or not.
                surface_colour (numpy.array): RGB 0-1 array representing
                    the colour of the surface at the hit point
                scattered_ray (Ray): The ray that bounced off the
                    surface.
        """

        ret_colour = self.colour
        absorbed = False
        reflected_direction = (
            in_ray.direction
            - (2 * in_ray.direction.dot(hit_record.normal)) * hit_record.normal
        )

        if self.fuzziness < 0.00001:
            reflected_ray = Ray(
                hit_record.hit_point,
                reflected_direction
            )
        else:
            reflected_ray = Ray(
                hit_record.hit_point,
                reflected_direction + self.fuzziness * random_vec_in_unit_sphere()
            )

        # With fuziness, a reflected ray can end up scattering below/into
        # the surface - this is a catch for that.
        # If the angle between them is > 90 degrees, the cos of the
        # angle is less than 0, and this means we've scattered below the
        # surface.
        
        if reflected_ray.direction.dot(hit_record.normal) < 0.00001:
            absorbed = True

            # Forcing the return colour to be the colour of the metal
            # creates a bit of a halo/edge around the object with the
            # given colour. Perhaps the scatter can just be tried again 
            # if this happens instead?
            ret_colour = numpy.array([0.0, 0.0, 0.0])

        return (
            absorbed,
            ret_colour,
            reflected_ray,
        )


def random_vec_in_unit_hemisphere(hemisphere_direction):
    """
    Generate a vector in a unit hemishpere.

    Args:
        hemisphere_direction (numpy.array): Direction for the
            hemisphere. Doesn't need to be normalised. The flat side
            of the hemisphere is at the base of the vector,
            the top of the curved part is at the tip.
    """

    in_unit_sphere = random_vec_in_unit_sphere()
    if in_unit_sphere.dot(hemisphere_direction) > 0.0:
        return in_unit_sphere
    else:
        return -in_unit_sphere


def random_vec_in_unit_sphere():
    """
    Generate a vector in a sphere with radius 1.
    """
    while True:
        random_vec = RNG.uniform(low=-1, high=1, size=3)
        # If the length of the vector squared (thanks dot product of
        # a vector with itself!) is greater than 1 then we're not in
        # a unit sphere.
        if random_vec.dot(random_vec) > 1:
            continue
        else:
            return random_vec


def random_unit_vec():
    """
    Return unit length vector pointing in a random direction
    """

    while True:
        vec_in_unit_sphere = random_vec_in_unit_sphere()
        len_squared = vec_in_unit_sphere.dot(vec_in_unit_sphere)
        if len_squared > 0.00001:
            unit_vec = vec_in_unit_sphere / numpy.sqrt(len_squared)
            return unit_vec
        else:
            continue
