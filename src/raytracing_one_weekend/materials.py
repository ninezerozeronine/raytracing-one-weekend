"""
Materials for the objects in a scene.

(As per the renderable we don't use a base class because that slows down execution)
"""
import numpy

from .ray import Ray

RNG = numpy.random.default_rng()

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