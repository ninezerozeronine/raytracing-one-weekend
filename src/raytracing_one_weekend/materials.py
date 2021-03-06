"""
Materials for the objects in a scene.

(As per the renderable we don't use a base class because that slows down execution)
"""
import random
import math

import numpy

from .ray import Ray
from . import renderable

RNG = numpy.random.default_rng()
AXIS_COLOUR_PAIRS = [
    # +X : Red
    (numpy.array([1.0, 0.0, 0.0], dtype=numpy.single), numpy.array([1.0, 0.0, 0.0], dtype=numpy.single)),

    # +Y : Green
    (numpy.array([0.0, 1.0, 0.0], dtype=numpy.single), numpy.array([0.0, 1.0, 0.0], dtype=numpy.single)),

    # +Z : Blue
    (numpy.array([0.0, 0.0, 1.0]), numpy.array([0.0, 0.0, 1.0], dtype=numpy.single)),

    # -X : Pink
    (numpy.array([-1.0, 0.0, 0.0], dtype=numpy.single), numpy.array([1.0, 0.0, 1.0], dtype=numpy.single)),

    # -Y : Yellow
    (numpy.array([0.0, -1.0, 0.0], dtype=numpy.single), numpy.array([1.0, 1.0, 0.0], dtype=numpy.single)),

    # -Z : Cyan
    (numpy.array([0.0, 0.0, -1.0], dtype=numpy.single), numpy.array([0.0, 1.0, 1.0], dtype=numpy.single)),
]



class PointOnHemiSphereCheckerboardMaterial():
    """
    Scatter rays towards points on a hemisphere at the hit point.

    This provides a good approximation to the lambert shading model.

    This comes from https://raytracing.github.io/books/RayTracingInOneWeekend.html#diffusematerials/analternativediffuseformulation.

    A scattered ray bounces off the hitpoint, aiming toward a
    random point on the surface of a hemisphere with the centre of it's
    flat side at the hit point, and the centre/top of the dome pointing
    in the direction of the normal at that point of the surface.
    """

    def __init__(self, scale, offset, colour_a, colour_b):
        """
        Initialise the object.

        Args:
        """

        self.scale = scale
        self.offset = offset
        self.colour_a = colour_a
        self.colour_b = colour_b

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

        ret_colour = None
        if (
                (abs(math.floor(hit_record.hit_point[0] * self.scale[0] + self.offset[0])) % 2)
                ^ (abs(math.floor(hit_record.hit_point[1] * self.scale[1] + self.offset[1])) % 2)
                ^ (abs(math.floor(hit_record.hit_point[2] * self.scale[2] + self.offset[2])) % 2)
                ):
            ret_colour = self.colour_a
        else:
            ret_colour = self.colour_b

        return (
            absorbed,
            ret_colour,
            scattered_ray,
        )


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
        if scatter_direction.dot(scatter_direction) < 0.00001:
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


class NumpyPointOnHemiSphereMaterial():
    def __init__(self, colour):
        """
        Initialise the object.

        Args:
            colour (numpy.array): An RGB 0-1 array representing the
                colour of the material.
        """
        self.colour = colour



    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_backfaces):

        # Generate points in unit hemispheres pointing in the normal direction
        ray_dirs = numpy_random_unit_vecs(hit_points.shape[0])

        # Reverse any points in the wrong hemisphere
        cosine_angles = numpy.einsum("ij,ij->i", ray_dirs, hit_normals)
        facing_wrong_way = cosine_angles < 0.0
        ray_dirs[facing_wrong_way] *= -1.0

        # Bounce ray origins are the hit points we fed in
        # Bounce ray directions are the random points in the hemisphere.

        hit_cols = numpy.full((hit_points.shape[0], 3), self.colour, dtype=numpy.single)

        absorbtions = numpy.full((hit_points.shape[0]), False)

        return hit_points, ray_dirs, hit_cols, absorbtions


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


class NumpyNormalToDiscreteRGBMaterial():
    """
    Colour the surface based on the discretised world normal at that
    point.
    """

    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_backfaces):

        # Generate points in unit hemispheres pointing in the normal direction
        ray_dirs = numpy_random_unit_vecs(hit_points.shape[0])

        # Reverse any points in the wrong hemisphere
        cosine_angles = numpy.einsum("ij,ij->i", ray_dirs, hit_normals)
        facing_wrong_way = cosine_angles < 0.0
        ray_dirs[facing_wrong_way] *= -1.0


        hit_cols = numpy.full((hit_points.shape[0], 3), [0.4, 0.4, 0.4], dtype=numpy.single)
        for axis, colour in AXIS_COLOUR_PAIRS:
            cos_angles = numpy.einsum("j,ij->i", axis, hit_normals)
            hit_cols[cos_angles > 0.8] = colour

        absorbtions = numpy.full((hit_points.shape[0]), False)

        return hit_points, ray_dirs, hit_cols, absorbtions


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
        reflected_direction = reflect(in_ray.direction, hit_record.normal)

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


class NumpyMetalMaterial():
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


    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_backfaces):
        reflected_dirs = numpy_reflect(hit_raydirs, hit_normals)

        hit_cols = numpy.full((hit_points.shape[0], 3), self.colour, dtype=numpy.single)
        absorbtions = numpy.full((hit_points.shape[0]), False)

        if self.fuzziness > 0.0001:
            fuzself.z_offests = numpy_random_unit_vecs(hit_points.shape[0]) * fuzziness
            reflected_dirs += fuzz_offests
            reflected_dirs /= numpy.sqrt(numpy.einsum("ij,ij->i", reflected_dirs, reflected_dirs))[..., numpy.newaxis]

            cos_angles = numpy.einsum("ij,ij->i", reflected_dirs, hit_normals)
            scattered_inside = cos_angles < 0.00001
            hit_cols[scattered_inside] = 0.0
            absorbtions[scattered_inside] = True


        return hit_points, reflected_dirs, hit_cols, absorbtions


def numpy_random_unit_vecs(num_vecs):
    """
    Generate random unit length vectors
    """

    # Start by generating points in the cube that bound the unit sphere
    vecs = RNG.uniform(low=-1.0, high=1.0, size=(num_vecs, 3))
    vecs = vecs.astype(numpy.single)

    # Would be good to optimise this so that we only check the newly
    # regenerated points
    while True:
        lengths_squared = numpy.einsum("ij,ij->i", vecs, vecs)

        # Catch points that lie outside the sphere or very close to the
        # centre.
        invalid_pts = numpy.logical_or(
            lengths_squared > 1.0,
            lengths_squared < 0.00001
        )
        num_bad_pts = numpy.count_nonzero(invalid_pts)
        if num_bad_pts == 0:
            break
        new_pts = RNG.uniform(low=-1.0, high=1.0, size=(num_bad_pts, 3))
        new_pts = new_pts.astype(numpy.single)
        vecs[invalid_pts] = new_pts

    # Normalise all the results
    vecs /= numpy.sqrt(numpy.einsum("ij,ij->i", vecs, vecs))[..., numpy.newaxis]

    return vecs


class DielectricMaterial():
    """
    A dielectic material description
    """

    def __init__(self, index_of_refraction):
        """
        Initialise class.

        Args:
            index_of_refraction (float): The index of refraction of the
                material. E.g. glass = 1.5, water = 1.3
        """

        self.ior = index_of_refraction

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
        colour = numpy.array([1.0, 1.0, 1.0])

        refraction_ratio = self.ior
        if hit_record.side == renderable.Side.FRONT:
            refraction_ratio = 1/self.ior

        cos_theta = min(-in_ray.direction.dot(hit_record.normal), 1.0)
        sin_theta = numpy.sqrt(1.0 - cos_theta**2)
        cannot_refract = refraction_ratio * sin_theta > 1.0

        reflectance = self.reflectance(cos_theta, refraction_ratio)
        reflectance_too_high = reflectance > random.random()

        if cannot_refract or reflectance_too_high:
            scattered_dir = reflect(in_ray.direction, hit_record.normal)
        else:
            scattered_dir = self.refract(
                in_ray.direction, hit_record.normal, refraction_ratio
            )

        # refracted_dir = self.refract(
        #     in_ray.direction, hit_record.normal, refraction_ratio
        # )

        scattered_ray = Ray(hit_record.hit_point, scattered_dir)

        return (
            absorbed,
            colour,
            scattered_ray,
        )

    def refract(self, in_direction, normal, etai_over_etat):
        """
        Calculate the refracted ray.

        I have almost no idea what's going on in here :(. Stolen from
        https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/snell'slaw

        Args:
            in_direction (numpy.array): The direction of the incoming
                ray (needs to be unit length)
            normal (numpy.array): The normal of the surface at the hit
                point.
            etai_over_etat (float): A way of describing the refractive
                indecies of the materials on either side of the boundary
                between them.
        Returns:
            numpy.array: The direction of the refracted ray
        """

        cos_theta = min(-in_direction.dot(normal), 1.0)
        r_out_perp = etai_over_etat * (in_direction + cos_theta * normal)
        r_out_perp_len_squared = r_out_perp.dot(r_out_perp)
        r_out_parallel = -numpy.sqrt(abs(1.0 - r_out_perp_len_squared)) * normal
        return r_out_perp + r_out_parallel

    def reflectance(self, cosine, ref_idx):
        """
        Calculate the reflectance using Schlick's approximation.

        I have no idea whats going on in here. Stolen from:
        https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/schlickapproximation

        Args:
            cosine (float): Cosine of ... some angle :(
            ref_idx (float): A way of describing the refractive
                indecies of the materials on either side of the boundary
                between them.
        Returns:
            float: A reflectance angle?
        """

        r0 = (1 - ref_idx) / (1 + ref_idx)
        r0 = r0**2
        return r0 + (1 - r0) * (1 - cosine)**5


class NumpyDielectricMaterial():
    """
    A dielectic material description
    """

    def __init__(self, index_of_refraction):
        """
        Initialise class.

        Args:
            index_of_refraction (float): The index of refraction of the
                material. E.g. glass = 1.5, water = 1.3
        """

        self.ior = index_of_refraction

    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_backfaces):
        """

        """

        refraction_ratios = numpy.full(hit_raydirs.shape[0], self.ior, dtype=numpy.single)
        frontfaces = numpy.logical_not(hit_backfaces)
        refraction_ratios = numpy.where(frontfaces, 1.0/refraction_ratios, refraction_ratios)

        cos_thetas = numpy.minimum(
            numpy.einsum("ij,ij->i", (-1.0 * hit_raydirs), hit_normals),
            1.0
        )
        sin_thetas = numpy.sqrt(1.0 - cos_thetas ** 2)
        cannot_refract = (refraction_ratios * sin_thetas) > 1.0

        reflectances = self.reflectance(cos_thetas, refraction_ratios)
        reflectance_too_high = reflectances > RNG.uniform(low=0.0, high=1.0, size=(hit_raydirs.shape[0]))

        to_reflect = numpy.logical_or(cannot_refract, reflectance_too_high)
        to_refract = numpy.logical_not(to_reflect)

        scattered_dirs = numpy.full((hit_raydirs.shape[0], 3), 0.0, dtype=numpy.single)
        scattered_dirs[to_reflect] = numpy_reflect(hit_raydirs[to_reflect], hit_normals[to_reflect])

        scattered_dirs[to_refract] = self.refract(
            hit_raydirs[to_refract],
            hit_normals[to_refract],
            refraction_ratios[to_refract]
        )

        hit_cols = numpy.full((hit_points.shape[0], 3), 1.0, dtype=numpy.single)
        absorbtions = numpy.full((hit_points.shape[0]), False)

        return hit_points, scattered_dirs, hit_cols, absorbtions

    def reflectance(self, cosines, ref_idxs):
        """
        Calculate the reflectance using Schlick's approximation.

        I have no idea whats going on in here. Stolen from:
        https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/schlickapproximation

        Args:
            cosines (numpy.ndarray): Cosine of ... some angle :(. 1D array
                of floats.
            ref_idxs (numpy.ndarray): A way of describing the refractive
                indecies of the materials on either side of the boundary
                between them. 1D array of floats
        Returns:
            numpy.ndarray: A reflectance angle? 1D array of floads
        """

        r0 = (1.0 - ref_idxs) / (1.0 + ref_idxs)
        r0 = r0 ** 2
        return r0 + ((1.0 - r0) * ((1.0 - cosines) ** 5))

    def refract(self, in_directions, normals, etai_over_etats):
        """
        Calculate the refracted ray.

        I have almost no idea what's going on in here :(. Stolen from
        https://raytracing.github.io/books/RayTracingInOneWeekend.html#dielectrics/snell'slaw

        Args:
            in_directions (numpy.ndarray): The direction of the incoming
                ray (needs to be unit length). Array of floats, shape
                (n,3).
            normals (numpy.ndarray): The normal of the surface at the hit
                point. Array of floats, shape (n,3).
            etai_over_etats (numpy.ndarray): A way of describing the refractive
                indecies of the materials on either side of the boundary
                between them. 1D array of floats.
        Returns:
            numpy.ndarray: The direction of the refracted ray
        """

        cos_thetas = numpy.minimum(
            numpy.einsum("ij,ij->i", (-1.0 * in_directions), normals),
            1.0
        )
        r_out_perps = etai_over_etats[..., numpy.newaxis] * (in_directions + (cos_thetas[..., numpy.newaxis] * normals))
        r_out_perps_len_squareds = numpy.einsum("ij,ij->i", r_out_perps, r_out_perps)
        r_out_parallels = (-1.0 * numpy.sqrt(numpy.abs(1.0 - r_out_perps_len_squareds)))[..., numpy.newaxis] * normals
        return r_out_perps + r_out_parallels


def reflect(in_direction, surface_normal):
    """
    Reflect a ray off a surface facing a given direction.

    Args:
        in_direction (numpy.array): The direction of the incoming ray (
            must be normalised)

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
    """
    return (
        in_direction
        - (2 * in_direction.dot(surface_normal)) * surface_normal
    )


def numpy_reflect(ray_dirs, surface_normals):
    """
    Find the direction of reflection for a ray hitting a surface with a
    given normal.
    """
    return ray_dirs - (surface_normals * 2.0 * numpy.einsum("ij,ij->i", ray_dirs, surface_normals)[..., numpy.newaxis])


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
        if random_vec.dot(random_vec) > 1.0:
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
