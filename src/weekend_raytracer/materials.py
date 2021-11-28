"""
Materials for the objects in a scene.

(As per the renderable we don't use a base class because that slows down execution)
"""
import random
import math

import numpy
from PIL import Image

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


class Diffuse():
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



    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_uvs, hit_backfaces):

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


class TexturedDiffuse():
    def __init__(self, texture_path):
        """
        Initialise the object.

        Args:
            texture_path (str): Path to texture
        """
        texture = Image.open(texture_path)
        width = texture.width
        height = texture.height
        self.smallest_side = float(min((width, height)))

        tex_mode = texture.mode
        tex_mode_map = {
            "RGB": 3,
            "RGBA": 4
        }
        if tex_mode not in tex_mode_map:
            raise Exception(f"Unsupported texture image mode: {tex_mode}")

        self.texture_pixels = numpy.array(texture.getdata(), dtype=numpy.single)
        self.texture_pixels /= 255.0
        self.texture_pixels = self.texture_pixels.reshape(
            (height, width, tex_mode_map[tex_mode])
        )
        self.texture_pixels = self.texture_pixels[:, :, 0:3]

        self.texture_pixels = numpy.flipud(self.texture_pixels)
        # self.texture_pixels = numpy.fliplr(self.texture_pixels)

    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_uvs, hit_backfaces):

        # Generate points in unit hemispheres pointing in the normal direction
        ray_dirs = numpy_random_unit_vecs(hit_points.shape[0])

        # Reverse any points in the wrong hemisphere
        cosine_angles = numpy.einsum("ij,ij->i", ray_dirs, hit_normals)
        facing_wrong_way = cosine_angles < 0.0
        ray_dirs[facing_wrong_way] *= -1.0

        # Bounce ray origins are the hit points we fed in
        # Bounce ray directions are the random points in the hemisphere.

        clipped_uvs = numpy.clip(hit_uvs, 0.0, 1.0)
        mapped_uvs = clipped_uvs * (self.smallest_side - 1.0)
        discretised_uvs = mapped_uvs.astype(numpy.intc)
        hit_cols = self.texture_pixels[
            discretised_uvs[:, 1],
            discretised_uvs[:, 0],
        ]

        # col_choice = hit_uvs[:, 0] > 0.5
        # col_choice = hit_uvs[:, 1] > 0.5
        # hit_cols = numpy.where(
        #     col_choice[:, numpy.newaxis],
        #     numpy.array([0.1, 0.8, 0.2]),
        #     numpy.array([0.2, 0.1, 0.1])
        # )

        absorbtions = numpy.full((hit_points.shape[0]), False)

        return hit_points, ray_dirs, hit_cols, absorbtions


class CheckerboardDiffuse():
    def __init__(self, scale, offset, colour_a, colour_b):
        """
        Initialise the object.

        Args:
        """
        self.scale = scale
        self.offset = offset
        self.colour_a = colour_a
        self.colour_b = colour_b

    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_uvs, hit_backfaces):

        # Generate points in unit hemispheres pointing in the normal direction
        ray_dirs = numpy_random_unit_vecs(hit_points.shape[0])

        # Reverse any points in the wrong hemisphere
        cosine_angles = numpy.einsum("ij,ij->i", ray_dirs, hit_normals)
        facing_wrong_way = cosine_angles < 0.0
        ray_dirs[facing_wrong_way] *= -1.0

        # Bounce ray origins are the hit points we fed in
        # Bounce ray directions are the random points in the hemisphere.

        Xs = numpy.remainder(numpy.fabs(numpy.floor(hit_points[:, 0] * self.scale[0] + self.offset[0])), 2)
        Ys = numpy.remainder(numpy.fabs(numpy.floor(hit_points[:, 1] * self.scale[1] + self.offset[1])), 2)
        Zs = numpy.remainder(numpy.fabs(numpy.floor(hit_points[:, 2] * self.scale[2] + self.offset[2])), 2)
        col_choice = numpy.logical_xor(Xs, numpy.logical_xor(Ys, Zs))
        hit_cols = numpy.where(
            col_choice[:, numpy.newaxis],
            self.colour_a,
            self.colour_b
        )

        absorbtions = numpy.full((hit_points.shape[0]), False)

        return hit_points, ray_dirs, hit_cols, absorbtions


class NormalToRGBDiffuse():
    """
    Colour the surface based on the world normal at that
    point.
    """

    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_uvs, hit_backfaces):

        # Generate points in unit hemispheres pointing in the normal direction
        ray_dirs = numpy_random_unit_vecs(hit_points.shape[0])

        # Reverse any points in the wrong hemisphere
        cosine_angles = numpy.einsum("ij,ij->i", ray_dirs, hit_normals)
        facing_wrong_way = cosine_angles < 0.0
        ray_dirs[facing_wrong_way] *= -1.0


        hit_cols = (hit_normals + 1.0) * 0.5

        absorbtions = numpy.full((hit_points.shape[0]), False)

        return hit_points, ray_dirs, hit_cols, absorbtions


class NormalToDiscreteRGBDiffuse():
    """
    Colour the surface based on the discretised world normal at that
    point.
    """

    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_uvs, hit_backfaces):

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


class Metal():
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


    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_uvs, hit_backfaces):
        """
        To simulate the fuziness the end point of the reflected ray is
        moved to a random location in a sphere centered at the tip of
        the reflected ray. I _believe_ this gives a non uniform result,
        instead picking points on a disk perpendicular to the ray may
        give more uniform results.
        """

        reflected_dirs = numpy_reflect(hit_raydirs, hit_normals)

        hit_cols = numpy.full((hit_points.shape[0], 3), self.colour, dtype=numpy.single)
        absorbtions = numpy.full((hit_points.shape[0]), False)

        if self.fuzziness > 0.0001:
            fuzself.z_offests = numpy_random_unit_vecs(hit_points.shape[0]) * fuzziness
            reflected_dirs += fuzz_offests
            reflected_dirs /= numpy.sqrt(numpy.einsum("ij,ij->i", reflected_dirs, reflected_dirs))[..., numpy.newaxis]

            # With fuziness, a reflected ray can end up scattering below/into
            # the surface - this is a catch for that.
            # If the angle between them is > 90 degrees, the cos of the
            # angle is less than 0, and this means we've scattered below the
            # surface.

            cos_angles = numpy.einsum("ij,ij->i", reflected_dirs, hit_normals)
            scattered_inside = cos_angles < 0.00001

            # Forcing the return colour to be the colour of the metal (rather than black)
            # creates a bit of a halo/edge around the object with the
            # given colour. Perhaps the scatter can just be tried again 
            # if this happens instead?
            hit_cols[scattered_inside] = 0.0
            absorbtions[scattered_inside] = True


        return hit_points, reflected_dirs, hit_cols, absorbtions


class Dielectric():
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

    def scatter(self, hit_raydirs, hit_points, hit_normals, hit_uvs, hit_backfaces):
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


def numpy_reflect(ray_dirs, surface_normals):
    """
    Find the direction of reflection for a ray hitting a surface with a
    given normal.

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
    return ray_dirs - (surface_normals * 2.0 * numpy.einsum("ij,ij->i", ray_dirs, surface_normals)[..., numpy.newaxis])
