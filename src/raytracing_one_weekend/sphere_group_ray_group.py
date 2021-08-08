"""
A group of spheres that can collide with a group of rays.
"""

import numpy

class SphereGroupRayGroup():
    """
    Note that we don't use a base class because that slows things down.
    """
    def __init__(self):
        """
        Initialise the object   
        """

        self.centres = None
        self.radii = None
        self.colours = None

    def add_sphere(self, centre, radius, colour):
        if self.centres is None:
            self.centres = numpy.array([centre], dtype=numpy.single)
            self.radii = numpy.array([radius], dtype=numpy.single)
            self.colours = numpy.array([colour], dtype=numpy.single)
        else:
            self.centres = numpy.append(
                self.centres, numpy.array([centre], dtype=numpy.single), axis=0
            )
            self.radii = numpy.append(
                self.radii, numpy.array([radius], dtype=numpy.single), axis=0
            )
            self.colours = numpy.append(
                self.colours, numpy.array([colour], dtype=numpy.single), axis=0
            )

    def get_hits(self, ray_origins, ray_dirs, t_min, t_max):

        # This is a grid of vectors num_rays by num_spheres in size
        # It's every origin minus every sphere centre
        #
        #    C0    C1    C2
        #    -----------------
        # R0|R0-C0 R0-C1 R0-C2
        # R1|R1-C0 R1-C1 R1-C2
        C_to_Os = ray_origins[:, numpy.newaxis] - self.centres

        # This is a grid of scalars num_rays by num_spheres in size
        # It's as if we take the C_to_Os gris, and then for each row
        # (which corresponds to a ray), find the dot product of the ray for
        # that row and each C_to_O
        #
        #    C0         C1         C2
        #    --------------------------------
        # R0|R0.(R0-C0) R0.(R0-C1) R0.(R0-C2)
        # R1|R1.(R1-C0) R1.(R1-C1) R1.(R1-C2)
        Hs = numpy.einsum("...i,...ki", ray_dirs, C_to_Os)

        # This is a grid of scalars num_rays by num_spheres in size.
        # It's the dot product of each C_to_O with itself, minus the radius
        # of the sphere for that column squared
        #
        #    S0                 S1                 S2
        #    ------------------------------------------------------
        # R0|C20.C20 - S0.r^2   C20.C20 - S1.r^2   C20.C20 - S2.r^2
        # R1|C20.C20 - S0.r^2   C20.C20 - S1.r^2   C20.C20 - S2.r^2
        Cs = numpy.einsum("...ij,...ij->...i", C_to_Os, C_to_Os) - self.radii**2

        # This is a grid of scalars num_rays by num_spheres in size.
        # To avoid nans from negative discriminants, use the max value
        # between the orig disc, and a small +ve num.
        discriminants = numpy.square(Hs) - Cs
        # This saves some memory
        del Cs


        mask = discriminants > 0.00001
        smaller_ts = numpy.full_like(discriminants, t_max + 1.0)
        smaller_ts[mask] = -Hs[mask] - numpy.sqrt(discriminants[mask])
        larger_ts = numpy.full_like(discriminants, t_max + 1.0)
        larger_ts[mask] = -Hs[mask] + numpy.sqrt(discriminants[mask])

        # sqrt_discriminants = numpy.sqrt(numpy.maximum(0.00001, discriminants))

        # # Also a grid of scalars num_rays by num_spheres in size.
        # # For each ray (row) it lists (column) the value of t where that
        # # ray hit the sphere. If it didn't it gets set to a large number.
        # smaller_ts = -Hs - sqrt_discriminants
        # larger_ts = -Hs + sqrt_discriminants

        # This takes the smaller of the two, as long as it's positive
        all_ts = numpy.where(
            (smaller_ts > 0.0) & (smaller_ts < larger_ts),
            smaller_ts,
            larger_ts
        )
        # This saves some memory
        del smaller_ts
        del larger_ts
        # Here we filter out any discriminants that were less than 0
        t_filter = (discriminants > 0.00001) & (all_ts > t_min) & (all_ts < t_max)
        final_ts = numpy.where(t_filter, all_ts, t_max + 1.0)

        # A 1D array num_rays long that contains the index of the
        # sphere with the smallest t
        smallest_t_indecies = numpy.argmin(final_ts, axis=1)

        # A 1D array num_rays long containing the t values for each ray
        smallest_ts = final_ts[numpy.arange(ray_origins.shape[0]), smallest_t_indecies]

        # A 1D array num_rays long that contains the index of the
        # sphere with the smallest t, or -1 if the ray hit no spheres
        sphere_hit_indecies = numpy.where(
            smallest_ts < t_max,
            smallest_t_indecies,
            -1
        )

        ray_hits = smallest_ts < t_max

        # Unlike the other arrays, this isn't num_rays long. There's
        # an entry for each ray that hit
        hit_points = ray_origins[ray_hits] + ray_dirs[ray_hits] * smallest_ts[ray_hits][..., numpy.newaxis]

        # Dividing by the radius is a quick way to normalise!
        hit_normals = (hit_points - self.centres[sphere_hit_indecies[ray_hits]]) / self.radii[sphere_hit_indecies[ray_hits]][..., numpy.newaxis]

        # Displace hit points along normal a tiny bit
        # If you don't do this you get artefacts on large spheres.
        hit_points += hit_normals * 0.0001

        # cosines = numpy.einsum("...ij,...ij->...i", hit_normals, ray_dirs[ray_hits])
        # hit_normals[cosines > 0.0] *= -1.0

        return sphere_hit_indecies, smallest_ts, hit_points, hit_normals