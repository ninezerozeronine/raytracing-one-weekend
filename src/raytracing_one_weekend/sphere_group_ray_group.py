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
        self.material_indecies = None

    def add_sphere(self, centre, radius, colour, material_index):
        if self.centres is None:
            self.centres = numpy.array([centre], dtype=numpy.single)
            self.radii = numpy.array([radius], dtype=numpy.single)
            self.colours = numpy.array([colour], dtype=numpy.single)
            self.material_indecies = numpy.array([material_index], dtype=numpy.ubyte)
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
            self.material_indecies = numpy.append(
                self.material_indecies, numpy.array([material_index], dtype=numpy.ubyte), axis=0
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

        # Free some memory
        del C_to_Os


        # This is a grid of scalars num_rays by num_spheres in size.
        # To avoid nans from negative discriminants, use the max value
        # between the orig disc, and a small +ve num.
        discriminants = numpy.square(Hs) - Cs

        # Free some memory
        del Cs

        # Also a grid of scalars num_rays by num_spheres in size.
        # For each ray (row) it lists (column) the value of t where that
        # ray hit the sphere. If it didn't it gets set to a large number.
        mask = discriminants > 0.00001
        smaller_ts = numpy.full_like(discriminants, t_max + 1.0)
        smaller_ts[mask] = -Hs[mask] - numpy.sqrt(discriminants[mask])
        larger_ts = numpy.full_like(discriminants, t_max + 1.0)
        larger_ts[mask] = -Hs[mask] + numpy.sqrt(discriminants[mask])

        # Free somememory
        del Hs

        # If the value is less than t_min, set it to a value thats too big
        smaller_ts[smaller_ts < t_min] = t_max + 1
        larger_ts[larger_ts < t_min] = t_max + 1

        # Take the smaller of the two
        smallest_ts = numpy.minimum(smaller_ts, larger_ts)

        # This saves some memory
        del smaller_ts
        del larger_ts

        # A 1D array num_rays long that contains the index of the
        # sphere with the smallest t
        smallest_t_indecies = numpy.argmin(smallest_ts, axis=1)

        # A 1D array num_rays long containing the smallest t values for each ray
        final_ts = smallest_ts[numpy.arange(smallest_ts.shape[0]), smallest_t_indecies]

        # A 1D array num_rays long containing a true/false for whether
        # the ray hit the sphere
        ray_hits = final_ts < t_max

        # Array of points (one point for each ray) where the rays hit.
        # If the ray didn't hit anything, point gets set to 0
        hit_points = numpy.zeros((ray_origins.shape[0], 3), dtype=numpy.single)
        hit_points[ray_hits] = ray_origins[ray_hits] + ray_dirs[ray_hits] * final_ts[ray_hits][..., numpy.newaxis]

        # A 1D array num_rays long that contains the index of the
        # sphere with the smallest t, or -1 if the ray hit no spheres
        sphere_hit_indecies = numpy.where(
            ray_hits,
            smallest_t_indecies,
            -1
        )

        # Array of normals (one normal for each ray) where the rays hit.
        # If the ray didn't hit anything, normal gets set to 0
        # Dividing by the radius is a quick way to normalise!
        hit_normals = numpy.zeros((ray_origins.shape[0], 3), dtype=numpy.single)
        hit_normals[ray_hits] = (hit_points[ray_hits] - self.centres[sphere_hit_indecies[ray_hits]]) / self.radii[sphere_hit_indecies[ray_hits]][..., numpy.newaxis]

        # Displace hit points along normal a tiny bit
        # If you don't do this you get artefacts on large spheres.
        # hit_points += hit_normals * 0.0001

        # Find out if any of the rays hit the back of the sphere
        cosines = numpy.einsum("ij,ij->i", hit_normals, ray_dirs)
        back_facing = cosines > 0.0
        # If they did, reverse the normal so it's facing the incoming ray
        hit_normals[back_facing] *= -1.0

        # A 1D array num rays long that contains the index of the
        # material that the ray hit. If it didn't hit, -1.
        hit_material_indecies = numpy.where(
            ray_hits,
            self.material_indecies[sphere_hit_indecies],
            -1
        )

        return ray_hits, final_ts, hit_points, hit_normals, hit_material_indecies, back_facing