import math

import numpy

from . import renderable


class MTTriangleGroupRayGroup():
    """
    This is a direct implementation of The Muller-Tumbore Algoritm as descibed
    in https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection    

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

    def __init__(self, material_id):
        self.pt0s = None
        self.pt1s = None
        self.pt2s = None
        self.normals = None
        self.As = None
        self.Bs = None
        self.material_id = material_id
        self.bounds_centre = None
        self.bounds_radius = None

    def add_triangle(self, pt0, pt1, pt2):
        """
        Add a triangle to the group
        """

        A = pt1 - pt0
        B = pt2 - pt0
        normal = numpy.cross(A, B)
        normal /= numpy.linalg.norm(normal)

        if self.pt0s is None:
            self.pt0s = numpy.array([pt0])
            self.pt1s = numpy.array([pt1])
            self.pt2s = numpy.array([pt2])

            self.As = numpy.array([A])
            self.Bs = numpy.array([B])

            self.normals = numpy.array([normal])
        else:
            self.pt0s = numpy.append(self.pt0s, numpy.array([pt0]), axis=0)
            self.pt1s = numpy.append(self.pt1s, numpy.array([pt1]), axis=0)
            self.pt2s = numpy.append(self.pt2s, numpy.array([pt2]), axis=0)

            self.As = numpy.append(self.As, numpy.array([A]), axis=0)
            self.Bs = numpy.append(self.Bs, numpy.array([B]), axis=0)

            self.normals = numpy.append(
                self.normals, numpy.array([normal]), axis=0
            )

        self.update_sphere_bounds()

    def update_sphere_bounds(self):
        """
        Update the definition of the sphere that encompasses all the
        triangles.
        """

        all_pts = numpy.concatenate((self.pt0s, self.pt1s, self.pt2s), axis=0)
        average_pos = numpy.mean(all_pts, axis=0)
        avg_to_pt = all_pts - average_pos
        distances_from_average = numpy.sqrt(numpy.einsum("ij,ij->i", avg_to_pt, avg_to_pt))
        max_dist = numpy.max(distances_from_average)
        self.bounds_centre = average_pos
        self.bounds_radius = max_dist * 1.01

    def get_hits(self, ray_origins, ray_dirs, t_min, t_max):
        """
        See if any of the rays hit any of the triangles
        """

        # This is a 2D grid of vectors num rays by num triangles in size
        # where each element is the cross product of the dir and B
        #
        # https://stackoverflow.com/questions/49881468/efficient-way-of-computing-the-cross-products-between-two-sets-of-vectors-numpy
        # This reshapes ray_dirs and self.Bs to:
        # - (<len dirs>, 1, 3)
        # - (1, <len Bs>, 3)
        # By the rules of broadcasting this will be interpreted like two
        # arrays of shape (<len_dirs>, <len_bs>, 3) where dir is
        # repeated along axis 1 and <b is repeated along axis 0
        p_vecs = numpy.cross(
            ray_dirs[:, numpy.newaxis, :],
            self.Bs[numpy.newaxis, :, :]
        )

        # This is a grid of scalars num rays by num triangles.
        # Each element is the dot product of:
        # - The pvec for that row (ray) and column (triangle)
        # - The A for that column (triangle) 
        dets = numpy.einsum("ij,...ij->...i", self.As, p_vecs)
        
        # Find the inverse of all the determinants
        # Another 2D array of scalars
        inv_dets = dets.copy()
        tris_parallel_to_rays = numpy.absolute(dets) < 0.00001
        inv_dets[tris_parallel_to_rays] = 1.0 
        inv_dets = 1.0/inv_dets

        # This is a 2D grid of vectors num rays by num triangles in size
        # where each element is the origin for the ray in each row minus
        # the pt0 for the triangle in each column 
        t_vecs = ray_origins[:, numpy.newaxis] - self.pt0s

        # A 2D gris of scalars num rays by num tris. Each element is the
        # dot product of the vectors in the corresponding position of
        # the input arrays
        Us = numpy.einsum("...ij,...ij->...i", t_vecs, p_vecs) * inv_dets

        # Free some memory
        del p_vecs

        # A 2D grid of vectors num rays by num tris. Each element is
        # the cross product of the elemnt from the corresponding
        # position of the t_vec array, crossed with the A of the
        # triangle for the column
        q_vecs = numpy.cross(
            t_vecs,
            self.As[numpy.newaxis, :]
        )

        # Free some memory
        del t_vecs

        # A 2D grid of scalars num rays by num tris. Each element is
        # the dot product of:
        # - The q_vec from the corresponding position
        # - The ray dir for that row.
        # Then multiplied by the inverse determinant for the
        # corresponding position
        Vs = numpy.einsum("...j,...ij->...i", ray_dirs, q_vecs) * inv_dets

        # A 2D grid of scalars num rays by num tris. Each element is
        # the dot product of:
        # - The Q_vec from the corresponding position
        # - The B for the tirangle for that column
        # Then multiplied by the inverse determinant for the
        # corresponding position
        Ts = numpy.einsum("ij,...ij->...i", self.Bs, q_vecs) * inv_dets

        # Free some memory
        del q_vecs

        miss_by_det = numpy.absolute(dets) < 0.00001

        miss_by_Us = (Us > 1.0) | (Us < 0.0)

        miss_by_Vs = (Vs < 0.0) | ((Us + Vs) > 1.0)

        miss_by_Ts = (Ts < t_min) | (Ts > t_max)

        misses = miss_by_det | miss_by_Us | miss_by_Vs | miss_by_Ts

        Ts[misses] = t_max + 1

        smallest_t_indecies = numpy.argmin(Ts, axis=1)
        final_ts = Ts[numpy.arange(Ts.shape[0]), smallest_t_indecies]
        ray_hits = final_ts < t_max

        hit_points = numpy.zeros((ray_origins.shape[0], 3), dtype=numpy.single)
        hit_points[ray_hits] = ray_origins[ray_hits] + ray_dirs[ray_hits] * final_ts[ray_hits][..., numpy.newaxis]

        hit_normals = numpy.zeros((ray_origins.shape[0], 3), dtype=numpy.single)
        hit_normals[ray_hits] = self.normals[smallest_t_indecies[ray_hits]]

        back_facing = dets[numpy.arange(ray_origins.shape[0]), smallest_t_indecies] < 0.0
        hit_normals[back_facing] *= -1.0

        hit_material_ids = numpy.full((ray_origins.shape[0]), self.material_id, dtype=numpy.ubyte)

        # ray_hits = numpy.full((ray_origins.shape[0]), False)

        return ray_hits, final_ts, hit_points, hit_normals, hit_material_ids, back_facing

        # ray_hits = numpy.any(numpy.less(Ts, t_max), axis=1)


        # Rays that hit (1D array of bools, num rays long)
        # Hit Ts (1D array of scalars, num rays long)
        # Hit Points (1D array of vectors, num rays long)
        # Hit Normals (1D array of vectors, num rays long)
        # Hit Material indecies (1D array of ints, num rays long)
        # Hit Backfaces (1D array of bools, num rays long)

