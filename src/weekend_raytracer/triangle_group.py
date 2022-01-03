import math

import numpy
import psutil
import humanize

from .ray_results import RayResults

class TriangleGroup():
    r"""
    This is a direct implementation of The Muller-Tumbore Algoritm as descibed
    in https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection    

    The points are defined in a counter clockwise order when looking
    from the direction the normal should point in::

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
        self.uv0s = None
        self.uv1s = None
        self.uv2s = None
        self.normals = None
        self.As = None
        self.Bs = None
        self.material_id = material_id
        self.bounds_centre = None
        self.bounds_radius = None
        self.num_triangles = 0

    def add_triangle(
            self,
            pt0,
            pt1,
            pt2,
            uv0=(0.0, 0.0),
            uv1=(0.0, 0.0),
            uv2=(0.0, 0.0),
            normal0=(0,1,0),
            normal1=(0,1,0),
            normal2=(0,1,0)
        ):
        """
        Add a triangle to the group
        """

        A = pt1 - pt0
        B = pt2 - pt0
        normal = numpy.cross(A, B)
        normal /= numpy.linalg.norm(normal)

        if self.num_triangles == 0:
            self.pt0s = numpy.array([pt0], dtype=numpy.single)
            self.pt1s = numpy.array([pt1], dtype=numpy.single)
            self.pt2s = numpy.array([pt2], dtype=numpy.single)

            self.uv0s = numpy.array([uv0], dtype=numpy.single)
            self.uv1s = numpy.array([uv1], dtype=numpy.single)
            self.uv2s = numpy.array([uv2], dtype=numpy.single)

            self.normal0s = numpy.array([normal0], dtype=numpy.single)
            self.normal1s = numpy.array([normal1], dtype=numpy.single)
            self.normal2s = numpy.array([normal2], dtype=numpy.single)

            self.As = numpy.array([A], dtype=numpy.single)
            self.Bs = numpy.array([B], dtype=numpy.single)

            self.normals = numpy.array([normal], dtype=numpy.single)
        else:
            self.pt0s = numpy.append(self.pt0s, numpy.array([pt0], dtype=numpy.single), axis=0)
            self.pt1s = numpy.append(self.pt1s, numpy.array([pt1], dtype=numpy.single), axis=0)
            self.pt2s = numpy.append(self.pt2s, numpy.array([pt2], dtype=numpy.single), axis=0)

            self.uv0s = numpy.append(self.uv0s, numpy.array([uv0], dtype=numpy.single), axis=0)
            self.uv1s = numpy.append(self.uv1s, numpy.array([uv1], dtype=numpy.single), axis=0)
            self.uv2s = numpy.append(self.uv2s, numpy.array([uv2], dtype=numpy.single), axis=0)

            self.normal0s = numpy.append(self.normal0s, numpy.array([normal0], dtype=numpy.single), axis=0)
            self.normal1s = numpy.append(self.normal1s, numpy.array([normal1], dtype=numpy.single), axis=0)
            self.normal2s = numpy.append(self.normal2s, numpy.array([normal2], dtype=numpy.single), axis=0)

            self.As = numpy.append(self.As, numpy.array([A], dtype=numpy.single), axis=0)
            self.Bs = numpy.append(self.Bs, numpy.array([B], dtype=numpy.single), axis=0)

            self.normals = numpy.append(
                self.normals, numpy.array([normal], dtype=numpy.single), axis=0
            )
        self.num_triangles += 1

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

    def _get_num_chunks(self, num_rays):

        p_vecs = num_rays * self.num_triangles * 3 * 4
        dets = num_rays * self.num_triangles * 4
        inv_dets = num_rays * self.num_triangles * 4
        t_vecs = num_rays * self.num_triangles * 3 * 4
        Us = num_rays * self.num_triangles * 4

        total_bytes_reqd = p_vecs + dets + inv_dets + t_vecs + Us

        mem_info = psutil.virtual_memory()

        # Plan to use all free memory in the system minus 10% of the
        # total memory as a safety buffer.
        allowable_bytes = mem_info.available - (mem_info.total * 0.1)

        print(f"Available mem: {humanize.naturalsize(mem_info.available, binary=True)}")
        print(f"Allowable mem: {humanize.naturalsize(allowable_bytes, binary=True)}")
        print(f"Total mem required: {humanize.naturalsize(total_bytes_reqd, binary=True)}")

        # If there's enough space, do it all in one chunk
        if total_bytes_reqd < allowable_bytes:
            return 1

        # Otherwise divide into chunks such that each chunk is smaller
        # than the allowable size 
        return math.ceil(total_bytes_reqd/allowable_bytes)


    def get_hits(self, ray_origins, ray_dirs, t_min, t_max):
        """
        See if any of the rays hit any of the triangles
        """

        num_rays = ray_origins.shape[0]

        # First check to see if any of the rays hit the bounding sphere
        C_to_Os = ray_origins - self.bounds_centre
        Hs = numpy.einsum("ij,ij->i", ray_dirs, C_to_Os)
        Cs = numpy.einsum("ij,ij->i", C_to_Os, C_to_Os) - self.bounds_radius**2
        discriminants = Hs**2 - Cs

        # This isn't correct! Need to calculate the T values
        # otherwise we might be hitting a sphere behind the ray.
        sphere_hits = discriminants > 0.0001
        num_sphere_hits = numpy.sum(sphere_hits)

        print(f"Num rays: {num_rays}")
        print(f"Num sphere hits: {num_sphere_hits}")

        all_ray_results = RayResults()
        all_ray_results.set_blank(num_rays, t_max + 1, self.material_id)

        # all_ray_hits = numpy.full(num_rays, False)
        # all_hit_ts = numpy.full(num_rays, t_max + 1)
        # all_hit_pts = numpy.zeros((num_rays, 3), dtype=numpy.single)
        # all_hit_normals = numpy.zeros((num_rays, 3), dtype=numpy.single)
        # all_hit_uvs = numpy.zeros((num_rays, 2), dtype=numpy.single)
        # all_back_facing = numpy.full(num_rays, False)
        # all_hit_material_indecies = numpy.full((num_rays), self.material_id, dtype=numpy.ubyte)

        if num_sphere_hits == 0:
            return all_ray_results
            # return (
            #     all_ray_hits,
            #     all_hit_ts,
            #     all_hit_pts,
            #     all_hit_normals,
            #     all_hit_uvs,
            #     all_hit_material_indecies,
            #     all_back_facing
            # )
        ray_origins_sphere_hits = ray_origins[sphere_hits]
        ray_dirs_sphere_hits = ray_dirs[sphere_hits]

        num_chunks = self._get_num_chunks(int(num_sphere_hits))

        if num_chunks == 1:
            sphere_hit_results = self._get_hits(
                ray_origins_sphere_hits,
                ray_dirs_sphere_hits,
                t_min,
                t_max
            )
            # (
            #     ray_hits,
            #     hit_ts,
            #     hit_pts,
            #     hit_normals,
            #     hit_uvs,
            #     hit_material_indecies,
            #     back_facing
            # ) = self._get_hits(ray_origins_sphere_hits, ray_dirs_sphere_hits, t_min, t_max)
        else:
            ray_origins_chunks = numpy.array_split(ray_origins_sphere_hits, num_chunks)
            ray_dirs_chunks = numpy.array_split(ray_dirs_sphere_hits, num_chunks)

            print(f"Chunk 1 of {num_chunks}")
            sphere_hit_results = self._get_hits(
                ray_origins_chunks[0],
                ray_dirs_chunks[0],
                t_min,
                t_max
            )
            # (
            #     ray_hits,
            #     hit_ts,
            #     hit_pts,
            #     hit_normals,
            #     hit_uvs,
            #     hit_material_indecies,
            #     back_facing
            # ) = self._get_hits(
            #     ray_origins_chunks[0],
            #     ray_dirs_chunks[0],
            #     t_min,
            #     t_max
            # )

            for chunk_index in range(1, num_chunks):
                print(f"Chunk {chunk_index + 1} of {num_chunks}")
                sphere_hit_chunk = self._get_hits(
                    ray_origins_chunks[chunk_index],
                    ray_dirs_chunks[chunk_index],
                    t_min,
                    t_max
                )
                # (
                #     ray_hits_chunk,
                #     hit_ts_chunk,
                #     hit_pts_chunk,
                #     hit_normals_chunk,
                #     hit_uvs_chunk,
                #     hit_material_indecies_chunk,
                #     back_facing_chunk
                # ) = self._get_hits(
                #     ray_origins_chunks[chunk_index],
                #     ray_dirs_chunks[chunk_index],
                #     t_min,
                #     t_max
                # )

                sphere_hit_results.concatenate(sphere_hit_chunk)

                # ray_hits = numpy.concatenate((ray_hits, ray_hits_chunk), axis=0)
                # hit_ts = numpy.concatenate((hit_ts, hit_ts_chunk), axis=0)
                # hit_pts = numpy.concatenate((hit_pts, hit_pts_chunk), axis=0)
                # hit_normals = numpy.concatenate((hit_normals, hit_normals_chunk), axis=0)
                # hit_uvs = numpy.concatenate((hit_uvs, hit_uvs_chunk), axis=0)
                # hit_material_indecies = numpy.concatenate((hit_material_indecies, hit_material_indecies_chunk), axis=0)
                # back_facing = numpy.concatenate((back_facing, back_facing_chunk), axis=0)

        all_ray_results.merge(sphere_hits, sphere_hit_results)

        # all_ray_hits[sphere_hits] = ray_hits
        # all_hit_ts[sphere_hits] = hit_ts
        # all_hit_pts[sphere_hits] = hit_pts
        # all_hit_normals[sphere_hits] = hit_normals
        # all_hit_uvs[sphere_hits] = hit_uvs
        # all_hit_material_indecies[sphere_hits] = hit_material_indecies
        # all_back_facing[sphere_hits] = back_facing

        return all_ray_results

        # return (
        #     all_ray_hits,
        #     all_hit_ts,
        #     all_hit_pts,
        #     all_hit_normals,
        #     all_hit_uvs,
        #     all_hit_material_indecies,
        #     all_back_facing
        # )


    def _get_hits(self, ray_origins, ray_dirs, t_min, t_max):

        # This is more or less magic which I don't understand - there's
        # lots of detail in the scratachapixel article about how this
        # relates to transforming the triangle into barycentric
        # coordinate space and calculating matrix determinants - all
        # very fancy.
        num_rays = ray_origins.shape[0]

        mem_info = psutil.virtual_memory()
        begin_avail = mem_info.available

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
            ray_dirs[:, numpy.newaxis],
            self.Bs[numpy.newaxis, :]
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

        mem_info = psutil.virtual_memory()
        # print(f"After Cs Available mem: {humanize.naturalsize(mem_info.available, binary=True)}")

        end_avail = mem_info.available

        used_mem = begin_avail - end_avail
        print(f"Used mem: {humanize.naturalsize(used_mem, binary=True)}")


        # triangle_misses = numpy.sum((Us > 1.0) | (Us < 0.0), axis=0) == num_sphere_hits
        # print(f"triangle_misses: {triangle_misses}")
        # triangle_hits = numpy.logical_not(triangle_misses)

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
        # q_vecs = numpy.cross(
        #     t_vecs[:, triangle_hits, :],
        #     self.As[numpy.newaxis, triangle_hits]
        # )
        # print(f"q_vecs shape: {q_vecs.shape}")

        # Free some memory
        del t_vecs

        # A 2D grid of scalars num rays by num tris. Each element is
        # the dot product of:
        # - The q_vec from the corresponding position
        # - The ray dir for that row.
        # Then multiplied by the inverse determinant for the
        # corresponding position
        Vs = numpy.einsum("...j,...ij->...i", ray_dirs, q_vecs) * inv_dets
        # Vs = numpy.einsum("...j,...ij->...i", ray_dirs[sphere_hits], q_vecs) * inv_dets[:, triangle_hits]

        # A 2D grid of scalars num rays by num tris. Each element is
        # the dot product of:
        # - The Q_vec from the corresponding position
        # - The B for the triangle for that column
        # Then multiplied by the inverse determinant for the
        # corresponding position
        Ts = numpy.einsum("ij,...ij->...i", self.Bs, q_vecs) * inv_dets
        # Ts = numpy.einsum("ij,...ij->...i", self.Bs[triangle_hits], q_vecs) * inv_dets[:, triangle_hits]


        # Free some memory
        del q_vecs

        miss_by_det = numpy.absolute(dets) < 0.00001
        # miss_by_det = numpy.absolute(dets[:, triangle_hits]) < 0.00001
        # print(f"miss_by_det shape: {miss_by_det.shape}")
        # print(f"triangles missed per ray by det: {numpy.sum(miss_by_det, axis=1)}")

        miss_by_Us = (Us > 1.0) | (Us < 0.0)
        # miss_by_Us = (Us[:, triangle_hits] > 1.0) | (Us[:, triangle_hits] < 0.0)
        # print(f"miss_by_Us shape: {miss_by_Us.shape}")
        # print(f"triangles missed per ray by Us: {numpy.sum(miss_by_Us, axis=1)}")
        # print(f"Rays missed per triangle by Us: {numpy.sum(miss_by_Us, axis=0)}")

        miss_by_Vs = (Vs < 0.0) | ((Us + Vs) > 1.0)
        # miss_by_Vs = (Vs < 0.0) | ((Us[:, triangle_hits] + Vs) > 1.0)
        # print(f"miss_by_Vs shape: {miss_by_Vs.shape}")
        # print(f"triangles missed per ray by Vs: {numpy.sum(miss_by_Vs, axis=1)}")
        # print(f"Rays missed per triangle by Vs: {numpy.sum(miss_by_Vs, axis=0)}")

        miss_by_Ts = (Ts < t_min) | (Ts > t_max)
        # print(f"miss_by_Ts shape: {miss_by_Ts.shape}")
        # print(f"triangles missed per ray by Ts: {numpy.sum(miss_by_Ts, axis=1)}")

        misses = miss_by_det | miss_by_Us | miss_by_Vs | miss_by_Ts

        Ts[misses] = t_max + 1

        # Array num_rays in length. The index is into the list
        # of triangles that hit, not the overall list of triangles
        smallest_t_indecies = numpy.argmin(Ts, axis=1)

        # Array num_rays in length, contains the smallest t value
        # for that ray
        final_ts = Ts[numpy.arange(num_rays), smallest_t_indecies]

        # Array num rays long
        # final_ts = numpy.full(num_rays, t_max + 1)

        # Splice the ts from the ray hits into all the rays
        # final_ts[sphere_hits] = smallest_ts

        ray_hits = numpy.full(num_rays, False)
        ray_hits = final_ts < t_max

        hit_points = numpy.zeros((num_rays, 3), dtype=numpy.single)
        hit_points[ray_hits] = ray_origins[ray_hits] + ray_dirs[ray_hits] * final_ts[ray_hits][..., numpy.newaxis]
        # hit_points[sphere_hits] = ray_origins[sphere_hits] + ray_dirs[sphere_hits] * smallest_ts[..., numpy.newaxis]

        hit_normals = numpy.zeros((num_rays, 3), dtype=numpy.single)
        # hit_normals[ray_hits] = self.normals[smallest_t_indecies[ray_hits]]
        # hit_normals[sphere_hits] = self.normals[smallest_t_indecies]
        # hit_normals[sphere_hits] = self.normals[triangle_hits][smallest_t_indecies]

        # print(f"self.normal0s[smallest_t_indecies[ray_hits]].shape: {self.normal0s[smallest_t_indecies[ray_hits]].shape}")

        hit_normals[ray_hits] = (
            self.normal0s[smallest_t_indecies[ray_hits]] * (1.0 - Us[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis] - Vs[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis])
            + self.normal1s[smallest_t_indecies[ray_hits]] * Us[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis]
            + self.normal2s[smallest_t_indecies[ray_hits]] * Vs[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis]
        )

        hit_normals[ray_hits] /= numpy.linalg.norm(hit_normals[ray_hits], axis=1)[:, numpy.newaxis]

        # print("Us:")
        # print(Us)
        # print("Vs:")
        # print(Vs)

        hit_uvs = numpy.zeros((num_rays, 2), dtype=numpy.single)
        hit_uvs[ray_hits] = (
            self.uv0s[smallest_t_indecies[ray_hits]] * (1.0 - Us[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis] - Vs[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis])
            + self.uv1s[smallest_t_indecies[ray_hits]] * Us[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis]
            + self.uv2s[smallest_t_indecies[ray_hits]] * Vs[ray_hits, smallest_t_indecies[ray_hits], numpy.newaxis]
        )

        # back_facing = numpy.full(num_rays, False)
        back_facing = dets[numpy.arange(num_rays), smallest_t_indecies] < 0.0
        # back_facing[sphere_hits] = dets[numpy.arange(Ts.shape[0]), smallest_t_indecies] < 0.0
        hit_normals[back_facing] *= -1.0

        hit_material_ids = numpy.full((num_rays), self.material_id, dtype=numpy.ubyte)

        return RayResults(
            ray_hits,
            final_ts,
            hit_points,
            hit_normals,
            hit_uvs,
            hit_material_ids,
            back_facing
        )