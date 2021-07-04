import math

import numpy

from . import renderable


class MTTriangleGroup():
    def __init__(self):
        self.pt0s = None
        self.pt1s = None
        self.pt2s = None
        self.normals = None
        self.As = None
        self.Bs = None
        self.materials = []
        self.bounds_centre = None
        self.bounds_radius = None

    def add_triangle(self, pt0, pt1, pt2, material):
        """
        Add a triangle to the group
        """

        A = pt1 - pt0
        B = pt2 - pt0
        normal = numpy.cross(A, B)
        normal /= numpy.linalg.norm(normal)

        self.materials.append(material)

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

    def hits_bounding_sphere(self, ray, t_min, t_max):
        C_to_O = ray.origin - self.bounds_centre

        H = ray.direction.dot(C_to_O)
        C = C_to_O.dot(C_to_O) - self.bounds_radius**2
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
            sqrt_discriminant = math.sqrt(discriminant)
            t = -H - sqrt_discriminant
            if t < t_min or t > t_max:
                t = -H + sqrt_discriminant
                if t < t_min or t > t_max:
                    # Neither root was suitable.
                    return False, None

            # Ray hit sphere
            return True

    def hit_test(self, ray, t_min, t_max):
        """
        See if the ray hits this group of triangles
        """
        #Bail if we don;t hit the bounding sphere
        if not self.hits_bounding_sphere(ray, t_min, t_max):
            return False, None

        p_vecs = numpy.cross(ray.direction, self.Bs)
        dets = numpy.einsum("ij,ij->i", self.As, p_vecs)

        # Indexes of triangles which are not parallel to ray.
        valid_tri_idxs = numpy.absolute(dets) > 0.00001

        # Bail if there are now no valid triangles
        if not valid_tri_idxs.any():
            return False, None

        inv_dets = numpy.zeros_like(dets)
        inv_dets[valid_tri_idxs] = 1.0/dets[valid_tri_idxs]

        t_vecs = ray.origin - self.pt0s

        Us = numpy.zeros_like(dets)
        Us[valid_tri_idxs] = numpy.einsum(
            "ij,ij->i",
            t_vecs[valid_tri_idxs],
            p_vecs[valid_tri_idxs]
        ) * inv_dets[valid_tri_idxs]
        valid_tri_idxs = numpy.logical_and(
            valid_tri_idxs,
            numpy.logical_not(numpy.logical_or(Us > 1.0, Us < 0))
        )

        # Bail if there are now no valid triangles
        if not valid_tri_idxs.any():
            return False, None

        q_vecs = numpy.cross(t_vecs, self.As)
        Vs = numpy.zeros_like(dets)
        Vs[valid_tri_idxs] = numpy.einsum(
            "ij,j->i",
            q_vecs[valid_tri_idxs],
            ray.direction
        ) * inv_dets[valid_tri_idxs]
        valid_tri_idxs = numpy.logical_and(
            valid_tri_idxs,
            numpy.logical_not(numpy.logical_or(Vs < 0.0, (Us + Vs) > 1.0))
        )

        # Bail if there are now no valid triangles
        if not valid_tri_idxs.any():
            return False, None

        Ts = numpy.full_like(dets, t_max + 1)
        Ts[valid_tri_idxs] = numpy.einsum(
            "ij,ij->i",
            self.Bs[valid_tri_idxs],
            q_vecs[valid_tri_idxs]
        ) * inv_dets[valid_tri_idxs]

        valid_t_idxs = numpy.asarray(
            numpy.logical_not(numpy.logical_or(Ts < t_min, Ts > t_max))
        ).nonzero()[0]
        if valid_t_idxs.size > 0:
            smallest_t_index = valid_t_idxs[Ts[valid_t_idxs].argmin()]
        else:
            # T was not within range
            return False, None

        t = Ts[smallest_t_index]
        hit_point = ray.at(t)
        side = renderable.Side.FRONT
        normal = self.normals[smallest_t_index]
        # Check if the triangle is back facing
        if dets[smallest_t_index] < 0:
            side = renderable.Side.BACK
            normal = self.normals[smallest_t_index] * -1.0

        return True, renderable.HitRecord(
            hit_point,
            normal,
            t,
            side,
            self.materials[smallest_t_index]
        )
