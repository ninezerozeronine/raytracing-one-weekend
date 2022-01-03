"""
A circular disk
"""

import numpy

from .ray_results import RayResults


class Disk():
    """
    A circular disk

    https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-plane-and-ray-disk-intersection
    """
    def __init__(self, centre, radius, normal, material_index, up=None):
        """

        """

        if up is None:
            self.up = numpy.array([0,1,0], dtype=numpy.single)
        else:
            self.up = numpy.array(up, dtype=numpy.single)
            self.up = self.up / numpy.linalg.norm(self.up)
            self.up = self.up.astype(numpy.single)

        self.centre = centre
        self.radius = radius
        self.radius_squared = radius ** 2
        self.normal = normal / numpy.linalg.norm(normal)
        self.normal = self.normal.astype(numpy.single)
        self.material_index = material_index

        # Construct an orthonormal set of vectors that describe the
        # coordinate system of the disk: U, V and W.
        # - W is the normal of the disk
        # - U points to the right of the disk, when looking from the
        #       direction the normal points in
        # - V points to the top of the disk, when looking from the
        #       direction the normal points in

        # To get U, we can cross the up vector with W (the normal).
        self.U = numpy.cross(self.up, self.normal)
        self.U = self.U / numpy.linalg.norm(self.U)

        # V is W (the normal) cross U
        self.V = numpy.cross(self.normal, self.U)


    def get_hits(self, ray_origins, ray_dirs, t_min, t_max):
 
        num_rays = ray_origins.shape[0]

        D_dot_ns = numpy.einsum("ij,j->i", ray_dirs, self.normal)

        C_minus_Os = self.centre - ray_origins

        C_minus_Os_dot_ns = numpy.einsum("ij,j->i", C_minus_Os, self.normal)

        hits = numpy.abs(D_dot_ns) > 0.0001

        ts = numpy.full(num_rays, t_max + 1.0, dtype=numpy.single)
        ts[hits] = C_minus_Os_dot_ns[hits]/D_dot_ns[hits]

        hits = hits & (ts > t_min) & (ts < t_max)

        hit_pts = ray_origins + ray_dirs * ts[:, numpy.newaxis]

        pts_minus_C = hit_pts - self.centre
        lengths_squared = numpy.einsum("ij,ij->i", pts_minus_C, pts_minus_C)
        hits = hits & (lengths_squared < self.radius_squared)

        hit_normals = numpy.full((num_rays, 3), self.normal, dtype=numpy.single)

        # Use the dot product to project the vector from the centre to
        # the hit point onto the U/V vector to see how much U or V makes
        # up that position.
        centre_to_pt = hit_pts - self.centre
        centre_to_pt = centre_to_pt.astype(numpy.single)
        # Also normalise the U/V amount based on disk radius
        U_components = numpy.einsum("ij,j->i", centre_to_pt, self.U / self.radius)
        V_components = numpy.einsum("ij,j->i", centre_to_pt, self.V / self.radius)
        # Offset the UV coord so 0.5, 0.5 is at the centre of the disk
        U_components = (U_components + 1.0) / 2.0
        V_components = (V_components + 1.0) / 2.0
        hit_uvs = numpy.column_stack((U_components, V_components))

        hit_material_indecies = numpy.full(num_rays, self.material_index, dtype=numpy.ubyte)
        
        cosines = numpy.einsum("ij,j->i", ray_dirs, self.normal)
        back_facing = cosines > 0.0
        # Flip any normals that were back facing so the normal always
        # faces the ray origin
        hit_normals[back_facing] *= -1.0

        return RayResults(
            hits,
            ts,
            hit_pts,
            hit_normals,
            hit_uvs,
            hit_material_indecies,
            back_facing
        )