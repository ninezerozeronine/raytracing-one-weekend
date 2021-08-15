import math

import numpy

from .ray import Ray

RNG = numpy.random.default_rng()


class Camera():
    """
    Represents the camera in the scene.
    """

    def __init__(self, position, lookat, focus_dist, aperture, aspect_ratio, horizontal_fov):
        """
        Initialise the camera.

        Args:
            position (numpy.array): The position in space of the camera.
            lookat (numpy.array): The position on space for the camera
                to look at.
            focus_dist (float): The distance (along the line between the
                camera position and lookat position) to the focal plane
                (where in space the image is sharp).
            apeture (float): The size of the aperture of the virtual
                lens. It's the distance the origin of the ray can be
                peturbed (perpendicular to the ray direction) from the
                camera position.
                An aperture of zero gives a perfectly sharp image,
                and if this is the case, the focal distance is somewhat
                irrelevant, it can be set to 1.
            aspect_ratio (float): The aspect ratio (width/height) of the
                image to be rendered.
            horizontal_fov (float): The horizontal field of view in
                degrees.
        """

        # Good old So/h Ca/h To/a :)
        viewport_width = math.tan(math.radians(horizontal_fov/2.0)) * 2.0
        viewport_height = viewport_width/aspect_ratio
        focalplane_dist = 1.0

        # We build an orthonormal set of vectors that describe the
        # camera: U, V and W. U is to screen right, V is to screen top,
        # and W the the camera ooks in -W.

        # We know that the camera is looking in -W so calculate that
        # first.
        W_dir = position - lookat
        self.W = W_dir / numpy.sqrt(W_dir.dot(W_dir))

        # To get U, we can cross W with the up vector.
        self.U = numpy.cross(numpy.array([0.0, 1.0, 0.0]), self.W)

        # Finally V is W cross U
        self.V = numpy.cross(self.W, self.U)

        # Note that we're multiplying all these measurements by the focus 
        # distance to scale out the viewport to the point in space where
        # the focal plane should be.
        self.viewport_horizontal = viewport_width * self.U * focus_dist
        self.viewport_vertical = viewport_height * self.V * focus_dist
        self.camera_pos = position
        self.bottomleft_focalplane_pos = (
            # Start at camera position
            self.camera_pos

            # Move out to the focal plane in -W, this puts us in the centre
            # of the focal plane
            - self.W * focus_dist

            # Move to the bottom of the focalplane
            - self.viewport_vertical * 0.5

            # Move to the left of the focalplane
            - self.viewport_horizontal * 0.5
        )

        self.lens_radius = aperture / 2

    def get_ray(self, horizontal, vertical):
        """
        Get a render ray from the camera

        Args:
            horizontal (float): How far from the left of the
                focal plane the ray should be. The left side is 0,
                the right side is 1.
            vertical (float): How far from the bottom of the
                focal plane the ray should be. The bottom side is 0,
                the top side is 1.
        """

        # Calculate how far the origin of the ray will be peturbed from
        # The camera position.
        offset_amount = self.lens_radius * point_in_unit_xy_disk()
        offset_vec = self.U * offset_amount[0] + self.V * offset_amount[1]

        pt_on_viewport = (
            self.bottomleft_focalplane_pos
            + self.viewport_horizontal * horizontal
            + self.viewport_vertical * vertical
        )
        ray_direction = pt_on_viewport - (self.camera_pos + offset_vec)

        return Ray((self.camera_pos + offset_vec), ray_direction)

    def get_ray_components(self, width, height, samples):
        """

        """

        # Turn width, height and samples into a list of 0-1 positions
        # on the focal plane.
        #
        # Create an array w x h x samples
        # Set w and h to be the index on that axis
        # Add a [0-1) value to x and y for the sample offset
        # Divide X values by width and Y values be height

        pixel_positions = numpy.zeros(
            (width, height, samples, 3), dtype=numpy.single
        )
        for x_coord in range(width):
            pixel_positions[x_coord,...,0] = x_coord

        for y_coord in range(height):
            pixel_positions[:,y_coord,...,1] = y_coord

        sample_offsets = RNG.uniform(low=0.0, high=1.0, size=(width, height, samples, 3))
        sample_offsets = sample_offsets.astype(numpy.single)
        pixel_positions += sample_offsets

        pixel_positions[...,0] /= width
        pixel_positions[...,1] /= height
        viewport_percentages = pixel_positions

        # Create an array of offests in an xy disk for every sample.
        xy_disk_coords = RNG.uniform(low=-1.0, high=1.0, size=(width, height, samples, 3))
        flattened = xy_disk_coords.reshape(-1, 3)
        flattened[..., 2] = 0
        while True:
            dots = numpy.einsum("ij,ij->i", flattened, flattened)
            if not numpy.any(dots > 1.0):
                break
            new_coords = RNG.uniform(low=-1.0, high=1.0, size=(width, height, samples, 3))
            new_flattened = new_coords.reshape(-1, 3)
            new_flattened[..., 2] = 0
            flattened[dots > 1.0] = new_flattened[dots > 1.0]
        xy_disk_coords = flattened.reshape(width, height, samples, 3)

        # Scale the disk offest vecs by lens radius
        offset_vecs = self.lens_radius * xy_disk_coords

        # Orient the offsets to the wat the camera is facing
        offset_vecs_oriented = (
            offset_vecs[..., 0, numpy.newaxis] * self.U
            + offset_vecs[..., 1, numpy.newaxis] * self.V
        )

        # Turn the viewport percentages into positions in space on the viewport
        pts_on_viewport = (
            self.bottomleft_focalplane_pos
            + self.viewport_horizontal * viewport_percentages[..., 0, numpy.newaxis]
            + self.viewport_vertical * viewport_percentages[..., 1, numpy.newaxis]
        )

        ray_origins = offset_vecs_oriented + self.camera_pos
        ray_dirs = pts_on_viewport - ray_origins
        ray_dirs /= numpy.sqrt(numpy.einsum("...ij,...ij->...i", ray_dirs, ray_dirs))[..., numpy.newaxis]
        # ray_dirs /= numpy.sqrt((ray_dirs ** 2).sum(axis=-1))[..., numpy.newaxis]


        # pts_on_viewport = (
        #     self.bottomleft_focalplane_pos
        #     + self.viewport_horizontal * viewport_percentages[..., 0, numpy.newaxis]
        #     + self.viewport_vertical * viewport_percentages[..., 1, numpy.newaxis]
        # )

        # ray_origins = numpy.zeros(
        #     (width, height, samples, 3), dtype=numpy.single
        # )
        # ray_origins[..., :] = self.camera_pos
        # ray_dirs = pts_on_viewport - self.camera_pos
        # # ray_dirs /= numpy.sqrt((ray_dirs ** 2).sum(axis=-1))[..., numpy.newaxis]
        # ray_dirs /= numpy.sqrt(numpy.einsum("...ij,...ij->...i", ray_dirs, ray_dirs))[..., numpy.newaxis]

        return ray_origins, ray_dirs

def point_in_unit_xy_disk():
    """
    Get a point in a disk in the XY plane with radius 1.

    Returns:
        numpy.array: Point in the disk on the XY plane
    """
    while True:
        random_vec = RNG.uniform(low=-1, high=1, size=3)
        random_vec[2] = 0.0
        # If the length of the vector squared (thanks dot product of
        # a vector with itself!) is greater than 1 then we're not in
        # a unit sphere.
        if random_vec.dot(random_vec) > 1:
            continue
        else:
            return random_vec