import math

import numpy

from .ray import Ray


class Camera():
    """
    Represents the camera in the scene.
    """

    def __init__(self, position, lookat, aspect_ratio, horizontal_fov):
        """
        Initialise the camera.

        Args:
            position (numpy.array): The position in space of the camera.
            lookat (numpy.array): The position on space for the camera
                to look at.
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
        W = W_dir / numpy.linalg.norm(W_dir)

        # To get U, we can cross W with the up vector.
        U = numpy.cross(numpy.array([0.0, 1.0, 0.0]), W)

        # Finally V is W cross U
        V = numpy.cross(W, U)

        self.viewport_horizontal = viewport_width * U
        self.viewport_vertical = viewport_height * V
        self.camera_pos = position
        self.bottomleft_focalplane_pos = (
            # Start at camera position
            self.camera_pos

            # Move out to the focal plane in -W, this puts us in the centre
            # of the focal plane
            - W

            # Move to the bottom of the focalplane
            - self.viewport_vertical * 0.5

            # Move to the left of the focalplane
            - self.viewport_horizontal * 0.5
        )

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

        pt_on_viewport = (
            self.bottomleft_focalplane_pos
            + self.viewport_horizontal * horizontal
            + self.viewport_vertical * vertical
        )
        ray_direction = pt_on_viewport - self.camera_pos

        return Ray(self.camera_pos, ray_direction)
