from .vec3 import Vec3
from .ray import Ray


class Camera():
    """
    Represents the camera in the scene.
    """

    def __init__(self, aspect_ratio):
        """
        Initialise the camera.

        Args:
            aspect_ratio (float): The aspect ratio (width/height) of the
                image to be rendered.
        """

        viewport_width = 2.0
        focalplane_dist = 1.0

        self.viewport_horizontal = Vec3(viewport_width, 0, 0)
        self.viewport_vertical = Vec3(0, viewport_width/aspect_ratio, 0)
        self.camera_pos = Vec3(0, 0, 0)
        self.bottomleft_focalplane_pos = (
            # Start at camera position
            self.camera_pos

            # The camera is looking in -Z, this way X is to the right and Y
            # is up like a typical X/Y graph.
            # Move out to the focal plane in -Z, this puts us in the centre
            # of the focal plane
            + Vec3(0, 0, (focalplane_dist * -1))

            # Move to the bottom of the focalplane
            + self.viewport_vertical * -0.5

            # Move to the left of the focalplane
            + self.viewport_horizontal * -0.5
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
