"""
Main entry point for renderer functionality
"""

from PIL import Image, ImageDraw

from .vec3 import Vec3
from .ray import Ray

IMG_HEIGHT = 90
IMG_WIDTH = 160
HORIZON_COLOUR = Vec3(1, 1, 1)
SKY_COLOUR = Vec3(0.5, 0.7, 1)


def generate_test_image():
    """
    Generate a test image.
    """

    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(image)

    for row in range(IMG_HEIGHT):
        for column in range(IMG_WIDTH):
            r = int(255 * (column / (IMG_WIDTH - 1)))
            g = int(255 * (row / (IMG_HEIGHT - 1)))
            b = int(255 * 0.25)
            draw.point((column, row), fill=(r, g, b))

    image.show()


def generate_image_from_data(img_data):
    """

    """

    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(image)

    for row in range(IMG_HEIGHT):
        for column in range(IMG_WIDTH):
            # Flip row to account for image data having 0, 0 at bottom
            # left, rather than top left in a PIL image.
            flipped_row = IMG_HEIGHT - 1 - row
            draw.point(
                (column, row),
                fill=(
                    int(255 * img_data[(column, flipped_row)].r),
                    int(255 * img_data[(column, flipped_row)].g),
                    int(255 * img_data[(column, flipped_row)].b),
                )
            )

    image.show()
    image.save("tmp_image.png")


def render():
    viewport_width = 2.0
    viewport_height = viewport_width * (IMG_HEIGHT/IMG_WIDTH)
    viewport_vertical = Vec3(0, viewport_height, 0)
    viewport_horizontal = Vec3(viewport_width, 0, 0)
    focalplane_dist = 1.0

    camera_pos = Vec3(0, 0, 0)
    bottomleft_focalplane_pos = (
        # Start at camera position
        camera_pos

        # The camera is looking in -Z, this way X is to the right and Y
        # is up like a typical X/Y graph.
        # Move out to the focal plane in -Z, this puts us in the centre
        # of the focal plane
        + Vec3(0, 0, (focalplane_dist * -1))

        # Move to the bottom of the focalplane
        + viewport_vertical * -0.5

        # Move to the left of the focalplane
        + viewport_horizontal * -0.5
    )

    img_data = {}

    for y_index in range(IMG_HEIGHT):
        y_progress = y_index/IMG_HEIGHT
        for x_index in range(IMG_WIDTH):
            x_progress = x_index/IMG_WIDTH
            pt_on_viewport = (
                bottomleft_focalplane_pos
                + viewport_vertical * y_progress
                + viewport_horizontal * x_progress
            )
            ray_direction = pt_on_viewport - camera_pos
            pixel_ray = Ray(camera_pos, ray_direction)
            colour = get_ray_colour(pixel_ray)
            img_data[(x_index, y_index)] = colour

    return img_data


def get_ray_colour(ray):
    """
    Given a ray, get the colour from the scene
    """

    normalised_ray = ray.direction.normalised()

    # Y component will now be somewhere between -1 and 1. Map it into
    # a 0 to 1 range.
    t = 0.5 * (normalised_ray.y + 1)

    # Lerp between white and blue based on mapped Y
    return (1.0 - t) * HORIZON_COLOUR + t * SKY_COLOUR


def main():
    print("Start render")
    img_data = render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
