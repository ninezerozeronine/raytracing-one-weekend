"""
Main entry point for renderer functionality
"""
import math

from PIL import Image, ImageDraw

from .vec3 import Vec3
from .ray import Ray
from .renderable import World
from .sphere import Sphere

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
    Write out an image to disk.
    """

    image = Image.new("RGB", (IMG_WIDTH, IMG_HEIGHT))
    draw = ImageDraw.Draw(image)

    for row in range(IMG_HEIGHT):
        for column in range(IMG_WIDTH):
            # Flip row to account for image data having 0, 0 at bottom
            # left, rather than top left in a PIL image.
            flipped_row = IMG_HEIGHT - 1 - row
            if (column, flipped_row) in img_data:
                draw.point(
                    (column, row),
                    fill=(
                        int(255 * img_data[(column, flipped_row)].r),
                        int(255 * img_data[(column, flipped_row)].g),
                        int(255 * img_data[(column, flipped_row)].b),
                    )
                )
            else:
                draw.point(
                    (column, row),
                    fill=(255, 255, 255)
                )

    image.show()
    image.save("tmp_image.png")


def render():
    """
    Do the rendering of the image.
    """

    # Camera setup
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

    # World setup
    world = World()
    world.renderables.append(Sphere(Vec3(-3, 0, -7), 3))
    world.renderables.append(Sphere(Vec3(0, 0, -10), 3))
    world.renderables.append(Sphere(Vec3(3, 0, -13), 3))
    world.renderables.append(Sphere(Vec3(6, 0, -17), 3))

    img_data = {}
    pixel_coords = (
        (x, y) for y in range(IMG_HEIGHT) for x in range(IMG_WIDTH)
    )
    # pixel_coords = (
    #     (50, 45),
    #     (80, 75),
    #     (110, 45),
    #     (140, 45),
    #     (80, 15),
    #     (80, 45),
    # )

    for x_coord, y_coord in pixel_coords:
        x_progress = x_coord/IMG_WIDTH
        y_progress = y_coord/IMG_HEIGHT

        pt_on_viewport = (
            bottomleft_focalplane_pos
            + viewport_vertical * y_progress
            + viewport_horizontal * x_progress
        )
        ray_direction = pt_on_viewport - camera_pos
        pixel_ray = Ray(camera_pos, ray_direction)
        colour = get_ray_colour(pixel_ray, world)
        img_data[(x_coord, y_coord)] = colour

    return img_data


def get_ray_colour(ray, world):
    """
    Given a ray, get the colour from the scene
    """

    hit, hit_record = world.hit(ray, 0, 5000)
    if hit:
        return normal_to_rgb(hit_record.normal)
    else:
        normalised_ray = ray.direction.normalised()

        # Y component will now be somewhere between -1 and 1. Map it into
        # a 0 to 1 range.
        t = 0.5 * (normalised_ray.y + 1)

        # Lerp between white and blue based on mapped Y
        return (1.0 - t) * HORIZON_COLOUR + t * SKY_COLOUR


def normal_to_rgb(normal):
    """
    Convert a normal to an rgb colour.

    Expects unit length normal.
    """

    return Vec3(
            normal.x + 1,
            normal.y + 1,
            normal.z + 1,
        ) * 0.5


def normal_to_discrete_rgb(normal):
    """
    Given a normal, return a colour based on whether it's close
    to an axis.

    E.g. if the normal is approximately +X, the colour is red, +Y the
    colour is green.

    Expects unit length normal.
    """

    axis_colour_pairs = [
        # +X : Red
        (Vec3(1, 0, 0), Vec3(1, 0, 0)),

        # +Y : Green
        (Vec3(0, 1, 0), Vec3(0, 1, 0)),

        # +Z : Blue
        (Vec3(0, 0, 1), Vec3(0, 0, 1)),

        # -X : Pink
        (Vec3(-1, 0, 0), Vec3(1, 0, 1)),

        # -Y : Yellow
        (Vec3(0, -1, 0), Vec3(1, 1, 0)),

        # -Z : Cyan
        (Vec3(0, 0, -1), Vec3(0, 1, 1)),
    ]

    for axis, colour in axis_colour_pairs:
        cos_angle = axis.dot(normal)
        if cos_angle > 0.8:
            return colour
    else:
        return Vec3(0, 0, 0)


def main():
    print("Start render")
    img_data = render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
