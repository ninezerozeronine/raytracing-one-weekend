"""
Main entry point for renderer functionality
"""
import math
from random import random

from PIL import Image, ImageDraw
import numpy

from .ray import Ray
from .renderable import World
from .sphere import Sphere
from .camera import Camera

IMG_HEIGHT = 90
IMG_WIDTH = 160
PIXEL_SAMPLES = 50
HORIZON_COLOUR = numpy.array([1.0, 1.0, 1.0])
SKY_COLOUR = numpy.array([0.5, 0.7, 1.0])
RNG = numpy.random.default_rng()
MAX_DEPTH = 50


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
                        int(255 * img_data[(column, flipped_row)][0]),
                        int(255 * img_data[(column, flipped_row)][1]),
                        int(255 * img_data[(column, flipped_row)][2]),
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

    camera = Camera(IMG_WIDTH/IMG_HEIGHT)

    # World setup
    world = World()
    # world.renderables.append(Sphere(numpy.array([-3.0, 0.0, -7.0]), 3.0))
    world.renderables.append(Sphere(numpy.array([0.0, 0.0, -10.0]), 3.0))
    # world.renderables.append(Sphere(numpy.array([3.0, 0.0, -13.0]), 3.0))
    # world.renderables.append(Sphere(numpy.array([6.0, 0.0, -17.0]), 3.0))
    world.renderables.append(Sphere(numpy.array([0.0, -103.0, -10.0]), 100.0))

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
        print(f"Rendering pixel at ({x_coord}, {y_coord})")
        total_colour = numpy.array([0.0, 0.0, 0.0])
        for _ in range(PIXEL_SAMPLES):
            x_progress = (x_coord + random()) / IMG_WIDTH
            y_progress = (y_coord + random()) / IMG_HEIGHT
            ray = camera.get_ray(x_progress, y_progress)
            total_colour += get_ray_colour(ray, world, MAX_DEPTH)
            # The squareroot is for a 2.0 gamma correction
        img_data[(x_coord, y_coord)] = numpy.sqrt(total_colour/PIXEL_SAMPLES)

    return img_data


def get_ray_colour(ray, world, depth):
    """
    Given a ray, get the colour from the scene
    """

    if depth <= 0:
        return numpy.array([0.0, 0.0, 0.0])

    hit, hit_record = world.hit(ray, 0.0, 5000.0)
    if hit:
        # return normal_to_rgb(hit_record.normal)
        dir_target = (
            hit_record.hit_point
            + hit_record.normal
            + random_vec_in_unit_sphere()
        )

        bounce_ray = Ray(
            hit_record.hit_point,
            dir_target - hit_record.hit_point
        )
        return 0.5 * get_ray_colour(bounce_ray, world, depth - 1)

    else:
        # Y component is somewhere between -1 and 1. Map it into
        # a 0 to 1 range.
        t = 0.5 * (ray.direction[1] + 1.0)

        # Lerp between white and blue based on mapped Y
        return (1.0 - t) * HORIZON_COLOUR + t * SKY_COLOUR


def random_vec_in_unit_sphere():
    """
    Generate a vector in a sphere with radius 1.
    """
    while True:
        random_vec = RNG.uniform(low=-1, high=1, size=3)
        # If the length of the vector squared (thanks dot product of
        # a vector with itself!) is greater than 1 then we're not in
        # a unit sphere.
        if random_vec.dot(random_vec) > 1:
            continue
        else:
            return random_vec

def normal_to_rgb(normal):
    """
    Convert a normal to an rgb colour.

    Expects unit length normal.
    """

    return numpy.array([
            normal[0] + 1.0,
            normal[1] + 1.0,
            normal[2] + 1.0,
        ]) * 0.5


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
        (numpy.array([1.0, 0.0, 0.0]), numpy.array([1.0, 0.0, 0.0])),

        # +Y : Green
        (numpy.array([0.0, 1.0, 0.0]), numpy.array([0.0, 1.0, 0.0])),

        # +Z : Blue
        (numpy.array([0.0, 0.0, 1.0]), numpy.array([0.0, 0.0, 1.0])),

        # -X : Pink
        (numpy.array([-1.0, 0.0, 0.0]), numpy.array([1.0, 0.0, 1.0])),

        # -Y : Yellow
        (numpy.array([0.0, -1.0, 0.0]), numpy.array([1.0, 1.0, 0.0])),

        # -Z : Cyan
        (numpy.array([0.0, 0.0, -1.0]), numpy.array([0.0, 1.0, 1.0])),
    ]

    for axis, colour in axis_colour_pairs:
        cos_angle = axis.dot(normal)
        if cos_angle > 0.8:
            return colour
    else:
        return numpy.array([0.0, 0.0, 0.0])


def main():
    print("Start render")
    img_data = render()
    print("End render")
    generate_image_from_data(img_data)


if __name__ == "__main__":
    main()
